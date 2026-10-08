"""Automatic circuit-and-head study across datasets, run unattended.

    python -m elements.experiments.autostudy run      # stages A -> B -> C, then report
    python -m elements.experiments.autostudy plan     # print every stage's plan
    python -m elements.experiments.autostudy report   # rebuild the report from disk

Three stages per task (a dataset x supervised / novelty), each a normal
config under ``configs/auto/`` run by ``elements.run`` (resumable):

A  circuit screen  every circuit in ``CIRCUITS`` with a cheap head pair, next
                   to its classical controls (dials, rff of equal width, raw).
B  head scaling    the circuit A picked (and its no-memory twin) under heads
                   from linear to a 2-layer MLP: how big must the classical
                   part be, relative to the quantum readout, per task?
C  seeds           the picked circuit and the controls under the head B found
                   best, on two more seeds (the split changes with the seed).

Selection is automatic and fixed in advance (``primary``): supervised = mean
macro-F1 over the label schemes under ``mlp256``; novelty = macro detection at a
1% false-alarm budget under the ``ae256`` detector. The circuit with the best
quantum score goes on; "gain" (quantum minus the best control under the same
head) is reported but not selected on, so the pick is not tuned on the controls.
Everything is one seed until stage C.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from elements.config import REPO_ROOT

CFG_DIR = REPO_ROOT / "configs" / "auto"
RUNS = REPO_ROOT / "runs"
STATE = RUNS / "autostudy_state.json"
REPORT = REPO_ROOT / "docs" / "findings_autostudy.md"
PREFIX = "auto"  # "smoke" under --smoke

ENCODING = {"scheme": "tanh", "width": 3.0}
SPLIT = {"test_size": 0.30, "val_frac": 0.10}

#: task name -> (task kind, data block)
TASKS = {
    "nsl_sup": ("supervised", {"dataset": "nslkdd", "schemes": ["families6", "all23"]}),
    "nsl_nov": ("novelty", {"dataset": "nslkdd"}),
    "unsw_sup": ("supervised", {"dataset": "nf_unsw_nb15_v2", "dedup": True,
                                "schemes": ["binary", "attack10"]}),
    "unsw_nov": ("novelty", {"dataset": "nf_unsw_nb15_v2", "dedup": True}),
    "cic_sup": ("supervised", {"dataset": "nf_cse_cic_ids2018_v2", "dedup": True,
                               "max_rows": 200000, "schemes": ["binary", "family7"]}),
    "cic_nov": ("novelty", {"dataset": "nf_cse_cic_ids2018_v2", "dedup": True,
                            "max_rows": 200000}),
}


def _ion(**kw):
    return {"kind": "iontrap", "n_in": 4, "n_mem": 4, "steps": 8, "dt": 0.3,
            "topology": "all_to_all", **kw}


#: Stage A circuits: one knob moved at a time from the 4+4 L8 baseline.
CIRCUITS = [
    _ion(),                                   # baseline
    _ion(dt=0.15), _ion(dt=0.6),              # Trotter step: weak / strong mixing
    _ion(j_scale=2.0),                        # stronger couplings
    _ion(topology="ring"),                    # nearest-neighbour instead of all-to-all
    _ion(n_mem=2), _ion(n_mem=6),             # less / more memory (6 / 10 qubits)
    _ion(n_in=6),                             # wider input register (10 qubits)
    _ion(n_in=2, n_mem=6, steps=16),          # narrow input, long sequence
    _ion(steps=16),                           # twice the rounds and readout
    _ion(memory=False),                       # no-memory control of the baseline
    {"photons": 3, "modes": 12},              # photonic references
    {"photons": 3, "modes": 20},
]

HEADS_A = {
    "supervised": [{"kind": "lbfgs", "l2": 1.0e-4, "max_iter": 200},
                   {"kind": "mlp", "hidden": [256]}],
    "novelty": [{"kind": "pca", "n_components": 32},
                {"kind": "ae", "hidden": [256], "bottleneck": 32}],
}
HEADS_B = {
    "supervised": [{"kind": "lbfgs", "l2": 1.0e-4, "max_iter": 200}]
    + [{"kind": "mlp", "hidden": h} for h in ([32], [128], [512], [2048], [512, 512])],
    "novelty": [{"kind": "pca", "n_components": 32}]
    + [{"kind": "ae", "hidden": h, "bottleneck": 32} for h in ([64], [256], [1024], [512, 512])],
}
PRIMARY_HEAD = {"supervised": "mlp256", "novelty": "ae256-b32"}
METRIC = {"supervised": "macro_f1", "novelty": "macro_det@0.01"}
SEEDS_C = [43, 44]


#: Stage K: standalone classical models on the raw features, same split and
#: seed as stage A -- the classical reference every plot draws.
CLASSICAL = {
    "supervised": [{"kind": "logreg"}, {"kind": "knn", "n_neighbors": 5},
                   {"kind": "rf", "n_estimators": 300},
                   {"kind": "extratrees", "n_estimators": 300},
                   {"kind": "xgb", "n_estimators": 300},
                   {"kind": "mlp", "hidden": [512, 512]}],
    "novelty": [{"kind": "iforest"}, {"kind": "ocsvm", "nu": 0.05, "max_train_rows": 10000},
                {"kind": "lof", "n_neighbors": 20}, {"kind": "knndist", "n_neighbors": 5},
                {"kind": "gmm", "n_components": 8},
                {"kind": "ae", "hidden": [512], "bottleneck": 32}],
}
#: Earlier runs that already hold stage K for a task (same split and seed).
CLASSICAL_DONE = {"nsl_sup": "classical_supervised", "nsl_nov": "classical_novelty",
                  "unsw_sup": "nfunsw_dedup_classical"}


# ------------------------------------------------------------------- configs


def _tag(res: dict) -> str:
    from elements.config import reservoir_from_dict

    return reservoir_from_dict(res).tag


def _config(name, task, reservoirs, heads, seed=42, arms=("quantum", "dials", "rff", "raw"),
            baseline=None):
    kind, data = TASKS[task]
    cfg = {"name": name, "task": kind, "seed": seed, "data": data, "split": SPLIT,
           "encoding": ENCODING, "arms": list(arms),
           "grid": {"reservoirs": reservoirs, "heads": heads},
           "device": "auto", "out": "runs",
           "notes": "generated by elements.experiments.autostudy"}
    if baseline:
        cfg["baseline_heads"] = baseline
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{name}.yaml"
    path.write_text("# generated by elements.experiments.autostudy -- do not edit\n"
                    + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def classical_config(task):
    kind, data = TASKS[task]
    cfg = {"name": f"{PREFIX}_K_{task}", "task": kind, "seed": 42, "data": data,
           "split": SPLIT, "arms": ["raw"], "models": CLASSICAL[kind],
           "device": "auto", "out": "runs",
           "notes": "generated by elements.experiments.autostudy (classical reference)"}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{cfg['name']}.yaml"
    path.write_text("# generated by elements.experiments.autostudy -- do not edit\n"
                    + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def classical_run(task: str) -> str:
    return CLASSICAL_DONE.get(task, f"{PREFIX}_K_{task}")


def stage_a_configs():
    out = []
    for task, (kind, _) in TASKS.items():
        base = [{"kind": "iforest", "arms": ["raw"]}] if kind == "novelty" else None
        out.append(_config(f"{PREFIX}_A_{task}", task, CIRCUITS, HEADS_A[kind], baseline=base))
    return out


def stage_b_config(task, pick):
    kind = TASKS[task][0]
    res = [pick] + ([{**pick, "memory": False}] if pick.get("kind") == "iontrap"
                    and pick.get("memory", True) else [])
    return _config(f"{PREFIX}_B_{task}", task, res, HEADS_B[kind])


def stage_c_configs(task, pick, head):
    kind = TASKS[task][0]
    res = [pick] + ([{**pick, "memory": False}] if pick.get("kind") == "iontrap"
                    and pick.get("memory", True) else [])
    spec = next(h for h in HEADS_B[kind] + HEADS_A[kind] if _head_label(h) == head)
    return [_config(f"{PREFIX}_C_{task}_s{s}", task, res, [spec], seed=s) for s in SEEDS_C]


def _head_label(h: dict) -> str:
    from elements.config import _build, HeadSpec

    return _build(HeadSpec, h).label


# ------------------------------------------------------------------- results


def load_results(name: str) -> pd.DataFrame:
    f = RUNS / name / "results.jsonl"
    if not f.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in f.open(encoding="utf-8")])


def _score(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One row per (config, arm, feature_id, head): the primary metric,
    averaged over label schemes for supervised tasks."""
    m = METRIC[kind]
    g = df.groupby(["config", "arm", "feature_id", "head"], as_index=False)
    return g.agg(score=(m, "mean"), feature_dim=("feature_dim", "first"),
                 n_params=("n_params", "first"))


def screen(task: str) -> pd.DataFrame:
    """Stage A table: per circuit, quantum score and gain over its controls.

    Controls are looked up by feature id, not by circuit: the runner trains a
    view such as ``dials64`` or ``rff84_64m`` once and every circuit of that
    input / readout width shares it."""
    kind = TASKS[task][0]
    df = load_results(f"{PREFIX}_A_{task}")
    if df.empty:
        return df
    m = METRIC[kind]
    by_fid = df.groupby(["feature_id", "head"])[m].mean()
    q = df[df.arm == "quantum"].groupby(["config", "head"], as_index=False).agg(
        quantum=(m, "mean"), readout=("feature_dim", "first"), width=("n_modes", "first"))
    if q.empty:
        return pd.DataFrame()

    def ctrl(fid, head):
        return by_fid.get((fid, head), np.nan)

    q["dials"] = [ctrl(f"dials{w}", h) for w, h in zip(q.width, q["head"])]
    q["rff"] = [ctrl(f"rff{d}_{w}m", h) for d, w, h in zip(q.readout, q.width, q["head"])]
    q["gain"] = q.quantum - q[["dials", "rff"]].max(axis=1)
    q["raw"] = [ctrl("raw", h) for h in q["head"]]
    q["readout"] = q.readout.astype(int)
    q = q.drop(columns="width")
    return q.sort_values(["head", "quantum"], ascending=[True, False])


def pick_circuit(task: str) -> dict:
    kind = TASKS[task][0]
    t = screen(task)
    t = t[t["head"] == PRIMARY_HEAD[kind]]
    best = t.sort_values("quantum", ascending=False).iloc[0].config
    return next(c for c in CIRCUITS if _tag(c) == best)


def stage_rows(task: str, stage: str, seed: int | None = None) -> pd.DataFrame:
    """Stage B or C records restricted to the CURRENT pick and its controls.

    A run directory keeps every fit it ever made, so when a later screen changes
    the pick (e.g. a circuit added to stage A), the old pick's rows stay on disk
    next to the new ones. Keep the picked circuit, its no-memory twin, the dials
    / RFF of matching width (+-1: a column constant on one seed's split can drop
    the PCA width by one) and raw."""
    name = f"{PREFIX}_{stage}_{task}" + (f"_s{seed}" if seed is not None else "")
    df = load_results(name)
    pick = _state().get("pick", {}).get(task)
    if df.empty or not pick:
        return df
    tags = {_tag(pick)}
    if pick.get("kind") == "iontrap":
        tags.add(_tag({**pick, "memory": False}))
    a = load_results(f"{PREFIX}_A_{task}")
    q = a[(a.arm == "quantum") & (a.config == _tag(pick))] if not a.empty else a
    if q.empty:
        return df[(df.arm != "quantum") | df.config.isin(tags)]
    dim, width = int(q.feature_dim.iloc[0]), int(q.n_modes.iloc[0])
    near = (df.n_modes - width).abs() <= 1
    keep = (((df.arm == "quantum") & df.config.isin(tags))
            | ((df.arm == "dials") & near)
            | ((df.arm == "rff") & (df.feature_dim == dim) & near)
            | (df.arm == "raw"))
    return df[keep]


def head_table(task: str) -> pd.DataFrame:
    kind = TASKS[task][0]
    df = stage_rows(task, "B")
    if df.empty:
        return df
    s = _score(df, kind)
    s["view"] = np.where(s.arm == "quantum", s.config, s.feature_id)
    return s.pivot_table(index="view", columns="head", values="score")


def best_head(task: str) -> str:
    """The head with the best quantum score in stage B (ties -> smaller head)."""
    kind = TASKS[task][0]
    s = _score(stage_rows(task, "B"), kind)
    q = s[s.arm == "quantum"]
    if (~q.config.str.endswith("_nomem")).any():
        q = q[~q.config.str.endswith("_nomem")]
    q = q.sort_values(["score", "n_params"], ascending=[False, True])
    return q.iloc[0]["head"]


def saturation(task: str, frac: float = 0.99) -> pd.DataFrame:
    """Per view: the smallest head reaching ``frac`` of that view's best score,
    and its width relative to the view's input (readout) width."""
    kind = TASKS[task][0]
    s = _score(stage_rows(task, "B"), kind)
    s["view"] = np.where(s.arm == "quantum", s.config, s.feature_id)
    rows = []
    for view, g in s.groupby("view"):
        best = g.score.max()
        ok = g[g.score >= frac * best].sort_values("n_params")
        h = ok.iloc[0]
        width = _head_width(h["head"])
        rows.append(dict(view=view, input_width=int(h.feature_dim), best=best,
                         best_head=g.sort_values("score").iloc[-1]["head"],
                         smallest_99=h["head"], head_params=int(h.n_params),
                         width_over_input=(width / h.feature_dim) if width else 0.0))
    return pd.DataFrame(rows).sort_values("best", ascending=False)


def _head_width(label: str) -> int:
    import re

    m = re.match(r"(?:mlp|ae)([\dx]+)", label)
    return int(m.group(1).split("x")[0]) if m else 0


def seeds_table(task: str) -> pd.DataFrame:
    kind = TASKS[task][0]
    st = _state()
    head = st.get("head", {}).get(task)
    frames = []
    a = stage_rows(task, "B")
    if not a.empty:
        frames.append(a[a["head"] == head])
    for s in SEEDS_C:
        frames.append(stage_rows(task, "C", s))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    sc = df.groupby(["seed", "arm", "config", "feature_id"], as_index=False)[METRIC[kind]].mean()
    sc["view"] = np.where(sc.arm == "quantum", sc.config, sc.arm)
    wide = sc.pivot_table(index="seed", columns="view", values=METRIC[kind])
    return wide


# ------------------------------------------------------------------- driver


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


def run_config(path: Path, plan: bool = False) -> int:
    name = path.stem
    cmd = [sys.executable, "-m", "elements.run", str(path)] + (["--plan"] if plan else [])
    log = RUNS / f"{name}.out"
    print(f"[{time.strftime('%H:%M:%S')}] {'plan' if plan else 'run '} {name}", flush=True)
    if plan:
        return subprocess.call(cmd, cwd=REPO_ROOT)
    with log.open("a", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}]   exit {rc}", flush=True)
    return rc


def run_all(only: list[str] | None = None) -> None:
    tasks = only or list(TASKS)
    st = _state()
    st.setdefault("pick", {})
    st.setdefault("head", {})
    # A: every task's screen first, so a crash later still leaves the full screen
    for p in stage_a_configs():
        if p.stem.removeprefix(f"{PREFIX}_A_") in tasks:
            run_config(p)
    for task in tasks:
        try:
            pick = pick_circuit(task)
        except Exception as e:  # noqa: BLE001 -- keep the night going
            print(f"  {task}: no stage-A result ({e}); skipping B/C", flush=True)
            continue
        st["pick"][task] = pick
        _save_state(st)
        print(f"  {task}: picked {_tag(pick)}", flush=True)
        run_config(stage_b_config(task, pick))
        try:
            st["head"][task] = best_head(task)
        except Exception as e:  # noqa: BLE001
            print(f"  {task}: no stage-B result ({e}); skipping C", flush=True)
            continue
        _save_state(st)
        print(f"  {task}: head {st['head'][task]}", flush=True)
        for p in stage_c_configs(task, pick, st["head"][task]):
            run_config(p)
        _safe_report()
    _safe_report()


def _safe_report() -> None:
    try:
        write_report()
    except Exception as e:  # noqa: BLE001 -- a report bug must not stop the night
        print(f"  report failed: {e!r}", flush=True)


# ------------------------------------------------------------------- report


def _md(df: pd.DataFrame, floatfmt=".4f") -> str:
    if df.empty:
        return "_no results yet_\n"
    return df.to_markdown(floatfmt=floatfmt) + "\n"


def write_report() -> Path:
    st = _state()
    out = ["# Automatic circuit and head study (generated)\n",
           f"Written {time.strftime('%Y-%m-%d %H:%M')} by "
           "`python -m elements.experiments.autostudy report`.\n",
           "Primary metric: supervised = mean macro-F1 over the label schemes; "
           "novelty = macro detection at 1% false alarms. **gain** = quantum minus "
           "the better of dials / rff under the same head. Stages A and B are one "
           "seed (42); differences under ~0.01-0.02 are not established.\n"]
    for task, (kind, data) in TASKS.items():
        out.append(f"\n## {task} ({kind}, {data['dataset']}"
                   f"{', dedup' if data.get('dedup') else ''}"
                   f"{', ' + str(data['max_rows']) + ' rows' if data.get('max_rows') else ''})\n")
        t = screen(task)
        if t.empty:
            out.append("_stage A not run yet_\n")
            continue
        out.append("\n### A. Circuit screen\n")
        for head, h in t.groupby("head"):
            out.append(f"\nHead `{head}`:\n\n")
            out.append(_md(h.drop(columns="head").set_index("config")))
        if task in st.get("pick", {}):
            out.append(f"\nPicked: `{_tag(st['pick'][task])}`\n")
        b = head_table(task)
        if not b.empty:
            out.append("\n### B. Head scaling (score per head)\n\n")
            out.append(_md(b))
            out.append("\nSmallest head within 1% of each view's best, and its width "
                       "relative to the view's input width:\n\n")
            out.append(_md(saturation(task).set_index("view"), floatfmt=".3f"))
        c = seeds_table(task)
        if not c.empty:
            out.append(f"\n### C. Seeds (head `{st.get('head', {}).get(task)}`)\n\n")
            summary = pd.DataFrame({"mean": c.mean(), "std": c.std(ddof=1), "n": c.count()})
            out.append(_md(c))
            out.append("\n")
            out.append(_md(summary.sort_values("mean", ascending=False)))
    REPORT.write_text("".join(out), encoding="utf-8")
    print(f"report -> {REPORT}", flush=True)
    return REPORT


def _smoke() -> None:
    global PREFIX, STATE, REPORT, CIRCUITS, TASKS
    PREFIX, STATE = "smoke", RUNS / "smoke_state.json"
    REPORT = RUNS / "smoke_report.md"
    CIRCUITS = [CIRCUITS[0], CIRCUITS[10], CIRCUITS[11]]  # baseline, its _nomem, 3p12m
    TASKS = {k: (kind, {**data, "max_rows": 3000}) for k, (kind, data) in TASKS.items()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "plan", "report", "classical"])
    p.add_argument("--tasks", nargs="*", help=f"subset of {list(TASKS)}")
    p.add_argument("--smoke", action="store_true",
                   help="tiny data and two circuits under smoke_* names, to test the pipeline")
    a = p.parse_args(argv)
    if a.smoke:
        _smoke()
    if a.action == "plan":
        for path in stage_a_configs():
            if not a.tasks or path.stem.removeprefix(f"{PREFIX}_A_") in a.tasks:
                run_config(path, plan=True)
    elif a.action == "run":
        run_all(a.tasks)
    elif a.action == "classical":
        for task in a.tasks or list(TASKS):
            if task not in CLASSICAL_DONE:
                run_config(classical_config(task))
    else:
        write_report()


if __name__ == "__main__":
    main()
