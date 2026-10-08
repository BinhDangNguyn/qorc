"""Run an ``ExperimentConfig``: every step x feature view x head, resumably.

Supervised and novelty share the loop; they differ only in how a head is
trained and scored. ``plan(cfg)`` prints what a run will do -- steps, feature
widths, head sizes, memory and a rough time estimate -- without training
anything, so a long run can be checked before it starts.

Record schema (one JSON line per trained head), shared with ``elements.viz``:
dataset, scheme, seed, step, config, n_photons, n_modes, arm, feature_id,
feature_dim, head, head_kind, head_role, hidden, n_params, <metrics>, fit_s.
"""

from __future__ import annotations

import gc
import time
import warnings

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix

from elements.config import ExperimentConfig, HeadSpec
from elements.eval.metrics import classification_metrics, detection_metrics
from elements.eval.results import Results
from elements.experiments.views import (
    FrontEnd,
    Task,
    append_dials,
    dials_view,
    input_width,
    iontrap_view,
    load_novelty_task,
    load_supervised_task,
    quantum_view,
    rff_view,
)
from elements.heads.factory import make_classifier, make_detector

MIN_TYPE_ROWS = 20  # novelty: attack types smaller than this are not averaged

# Tiny all23 classes are often absent from a test split; sklearn warns on every
# fit. The metric handles it (zero_division=0), so the warning is only noise.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_task(cfg: ExperimentConfig) -> Task:
    if cfg.task == "novelty":
        return load_novelty_task(cfg.data, cfg.split, cfg.seed)
    return load_supervised_task(cfg.data, cfg.split, cfg.seed)


# --------------------------------------------------------------------- training


def _supervised(task, views, meta, heads, role, cfg, device, results):
    Xtr, Xva, Xte = views
    for scheme, (ytr, yva, yte, names) in task.labels.items():
        for spec in heads:
            key = dict(meta, dataset=task.name, scheme=scheme, seed=cfg.seed, head=spec.label)
            if Results.key(**key) in results.done:
                continue
            t0 = time.time()
            head = make_classifier(spec, len(names), device, cfg.seed).fit(Xtr, ytr, Xva, yva)
            pred = head.predict(Xte)
            m = classification_metrics(yte, pred, head.predict_proba(Xte), names)
            cm = confusion_matrix(yte, pred, labels=range(len(names)))
            rec = dict(
                key, head_kind=spec.kind, head_role=role.get(spec.label, "scaled"),
                hidden=spec.hidden, n_params=spec.n_params(Xtr.shape[1], len(names)),
                accuracy=m["accuracy"], balanced_acc=m["balanced_acc"],
                macro_f1=m["macro_f1"], weighted_f1=m["weighted_f1"],
                macro_auc=m["macro_auc_ovr"], fit_s=round(time.time() - t0, 1),
                per_class_f1={k: v["f1"] for k, v in m["per_class"].items()},
                **operational_metrics(cm, names), confusion=cm.tolist(),
                **head.info,
            )
            results.append(rec)
            results.log(f"  {scheme:9s} {meta['feature_id']:>14s} {spec.label:12s} "
                        f"F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
                        f"({rec['fit_s']:.0f}s)")
            del head
            if device != "cpu":
                torch.cuda.empty_cache()


def operational_metrics(cm: np.ndarray, names: list[str]) -> dict:
    """What a security team reads off a confusion matrix, for any label scheme.

    false_alarm_rate  normal traffic predicted as some attack
    miss_rate         attack traffic predicted as normal
    attack_type_acc   of attacks caught AS attacks, how often the type is right
    """
    if names[0] != "normal":
        return {}
    cm = cm.astype(float)
    caught = cm[1:, 1:]
    return {
        "false_alarm_rate": float(cm[0, 1:].sum() / max(cm[0].sum(), 1)),
        "miss_rate": float(cm[1:, 0].sum() / max(cm[1:].sum(), 1)),
        "attack_type_acc": float(np.trace(caught) / max(caught.sum(), 1)),
    }


def novelty_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Pooled and per-attack-type detection metrics against held-out normal."""
    is_norm = labels == "normal"
    s_norm = scores[is_norm]
    pooled = detection_metrics(s_norm, scores[~is_norm])
    per_type = {}
    for t in np.unique(labels[~is_norm]):
        m = detection_metrics(s_norm, scores[labels == t])
        per_type[str(t)] = {"n": int((labels == t).sum()), "auc": m["auc"],
                            "det@0.01": m["det@0.01"]}
    big = [v for v in per_type.values() if v["n"] >= MIN_TYPE_ROWS]
    return {
        "macro_auc": float(np.mean([v["auc"] for v in big])),
        "macro_det@0.01": float(np.mean([v["det@0.01"] for v in big])),
        "pooled_auc": pooled["auc"], "det@0.001": pooled["det@0.001"],
        "det@0.01": pooled["det@0.01"], "det@0.05": pooled["det@0.05"],
        "per_type": per_type,
    }


def _novelty(task, views, meta, heads, role, cfg, device, results):
    Xtr, Xva, Xte = views
    labels = task.labels["novelty"][2]
    for spec in heads:
        key = dict(meta, dataset=task.name, scheme="novelty", seed=cfg.seed, head=spec.label)
        if Results.key(**key) in results.done:
            continue
        t0 = time.time()
        det = make_detector(spec, device, cfg.seed).fit(Xtr, Xva)
        m = novelty_metrics(det.score(Xte), labels)
        rec = dict(key, head_kind=spec.kind, head_role=role.get(spec.label, "scaled"),
                   hidden=spec.hidden, n_params=spec.n_params(Xtr.shape[1], 0), **m,
                   fit_s=round(time.time() - t0, 1), **det.info)
        results.append(rec)
        results.log(f"  novelty {meta['feature_id']:>14s} {spec.label:14s} "
                    f"macroAUC={m['macro_auc']:.4f} det@1%={m['det@0.01']:.4f} "
                    f"macro_det@1%={m['macro_det@0.01']:.4f} ({rec['fit_s']:.0f}s)")
        del det
        if device != "cpu":
            torch.cuda.empty_cache()


# ------------------------------------------------------------------------- the run


def _pending(task, fid, heads, cfg, results, arm) -> list[HeadSpec]:
    return [h for h in heads if h.applies_to(arm)
            and any((task.name, s, cfg.seed, fid, h.label) not in results.done
                    for s in task.labels)]


def _raw_heads(cfg: ExperimentConfig, steps) -> tuple[list[HeadSpec], dict]:
    """Heads for the raw arm: the standalone models, or every head of every step."""
    if cfg.models:
        heads = cfg.model_heads()
        return heads, {h.label: "classical" for h in heads}
    return _unique_heads(steps)


def _unique_heads(steps) -> tuple[list[HeadSpec], dict]:
    heads, role = [], {}
    for st in steps:
        for h in st.heads:
            if h.label not in role:
                heads.append(h)
                role[h.label] = st.role[h.label]
    return heads, role


def run(cfg: ExperimentConfig) -> Results:
    device = resolve_device(cfg.device)
    results = Results(cfg.out_dir)
    (cfg.out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    train = _novelty if cfg.task == "novelty" else _supervised
    steps = cfg.steps()

    results.log(f"=== {cfg.name}: {cfg.task}, seed {cfg.seed}, {len(steps)} steps, "
                f"arms {cfg.arms}, device {device} ===")
    task = load_task(cfg)
    results.log(f"train {task.X_train.shape} val {task.X_val.shape} test {task.X_test.shape}")
    fe = FrontEnd(task, cfg.seed, drop_constant=cfg.task == "supervised")
    t_run = time.time()

    if "raw" in cfg.arms:
        heads, role = _raw_heads(cfg, steps)
        todo = _pending(task, "raw", heads, cfg, results, "raw")
        if todo:
            meta = dict(step=-1, config="raw", n_modes=0, n_photons=0, arm="raw",
                        feature_id="raw", feature_dim=fe.n_features, pca_var=1.0)
            train(task, fe.raw(), meta, todo, role, cfg, device, results)

    for i, st in enumerate(steps):
        res, dim = st.reservoir, st.reservoir.dim
        width = input_width(fe, res)
        _, explained = fe.pca(width)
        results.log(f"[step {i + 1}/{len(steps)}] {res.tag}: readout {dim:,}, "
                    f"PCA({width}) {explained:.1%}, heads "
                    f"{[h.label for h in st.heads]}")
        base = dict(step=i, config=res.tag, n_modes=width, n_photons=res.photons,
                    pca_var=explained, reservoir_kind=res.kind)

        fid = f"dials{width}"
        todo = _pending(task, fid, st.heads, cfg, results, "dials") if "dials" in cfg.arms else []
        if todo:
            meta = dict(base, arm="dials", feature_id=fid, feature_dim=width)
            train(task, dials_view(fe, width), meta, todo, st.role, cfg, device, results)

        fq, fqd = f"q{res.tag}", f"qd{res.tag}"
        todo_q = (_pending(task, fq, st.heads, cfg, results, "quantum")
                  if "quantum" in cfg.arms else [])
        todo_qd = (_pending(task, fqd, st.heads, cfg, results, "qdials")
                   if "qdials" in cfg.arms else [])
        if todo_q or todo_qd:
            t = time.time()
            if res.kind == "iontrap":
                views = iontrap_view(fe, res, cfg.encoding, cfg.seed, device,
                                     save_to=cfg.out_dir / "reservoirs" / f"{res.tag}.npz")
            else:
                views = quantum_view(fe, res, cfg.encoding, cfg.seed, device)
            results.log(f"  reservoir {res.tag} built in {time.time() - t:.0f}s")
            if todo_q:
                meta = dict(base, arm="quantum", feature_id=fq, feature_dim=dim)
                train(task, views, meta, todo_q, st.role, cfg, device, results)
            if todo_qd:
                append_dials(views, dials_view(fe, width))
                meta = dict(base, arm="qdials", feature_id=fqd, feature_dim=dim + width)
                train(task, views, meta, todo_qd, st.role, cfg, device, results)
            del views
            gc.collect()

        fr, frd = f"rff{dim}_{width}m", f"rffd{dim}_{width}m"
        todo_r = _pending(task, fr, st.heads, cfg, results, "rff") if "rff" in cfg.arms else []
        todo_rd = (_pending(task, frd, st.heads, cfg, results, "rffdials")
                   if "rffdials" in cfg.arms else [])
        if todo_r or todo_rd:
            views = rff_view(fe, width, dim, cfg.seed)
            if todo_r:
                meta = dict(base, arm="rff", feature_id=fr, feature_dim=dim)
                train(task, views, meta, todo_r, st.role, cfg, device, results)
            if todo_rd:
                append_dials(views, dials_view(fe, width))
                meta = dict(base, arm="rffdials", feature_id=frd, feature_dim=dim + width)
                train(task, views, meta, todo_rd, st.role, cfg, device, results)
            del views
            gc.collect()

    results.log(f"=== {cfg.name} done in {(time.time() - t_run) / 60:.1f} min -> "
                f"{results.path} ===")
    return results


# ------------------------------------------------------------------------- the plan

# Rough throughput on the RTX 2070 this project was measured on. The plan's
# times are order-of-magnitude guides, not promises.
_GPU_FLOPS = 2.5e12  # effective fp32 for MLP / AE training
_HOST_BW = 5e9  # bytes/s streamed per L-BFGS iteration (design matrix stays in RAM)
_RESERVOIR_RATE = 6e7  # (rows x readout dim) per second through MerLin
_IONTRAP_RATE = 2e10  # complex MACs per second, exact density-matrix simulation
_DENSE_RATE = 2.5e11  # complex MACs per second, batched dense matmuls
_TRAJ_COST = 1.4e-9  # seconds per (trajectory x step x amplitude), measured at 20q
_EPOCHS = 40  # typical epochs before early stopping


def _fit_seconds(spec: HeadSpec, n: int, d: int, n_out: int) -> float:
    if spec.kind == "lbfgs":
        return spec.max_iter * 1.2 * n * d * 4 / _HOST_BW + 2
    if spec.kind == "pca":
        return 2 + n * d * spec.n_components * 4 / 5e10
    if spec.kind == "iforest":
        return 5
    n_fit = min(n, spec.max_train_rows or n)
    if spec.kind == "ocsvm":  # kernel SVM: roughly quadratic in rows
        return 30 * (n_fit / 10_000) ** 2
    if spec.kind in ("lof", "knndist", "knn"):  # neighbour search dominates
        return 20 + n * d * 2e-6
    if spec.kind in ("rf", "extratrees", "xgb", "hgb"):
        return 5 + spec.n_estimators * n * d * 4e-9 * max(1, n_out / 4)
    if spec.kind in ("logreg", "svm", "gmm"):
        return 10 + n * d * n_out * 5e-8
    params = spec.n_params(d, n_out)
    per_epoch = 6 * params * n / _GPU_FLOPS + (n / spec.batch_size) * 0.004 + 0.2
    return min(spec.max_epochs, _EPOCHS) * per_epoch


def _reservoir_seconds(res, n_rows: int) -> float:
    if res.kind != "iontrap":
        return n_rows * res.dim / _RESERVOIR_RATE
    A, M = 2 ** res.n_in, 2 ** res.n_mem
    if res.trajectories:
        k = res.trajectories if res.memory else 1
        return 2 + n_rows * k * res.steps * A * M * _TRAJ_COST
    if res.uploads > 1 or res.encode_all:  # ~4 dense D x D products per upload
        return 2 + n_rows * res.steps * res.uploads * 4 * (A * M) ** 3 / _DENSE_RATE
    return 2 + n_rows * res.steps * (A * A * M * M + 2 * A * M ** 3) / _IONTRAP_RATE


def plan(cfg: ExperimentConfig) -> str:
    """Describe the run without training anything. Loads the data for its shape."""
    steps = cfg.steps()
    task = load_task(cfg)
    fe = FrontEnd(task, cfg.seed, drop_constant=cfg.task == "supervised")
    n_tr, n_all = len(task.X_train), len(task.X_train) + len(task.X_val) + len(task.X_test)
    schemes = list(task.labels)
    n_out = max((len(v[3]) for v in task.labels.values() if v[3]), default=0)
    results = Results(cfg.out_dir) if cfg.out_dir.exists() else None
    done = results.done if results else set()

    lines = [
        f"PLAN  {cfg.name}  ({cfg.task}, seed {cfg.seed}, device {resolve_device(cfg.device)})",
        f"data  {cfg.data.dataset} {cfg.data.path or '(default path)'}",
        f"      train {n_tr:,} / val {len(task.X_val):,} / test {len(task.X_test):,} rows, "
        f"{fe.n_features} features, label sets {schemes}",
        f"arms  {cfg.arms}",
        f"out   {cfg.out_dir}",
        "",
        f"{'step':>4}  {'reservoir':10} {'readout':>8} {'feat GB':>7}  "
        f"{'head':14} {'role':8} {'params (quantum)':>16} {'~fit':>7}",
    ]
    total_s, n_fits, n_done = 0.0, 0, 0
    seen: set = set()  # dials of the same width are trained once, as in run()

    def account(fid, d, spec):
        nonlocal total_s, n_fits, n_done
        for s in schemes:
            if (fid, spec.label, s) in seen:
                continue
            seen.add((fid, spec.label, s))
            n_fits += 1
            if (task.name, s, cfg.seed, fid, spec.label) in done:
                n_done += 1
            else:
                total_s += _fit_seconds(spec, n_tr, d, n_out)

    if "raw" in cfg.arms:
        heads, _ = _raw_heads(cfg, steps)
        for h in heads:
            if h.applies_to("raw"):
                account("raw", fe.n_features, h)

    for i, st in enumerate(steps):
        res, dim = st.reservoir, st.reservoir.dim
        width = input_width(fe, res)
        gb = n_all * dim * 4 / 1e9
        if "quantum" in cfg.arms or "qdials" in cfg.arms:
            total_s += _reservoir_seconds(res, n_all)
        shown = [h for h in st.heads if any(h.applies_to(a) for a in cfg.arms if a != "raw")]
        for j, h in enumerate(shown):
            est = _fit_seconds(h, n_tr, dim, n_out)
            lines.append(
                f"{(i + 1) if j == 0 else '':>4}  {res.tag if j == 0 else '':10} "
                f"{dim if j == 0 else '':>8} {f'{gb:.2f}' if j == 0 else '':>7}  "
                f"{h.label:14} {st.role[h.label]:8} {h.n_params(dim, n_out):>16,} "
                f"{est / 60:>6.1f}m")
            fids = {"quantum": (f"q{res.tag}", dim), "dials": (f"dials{width}", width),
                    "rff": (f"rff{dim}_{width}m", dim),
                    "qdials": (f"qd{res.tag}", dim + width),
                    "rffdials": (f"rffd{dim}_{width}m", dim + width)}
            for arm in cfg.arms:
                if arm in fids and h.applies_to(arm):
                    account(*fids[arm], h)
    if cfg.models:
        for h in cfg.model_heads():
            est = _fit_seconds(h, n_tr, fe.n_features, n_out)
            lines.append(f"{'':>4}  {'raw':10} {fe.n_features:>8} {'':>7}  "
                         f"{h.label:14} {'classical':8} {'-':>16} {est / 60:>6.1f}m")
    peak = max([n_all * st.reservoir.dim * 4 / 1e9 for st in steps]
               + [n_all * fe.n_features * 4 / 1e9])
    lines += [
        "",
        f"fits  {n_fits} total ({len(schemes)} label set(s) x heads x arms), "
        f"{n_done} already on disk, {n_fits - n_done} to run",
        "time  ~" + (f"{total_s / 3600:.1f} h" if total_s >= 3600 else f"{total_s / 60:.0f} min")
        + " remaining (rough: +-50%, one GPU, nothing else running)",
        f"RAM   peak feature matrix {peak:.1f} GB"
        + (" (x2 while building qdials / rffdials)"
           if {"qdials", "rffdials"} & set(cfg.arms) else ""),
    ]
    return "\n".join(lines)
