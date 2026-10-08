"""Interactive Plotly views of the automatic circuit / head study.

``load_all()`` builds one tidy frame of every fit the study made (stages A, B,
C and the classical stage K); the figure functions and the widget explorer only
filter it. Colours follow ``elements.viz.style`` (quantum blue, dials orange,
RFF aqua, raw yellow); the wide faint grey band is the best standalone
classical model on the raw features, same split and seed.

Series names: a quantum view is its circuit tag (``it4+4_a2a_L16_dt0.3``,
``3p20m``); a control is its feature id (``dials121``, ``rff164_121m``, ``raw``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from elements.experiments import autostudy as A
from elements.viz.autostudy import HEAD_ORDER, TITLES, VIEW_LABEL, _arm_of
from elements.viz.style import ARM_COLOR

TEMPLATE = "plotly_white"
REF_RGBA = "rgba(138,137,132,0.28)"
SYMBOL = {"quantum": "circle", "dials": "square", "rff": "triangle-up", "raw": "diamond",
          "classical": "x"}
#: quantum series beyond the first get darker / lighter blues, never another hue
QUANTUM_SHADES = ["#2a78d6", "#104281", "#86b6ef", "#1c5cab", "#5598e7", "#0d366b"]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash"]

METRICS = {
    "supervised": ["macro_f1", "weighted_f1", "accuracy", "balanced_acc",
                   "false_alarm_rate", "miss_rate"],
    "novelty": ["macro_det@0.01", "macro_auc", "pooled_auc", "det@0.001", "det@0.01",
                "det@0.05"],
}
LOWER_IS_BETTER = {"false_alarm_rate", "miss_rate"}
METRIC_LABEL = {"macro_f1": "macro-F1", "weighted_f1": "weighted F1", "accuracy": "accuracy",
                "balanced_acc": "balanced accuracy", "false_alarm_rate": "false-alarm rate",
                "miss_rate": "miss rate", "macro_det@0.01": "detection at 1% FA (macro)",
                "macro_auc": "AUC (macro over attack types)", "pooled_auc": "AUC (pooled)",
                "det@0.001": "detection at 0.1% FA", "det@0.01": "detection at 1% FA",
                "det@0.05": "detection at 5% FA"}


def _kind(task: str) -> str:
    return A.TASKS[task][0]


# ------------------------------------------------------------------- data


def load_all() -> pd.DataFrame:
    """Every record of stages A, B, C and K, one row per (fit, label scheme)."""
    frames = []
    for task in A.TASKS:
        parts = [("A", A.load_results(f"{A.PREFIX}_A_{task}")),
                 ("B", A.stage_rows(task, "B")),          # current pick only
                 ("K", A.load_results(A.classical_run(task)))]
        parts += [("C", A.stage_rows(task, "C", s)) for s in A.SEEDS_C]
        for stage, df in parts:
            if not df.empty:
                frames.append(df.assign(task=task, stage=stage))
    d = pd.concat(frames, ignore_index=True)
    d["series"] = np.where(d.arm == "quantum", d.config, d.feature_id)
    k = d.stage == "K"
    d.loc[k, "arm"] = "classical"
    d.loc[k, "series"] = "classical " + d.loc[k, "head"]
    return d


def schemes(d: pd.DataFrame, task: str) -> list[str]:
    s = sorted(d[d.task == task].scheme.dropna().unique())
    return s if s == ["novelty"] else ["mean"] + s


def values(d: pd.DataFrame, task: str, metric: str, scheme: str = "mean") -> pd.DataFrame:
    """One value per fit: the metric on one scheme, or averaged over schemes."""
    x = d[(d.task == task) & d[metric].notna()] if metric in d else d.iloc[:0]
    if scheme != "mean":
        x = x[x.scheme == scheme]
    keys = ["stage", "seed", "arm", "config", "series", "head", "feature_id",
            "feature_dim", "n_modes"]
    return x.groupby(keys, as_index=False, dropna=False)[metric].mean().rename(
        columns={metric: "value"})


def classical_ref(d, task, metric, scheme="mean") -> tuple[float, str]:
    v = values(d, task, metric, scheme)
    v = v[v.stage == "K"]
    if v.empty:
        return np.nan, "not run"
    i = v.value.idxmin() if metric in LOWER_IS_BETTER else v.value.idxmax()
    return float(v.value[i]), str(v["head"][i])


def default_metric(task: str) -> str:
    return A.METRIC[_kind(task)]


# ------------------------------------------------------------------- styling


def _style(arm: str, i: int = 0) -> dict:
    color = QUANTUM_SHADES[i % len(QUANTUM_SHADES)] if arm == "quantum" else ARM_COLOR.get(
        arm, "#52514e")
    return {"color": color, "symbol": SYMBOL.get(arm, "circle"),
            "dash": DASHES[0] if arm == "quantum" else DASHES[i % len(DASHES)]}


def _series_styles(series: list[str]) -> dict:
    """Stable style per series: arm colour, shades / dashes within an arm."""
    seen: dict[str, int] = {}
    out = {}
    for s in series:
        arm = _arm_of(s)
        i = seen.get(arm, 0)
        seen[arm] = i + 1
        st = _style(arm, i)
        if ("_nomem" in s):
            st.update(color=QUANTUM_SHADES[0], dash="dash", hollow=True)
        out[s] = st
    return out


def qubits(tag: str) -> str:
    """'8q' for an ion trap tag (input + memory qubits), 'photonic' otherwise."""
    import re

    m = re.match(r"(?:A·)?it(\d+)\+(\d+)", tag)
    if m:
        return f"{int(m.group(1)) + int(m.group(2))}q"
    return "photonic" if re.match(r"(?:A·)?\d+p\d+m", tag) else ""


def _short(series: str) -> str:
    """Compact axis label; the full name is in the legend and the hover."""
    arm = _arm_of(series)
    if arm != "quantum":
        return {"dials": "dials", "rff": "RFF", "raw": "raw"}[arm]
    return f"quantum {qubits(series)}" + (" no mem" if "_nomem" in series else "")


def _label(series: str) -> str:
    arm = _arm_of(series)
    if arm == "quantum":
        return (f"quantum {series} ({qubits(series)})"
                + (" no memory" if ("_nomem" in series) else ""))
    return VIEW_LABEL[arm] if series == arm else f"{VIEW_LABEL[arm]} [{series}]"


def _layout(fig, title, height, ylab=None, xlab=None):
    fig.update_layout(template=TEMPLATE, title=dict(text=title, font=dict(size=15)),
                      height=height, margin=dict(l=70, r=30, t=90, b=60),
                      legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
                      hoverlabel=dict(font_size=11))
    if ylab:
        fig.update_yaxes(title_text=ylab, title_font=dict(size=11))
    if xlab:
        fig.update_xaxes(title_text=xlab, title_font=dict(size=11))
    return fig


# ------------------------------------------------------------------- figures


def fig_seeds(d, tasks=None, metric=None, scheme="mean", series=None, show_ref=True):
    """Stage C: each view on every seed (small markers) and its mean +- sd.
    Seed 42 comes from stage B under the head stage C used."""
    tasks = tasks or list(A.TASKS)
    st = A._state().get("head", {})
    rows = int(np.ceil(len(tasks) / 2))
    titles = []
    for t in tasks:
        m = metric or default_metric(t)
        v, name = classical_ref(d, t, m, scheme)
        ref = f"grey band = classical best: {name} ({v:.3f})" if show_ref and not np.isnan(v) else ""
        titles.append(f"{TITLES[t]}, head {st.get(t)}<br><sup>{ref}</sup>")
    fig = make_subplots(rows=rows, cols=2, subplot_titles=titles, vertical_spacing=0.16)
    legend_done = set()
    for k, t in enumerate(tasks):
        r, c = k // 2 + 1, k % 2 + 1
        m = metric or default_metric(t)
        v = values(d, t, m, scheme)
        v = _seed_rows(v, st.get(t))
        names = series or sorted(v.series.unique(), key=_order_key)
        styles = _series_styles(names)
        for s in names:
            y = v[v.series == s].sort_values("seed")
            if y.empty:
                continue
            sty, lab = styles[s], _label(s)
            short = _short(s)
            fill = "white" if sty.get("hollow") else sty["color"]
            fig.add_trace(go.Scatter(
                x=[short] * len(y), y=y.value, mode="markers", opacity=0.45,
                marker=dict(symbol=sty["symbol"], size=7, color=fill,
                            line=dict(color=sty["color"], width=1.5)),
                customdata=y.seed, hovertemplate="seed %{customdata}: %{y:.4f}<extra></extra>",
                showlegend=False), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=[short], y=[y.value.mean()], mode="markers+text", cliponaxis=False,
                text=[f"{y.value.mean():.3f}"], textposition="middle right",
                error_y=dict(type="data", array=[y.value.std(ddof=1) if len(y) > 1 else 0],
                             color=sty["color"], thickness=1.5, width=4),
                marker=dict(symbol=sty["symbol"], size=12, color=fill,
                            line=dict(color=sty["color"], width=2)),
                name=lab, legendgroup=lab, showlegend=lab not in legend_done,
                hovertemplate=f"{lab}<br>mean %{{y:.4f}} over {len(y)} seeds<extra></extra>"),
                row=r, col=c)
            legend_done.add(lab)
        if show_ref:
            ref, _ = classical_ref(d, t, m, scheme)
            if not np.isnan(ref):
                fig.add_hline(y=ref, line=dict(color=REF_RGBA, width=10), layer="below",
                              row=r, col=c)
        fig.update_yaxes(title_text=METRIC_LABEL.get(m, m), title_font=dict(size=10),
                         row=r, col=c)
        fig.update_xaxes(tickfont=dict(size=9), row=r, col=c)
    return _layout(fig, "Stage C: picked circuit vs controls, seeds 42-44 (mean +- sd)",
                   330 * rows + 120)


def fig_screen(d, task, head=None, metric=None, scheme="mean", circuits=None,
               arms=("quantum", "dials", "rff", "raw"), show_ref=True):
    """Stage A: one row per circuit, quantum next to its matched controls."""
    m = metric or default_metric(task)
    head = head or A.PRIMARY_HEAD[_kind(task)]
    v = values(d, task, m, scheme)
    a = v[(v.stage == "A") & (v["head"] == head)]
    by_fid = a.groupby("feature_id").value.mean()
    q = a[a.arm == "quantum"].copy()
    if circuits:
        q = q[q.config.isin(circuits)]
    q["dials"] = [by_fid.get(f"dials{int(w)}", np.nan) for w in q.n_modes]
    q["rff"] = [by_fid.get(f"rff{int(f)}_{int(w)}m", np.nan)
                for f, w in zip(q.feature_dim, q.n_modes)]
    q = q.sort_values("value", ascending=m in LOWER_IS_BETTER)
    fig = go.Figure()
    for arm, col in (("dials", "dials"), ("rff", "rff"), ("quantum", "value")):
        if arm not in arms:
            continue
        sty = _style(arm)
        fig.add_trace(go.Scatter(
            x=q[col], y=q.config, mode="markers+text" if arm == "quantum" else "markers",
            text=[f"{x:.3f}" for x in q[col]] if arm == "quantum" else None,
            textposition="middle right",
            name=VIEW_LABEL[arm] if arm != "quantum" else "quantum readout", marker=dict(symbol=sty["symbol"], color=sty["color"],
                                           size=12 if arm == "quantum" else 9,
                                           opacity=1 if arm == "quantum" else 0.75),
            hovertemplate="%{y}<br>" + arm + ": %{x:.4f}<extra></extra>"))
    if "raw" in arms and "raw" in by_fid:
        fig.add_vline(x=by_fid["raw"], line=dict(color=ARM_COLOR["raw"], width=1.5, dash="dot"),
                      annotation_text="raw input", annotation_font_size=10)
    ref, name = classical_ref(d, task, m, scheme)
    sub = ""
    if show_ref and not np.isnan(ref):
        fig.add_vline(x=ref, line=dict(color=REF_RGBA, width=10), layer="below")
        sub = f"grey band = classical best: {name} ({ref:.3f})"
    pick = A._state().get("pick", {}).get(task)
    note = A._state().get("notes", {}).get(task)
    title = (f"Stage A circuit screen - {TITLES[task]}<br><sup>head {head}"
             + (f" | picked {A._tag(pick)}" if pick else "") + f" | {sub}"
             + (f"<br>note: {note}" if note else "") + "</sup>")
    fig.update_yaxes(tickfont=dict(size=10))
    return _layout(fig, title, 32 * len(q) + 220, xlab=f"{METRIC_LABEL.get(m, m)}"
                   + ("" if scheme == "mean" else f" ({scheme})"))


def fig_heads(d, tasks=None, metric=None, scheme="mean", series=None,
              include_stage_a=False, show_ref=True):
    """Stage B: the metric against head size, one line per feature view.
    ``include_stage_a`` adds every stage-A circuit (its two heads only)."""
    tasks = tasks or list(A.TASKS)
    rows = int(np.ceil(len(tasks) / 2))
    titles = []
    for t in tasks:
        m = metric or default_metric(t)
        v, name = classical_ref(d, t, m, scheme)
        titles.append(f"{TITLES[t]}<br><sup>grey band = classical best: {name} ({v:.3f})</sup>"
                      if show_ref and not np.isnan(v) else TITLES[t])
    fig = make_subplots(rows=rows, cols=2, subplot_titles=titles, vertical_spacing=0.12)
    legend_done = set()
    for k, t in enumerate(tasks):
        r, c = k // 2 + 1, k % 2 + 1
        m = metric or default_metric(t)
        order = HEAD_ORDER[_kind(t)]
        v = values(d, t, m, scheme)
        lines = _head_lines(v, series, include_stage_a)
        styles = _series_styles([s for s, _ in lines])
        for s, y in lines:
            y = y.set_index("head").reindex([h for h in order if h in set(y["head"])])
            sty, lab = styles[s], _label(s)
            if include_stage_a and s.startswith("A·"):
                lab, sty = f"A · {s[2:]}", {**sty, "dash": "dot"}
            fig.add_trace(go.Scatter(
                x=y.index, y=y.value, mode="lines+markers", name=lab, legendgroup=lab,
                showlegend=lab not in legend_done,
                line=dict(color=sty["color"], width=2, dash=sty["dash"]),
                marker=dict(symbol=sty["symbol"], size=9,
                            color="white" if sty.get("hollow") else sty["color"],
                            line=dict(color=sty["color"], width=2)),
                hovertemplate=lab + "<br>%{x}: %{y:.4f}<extra></extra>"), row=r, col=c)
            legend_done.add(lab)
        if show_ref:
            ref, _ = classical_ref(d, t, m, scheme)
            if not np.isnan(ref):
                fig.add_hline(y=ref, line=dict(color=REF_RGBA, width=10), layer="below",
                              row=r, col=c)
        fig.update_xaxes(categoryorder="array", categoryarray=order, tickfont=dict(size=9),
                         row=r, col=c)
        fig.update_yaxes(title_text=METRIC_LABEL.get(m, m), title_font=dict(size=10),
                         row=r, col=c)
    return _layout(fig, "Stage B: how big must the classical head be? (seed 42)",
                   330 * rows + 120)


def _seed_rows(v: pd.DataFrame, head: str) -> pd.DataFrame:
    """Stage B (seed 42) + C rows under the stage-C head. Controls are named by
    their arm: the PCA width can shift by one between seeds' splits (a column
    constant on one train split), which would otherwise split ``dials121`` /
    ``dials122`` into two series."""
    v = v[v.stage.isin(["B", "C"]) & (v["head"] == head)].copy()
    v["series"] = np.where(v.arm == "quantum", v.series, v.arm)
    return v


def _head_lines(v, series, include_stage_a):
    b = v[(v.stage == "B") & (v.seed == 42)]
    names = series or sorted(b.series.unique(), key=_order_key)
    lines = [(s, b[b.series == s]) for s in names if s in set(b.series)]
    if include_stage_a:
        a = v[(v.stage == "A") & (v.arm == "quantum")]
        for s in sorted(a.series.unique(), key=_order_key):
            if series is None or f"A·{s}" in series:
                lines.append((f"A·{s}", a[a.series == s]))
    return lines


def _order_key(s: str):
    arm = _arm_of(s)
    return ({"quantum": 0, "dials": 1, "rff": 2, "raw": 3}.get(arm, 4), ("_nomem" in s), s)


# ------------------------------------------------------------------- tables


def circuits_table() -> pd.DataFrame:
    """Every stage-A circuit with what it changes relative to the baseline."""
    from elements.config import reservoir_from_dict

    base = A.CIRCUITS[0]
    rows = []
    for c in A.CIRCUITS:
        r = reservoir_from_dict(c)
        if r.kind == "iontrap":
            diff = [f"{k}={c[k]}" for k in c if c.get(k) != base.get(k)]
            rows.append({"circuit": r.tag, "type": "trapped ion", "input qubits": r.n_in,
                         "memory qubits": r.n_mem, "qubits": r.n_qubits, "rounds L": r.steps,
                         "dt": r.dt, "coupling": r.topology, "J scale": r.j_scale,
                         "memory": r.memory, "encoding slots": r.n_slots,
                         "readout": r.dim, "vs baseline": ", ".join(diff) or "baseline"})
        else:
            rows.append({"circuit": r.tag, "type": "photonic", "input qubits": None,
                         "memory qubits": None, "qubits": None, "rounds L": None,
                         "dt": None, "coupling": f"{r.modes} modes, {r.photons} photons",
                         "J scale": None, "memory": False, "encoding slots": r.modes,
                         "readout": r.dim, "vs baseline": "photonic reference"})
    return pd.DataFrame(rows).set_index("circuit")


def tasks_table(d: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for t, (kind, data) in A.TASKS.items():
        rows.append({"task": t, "kind": kind, "dataset": data["dataset"],
                     "dedup": data.get("dedup", False),
                     "row cap": data.get("max_rows") or "all",
                     "label schemes": ", ".join(data.get("schemes", ["novelty (normal-only)"])),
                     "primary metric": A.METRIC[kind],
                     "stage-A heads": ", ".join(A._head_label(h) for h in A.HEADS_A[kind]),
                     "stage-B heads": ", ".join(A._head_label(h) for h in A.HEADS_B[kind])})
    return pd.DataFrame(rows).set_index("task")


# ------------------------------------------------------------------- 8q vs 20q runs

#: Earlier ion-trap runs (configs/iontrap*.yaml), seed 42, NSL-KDD.
ION_RUNS = {
    "iontrap20_supervised": "NSL-KDD, 20k-row subsample: 8q (4+4) and 20q ring (8+12, 8 trajectories)",
    "iontrap20_novelty": "NSL-KDD zero-day, 20k-row subsample: 8q and 20q ring",
    "iontrap_supervised": "NSL-KDD, full data: 8q (4+4) and photonic 3p12m",
    "iontrap_novelty": "NSL-KDD zero-day, full data: 8q (4+4) and photonic 3p12m",
}
#: Full-data runs share split and seed with these classical runs (grey band).
#: The 20k-row runs have no classical reference on their subsample.
ION_CLASSICAL = {"iontrap_supervised": "classical_supervised",
                 "iontrap_novelty": "classical_novelty"}
HEAD_SYMBOL = {"lbfgs": "circle-open", "mlp256": "circle", "pca32": "square-open",
               "ae256-b32": "square", "iforest": "x"}


def load_ion_runs() -> pd.DataFrame:
    frames = []
    for run in ION_RUNS:
        df = A.load_results(run)
        if not df.empty:
            frames.append(df.assign(run=run))
    d = pd.concat(frames, ignore_index=True)
    d["series"] = np.where(d.arm == "quantum", d.config, d.feature_id)
    return d


def _run_kind(run: str) -> str:
    return "novelty" if run.endswith("novelty") else "supervised"


def _ion_order(s: str):
    q = qubits(s)
    return (_order_key(s)[0], -int(q[:-1]) if q.endswith("q") else 0, ("_nomem" in s), s)


def fig_qubits(r: pd.DataFrame, run: str, metric=None, schemes_=None, series=None,
               heads=None, show_ref=True):
    """One earlier ion-trap run: every view (x) under every head (marker),
    one panel per label scheme. Quantum views carry their qubit count."""
    kind = _run_kind(run)
    m = metric or A.METRIC[kind]
    x = r[(r.run == run) & r[m].notna()]
    schemes_ = schemes_ or sorted(x.scheme.unique())
    heads = heads or [h for h in HEAD_SYMBOL if h in set(x["head"])]
    names = series or sorted(x.series.unique(), key=_ion_order)
    styles = _series_styles(names)
    ref = A.load_results(ION_CLASSICAL[run]) if show_ref and run in ION_CLASSICAL else None
    titles = []
    for sc in schemes_:
        sub = sc
        if ref is not None:
            rr = ref[ref.scheme == sc].groupby("head")[m].mean()
            if len(rr):
                sub += f"<br><sup>grey band = classical best: {rr.idxmax()} ({rr.max():.3f})</sup>"
        titles.append(sub)
    fig = make_subplots(rows=1, cols=len(schemes_), subplot_titles=titles)
    for j, sc in enumerate(schemes_, start=1):
        y = x[x.scheme == sc]
        for h in heads:
            yh = y[y["head"] == h].set_index("series").reindex(names).dropna(subset=[m])
            if yh.empty:
                continue
            labels = [_label(s) for s in yh.index]
            colors = [styles[s]["color"] for s in yh.index]
            fig.add_trace(go.Scatter(
                x=labels, y=yh[m], mode="markers", name=h, legendgroup=h, showlegend=j == 1,
                marker=dict(symbol=HEAD_SYMBOL.get(h, "circle"), size=12, color=colors,
                            line=dict(color=colors, width=2)),
                hovertemplate="%{x}<br>" + h + ": %{y:.4f}<extra></extra>"), row=1, col=j)
        if ref is not None:
            rr = ref[ref.scheme == sc].groupby("head")[m].mean()
            if len(rr):
                fig.add_hline(y=rr.max(), line=dict(color=REF_RGBA, width=10), layer="below",
                              row=1, col=j)
        fig.update_xaxes(tickangle=35, tickfont=dict(size=9), row=1, col=j)
        fig.update_yaxes(title_text=METRIC_LABEL.get(m, m), title_font=dict(size=10),
                         row=1, col=j)
    fig = _layout(fig, f"{run}: {ION_RUNS[run]}<br><sup>marker = head "
                       "(filled = MLP / AE, open = linear / PCA); colour = view</sup>", 620)
    fig.update_layout(margin=dict(b=200), legend=dict(y=1.08, orientation="h"))
    return fig


def script20_table(path="D:/Dev/ai-ml/reservoir-computing/ion_traps/results_20q_ring_L16.json"):
    """The original 20-qubit script (ion_traps/qrc_iontrap_20q_ring.py), GPU run."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    r = json.loads(p.read_text())["results"]
    return pd.DataFrame({
        "quantum (20q ring, 596 features)": {
            "supervised AUC": r["supervised_quantum"], "zero-day AUC": r["zeroday_quantum"]["AUC"],
            "detection @1% FA": r["zeroday_quantum"]["@1%FA"]},
        "classical (same 122 PCA comps)": {
            "supervised AUC": r["supervised_classical"],
            "zero-day AUC": r["zeroday_classical"]["AUC"],
            "detection @1% FA": r["zeroday_classical"]["@1%FA"]},
    }).T


# ------------------------------------------------------------------- explorer


def explorer(d: pd.DataFrame, r: pd.DataFrame | None = None):
    """Widget explorer. ``d`` = ``load_all()`` (the study); ``r`` =
    ``load_ion_runs()`` (earlier 8q / 20q runs) enables the fourth view."""
    import ipywidgets as W
    from IPython.display import clear_output, display

    reg = globals().setdefault("_autostudy_explorer_widgets", {})

    def get(name, make):
        if name not in reg:
            reg[name] = make()
        return reg[name]

    QB = "8q vs 20q (earlier runs)"
    modes = ["Circuit screen (stage A)", "Head size (stages A+B)", "Seeds (stages B+C)"]
    modes += [QB] if r is not None else []
    tasks = list(A.TASKS)
    runs = [x for x in ION_RUNS if r is not None and x in set(r.run)]
    mode = get("mode", lambda: W.ToggleButtons(description="View:"))
    task = get("task", lambda: W.Dropdown(description="Task:", layout=W.Layout(width="260px")))
    run = get("run", lambda: W.Dropdown(description="Run:", layout=W.Layout(width="320px")))
    metric = get("metric", lambda: W.Dropdown(description="Metric:",
                                              layout=W.Layout(width="300px")))
    scheme = get("scheme", lambda: W.Dropdown(description="Scheme:",
                                              layout=W.Layout(width="240px")))
    head = get("head", lambda: W.Dropdown(description="Head:", layout=W.Layout(width="240px")))
    lines = get("lines", lambda: W.SelectMultiple(description="Lines:", rows=12,
                                                  layout=W.Layout(width="600px")))
    ref = get("ref", lambda: W.Checkbox(value=True, description="classical band"))
    btn = get("btn", lambda: W.Button(description="Plot", button_style="primary",
                                      layout=W.Layout(width="160px", height="34px")))
    hint = get("hint", lambda: W.HTML())
    out = get("out", lambda: W.Output())
    # drop only the observers an earlier run of this cell bound; unobserve_all()
    # would also strip ipywidgets' own option observers and break the dropdowns
    for w, fn in reg.pop("_bound", []):
        try:
            w.unobserve(fn, names="value")
        except ValueError:
            pass
    try:
        btn._click_handlers.callbacks.clear()
    except Exception:  # noqa: BLE001 -- ipywidgets 7
        btn._click_handlers.callbacks = []
    out.clear_output()
    # refresh options on reuse: the data may have changed since the last run
    mode.options = modes
    if mode.value not in modes:
        mode.value = modes[1]
    task.options = tasks
    if task.value not in tasks:
        task.value = tasks[0]
    run.options = runs
    if runs and run.value not in runs:
        run.value = runs[0]

    def set_metrics(kind, frame):
        mets = [m for m in METRICS[kind] if m in frame and frame[m].notna().any()]
        metric.options = [(METRIC_LABEL.get(m, m), m) for m in mets]
        metric.value = A.METRIC[kind]

    def refresh(_=None):
        md = mode.value
        qb = md == QB
        task.layout.display = "none" if qb else ""
        run.layout.display = "" if qb else "none"
        if qb:
            x = r[r.run == run.value]
            set_metrics(_run_kind(run.value), x)
            sc = sorted(x.scheme.unique())
            scheme.options = (["all"] if len(sc) > 1 else []) + sc
            scheme.value = scheme.options[0]
            hs = [h for h in HEAD_SYMBOL if h in set(x["head"])]
            head.options, head.disabled = ["all"] + hs, False
            head.value = "all"
            opts = sorted(x.series.unique(), key=_ion_order)
            lines.options = [(_label(o), o) for o in opts]
            lines.value = tuple(opts)
            hint.value = ("<i>" + ION_RUNS[run.value] + ". Seed 42. Grey band only on the "
                          "full-data runs (the 20k-row subsample has no classical run).</i>")
            return
        t = task.value
        kind = _kind(t)
        v = d[d.task == t]
        set_metrics(kind, v)
        scheme.options = schemes(d, t)
        scheme.value = scheme.options[0]
        if md.startswith("Circuit"):
            heads = [h for h in HEAD_ORDER[kind] if h in set(v[v.stage == "A"]["head"])]
            head.options, head.disabled = heads, False
            head.value = A.PRIMARY_HEAD[kind]
            opts = sorted(v[(v.stage == "A") & (v.arm == "quantum")].config.unique(),
                          key=_order_key)
            lines.options = [(f"{o} ({qubits(o)})", o) for o in opts]
            lines.value = tuple(opts)
            hint.value = ("<i>Lines = circuits to show (Ctrl/Shift-click). Each row shows "
                          "quantum, dials, RFF.</i>")
        elif md.startswith("Head"):
            head.options, head.disabled = ["(x axis)"], True
            b = sorted(v[v.stage == "B"].series.unique(), key=_order_key)
            a = sorted(v[(v.stage == "A") & (v.arm == "quantum")].series.unique(),
                       key=_order_key)
            lines.options = ([(f"B · {_label(s)}", s) for s in b]
                             + [(f"A · {_label(s)}, 2 heads only", f"A·{s}") for s in a])
            lines.value = tuple(b)
            hint.value = ("<i>B = picked circuit + controls across all heads; A = every "
                          "circuit, lbfgs/mlp256 (pca32/ae256) only.</i>")
        else:
            h = A._state().get("head", {}).get(t)
            head.options, head.disabled = [h or "?"], True
            s = sorted(_seed_rows(v, h).series.unique(), key=_order_key)
            lines.options = [(_label(x), x) for x in s]
            lines.value = tuple(s)
            hint.value = ("<i>Seed 42 from stage B, seeds 43-44 from stage C, under the head "
                          "stage C used.</i>")

    def plot(_=None):
        with out:
            clear_output(wait=True)
            sel = list(lines.value)
            if not sel:
                print("select at least one line")
                return
            md = mode.value
            if md == QB:
                fig = fig_qubits(r, run.value, metric=metric.value,
                                 schemes_=None if scheme.value == "all" else [scheme.value],
                                 series=sel, heads=None if head.value == "all" else [head.value],
                                 show_ref=ref.value)
            elif md.startswith("Circuit"):
                fig = fig_screen(d, task.value, head=head.value, metric=metric.value,
                                 scheme=scheme.value, circuits=sel, show_ref=ref.value)
            elif md.startswith("Head"):
                fig = fig_heads(d, [task.value], metric=metric.value, scheme=scheme.value,
                                series=sel, include_stage_a=any(s.startswith("A·") for s in sel),
                                show_ref=ref.value)
                fig.update_layout(height=520)
            else:
                fig = fig_seeds(d, [task.value], metric=metric.value, scheme=scheme.value,
                                series=sel, show_ref=ref.value)
                fig.update_layout(height=480)
            fig.show()

    for w in (task, mode, run):
        w.observe(refresh, names="value")
    reg["_bound"] = [(task, refresh), (mode, refresh), (run, refresh)]
    btn.on_click(plot)
    refresh()
    display(W.VBox([mode, W.HBox([task, run, metric, scheme, head]),
                    W.HBox([lines, W.VBox([ref, btn])]), hint, out]))
