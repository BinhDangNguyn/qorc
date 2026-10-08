"""Plots for the automatic circuit / head study (``elements.experiments.autostudy``).

Every panel carries one faint grey line: the best standalone classical model
on the raw features, trained on the same split and seed (stage K) -- the bar
any reservoir has to clear. The published numbers use the duplicate-sharing
protocol, so they are not drawn on the dedup tasks; ``paper_check`` shows that
the same pipeline reproduces them under the paper's own protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from elements.experiments import autostudy as A
from elements.viz.style import ARM_COLOR, ARM_MARKER, GRID, INK, MUTED, use_style

REF_COLOR = "#8a8984"  # neutral grey: reference, not a series

TITLES = {
    "nsl_sup": "NSL-KDD, supervised", "nsl_nov": "NSL-KDD, zero-day",
    "unsw_sup": "NF-UNSW-NB15-v2 (dedup), supervised",
    "unsw_nov": "NF-UNSW-NB15-v2 (dedup), zero-day",
    "cic_sup": "NF-CSE-CIC-IDS2018-v2 (dedup), supervised",
    "cic_nov": "NF-CSE-CIC-IDS2018-v2 (dedup), zero-day",
}
YLABEL = {"supervised": "macro-F1 (mean over label schemes)",
          "novelty": "detection at 1% false alarms (macro)"}
HEAD_ORDER = {
    "supervised": ["lbfgs", "mlp32", "mlp128", "mlp256", "mlp512", "mlp2048",
                   "mlp512x512"],
    "novelty": ["pca32", "ae64-b32", "ae256-b32", "ae1024-b32", "ae512x512-b32"],
}
VIEW_LABEL = {"dials": "dials (PCA input)", "rff": "RFF, same width", "raw": "raw input"}


def _kind(task: str) -> str:
    return A.TASKS[task][0]


def classical_reference(task: str) -> tuple[float, str]:
    """(score, model) of the best standalone classical model, primary metric."""
    df = A.load_results(A.classical_run(task))
    if df.empty:
        return np.nan, "not run"
    s = df.groupby("head")[A.METRIC[_kind(task)]].mean()
    return float(s.max()), str(s.idxmax())


def _ref_line(ax, task, horizontal=True) -> str:
    """Draw the faint classical reference; return its label for the title
    (a label inside the panel collides with the data labels)."""
    v, name = classical_reference(task)
    if np.isnan(v):
        return "classical reference not run"
    line = ax.axhline if horizontal else ax.axvline
    line(v, color=REF_COLOR, lw=6, alpha=0.25, zorder=0, solid_capstyle="butt")
    return f"grey band = classical best: {name} ({v:.3f})"


def _arm_of(view: str) -> str:
    for arm in ("dials", "rff", "raw"):
        if view.startswith(arm):
            return arm
    return "quantum"


# ------------------------------------------------------------------ figures


def fig_seeds(tasks=None):
    """Stage C: per task, each view's score on 3 seeds (dots) and their mean."""
    use_style()
    tasks = tasks or list(A.TASKS)
    ncol = 2
    fig, axes = plt.subplots(int(np.ceil(len(tasks) / ncol)), ncol,
                             figsize=(12, 3.6 * np.ceil(len(tasks) / ncol)), squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        c = A.seeds_table(task)
        if c.empty:
            ax.set_title(f"{TITLES[task]} - no stage C yet", fontsize=10)
            continue
        quantum = [v for v in c.columns if _arm_of(v) == "quantum"]
        order = sorted(quantum, key=lambda v: v.endswith("_nomem")) + [
            v for v in ("dials", "rff", "raw") if v in c.columns]
        for i, view in enumerate(order):
            arm = _arm_of(view)
            hollow = view.endswith("_nomem")
            y = c[view].dropna()
            ax.scatter(np.full(len(y), i), y, s=22, alpha=0.45, marker=ARM_MARKER[arm],
                       zorder=2, edgecolors=ARM_COLOR[arm],
                       facecolors="none" if hollow else ARM_COLOR[arm])
            ax.errorbar(i + 0.18, y.mean(), yerr=y.std(ddof=1) if len(y) > 1 else 0,
                        fmt=ARM_MARKER[arm], ms=8, color=ARM_COLOR[arm], capsize=3,
                        mfc="white" if hollow else ARM_COLOR[arm], zorder=3)
            ax.annotate(f"{y.mean():.3f}", (i + 0.18, y.mean()), xytext=(8, -3),
                        textcoords="offset points", fontsize=8, color=INK)
        ref = _ref_line(ax, task)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([("no memory" if v.endswith("_nomem") else v) if _arm_of(v) ==
                            "quantum" else VIEW_LABEL[v] for v in order],
                           rotation=15, ha="right", fontsize=8)
        ax.set_xlim(-0.5, len(order) - 0.2)
        ax.set_title(f"{TITLES[task]}, head {A._state().get('head', {}).get(task)}\n{ref}",
                     fontsize=9)
        ax.set_ylabel(YLABEL[_kind(task)], fontsize=8)
    for ax in axes.flat[len(tasks):]:
        ax.axis("off")
    fig.suptitle("Stage C: picked circuit vs controls, 3 seeds (dots) and mean +- sd",
                 fontsize=12)
    fig.tight_layout()
    return fig


def fig_screen(task: str):
    """Stage A: every circuit's quantum score next to its dials / rff controls."""
    use_style()
    kind = _kind(task)
    t = A.screen(task)
    t = t[t["head"] == A.PRIMARY_HEAD[kind]].sort_values("quantum")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.5))
    y = np.arange(len(t))
    for arm, dy in (("dials", -0.18), ("rff", 0.18)):
        ax.scatter(t[arm], y + dy, marker=ARM_MARKER[arm], s=30, color=ARM_COLOR[arm],
                   alpha=0.7, label=VIEW_LABEL[arm], zorder=2)
    ax.scatter(t.quantum, y, marker="o", s=55, color=ARM_COLOR["quantum"],
               label="quantum readout", zorder=3)
    for yi, v in zip(y, t.quantum):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(7, -3), textcoords="offset points",
                    fontsize=8, color=INK)
    raw = t.raw.iloc[0]
    ax.axvline(raw, color=ARM_COLOR["raw"], lw=1.2, ls=":", label=VIEW_LABEL["raw"])
    ref = _ref_line(ax, task, horizontal=False)
    ax.set_yticks(y)
    ax.set_yticklabels(t.config, fontsize=8)
    ax.set_xlabel(f"{YLABEL[kind]}, head {A.PRIMARY_HEAD[kind]}")
    pick = A._state().get("pick", {}).get(task)
    ax.set_title(f"Stage A circuit screen - {TITLES[task]}"
                 + (f"\npicked: {A._tag(pick)}" if pick else "") + f"\n{ref}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    return fig


def fig_heads(tasks=None):
    """Stage B: score against head size, one line per feature view."""
    use_style()
    tasks = tasks or list(A.TASKS)
    ncol = 2
    fig, axes = plt.subplots(int(np.ceil(len(tasks) / ncol)), ncol,
                             figsize=(12, 3.6 * np.ceil(len(tasks) / ncol)), squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        b = A.head_table(task)
        if b.empty:
            ax.set_title(f"{TITLES[task]} - no stage B yet", fontsize=10)
            continue
        heads = [h for h in HEAD_ORDER[_kind(task)] if h in b.columns]
        x = np.arange(len(heads))
        for view, row in b[heads].iterrows():
            arm = _arm_of(view)
            nomem = view.endswith("_nomem")
            ax.plot(x, row.values, marker=ARM_MARKER[arm], color=ARM_COLOR[arm], lw=2,
                    ms=7, ls="--" if nomem else "-", mfc="white" if nomem else None,
                    label=("quantum, no memory" if nomem else f"quantum {view}")
                    if arm == "quantum" else VIEW_LABEL[arm])
        ref = _ref_line(ax, task)
        ax.set_xticks(x)
        ax.set_xticklabels(heads, fontsize=8)
        ax.set_title(f"{TITLES[task]}\n{ref}", fontsize=9)
        ax.set_ylabel(YLABEL[_kind(task)], fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axes.flat[len(tasks):]:
        ax.axis("off")
    fig.suptitle("Stage B: how big must the classical head be? (seed 42)", fontsize=12)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ tables


def summary_table() -> pd.DataFrame:
    """One row per task: pick, head, quantum mean over seeds, best control,
    classical reference, and paired differences (seed by seed)."""
    rows = []
    for task in A.TASKS:
        c = A.seeds_table(task)
        if c.empty:
            continue
        q = next(v for v in c.columns if _arm_of(v) == "quantum" and not v.endswith("_nomem"))
        nm = [v for v in c.columns if v.endswith("_nomem")]
        ctrl = c[[v for v in ("dials", "rff", "raw") if v in c.columns]].max(axis=1)
        ref, ref_name = classical_reference(task)
        rows.append({
            "task": task, "circuit": q, "head": A._state()["head"][task],
            "quantum": c[q].mean(), "best control": ctrl.mean(),
            "quantum - control": (c[q] - ctrl).mean(),
            "quantum - no memory": (c[q] - c[nm[0]]).mean() if nm else np.nan,
            "classical best": ref, "classical model": ref_name, "seeds": len(c),
        })
    return pd.DataFrame(rows).set_index("task")


#: Sarhan, Layeghy & Portmann (MONET 2022), Extra Trees on NF-UNSW-NB15-v2,
#: random 70/30 split WITHOUT dedup (Table 8 binary, Table 9 multi-class).
PAPER_UNSW = {"accuracy": 0.9973, "attack F1": 0.97, "detection rate": 0.9707,
              "false alarm rate": 0.0016, "weighted F1 (10 classes)": 0.99}


def paper_check() -> pd.DataFrame:
    """Paper vs this pipeline under the paper's protocol (``nfunsw_classical``)
    and under dedup (``nfunsw_dedup_classical``), Extra Trees."""
    out = {"Sarhan et al. 2022": PAPER_UNSW}
    for label, run in (("ours, paper protocol", "nfunsw_classical"),
                       ("ours, dedup", "nfunsw_dedup_classical")):
        df = A.load_results(run)
        if df.empty:
            continue
        et = df[df["head"].str.startswith("extratrees")]
        b = et[et.scheme == "binary"].iloc[0]
        m = et[et.scheme == "attack10"].iloc[0]
        out[label] = {"accuracy": b.accuracy, "attack F1": b.per_class_f1["attack"],
                      "detection rate": 1 - b.miss_rate,
                      "false alarm rate": b.false_alarm_rate,
                      "weighted F1 (10 classes)": m.weighted_f1}
    return pd.DataFrame(out).T
