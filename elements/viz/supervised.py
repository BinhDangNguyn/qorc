"""Supervised-specific views: per-class F1, confusion matrices, operational rates.

Load with ``load_results(run, keep=("per_class_f1", "confusion"))`` so the
nested fields these need are kept.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elements.viz.scaling import DIV, _annotated, config_order, title
from elements.viz.style import INK, savefig, use_style


def _pick(df, arm, config, head, scheme):
    fid_col = df[(df.arm == arm) & (df["head"] == head) & (df.scheme == scheme)]
    if arm == "dials":
        n_modes = int(config.split("p")[1].rstrip("m").split("m")[0])
        return fid_col[fid_col.n_modes == n_modes]
    if arm == "raw":
        return fid_col
    return fid_col[fid_col.config == config]


def per_class_table(df, scheme, arm="quantum", head=None, min_support=0) -> pd.DataFrame:
    """Per-class F1 for one arm and head, one column per reservoir config."""
    sub = df[(df.scheme == scheme) & (df.arm == arm)]
    head = head or sub["head"].iloc[0]
    cols = {}
    for cfg in config_order(df):
        r = _pick(df, arm, cfg, head, scheme)
        if len(r):
            cols[cfg] = pd.Series(r.iloc[0]["per_class_f1"])
    return pd.DataFrame(cols)


def per_class_gain(df, scheme, head, control="dials", out=None):
    """Heatmap of per-class F1 gain (quantum - control), classes x configs."""
    use_style()
    q = per_class_table(df, scheme, "quantum", head)
    c = {}
    for cfg in q.columns:
        r = _pick(df, control, cfg, head, scheme)
        if len(r):
            c[cfg] = pd.Series(r.iloc[0]["per_class_f1"])
    g = (q - pd.DataFrame(c)).dropna(how="all")
    fig, ax = plt.subplots(figsize=(1.3 * g.shape[1] + 3, 0.36 * g.shape[0] + 1.8))
    lim = max(0.05, np.nanmax(np.abs(g.values)))
    _annotated(ax, g, DIV, "{:+.2f}", -lim, lim)
    ax.set_title(f"{title('nslkdd', scheme)}: per-class F1, quantum - {control} ({head})")
    fig.tight_layout()
    savefig(fig, out)
    return fig


def confusion(df, scheme, config, head, arms=("quantum", "dials"), out=None):
    """Row-normalised confusion matrices: of the true class, where did it go?"""
    use_style()
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 4.8), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        r = _pick(df, arm, config, head, scheme)
        if r.empty:
            ax.axis("off")
            continue
        cm = np.array(r.iloc[0]["confusion"], dtype=float)
        names = list(r.iloc[0]["per_class_f1"])
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        if len(names) <= 8:
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_title(f"{arm} / {config} / {head}", fontsize=10)
        ax.grid(False)
    axes[0][0].set_ylabel("true")
    fig.tight_layout()
    savefig(fig, out)
    return fig


def operational_table(df: pd.DataFrame) -> pd.DataFrame:
    """False-alarm rate, miss rate and attack-type accuracy for every trained head."""
    cols = ["scheme", "step", "config", "arm", "head", "false_alarm_rate", "miss_rate",
            "attack_type_acc", "macro_f1"]
    have = [c for c in cols if c in df]
    return df[have].sort_values([c for c in ("scheme", "step", "arm", "head") if c in have])
