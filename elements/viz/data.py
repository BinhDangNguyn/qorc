"""Dataset overview: class balance and how much PCA keeps at each mode count."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from elements.viz.style import INK, MUTED, QUANTUM, savefig, use_style


def class_table(ds, schemes=("families6", "all23")) -> pd.DataFrame:
    """Row counts per class for each label scheme of an ``NSLKDD`` object."""
    out = []
    for s in schemes:
        counts = ds.class_counts(s)
        out.append(pd.DataFrame({"scheme": s, "class": counts.index, "rows": counts.values,
                                 "share": counts.values / counts.values.sum()}))
    return pd.concat(out, ignore_index=True)


def class_bars(ds, scheme="all23", out=None):
    """Class sizes on a log axis: the imbalance every metric choice follows from."""
    use_style()
    counts = ds.class_counts(scheme).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(counts) + 1.5))
    ax.barh(counts.index, counts.values, color=QUANTUM, height=0.7)
    for y, v in enumerate(counts.values):
        ax.text(v * 1.08, y, f"{v:,}", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("rows (log scale)")
    ax.set_title(f"NSL-KDD {scheme}: {len(counts)} classes, "
                 f"{counts.max() / counts.min():,.0f}x largest / smallest")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    savefig(fig, out)
    return fig


def pca_variance(X: np.ndarray, marks=(12, 20, 24, 32, 64), out=None):
    """Cumulative variance retained against components -- i.e. against modes."""
    use_style()
    Z = StandardScaler().fit_transform(X)
    Z = Z[:, Z.std(0) > 1e-6]
    k = min(Z.shape[1], max(marks) * 2)
    cum = np.cumsum(PCA(n_components=k, random_state=0).fit(Z).explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(np.arange(1, k + 1), cum, color=QUANTUM, lw=2)
    for m in marks:
        if m <= k:
            ax.plot(m, cum[m - 1], "o", color=QUANTUM, mec="white", ms=7)
            ax.annotate(f"{m} modes: {cum[m - 1]:.0%}", (m, cum[m - 1]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK)
    ax.set_xlabel("PCA components = modes")
    ax.set_ylabel("cumulative variance retained")
    ax.set_ylim(0, 1.02)
    ax.set_title("How much of the input each mode count lets into the reservoir",
                 color=INK)
    ax.axhline(1, color=MUTED, lw=0.8)
    fig.tight_layout()
    savefig(fig, out)
    return fig
