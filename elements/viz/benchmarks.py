"""Image classification, image anomaly detection and forecasting benchmarks.

Reads the JSON files the benchmark CLIs write into ``runs/benchmarks``:

    python -m elements.experiments.images --task classify --dataset kmnist \\
        --out runs/benchmarks/classify_kmnist.json
    python -m elements.experiments.temporal
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elements.viz.style import (
    CLASSICAL, DIALS, INK, MUTED, NEGATIVE, POSITIVE, QUANTUM, RAW, RUNS, label_bars, load,
    pretty, savefig, use_style,
)


def image_table() -> pd.DataFrame:
    img = load("classify_*.json")
    piv = (img.pivot_table(index="dataset", columns="arm", values="macro_f1")
              .reindex(columns=["dials", "quantum", "logreg_raw", "raw"]))
    piv["gain"] = piv["quantum"] - piv["dials"]
    return piv.sort_values("gain", ascending=False)


def image_bars(out=None):
    """Quantum readout vs its own input vs a classical reference, per dataset."""
    use_style()
    piv = image_table()
    x, w = np.arange(len(piv)), 0.26
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for off, col, color, name in [(-w, "dials", DIALS, "PCA dials (control)"),
                                  (0, "quantum", QUANTUM, "Quantum readout"),
                                  (w, "logreg_raw", CLASSICAL, "Logistic regression, 784 px")]:
        label_bars(ax, ax.bar(x + off, piv[col], w * 0.92, label=name, color=color),
                   fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(d) for d in piv.index])
    ax.set_ylabel("macro F1")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncols=3)
    ax.set_title("Quantum readout vs its own input vs a classical reference")
    fig.tight_layout()
    savefig(fig, out)
    return fig


def image_gain(out=None):
    use_style()
    gains = image_table()["gain"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars = ax.bar([pretty(d) for d in gains.index], gains, 0.6,
                  color=[POSITIVE if g > 0 else NEGATIVE for g in gains])
    label_bars(ax, bars, fmt="{:+.3f}")
    ax.axhline(0, color=INK, lw=1.1)
    ax.set_ylabel("macro F1 gain\n(quantum - dials)")
    ax.set_title("Where the reservoir helps and where it hurts")
    ax.margins(y=0.22)
    fig.tight_layout()
    savefig(fig, out)
    return fig


def pca_rule(out=None):
    """Gain against the variance PCA(12) retains -- a cheap screen before any dataset."""
    use_style()
    rule = pd.DataFrame(json.loads((RUNS / "pca_rule.json").read_text(encoding="utf-8")))
    r = np.corrcoef(rule["pca12_var"], rule["gain"])[0, 1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.scatter(rule["pca12_var"], rule["gain"], s=150, zorder=3, edgecolor="white", lw=1.6,
               c=[POSITIVE if g > 0 else NEGATIVE for g in rule["gain"]])
    fit = np.polyfit(rule["pca12_var"], rule["gain"], 1)
    xs = np.linspace(0.28, 1.0, 50)
    ax.plot(xs, np.polyval(fit, xs), color=MUTED, lw=1.4, ls="--", zorder=2)
    ax.axhline(0, color=INK, lw=1.1)
    for _, row in rule.iterrows():
        ax.annotate(pretty(row["dataset"]), (row["pca12_var"], row["gain"]),
                    textcoords="offset points", xytext=(0, 12 if row["gain"] > 0 else -18),
                    ha="center", fontsize=9, color=INK)
    ax.set_xlabel("variance retained by PCA(12)")
    ax.set_ylabel("macro F1 gain (quantum - dials)")
    ax.set_title(f"The PCA rule: gain vs retained variance, r = {r:+.2f}")
    fig.tight_layout()
    savefig(fig, out)
    return fig


def anomaly_table() -> pd.DataFrame:
    anom = load("anomaly_*.json")
    return (anom.pivot_table(index="dataset", columns="arm", values="auc")
                .reindex(columns=["dials", "quantum", "raw", "iforest_raw"]))


def anomaly_bars(out=None):
    use_style()
    ap = anomaly_table()
    x, w = np.arange(len(ap)), 0.2
    fig, ax = plt.subplots(figsize=(10, 4))
    specs = [("dials", DIALS, "PCA dials (control)"), ("quantum", QUANTUM, "Quantum readout"),
             ("raw", RAW, "Raw pixels"), ("iforest_raw", CLASSICAL, "Isolation Forest")]
    for i, (col, color, name) in enumerate(specs):
        label_bars(ax, ax.bar(x + (i - 1.5) * w, ap[col], w * 0.9, label=name, color=color),
                   fmt="{:.2f}")
    ax.axhline(0.5, color=INK, lw=1.2, ls="--")
    ax.text(0.995, 0.52, "chance", transform=ax.get_yaxis_transform(), ha="right",
            fontsize=9, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(d) for d in ap.index])
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("Image anomaly detection")
    fig.tight_layout()
    savefig(fig, out)
    return fig


def temporal(out=None):
    """NARMA-10 error and linear memory capacity: memoryless vs delay vs feedback vs ESN."""
    use_style()
    t = pd.DataFrame(json.loads((RUNS / "temporal_benchmark.json").read_text(encoding="utf-8")))
    narma = t[t["task"] == "narma10"].set_index("arm")["nrmse"]
    mc = t[t["task"] == "memory_capacity"].set_index("arm")["memory_capacity"]
    arms = ["memoryless", "feedback", "delay", "esn"]
    names = ["Memoryless\n(current)", "Feedback\nloop", "Delay\nembedding", "Classical\nESN"]
    colors = [DIALS, QUANTUM, QUANTUM, CLASSICAL]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    label_bars(axes[0], axes[0].bar(names, [narma[a] for a in arms], 0.6, color=colors),
               fmt="{:.2f}")
    axes[0].axhline(1.0, color=INK, lw=1.2, ls="--")
    axes[0].set_ylabel("NRMSE   (lower is better)")
    axes[0].set_title("NARMA-10: needs 10 steps of memory")
    axes[0].margins(y=0.18)
    label_bars(axes[1], axes[1].bar(names, [mc[a] for a in arms], 0.6, color=colors),
               fmt="{:.1f}")
    axes[1].set_ylabel("linear memory capacity   (higher is better)")
    axes[1].set_title("How far back each arm can see")
    axes[1].margins(y=0.18)
    fig.tight_layout()
    savefig(fig, out)
    return fig
