"""Shared plotting setup and data loading for the results notebook.

Kept beside the notebook rather than inside it so the notebook stays readable
and the styling is defined once.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path("qrc_runs")

# Validated categorical palette (checked with the dataviz validator: lightness
# band, chroma floor, CVD separation and normal-vision floor all PASS).
# Contrast vs surface WARNs for aqua/yellow, which obliges visible value
# labels -- every chart below carries them.
QUANTUM = "#2a78d6"  # slot 1, blue    -- the reservoir readout
DIALS = "#eb6834"  # slot 2, orange -- its own PCA input (the control)
CLASSICAL = "#1baf7a"  # slot 3, aqua   -- classical reference
RAW = "#eda100"  # slot 4, yellow -- untouched input

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdcd8"
POSITIVE = "#1baf7a"
NEGATIVE = "#eb6834"


def use_style() -> None:
    """Recessive axes, no chartjunk, text in ink tokens rather than series colors."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load(pattern: str) -> pd.DataFrame:
    """Read every qrc_runs JSON matching a glob into one frame."""
    rows: list[dict] = []
    for path in sorted(RUNS.glob(pattern)):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(
            f"no results matched {pattern!r} in {RUNS}/. "
            "Run the benchmarks with --out first; see the notebook's first cell."
        )
    return pd.DataFrame(rows)


def label_bars(ax, bars, fmt="{:.3f}", pad=3, color=INK) -> None:
    """Value labels on every bar.

    Required, not decorative: two palette slots WARN on contrast against a
    white surface, and the validator's rule is that a contrast WARN obliges
    visible labels or a table view.
    """
    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = pad if height >= 0 else -pad
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            va=va,
            fontsize=8,
            color=color,
        )


PRETTY = {
    "kmnist": "K-MNIST",
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "organcmnist": "OrganC",
    "breastmnist": "Breast",
    "pneumoniamnist": "Pneumonia",
    "retinamnist": "Retina",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name)
