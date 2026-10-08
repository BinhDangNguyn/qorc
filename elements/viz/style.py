"""Shared plotting style: one validated palette, recessive axes, value labels.

Every figure in ``elements.viz`` uses these tokens so a series keeps its colour
across notebooks: quantum is always blue, its PCA dials always orange.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: Where the image / forecasting benchmark JSONs live (``load`` reads them).
RUNS = Path(__file__).resolve().parents[2] / "runs" / "benchmarks"

# Validated categorical palette (checked with the dataviz validator: lightness
# band, chroma floor, CVD separation and normal-vision floor all PASS).
# Contrast vs surface WARNs for aqua/yellow, which obliges visible value
# labels -- every chart below carries them.
QUANTUM = "#2a78d6"  # slot 1, blue    -- the reservoir readout
DIALS = "#eb6834"  # slot 2, orange -- its own PCA input (the control)
CLASSICAL = "#1baf7a"  # slot 3, aqua   -- classical reference
RAW = "#eda100"  # slot 4, yellow -- untouched input
QDIALS = "#4a3aa7"  # slot 7, violet  -- quantum readout + its dials
RFFDIALS = "#e87ba4"  # slot 5, magenta -- RFF + dials
# (six-slot set re-validated: all adjacent-pair checks PASS; contrast WARN on
# aqua / yellow / magenta -> distinct markers and table views accompany them)

ARM_COLOR = {"quantum": QUANTUM, "dials": DIALS, "rff": CLASSICAL, "raw": RAW,
             "qdials": QDIALS, "rffdials": RFFDIALS}
ARM_MARKER = {"quantum": "o", "dials": "s", "rff": "^", "raw": "D", "qdials": "P",
              "rffdials": "X"}
ARM_LABEL = {"quantum": "Quantum readout", "dials": "PCA dials (control)",
             "rff": "Random Fourier, same width", "raw": "Raw input (no bottleneck)",
             "qdials": "Quantum + dials", "rffdials": "RFF + dials"}

# Ordinal blue ramp (light -> dark) for ordered things such as head size.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

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
    """Read every benchmark JSON in runs/benchmarks matching a glob into one frame."""
    rows: list[dict] = []
    for path in sorted(RUNS.glob(pattern)):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(
            f"no results matched {pattern!r} in {RUNS}/. "
            "Run the benchmarks with --out first (see notebooks/40_*)."
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


def savefig(fig, path: str | Path | None) -> None:
    """Save next to the run it describes, when a path is given."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
