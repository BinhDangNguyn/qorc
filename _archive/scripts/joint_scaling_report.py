"""Figures and tables for the joint (quantum x classical) scaling sweep.

Reads ``qrc_runs/joint_scaling/results.jsonl`` -- nothing is hardcoded -- and
writes PNGs plus ``SUMMARY.md`` next to it. Seeds are averaged when more than
one is present; the std is reported beside every mean.

Run:  python joint_scaling_report.py [--run qrc_runs/joint_scaling]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from qrcpkg_results import CLASSICAL, DIALS, GRID, INK, MUTED, QUANTUM, RAW, use_style

ALL_HEADS = ["linear", "mlp128", "mlp512", "mlp2048", "mlp512x2",
             "pca8", "pca32", "ae128", "ae512", "ae2048", "ae512x2"]
HEAD_ORDER = ALL_HEADS  # narrowed to the heads present once the data is loaded
HEAD_LABEL = {"linear": "linear", "mlp128": "MLP 128", "mlp512": "MLP 512",
              "mlp2048": "MLP 2048", "mlp512x2": "MLP 512x2",
              "pca8": "PCA-8", "pca32": "PCA-32", "ae128": "AE 128", "ae512": "AE 512",
              "ae2048": "AE 2048", "ae512x2": "AE 512x2", "iforest": "IForest"}
# Slots 5-6 of the same validated order (light run PASSES all six adjacent-pair
# checks; contrast WARN on slots 3, 4, 6 -> distinct markers + SUMMARY.md table).
QDIALS = "#4a3aa7"  # violet  -- quantum readout + its dials
RFFDIALS = "#e87ba4"  # magenta -- RFF + dials, its classical counterpart
ARM_COLOR = {"quantum": QUANTUM, "dials": DIALS, "rff": CLASSICAL, "raw": RAW,
             "qdials": QDIALS, "rffdials": RFFDIALS}
ARM_LABEL = {"quantum": "Quantum readout", "dials": "PCA dials (control)",
             "rff": "Random Fourier, same dim", "raw": "Raw input (no bottleneck)",
             "qdials": "Quantum + dials", "rffdials": "RFF + dials"}
ARM_MARKER = {"quantum": "o", "dials": "s", "rff": "^", "raw": "D",
              "qdials": "P", "rffdials": "X"}

# Ordinal blue ramp (reference palette steps 250 -> 650): heads are ordered by
# size, so lightness carries order and identity is also in the legend.
HEAD_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

PANELS = [("nslkdd", "families6"), ("nslkdd", "all23"), ("kmnist", "all"),
          ("fashion_mnist", "all"), ("mnist", "all"), ("nslkdd_novelty", "novelty")]
TITLE = {("nslkdd", "families6"): "NSL-KDD families6 (7 cl.)",
         ("nslkdd", "all23"): "NSL-KDD all23 (23 cl.)",
         ("kmnist", "all"): "K-MNIST (10 cl.)",
         ("fashion_mnist", "all"): "Fashion-MNIST (10 cl.)",
         ("mnist", "all"): "MNIST (10 cl.)",
         ("nslkdd_novelty", "novelty"): "NSL-KDD zero-day (normal-only)"}


def load(run: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    df = pd.DataFrame(rows).drop(columns=["per_class_f1", "per_type"], errors="ignore")
    return df


def agg(df: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    g = df.groupby(keys)[metric]
    return g.agg(mean="mean", std="std", n="count").reset_index()


def config_order(df) -> list[str]:
    """Reservoir configs, smallest readout first."""
    q = df[df.arm == "quantum"].drop_duplicates("config")
    return list(q.sort_values(["feature_dim", "n_modes"])["config"])


def panels_present(df):
    q = df[df.arm == "quantum"]  # a dataset still running may only have `raw`
    have = set(zip(q.dataset, q.scheme))
    return [p for p in PANELS if p in have]


def raw_ceiling(df, ds, sc, metric):
    sub = agg(df[(df.dataset == ds) & (df.scheme == sc) & (df.arm == "raw")], ["head"], metric)
    return sub.loc[sub["mean"].idxmax()] if len(sub) else None


# ------------------------------------------------------------------------ figures


def fig_quantum_by_head(df, metric, out):
    """Headline: does a bigger head lift the plateau of the reservoir curve?"""
    panels = panels_present(df)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.9), squeeze=False)
    for ax, (ds, sc) in zip(axes[0], panels):
        q = agg(df[(df.dataset == ds) & (df.scheme == sc) & (df.arm == "quantum")],
                ["config", "feature_dim", "head"], metric)
        for color, head in zip(HEAD_RAMP, HEAD_ORDER):
            s = q[q["head"] == head].sort_values("feature_dim")
            if s.empty:
                continue
            ax.errorbar(s["feature_dim"], s["mean"], yerr=s["std"].fillna(0), color=color,
                        marker="o", markersize=6, linewidth=2, capsize=2,
                        markeredgecolor="white", markeredgewidth=1.2, label=HEAD_LABEL[head])
        ceil = raw_ceiling(df, ds, sc, metric)
        if ceil is not None:
            ax.axhline(ceil["mean"], color=MUTED, linestyle="--", linewidth=1.2)
            ax.annotate(f"raw input + {HEAD_LABEL[ceil['head']]}: {ceil['mean']:.3f}",
                        (0.02, ceil["mean"]), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 4), fontsize=8, color=MUTED)
        ax.set_xscale("log")
        dims = sorted(q["feature_dim"].unique())
        cfgs = q.drop_duplicates("feature_dim").set_index("feature_dim")["config"]
        ax.set_xticks(dims)
        ax.set_xticklabels([f"{cfgs[d]}\n{d:,}" for d in dims], fontsize=8)
        ax.minorticks_off()
        ax.set_title(TITLE[(ds, sc)])
        ax.set_xlabel("reservoir config / readout dim")
    axes[0][0].set_ylabel(metric.replace("_", " "))
    axes[0][-1].legend(loc="lower right", title="classical head", title_fontsize=8)
    fig.suptitle("Quantum readout: scaling the reservoir with each classical head size",
                 y=1.02, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(df, metric, out):
    """The full grid for the quantum arm: which corner is cheapest for a score?"""
    panels = panels_present(df)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (ds, sc) in zip(axes[0], panels):
        q = agg(df[(df.dataset == ds) & (df.scheme == sc) & (df.arm == "quantum")],
                ["config", "head"], metric)
        m = q.pivot(index="config", columns="head", values="mean")
        m = m.reindex(index=[c for c in config_order(df) if c in m.index],
                      columns=[h for h in HEAD_ORDER if h in m.columns])
        im = ax.imshow(m.values, cmap=SEQ, aspect="auto")
        vmin, vmax = np.nanmin(m.values), np.nanmax(m.values)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if np.isnan(v):
                    continue
                dark = (v - vmin) / (vmax - vmin + 1e-12) > 0.55
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else INK)
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels([HEAD_LABEL[h] for h in m.columns], rotation=30, ha="right")
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(m.index)
        ax.grid(False)
        ax.set_title(TITLE[(ds, sc)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0][0].set_ylabel("reservoir (photons p, modes m)")
    fig.suptitle(f"Quantum arm {metric.replace('_', ' ')}: reservoir size x head size",
                 y=1.03, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_gain(df, metric, control, out):
    """Quantum minus a control at the SAME reservoir config and head."""
    panels = panels_present(df)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (ds, sc) in zip(axes[0], panels):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        # dials depend only on n_modes and are recorded once per width, so
        # 4p24m is compared with the dials24 run logged under 3p24m.
        on = ["seed", "n_modes", "head"] if control == "dials" else ["seed", "config", "head"]
        q = sub[sub.arm == "quantum"]
        c = sub[sub.arm == control].drop_duplicates(on)[on + [metric]]
        mg = q.merge(c, on=on, suffixes=("", "_c"))
        if mg.empty:
            ax.set_visible(False)
            continue
        mg["gain"] = mg[metric] - mg[f"{metric}_c"]
        g = mg.groupby(["config", "head"])["gain"].mean().unstack("head")
        g = g.reindex(index=[x for x in config_order(df) if x in g.index],
                      columns=[h for h in HEAD_ORDER if h in g.columns])
        lim = max(0.01, np.nanmax(np.abs(g.values)))
        ax.imshow(g.values, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.6 * lim else INK)
        ax.set_xticks(range(g.shape[1]))
        ax.set_xticklabels([HEAD_LABEL[h] for h in g.columns], rotation=30, ha="right")
        ax.set_yticks(range(g.shape[0]))
        ax.set_yticklabels(g.index)
        ax.grid(False)
        ax.set_title(TITLE[(ds, sc)])
    axes[0][0].set_ylabel("reservoir config")
    name = {"dials": "its own PCA dials", "rff": "random Fourier features of equal width"}[control]
    fig.suptitle(f"Quantum gain over {name} (blue = quantum ahead)",
                 y=1.03, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_arms_vs_head(df, metric, config, out, arms=("quantum", "rff", "dials", "raw"),
                     title=None):
    """At one reservoir size: does the quantum lead survive a bigger head?"""
    panels = panels_present(df)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.7), squeeze=False)
    for ax, (ds, sc) in zip(axes[0], panels):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        n_modes = int(config.split("p")[1].rstrip("m"))
        pick = {
            "dials": lambda s: s[s.n_modes == n_modes],
            "raw": lambda s: s,
        }
        sel = {a: pick.get(a, lambda s: s[s.config == config])(sub[sub.arm == a])
               for a in arms}
        sel = {a: s for a, s in sel.items() if len(s)}
        x = np.arange(len(HEAD_ORDER))
        for arm, s in sel.items():
            a = agg(s.drop_duplicates(["seed", "head"]), ["head"], metric).set_index("head")
            a = a.reindex(HEAD_ORDER)
            ax.errorbar(x, a["mean"], yerr=a["std"].fillna(0), color=ARM_COLOR[arm],
                        marker=ARM_MARKER[arm], markersize=7, linewidth=2, capsize=2,
                        markeredgecolor="white", markeredgewidth=1.2, label=ARM_LABEL[arm])
        ax.set_xticks(x)
        ax.set_xticklabels([HEAD_LABEL[h] for h in HEAD_ORDER], rotation=30, ha="right")
        ax.set_title(TITLE[(ds, sc)])
    axes[0][0].set_ylabel(metric.replace("_", " "))
    axes[0][-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(title or f"Every feature view as the head grows ({config}: quantum "
                 f"and RFF at the same width)", y=1.03, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_vs_modes(df, metric, out, heads=("linear", "mlp512x2")):
    """Information in vs features out: score against the number of modes.

    Modes set how many PCA components enter the reservoir; photons set how
    many Fock features come out. Plotting against modes, one line per photon
    number, separates the two.
    """
    panels = panels_present(df)
    fig, axes = plt.subplots(len(panels), len(heads), figsize=(5.2 * len(heads), 3.9 * len(panels)),
                             squeeze=False)
    photon_color = {2: "#86b6ef", 3: "#2a78d6", 4: "#104281"}  # ordinal blue ramp
    for r, (ds, sc) in enumerate(panels):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        for c, head in enumerate(heads):
            ax = axes[r][c]
            h = sub[sub["head"] == head]
            for n_ph in sorted(h[h.arm == "quantum"].n_photons.unique()):
                a = agg(h[(h.arm == "quantum") & (h.n_photons == n_ph)], ["n_modes"], metric)
                ax.errorbar(a["n_modes"], a["mean"], yerr=a["std"].fillna(0), marker="o",
                            color=photon_color.get(int(n_ph), QUANTUM), linewidth=2,
                            markersize=6, markeredgecolor="white", capsize=2,
                            label=f"quantum, {int(n_ph)} photons")
            for arm in ("dials", "qdials"):
                s = h[h.arm == arm]
                if arm == "qdials":  # one reservoir per width: the most photons
                    s = s[s.n_photons == s.groupby("n_modes").n_photons.transform("max")]
                a = agg(s.drop_duplicates(["seed", "n_modes"]), ["n_modes"], metric)
                if len(a):
                    ax.errorbar(a["n_modes"], a["mean"], yerr=a["std"].fillna(0),
                                marker=ARM_MARKER[arm], color=ARM_COLOR[arm], linewidth=2,
                                markersize=7, markeredgecolor="white", capsize=2,
                                label=ARM_LABEL[arm] + (" (max photons)" if arm == "qdials" else ""))
            ceil = h[h.arm == "raw"][metric].mean()
            if not np.isnan(ceil):
                ax.axhline(ceil, color=MUTED, linestyle="--", linewidth=1.2)
                ax.annotate(f"raw input: {ceil:.3f}", (0.02, ceil), xycoords=("axes fraction", "data"),
                            textcoords="offset points", xytext=(0, 4), fontsize=8, color=MUTED)
            ax.set_title(f"{TITLE[(ds, sc)]} - {HEAD_LABEL[head]} head")
            ax.set_xlabel("modes = PCA components fed in")
        axes[r][0].set_ylabel(metric.replace("_", " "))
        axes[r][-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Score against the number of modes (= PCA components fed in)",
                 y=1.01, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- tables


def fmt(m, s):
    return f"{m:.4f}" if np.isnan(s) else f"{m:.4f} ± {s:.4f}"


def summary(df, metric) -> str:
    seeds = sorted(int(s) for s in df.seed.unique())
    lines = [f"# Joint scaling results ({metric})", "",
             f"Seeds: {seeds}. Mean ± std over seeds where more than one ran.", ""]
    for ds, sc in panels_present(df):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        lines += [f"## {TITLE[(ds, sc)]}", "", "Quantum arm, config x head:", ""]
        q = agg(sub[sub.arm == "quantum"], ["config", "head"], metric)
        heads = [h for h in HEAD_ORDER if h in set(q["head"])]
        lines.append("| config | dim | " + " | ".join(HEAD_LABEL[h] for h in heads) + " |")
        lines.append("|---|---|" + "---|" * len(heads))
        dims = sub[sub.arm == "quantum"].drop_duplicates("config").set_index("config")["feature_dim"]
        for cfg in [c for c in config_order(sub) if c in set(q["config"])]:
            row = q[q["config"] == cfg].set_index("head")
            cells = [fmt(row.loc[h, "mean"], row.loc[h, "std"]) if h in row.index else "-"
                     for h in heads]
            lines.append(f"| {cfg} | {int(dims[cfg]):,} | " + " | ".join(cells) + " |")
        lines += ["", "Best per feature view:", "",
                  "| view | best setting | " + metric + " |", "|---|---|---|"]
        for arm in ["quantum", "rff", "qdials", "rffdials", "dials", "raw"]:
            a = agg(sub[sub.arm == arm], ["feature_id", "head"], metric)
            if a.empty:
                continue
            b = a.loc[a["mean"].idxmax()]
            lines.append(f"| {arm} | {b['feature_id']} + {HEAD_LABEL[b['head']]} | "
                         f"{fmt(b['mean'], b['std'])} |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("qrc_runs/joint_scaling"))
    p.add_argument("--metric", default="macro_f1")
    p.add_argument("--also", type=Path, nargs="*", default=[],
                   help="other run dirs to merge (e.g. the mode-scaling run)")
    args = p.parse_args()

    use_style()
    df = pd.concat([load(args.run), *[load(a) for a in args.also]], ignore_index=True)
    # the same (dataset, scheme, seed, feature, head) may appear in two runs
    df = df.drop_duplicates(["dataset", "scheme", "seed", "feature_id", "head"])
    global HEAD_ORDER
    HEAD_ORDER = [h for h in ALL_HEADS if h in set(df["head"])]
    figs = args.run / "figs"
    figs.mkdir(exist_ok=True)
    fig_quantum_by_head(df, args.metric, figs / "1_quantum_by_head.png")
    fig_heatmap(df, args.metric, figs / "2_quantum_grid.png")
    fig_gain(df, args.metric, "dials", figs / "3_gain_vs_dials.png")
    fig_gain(df, args.metric, "rff", figs / "4_gain_vs_rff.png")
    fig_arms_vs_head(df, args.metric, "4p24m", figs / "5_arms_vs_head_4p24m.png")
    fig_arms_vs_head(df, args.metric, "3p12m", figs / "6_arms_vs_head_3p12m.png")
    if (df.arm == "qdials").any():
        hybrid = ("qdials", "rffdials", "quantum", "dials")
        for cfg in ("3p12m", "4p24m"):
            fig_arms_vs_head(df, args.metric, cfg, figs / f"7_hybrid_readout_{cfg}.png",
                             arms=hybrid, title=f"Readout sees reservoir AND its input "
                             f"({cfg})")
    if df.n_modes.max() > 24:
        fig_vs_modes(df, args.metric, figs / "8_vs_modes.png", heads=(HEAD_ORDER[0], HEAD_ORDER[-1]))
    (args.run / "SUMMARY.md").write_text(summary(df, args.metric), encoding="utf-8")
    print(f"wrote {figs}/ and {args.run / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
