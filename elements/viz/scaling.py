"""Figures and tables for scaling studies: the reservoir, the head, or both.

Every function takes the flat frame from ``elements.eval.results.load_results``
(one row per trained head) and returns a matplotlib figure or a DataFrame;
pass ``out=`` to also save the figure. ``metric`` is ``macro_f1`` for
supervised runs and ``macro_auc`` (or ``det@0.01``) for novelty runs.

Seeds are averaged when a run has several; with one seed the error bars
vanish and differences under ~0.01-0.02 should not be read as real.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from elements.viz.style import (
    ARM_COLOR, ARM_LABEL, ARM_MARKER, INK, MUTED, QUANTUM, RAMP, savefig, use_style,
)

SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

_KIND_RANK = {"lbfgs": 0, "linear": 1, "mlp": 2, "pca": 3, "ae": 4, "iforest": 5}
_TITLES = {("nslkdd", "families6"): "NSL-KDD families6 (7 classes)",
           ("nslkdd", "all23"): "NSL-KDD all23 (23 classes)",
           ("nslkdd_novelty", "novelty"): "NSL-KDD zero-day (normal-only)",
           ("kmnist", "all"): "K-MNIST", ("mnist", "all"): "MNIST",
           ("fashion_mnist", "all"): "Fashion-MNIST"}
ARM_ORDER = ["quantum", "qdials", "rff", "rffdials", "dials", "raw"]


def title(ds: str, sc: str) -> str:
    return _TITLES.get((ds, sc), f"{ds} / {sc}")


def _kind(label: str) -> str:
    for k in sorted(_KIND_RANK, key=len, reverse=True):
        if label.startswith(k):
            return k
    return label


def head_order(df: pd.DataFrame) -> list[str]:
    """Heads from simplest to largest: kind, then total hidden width, then depth."""
    def size(h):
        hid = df.loc[df["head"] == h, "hidden"].iloc[0] if "hidden" in df else []
        hid = hid if isinstance(hid, list) else []
        return (_KIND_RANK.get(_kind(h), 9), sum(hid), len(hid), h)
    return sorted(df["head"].unique(), key=size)


def config_order(df: pd.DataFrame) -> list[str]:
    """Reservoir configs, smallest readout first."""
    q = df[df.arm == "quantum"].drop_duplicates("config")
    return list(q.sort_values(["feature_dim", "n_modes"])["config"])


def panels(df: pd.DataFrame) -> list[tuple[str, str]]:
    q = df[df.arm == "quantum"]
    return list(dict.fromkeys(zip(q.dataset, q.scheme)))


def _agg(df, keys, metric):
    return df.groupby(keys)[metric].agg(mean="mean", std="std").reset_index()


def _axes(n, w=4.4, h=3.9):
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), squeeze=False)
    return fig, axes[0]


# ------------------------------------------------------------------------ ladder


def ladder_steps(sub: pd.DataFrame) -> pd.DataFrame:
    """One row per ladder step: its reservoir and its scaled head."""
    st = sub[(sub.step >= 0) & (sub.head_role == "scaled") & (sub.arm == "quantum")]
    return (st.drop_duplicates("step").sort_values("step")
              [["step", "config", "n_modes", "n_photons", "feature_dim", "head"]]
              .reset_index(drop=True))


def _ladder_value(sub, arm, step_row, head, metric):
    """Metric for one arm at one ladder step, trained with ``head``."""
    fid = {"quantum": f"q{step_row.config}", "qdials": f"qd{step_row.config}",
           "dials": f"dials{step_row.n_modes}", "raw": "raw",
           "rff": f"rff{step_row.feature_dim}_{step_row.n_modes}m",
           "rffdials": f"rffd{step_row.feature_dim}_{step_row.n_modes}m"}[arm]
    v = sub[(sub.feature_id == fid) & (sub["head"] == head)][metric]
    return v.mean() if len(v) else np.nan


def ladder_table(df: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Every arm at every ladder step, with the scaled head and each baseline."""
    rows = []
    for ds, sc in panels(df):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        base_heads = list(dict.fromkeys(sub.loc[sub.head_role == "baseline", "head"]))
        for _, st in ladder_steps(sub).iterrows():
            row = {"panel": title(ds, sc), "step": int(st.step) + 1, "reservoir": st.config,
                   "readout": int(st.feature_dim), "head": st["head"]}
            for arm in ARM_ORDER:
                v = _ladder_value(sub, arm, st, st["head"], metric)
                if not np.isnan(v):
                    row[arm] = v
            for b in base_heads:
                for arm in ("quantum", "dials"):
                    v = _ladder_value(sub, arm, st, b, metric)
                    if not np.isnan(v):
                        row[f"{arm} [{b}]"] = v
            if "quantum" in row and "dials" in row:
                row["q - dials"] = row["quantum"] - row["dials"]
            if "quantum" in row and "rff" in row:
                row["q - rff"] = row["quantum"] - row["rff"]
            rows.append(row)
    return pd.DataFrame(rows)


def ladder(df: pd.DataFrame, metric: str = "macro_f1", out=None):
    """The headline for a ladder run: every arm as reservoir AND head grow together.

    Solid lines use the step's scaled head; dashed lines are the baseline head
    (e.g. the original linear readout) on the same features.
    """
    use_style()
    pans = panels(df)
    fig, axes = _axes(len(pans), 4.9, 4.2)
    for ax, (ds, sc) in zip(axes, pans):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        steps = ladder_steps(sub)
        x = np.arange(len(steps))
        for arm in ARM_ORDER:
            y = [_ladder_value(sub, arm, st, st["head"], metric) for _, st in steps.iterrows()]
            if np.all(np.isnan(y)):
                continue
            ax.plot(x, y, color=ARM_COLOR[arm], marker=ARM_MARKER[arm], lw=2, ms=7,
                    mec="white", mew=1.2, label=f"{ARM_LABEL[arm]}, scaled head")
        for b in dict.fromkeys(sub.loc[sub.head_role == "baseline", "head"]):
            for arm in ("quantum", "dials"):
                y = [_ladder_value(sub, arm, st, b, metric) for _, st in steps.iterrows()]
                if np.all(np.isnan(y)):
                    continue
                ax.plot(x, y, color=ARM_COLOR[arm], marker=ARM_MARKER[arm], lw=1.3, ms=5,
                        ls="--", mfc="white", label=f"{ARM_LABEL[arm]}, {b} (baseline)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{st.config}\n{st.feature_dim:,}\n{st['head']}"
                            for _, st in steps.iterrows()], fontsize=8)
        ax.set_title(title(ds, sc))
        ax.set_xlabel("step: reservoir / readout dim / head")
    axes[0].set_ylabel(metric.replace("_", " "))
    fig.suptitle("Scaling the reservoir and the classical head together",
                 y=1.02, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(4, len(labels)), fontsize=8)
    savefig(fig, out)
    return fig


# ---------------------------------------------------------------------- grid runs


def quantum_by_head(df, metric="macro_f1", arm="quantum", out=None):
    """One line per head size: does a bigger head lift the reservoir's plateau?"""
    use_style()
    pans, heads = panels(df), head_order(df[df.arm == arm])
    fig, axes = _axes(len(pans), 4.4, 3.9)
    for ax, (ds, sc) in zip(axes, pans):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        q = _agg(sub[sub.arm == arm], ["config", "feature_dim", "head"], metric)
        ramp = RAMP[-len(heads):] if len(heads) <= len(RAMP) else RAMP
        for color, head in zip(ramp, heads):
            s = q[q["head"] == head].sort_values("feature_dim")
            if len(s):
                ax.errorbar(s["feature_dim"], s["mean"], yerr=s["std"].fillna(0),
                            color=color, marker="o", ms=6, lw=2, capsize=2,
                            mec="white", mew=1.2, label=head)
        raw = _agg(sub[sub.arm == "raw"], ["head"], metric)
        if len(raw):
            best = raw.loc[raw["mean"].idxmax()]
            ax.axhline(best["mean"], color=MUTED, ls="--", lw=1.2)
            ax.annotate(f"raw input + {best['head']}: {best['mean']:.3f}", (0.02, best["mean"]),
                        xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(0, 4), fontsize=8, color=MUTED)
        dims = sorted(q["feature_dim"].unique())
        cfg = q.drop_duplicates("feature_dim").set_index("feature_dim")["config"]
        ax.set_xscale("log")
        ax.set_xticks(dims)
        ax.set_xticklabels([f"{cfg[d]}\n{d:,}" for d in dims], fontsize=8)
        ax.minorticks_off()
        ax.set_title(title(ds, sc))
        ax.set_xlabel("reservoir / readout dim")
    axes[0].set_ylabel(metric.replace("_", " "))
    axes[-1].legend(loc="lower right", title="classical head", title_fontsize=8, fontsize=8)
    fig.suptitle(f"{ARM_LABEL[arm]}: reservoir size for each head size", y=1.02,
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    savefig(fig, out)
    return fig


def _annotated(ax, m: pd.DataFrame, cmap, fmt, vmin=None, vmax=None):
    im = ax.imshow(m.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    lo = np.nanmin(m.values) if vmin is None else vmin
    hi = np.nanmax(m.values) if vmax is None else vmax
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                continue
            t = (v - lo) / (hi - lo + 1e-12)
            dark = t > 0.55 if vmin is None else abs(v) > 0.6 * hi
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if dark else INK)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=30, ha="right")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index)
    ax.grid(False)
    return im


def grid_heatmap(df, metric="macro_f1", arm="quantum", out=None):
    """Reservoir config x head for one arm."""
    use_style()
    pans = panels(df)
    fig, axes = _axes(len(pans), 4.3, 3.7)
    for ax, (ds, sc) in zip(axes, pans):
        sub = df[(df.dataset == ds) & (df.scheme == sc) & (df.arm == arm)]
        m = sub.pivot_table(index="config", columns="head", values=metric)
        m = m.reindex(index=[c for c in config_order(df) if c in m.index],
                      columns=[h for h in head_order(sub) if h in m.columns])
        im = _annotated(ax, m, SEQ, "{:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title(ds, sc))
    axes[0].set_ylabel("reservoir (photons p, modes m)")
    fig.suptitle(f"{ARM_LABEL[arm]}: {metric.replace('_', ' ')}, reservoir x head",
                 y=1.03, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    savefig(fig, out)
    return fig


def gain_heatmap(df, metric="macro_f1", control="dials", out=None):
    """Quantum minus a control at the same reservoir config and the same head.

    dials depend only on n_modes and are recorded once per width, so they are
    matched on n_modes (4p24m pairs with the dials24 run logged under 3p24m).
    """
    use_style()
    pans = panels(df)
    fig, axes = _axes(len(pans), 4.3, 3.7)
    on = ["seed", "n_modes", "head"] if control == "dials" else ["seed", "config", "head"]
    for ax, (ds, sc) in zip(axes, pans):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        c = sub[sub.arm == control].drop_duplicates(on)[on + [metric]]
        mg = sub[sub.arm == "quantum"].merge(c, on=on, suffixes=("", "_c"))
        if mg.empty:
            ax.set_visible(False)
            continue
        mg["gain"] = mg[metric] - mg[f"{metric}_c"]
        g = mg.groupby(["config", "head"])["gain"].mean().unstack("head")
        g = g.reindex(index=[x for x in config_order(df) if x in g.index],
                      columns=[h for h in head_order(sub) if h in g.columns])
        lim = max(0.01, np.nanmax(np.abs(g.values)))
        _annotated(ax, g, DIV, "{:+.3f}", -lim, lim)
        ax.set_title(title(ds, sc))
    axes[0].set_ylabel("reservoir config")
    fig.suptitle(f"Quantum minus {ARM_LABEL[control].lower()} under the same head "
                 "(blue = quantum ahead)", y=1.03, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    savefig(fig, out)
    return fig


def arms_vs_head(df, metric="macro_f1", config=None, arms=("quantum", "rff", "dials", "raw"),
                 out=None):
    """At one reservoir size: every feature view as the head grows."""
    use_style()
    config = config or config_order(df)[-1]
    n_modes = int(config.split("p")[1].rstrip("m").split("m")[0])
    pans = panels(df)
    fig, axes = _axes(len(pans), 4.3, 3.8)
    for ax, (ds, sc) in zip(axes, pans):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        heads = head_order(sub[sub.arm == "quantum"])
        x = np.arange(len(heads))
        for arm in arms:
            s = sub[sub.arm == arm]
            s = s if arm == "raw" else (s[s.n_modes == n_modes] if arm == "dials"
                                        else s[s.config == config])
            if s.empty:
                continue
            a = _agg(s.drop_duplicates(["seed", "head"]), ["head"], metric).set_index("head")
            a = a.reindex(heads)
            ax.errorbar(x, a["mean"], yerr=a["std"].fillna(0), color=ARM_COLOR[arm],
                        marker=ARM_MARKER[arm], ms=7, lw=2, capsize=2, mec="white",
                        mew=1.2, label=ARM_LABEL[arm])
        ax.set_xticks(x)
        ax.set_xticklabels(heads, rotation=30, ha="right")
        ax.set_title(title(ds, sc))
    axes[0].set_ylabel(metric.replace("_", " "))
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Every feature view as the head grows ({config})", y=1.03,
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    savefig(fig, out)
    return fig


def vs_modes(df, metric="macro_f1", heads=None, out=None):
    """Score against the number of modes (= PCA components fed in), per photon number."""
    use_style()
    pans = panels(df)
    heads = heads or [head_order(df)[0], head_order(df)[-1]]
    fig, axes = plt.subplots(len(pans), len(heads), squeeze=False,
                             figsize=(5.2 * len(heads), 3.9 * len(pans)))
    photon_color = {2: RAMP[0], 3: RAMP[2], 4: RAMP[4], 5: RAMP[5]}
    for r, (ds, sc) in enumerate(pans):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        for c, head in enumerate(heads):
            ax, h = axes[r][c], sub[sub["head"] == head]
            for n_ph in sorted(h[h.arm == "quantum"].n_photons.unique()):
                a = _agg(h[(h.arm == "quantum") & (h.n_photons == n_ph)], ["n_modes"], metric)
                ax.errorbar(a["n_modes"], a["mean"], yerr=a["std"].fillna(0), marker="o",
                            color=photon_color.get(int(n_ph), QUANTUM), lw=2, ms=6,
                            mec="white", capsize=2, label=f"quantum, {int(n_ph)} photons")
            for arm in ("dials", "qdials"):
                s = h[h.arm == arm]
                if arm == "qdials" and len(s):  # one reservoir per width: most photons
                    s = s[s.n_photons == s.groupby("n_modes").n_photons.transform("max")]
                a = _agg(s.drop_duplicates(["seed", "n_modes"]), ["n_modes"], metric)
                if len(a):
                    ax.errorbar(a["n_modes"], a["mean"], yerr=a["std"].fillna(0),
                                marker=ARM_MARKER[arm], color=ARM_COLOR[arm], lw=2, ms=7,
                                mec="white", capsize=2, label=ARM_LABEL[arm])
            raw = h[h.arm == "raw"][metric].mean()
            if not np.isnan(raw):
                ax.axhline(raw, color=MUTED, ls="--", lw=1.2)
                ax.annotate(f"raw input: {raw:.3f}", (0.02, raw), xycoords=("axes fraction", "data"),
                            textcoords="offset points", xytext=(0, 4), fontsize=8, color=MUTED)
            ax.set_title(f"{title(ds, sc)} - {head}")
            ax.set_xlabel("modes = PCA components fed in")
        axes[r][0].set_ylabel(metric.replace("_", " "))
        axes[r][-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Score against the number of modes", y=1.01, fontsize=12,
                 fontweight="semibold", color=INK)
    fig.tight_layout()
    savefig(fig, out)
    return fig


def best_table(df: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Best setting per (panel, arm)."""
    rows = []
    for ds, sc in panels(df):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        for arm in ARM_ORDER:
            a = _agg(sub[sub.arm == arm], ["feature_id", "head"], metric)
            if len(a):
                b = a.loc[a["mean"].idxmax()]
                rows.append({"panel": title(ds, sc), "arm": arm,
                             "best": f"{b['feature_id']} + {b['head']}", metric: b["mean"]})
    return pd.DataFrame(rows)


def capacity(df, metric="macro_f1", head=None, arms=("quantum", "rff", "qdials", "dials"),
             out=None):
    """The original question for a single head: does a bigger reservoir help?

    x = reservoir readout dim; dials are matched to each config by n_modes, so
    the control line moves only when the mode count does. The raw input with
    the same head is the dashed ceiling.
    """
    use_style()
    pans = panels(df)
    fig, axes = _axes(len(pans), 4.9, 4.1)
    for ax, (ds, sc) in zip(axes, pans):
        sub = df[(df.dataset == ds) & (df.scheme == sc)]
        h = head or head_order(sub[sub.arm == "quantum"])[0]
        sub = sub[sub["head"] == h]
        q = sub[sub.arm == "quantum"].drop_duplicates("config").sort_values("feature_dim")
        for arm in arms:
            ys = []
            for _, r in q.iterrows():
                s = sub[sub.arm == arm]
                s = s[s.n_modes == r.n_modes] if arm == "dials" else s[s.config == r.config]
                ys.append(s[metric].mean() if len(s) else np.nan)
            if not np.all(np.isnan(ys)):
                ax.plot(q["feature_dim"], ys, color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                        lw=2, ms=7, mec="white", mew=1.2, label=ARM_LABEL[arm])
        raw = sub[sub.arm == "raw"][metric]
        if len(raw):
            ax.axhline(raw.mean(), color=MUTED, ls="--", lw=1.2)
            ax.annotate(f"raw input, same head: {raw.mean():.3f}", (0.02, raw.mean()),
                        xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(0, 4), fontsize=8, color=MUTED)
        ax.set_xscale("log")
        ax.set_xticks(q["feature_dim"])
        ax.set_xticklabels([f"{c}\n{d:,}" for c, d in zip(q.config, q.feature_dim)], fontsize=8)
        ax.minorticks_off()
        ax.set_title(f"{title(ds, sc)} - {h}")
        ax.set_xlabel("reservoir / readout dim")
    axes[0].set_ylabel(metric.replace("_", " "))
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Does a bigger reservoir help, with the head held fixed?", y=1.02,
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    savefig(fig, out)
    return fig
