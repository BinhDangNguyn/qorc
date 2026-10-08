"""Novelty-specific views: which zero-days are catchable, at what false-alarm cost.

Load with ``load_results(run, keep=("per_type",))`` so per-attack results stay.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elements.viz.scaling import SEQ, _annotated, config_order
from elements.viz.style import savefig, use_style

MIN_SUPPORT = 20


def per_attack(df: pd.DataFrame, arm: str = "quantum", head: str | None = None,
               value: str = "auc") -> pd.DataFrame:
    """Per-attack ``auc`` or ``det@0.01`` for one arm and detector, attacks x configs."""
    sub = df[df.arm == arm]
    head = head or sub["head"].iloc[0]
    sub = sub[sub["head"] == head]
    cols, n = {}, {}
    order = config_order(df) if arm != "raw" else ["raw"]
    for cfg in order:
        r = sub if arm == "raw" else (sub[sub.n_modes == int(cfg.split("p")[1].rstrip("m"))]
                                      if arm == "dials" else sub[sub.config == cfg])
        if len(r):
            pt = r.iloc[0]["per_type"]
            cols[cfg] = {a: v[value] for a, v in pt.items()}
            n.update({a: v["n"] for a, v in pt.items()})
    t = pd.DataFrame(cols)
    t.insert(0, "n", pd.Series(n))
    return t.sort_values("n", ascending=False)


def per_attack_ladder(df: pd.DataFrame, arm: str = "quantum", value: str = "auc") -> pd.DataFrame:
    """Per-attack ``value`` along a ladder: each column is one step with ITS scaled head."""
    from elements.viz.scaling import ladder_steps

    cols, n = {}, {}
    for _, st in ladder_steps(df).iterrows():
        fid = {"quantum": f"q{st.config}", "dials": f"dials{st.n_modes}", "raw": "raw",
               "rff": f"rff{st.feature_dim}_{st.n_modes}m"}[arm]
        r = df[(df.feature_id == fid) & (df["head"] == st["head"])]
        if len(r):
            pt = r.iloc[0]["per_type"]
            cols[f"{st.config}/{st['head']}"] = {a: v[value] for a, v in pt.items()}
            n.update({a: v["n"] for a, v in pt.items()})
    t = pd.DataFrame(cols)
    t.insert(0, "n", pd.Series(n))
    return t.sort_values("n", ascending=False)


def per_attack_heatmap(df, arm="quantum", head=None, value="auc", out=None):
    """Attacks with at least MIN_SUPPORT rows x reservoir configs."""
    use_style()
    t = per_attack_ladder(df, arm, value) if head == "ladder" else per_attack(df, arm, head, value)
    t = t[t["n"] >= MIN_SUPPORT]
    m = t.drop(columns="n")
    m.index = [f"{a} (n={int(t.loc[a, 'n']):,})" for a in m.index]
    fig, ax = plt.subplots(figsize=(1.2 * m.shape[1] + 3.5, 0.4 * m.shape[0] + 1.8))
    im = _annotated(ax, m, SEQ, "{:.2f}", 0.0 if value != "auc" else 0.5, 1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=value)
    head = head or df[df.arm == arm]["head"].iloc[0]
    ax.set_title(f"Per-attack zero-day {value}: {arm} / {head}")
    fig.tight_layout()
    savefig(fig, out)
    return fig


def detection_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled and macro detection rates at fixed false-alarm budgets."""
    cols = ["step", "config", "arm", "head", "macro_auc", "pooled_auc", "det@0.001",
            "det@0.01", "det@0.05", "macro_det@0.01"]
    have = [c for c in cols if c in df]
    return df[have].sort_values([c for c in ("step", "arm", "head") if c in have])


#: The canonical NSL-KDD attack families (KDD'99 taxonomy).
ATTACK_CATEGORY = {
    **dict.fromkeys(["back", "land", "neptune", "pod", "smurf", "teardrop"], "DoS"),
    **dict.fromkeys(["ipsweep", "nmap", "portsweep", "satan"], "Probe"),
    **dict.fromkeys(["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
                     "warezclient", "warezmaster"], "R2L"),
    **dict.fromkeys(["buffer_overflow", "loadmodule", "perl", "rootkit"], "U2R"),
}
_CATEGORY_ORDER = {"DoS": 0, "Probe": 1, "R2L": 2, "U2R": 3}


def attack_comparison(models: dict, value: str = "det@0.01") -> pd.DataFrame:
    """Per-attack comparison of several detectors, one column per detector.

    ``models`` maps a display name to one result row (a Series with ``per_type``).
    ``value`` is ``det@0.01`` -- the share of that attack caught when the
    threshold lets through 1% false alarms on normal traffic -- or ``auc``.
    Two summary rows follow the attacks: the mean over attack types with at
    least MIN_SUPPORT rows, and the pooled figure over every attack row.
    """
    first = next(iter(models.values()))["per_type"]
    t = pd.DataFrame({"category": {a: ATTACK_CATEGORY.get(a, "?") for a in first},
                      "n": {a: v["n"] for a, v in first.items()}})
    for name, rec in models.items():
        t[name] = pd.Series({a: v[value] for a, v in rec["per_type"].items()})
    t = t.sort_values(["category", "n"], key=lambda c: c.map(_CATEGORY_ORDER)
                      if c.name == "category" else -c)
    big = t[t["n"] >= MIN_SUPPORT]
    pooled_col = "det@0.01" if value == "det@0.01" else "pooled_auc"
    summary = pd.DataFrame(
        [{"category": "", "n": int(big["n"].sum()),
          **{m: big[m].mean() for m in models}},
         {"category": "", "n": int(t["n"].sum()),
          **{m: rec[pooled_col] for m, rec in models.items()}}],
        index=[f"MEAN over types (n>={MIN_SUPPORT})", "POOLED (all attack rows)"])
    return pd.concat([t, summary])
