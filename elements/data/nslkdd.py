"""Loading NSL-KDD and building its label schemes.

The 41 raw columns become 122 features: the numeric columns as-is, plus
one-hot encodings of ``protocol_type``, ``service`` and ``flag``. The encoder
is fitted on the whole file, which is safe -- it only enumerates categories,
it learns nothing from the labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET = "label"

#: The five attack types large enough to model individually. Everything else
#: collapses into ``other_attack`` under the ``families6`` scheme.
NAMED_FAMILIES = ["neptune", "satan", "ipsweep", "portsweep", "smurf"]

import os

_REPO = Path(__file__).resolve().parents[2]

#: Where the dataset usually sits, tried in order after $QORC_NSLKDD.
CSV_CANDIDATES = [
    _REPO / "data" / "NSL_KDD_labeled.csv",
    Path("D:/Dev/ai-ml/reservoir-computing/QRC/data/NSL_KDD_labeled.csv"),
    Path("NSL_KDD_labeled.csv"),
]


def find_csv(explicit: str | Path | None = None) -> Path:
    """Locate the dataset, preferring an explicit path."""
    explicit = explicit or os.environ.get("QORC_NSLKDD")
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"NSL-KDD csv not found at {p}")
        return p
    for cand in CSV_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return p
    raise FileNotFoundError(
        "NSL_KDD_labeled.csv not found; pass an explicit path. Tried: "
        + ", ".join(CSV_CANDIDATES)
    )


@dataclass
class NSLKDD:
    """The encoded dataset plus its raw labels.

    ``X`` is (n_rows, 122) float32; ``raw_label`` keeps the original strings so
    any label scheme can be derived without re-reading the file.
    """

    X: np.ndarray
    raw_label: np.ndarray
    feature_names: list[str]
    path: Path

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def labels(self, scheme: str) -> tuple[np.ndarray, list[str]]:
        """Return ``(y, class_names)`` for a label scheme.

        ``families6``  normal + 5 named families + other_attack  (7 classes)
        ``all23``      every distinct label in the file          (23 classes)
        ``binary``     normal vs attack                          (2 classes)

        The ``families6`` name is historical: it predates the ``normal`` class
        being counted, so the scheme actually yields 7. It is NOT the canonical
        DoS/Probe/R2L/U2R taxonomy -- those five are individual attack types.
        """
        raw = self.raw_label
        if scheme == "families6":
            mapped = np.where(
                raw == "normal",
                "normal",
                np.where(np.isin(raw, NAMED_FAMILIES), raw, "other_attack"),
            )
            names = ["normal"] + NAMED_FAMILIES + ["other_attack"]
        elif scheme == "all23":
            mapped = raw
            counts = pd.Series(raw).value_counts()
            names = ["normal"] + [c for c in counts.index if c != "normal"]
        elif scheme == "binary":
            mapped = np.where(raw == "normal", "normal", "attack")
            names = ["normal", "attack"]
        else:
            raise ValueError(f"unknown label scheme: {scheme!r}")

        index = {n: i for i, n in enumerate(names)}
        y = np.array([index[v] for v in mapped], dtype=np.int64)
        return y, names

    def class_counts(self, scheme: str) -> pd.Series:
        y, names = self.labels(scheme)
        counts = pd.Series(y).value_counts().sort_index()
        return pd.Series(
            [int(counts.get(i, 0)) for i in range(len(names))], index=names
        )


def load(path: str | Path | None = None) -> NSLKDD:
    """Read NSL-KDD and one-hot encode its categorical columns."""
    csv = find_csv(path)
    df = pd.read_csv(csv)

    missing = [c for c in [*CATEGORICAL, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"missing required NSL-KDD columns: {missing}")

    numeric = [c for c in df.columns if c not in [*CATEGORICAL, TARGET]]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(df[CATEGORICAL]).astype(np.float32)
    X_num = df[numeric].to_numpy(dtype=np.float32)

    names = numeric + [
        f"{col}={val}"
        for col, vals in zip(CATEGORICAL, ohe.categories_)
        for val in vals
    ]
    return NSLKDD(
        X=np.concatenate([X_num, X_cat], axis=1),
        raw_label=df[TARGET].to_numpy(),
        feature_names=names,
        path=csv,
    )
