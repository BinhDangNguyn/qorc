"""NetFlow-v2 NIDS datasets (Sarhan, Layeghy & Portmann, MONET 2022).

Every NF-*-v2 dataset carries the same 43 NetFlow v9 features, extracted with
nProbe from the original pcaps, plus ``Label`` (0/1) and ``Attack`` (name).
One loader serves them all:

``nf_unsw_nb15_v2``        2.39M flows, UNSW-NB15 testbed, 9 attack categories
``nf_cse_cic_ids2018_v2``  18.9M flows, CSE-CIC-IDS2018 testbed, 14 attack types

Preprocessing (the dataset authors' benchmark protocol, applied uniformly):

* drop the flow identifiers ``IPV4_SRC_ADDR``, ``IPV4_DST_ADDR``,
  ``L4_SRC_PORT``, ``L4_DST_PORT`` (a model would learn the testbed's hosts);
* drop ``MIN_TTL`` / ``MAX_TTL``: on testbeds TTL is set by the attacking
  machine's OS and hop count, a host fingerprint rather than traffic. The
  authors drop it for UNSW-NB15; it is dropped everywhere here so every
  dataset sees the same 37 features;
* ``log1p`` every remaining column (all non-negative counts, rates or codes;
  byte columns span ten orders of magnitude).

``dedup=True`` drops exact duplicate rows (features + label), judged on the raw
values. Once identifiers are gone most rows are duplicates (89.9% of
NF-UNSW-NB15-v2), so a random split otherwise shares flows between train and
test. Label ``Benign`` is renamed ``normal`` so the novelty protocol and
``operational_metrics`` apply unchanged.

The parsed file is cached next to the csv as ``<name>.npz`` (features float32,
labels as integer codes) and reused while the csv is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]

IDENTIFIERS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"]
LEAKY = ["MIN_TTL", "MAX_TTL"]
TARGETS = ["Label", "Attack"]

CIC2018_FAMILIES = {  # Table 6 of the dataset paper
    "DDOS attack-HOIC": "DDoS", "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DoS attacks-Hulk": "DoS", "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS", "DoS attacks-Slowloris": "DoS",
    "SSH-Bruteforce": "BruteForce", "FTP-BruteForce": "BruteForce",
    "Brute Force -Web": "Web", "Brute Force -XSS": "Web", "SQL Injection": "Web",
    "Bot": "Bot", "Infilteration": "Infiltration",
}


@dataclass(frozen=True)
class NetFlowInfo:
    name: str
    folder: str
    csv: str
    env: str
    #: scheme -> mapping raw attack name -> class (None = keep the raw name)
    schemes: dict = field(default_factory=dict)
    fine_scheme: str = "fine"


DATASETS = {
    "nf_unsw_nb15_v2": NetFlowInfo(
        "nf_unsw_nb15_v2", "nf_unsw_nb15_v2", "NF-UNSW-NB15-v2.csv", "QORC_NF_UNSW",
        schemes={"attack10": None}, fine_scheme="attack10"),
    "nf_cse_cic_ids2018_v2": NetFlowInfo(
        "nf_cse_cic_ids2018_v2", "nf_cse_cic_ids2018_v2", "NF-CSE-CIC-IDS2018-v2.csv",
        "QORC_NF_CIC2018",
        schemes={"attack15": None, "family7": CIC2018_FAMILIES}, fine_scheme="attack15"),
}


def find_csv(info: NetFlowInfo, explicit: str | Path | None = None) -> Path:
    explicit = explicit or os.environ.get(info.env)
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"{info.csv} not found at {p}")
        return p
    for cand in (_REPO / "data" / info.folder / info.csv, Path(info.csv)):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"{info.csv} not found; download it from "
        f"https://staff.itee.uq.edu.au/marius/NIDS_datasets/ into data/{info.folder}/")


@dataclass
class NetFlow:
    """Encoded flows plus raw labels; same interface as ``nslkdd.NSLKDD``."""

    X: np.ndarray
    raw_label: np.ndarray
    feature_names: list[str]
    path: Path
    info: NetFlowInfo
    n_duplicates_dropped: int = 0

    @property
    def fine_scheme(self) -> str:
        return self.info.fine_scheme

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def labels(self, scheme: str) -> tuple[np.ndarray, list[str]]:
        """``binary``: normal vs attack. Otherwise one of the dataset's schemes:
        the fine scheme keeps every attack name, a family scheme maps them."""
        raw = self.raw_label
        if scheme == "binary":
            names = ["normal", "attack"]
            mapped = np.where(raw == "normal", "normal", "attack")
        elif scheme in self.info.schemes:
            mapping = self.info.schemes[scheme]
            mapped = raw if mapping is None else np.array(
                [v if v == "normal" else mapping[v] for v in raw], dtype=object)
            counts = pd.Series(mapped[mapped != "normal"]).value_counts()
            names = ["normal"] + list(counts.index)
        else:
            raise ValueError(f"unknown label scheme for {self.info.name}: {scheme!r}; "
                             f"have binary, {', '.join(self.info.schemes)}")
        index = {n: i for i, n in enumerate(names)}
        return np.array([index[v] for v in mapped], dtype=np.int64), names


def _parse(csv: Path, cache: Path):
    drop = set(IDENTIFIERS + LEAKY)
    cols = [c for c in pd.read_csv(csv, nrows=0).columns if c not in drop]
    missing = [c for c in TARGETS if c not in cols]
    if missing:
        raise ValueError(f"{csv.name}: missing columns {missing}")
    feats = [c for c in cols if c not in TARGETS]
    df = pd.read_csv(csv, usecols=feats + ["Attack"],
                     dtype={c: np.float64 for c in feats})
    attack = df.pop("Attack")
    codes, classes = pd.factorize(attack)
    # duplicates judged on the raw values, not the float32 log1p
    dup = df.assign(_y=codes).duplicated().to_numpy()
    X = np.log1p(df.to_numpy(np.float64)).astype(np.float32)
    del df
    np.savez(cache, X=X, codes=codes.astype(np.int16), classes=np.array(classes, dtype=str),
             feature_names=np.array(feats), duplicate=dup)


def load(dataset: str, path: str | Path | None = None, dedup: bool = False) -> NetFlow:
    info = DATASETS[dataset]
    csv = find_csv(info, path)
    cache = csv.with_suffix(".npz")
    if not (cache.exists() and cache.stat().st_mtime >= csv.stat().st_mtime
            and "codes" in np.load(cache, allow_pickle=False).files):
        _parse(csv, cache)
    z = np.load(cache, allow_pickle=False)
    X, codes, dup = z["X"], z["codes"], z["duplicate"]
    classes = np.where(z["classes"] == "Benign", "normal", z["classes"]).astype(object)
    dropped = 0
    if dedup:
        keep = ~dup
        dropped = int(dup.sum())
        X, codes = X[keep], codes[keep]
    return NetFlow(X=X, raw_label=classes[codes], feature_names=z["feature_names"].tolist(),
                   path=csv, info=info, n_duplicates_dropped=dropped)
