"""The historical supervised protocol, kept exactly so it stays reproducible.

This is the pipeline of the original ``qorc_supervised_multiclass`` notebook
(per-scheme stratified split, L-BFGS head on the whole train split), pinned by
``tests/test_regression.py`` against ``runs/legacy/qorc_sup_runs``. New work
goes through ``elements.experiments.runner`` and a config file instead.

One (reservoir config x label scheme) supervised experiment, end to end.

Every experiment trains TWO heads on the same split:

``quantum``  the C(m, p)-dimensional reservoir readout
``dials``    the n_modes-dimensional PCA vector that DROVE that reservoir

``dials`` is the control that makes the comparison mean something. It shares
the reservoir's own front-end, so any gain is attributable to the reservoir
itself rather than to the scaling and PCA in front of it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from elements.core.encoding import PhaseEncoder, standardize_inplace
from elements.core.reservoir import Reservoir
from elements.eval.metrics import classification_metrics
from elements.heads.linear import SoftmaxHead


@dataclass
class SupervisedConfig:
    """Everything that defines one supervised experiment."""

    n_photons: int
    n_modes: int
    scheme: str = "families6"
    seed: int = 42
    test_size: float = 0.30
    encoding: str = "tanh"
    head_l2: float = 1e-4
    head_max_iter: int = 200
    device: str = "cpu"
    max_feature_gb: float = 6.0
    subsample: int = 0  # 0 = use every row; otherwise cap the dataset size

    @property
    def tag(self) -> str:
        return f"{self.n_photons}p{self.n_modes}m"

    @property
    def readout_dim(self) -> int:
        return math.comb(self.n_modes, self.n_photons)


@dataclass
class Features:
    """The two feature views of one split, plus the labels."""

    quantum_train: np.ndarray
    quantum_test: np.ndarray
    dials_train: np.ndarray
    dials_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pca_explained: float = 0.0
    t_quantum: float = 0.0
    views: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.views = {
            "quantum": (self.quantum_train, self.quantum_test),
            "dials": (self.dials_train, self.dials_test),
        }


def build_features(
    X: np.ndarray,
    y: np.ndarray,
    cfg: SupervisedConfig,
    log=print,
) -> Features:
    """Split, then scale -> PCA -> phase-encode -> reservoir, fitted on TRAIN.

    Every statistic (scaler, PCA, phase mu/sd, readout standardisation) comes
    from the training rows only; the test rows are transformed with them. That
    is what keeps the evaluation honest.
    """
    gb = X.shape[0] * cfg.readout_dim * 4 / 1e9
    if gb > cfg.max_feature_gb:
        raise MemoryError(
            f"{cfg.tag}: feature matrix would be {gb:.1f} GB "
            f"(limit {cfg.max_feature_gb} GB)"
        )

    # Stratify on the fine-grained label so rare classes land in both halves.
    counts = pd.Series(y).value_counts()
    strat = y if counts.min() >= 2 else None
    if strat is None:
        log("  a class has <2 rows; falling back to an unstratified split")
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)),
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=strat,
    )
    y_tr, y_te = y[idx_tr], y[idx_te]

    scaler = StandardScaler().fit(X[idx_tr])
    A = scaler.transform(X[idx_tr]).astype(np.float32)
    B = scaler.transform(X[idx_te]).astype(np.float32)
    pca = PCA(n_components=cfg.n_modes, random_state=cfg.seed).fit(A)
    Za, Zb = pca.transform(A), pca.transform(B)
    del A, B
    explained = float(pca.explained_variance_ratio_.sum())
    log(f"  front-end fitted, PCA explains {explained:.1%} of variance")

    encoder = PhaseEncoder(cfg.encoding).fit(Za)
    reservoir = Reservoir(
        n_modes=cfg.n_modes,
        n_photons=cfg.n_photons,
        seed=cfg.seed,
        device=cfg.device,
    )
    t0 = time.time()
    Qa = reservoir.transform(encoder.transform(Za))
    Qb = reservoir.transform(encoder.transform(Zb))
    t_quantum = time.time() - t0
    log(
        f"  Fock readout: {len(Qa):,}+{len(Qb):,} rows x {Qa.shape[1]} "
        f"in {t_quantum:.1f}s ({t_quantum / (len(Qa) + len(Qb)) * 1e3:.3f} ms/row)"
    )

    q_mean, q_std = standardize_inplace(Qa)
    Qb -= q_mean
    Qb /= q_std

    Ra = Za.astype(np.float32, copy=True)
    Rb = Zb.astype(np.float32, copy=True)
    r_mean, r_std = standardize_inplace(Ra)
    Rb -= r_mean
    Rb /= r_std

    return Features(Qa, Qb, Ra, Rb, y_tr, y_te, explained, t_quantum)


def run_supervised(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    cfg: SupervisedConfig,
    log=print,
    keep_confusion: bool = True,
) -> dict:
    """Run one experiment and return a record ready for ``results.jsonl``."""
    t_start = time.time()
    n_classes = len(class_names)
    log(
        f"--- {cfg.tag} / {cfg.scheme}: {n_classes} classes, "
        f"readout dim {cfg.readout_dim} ---"
    )

    feats = build_features(X, y, cfg, log=log)

    rec = {
        "config": cfg.tag,
        "n_photons": cfg.n_photons,
        "n_modes": cfg.n_modes,
        "readout_dim": cfg.readout_dim,
        "scheme": cfg.scheme,
        "seed": cfg.seed,
        "n_classes": n_classes,
        "class_names": class_names,
        "n_train": len(feats.y_train),
        "n_test": len(feats.y_test),
        "pca_explained_var": feats.pca_explained,
        "t_quantum_s": feats.t_quantum,
        "heads": {},
    }
    confusions: dict[str, np.ndarray] = {}

    for name, (Xtr, Xte) in feats.views.items():
        t0 = time.time()
        head = SoftmaxHead(
            n_classes,
            l2=cfg.head_l2,
            max_iter=cfg.head_max_iter,
            device=cfg.device,
        ).fit(Xtr, feats.y_train)
        pred = head.predict(Xte)
        proba = head.predict_proba(Xte)
        m = classification_metrics(feats.y_test, pred, proba, class_names)
        m["fit_time_s"] = time.time() - t0
        m["final_loss"] = head.final_loss
        m["input_dim"] = int(Xtr.shape[1])
        rec["heads"][name] = m
        if keep_confusion:
            confusions[name] = confusion_matrix(
                feats.y_test, pred, labels=range(n_classes)
            )
        log(
            f"  {name:8s} dim={Xtr.shape[1]:5d} acc={m['accuracy']:.4f} "
            f"bal_acc={m['balanced_acc']:.4f} macro_F1={m['macro_f1']:.4f} "
            f"({m['fit_time_s']:.1f}s)"
        )
        del head, proba

    q, d = rec["heads"]["quantum"], rec["heads"]["dials"]
    rec["gain_macro_f1"] = q["macro_f1"] - d["macro_f1"]
    rec["gain_balanced_acc"] = q["balanced_acc"] - d["balanced_acc"]
    rec["gain_accuracy"] = q["accuracy"] - d["accuracy"]
    rec["total_time_s"] = time.time() - t_start
    log(
        f"  GAIN quantum-dials: macro_F1 {rec['gain_macro_f1']:+.4f} | "
        f"bal_acc {rec['gain_balanced_acc']:+.4f} "
        f"(total {rec['total_time_s']:.1f}s)"
    )
    return (rec, confusions) if keep_confusion else rec
