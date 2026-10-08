"""Metric bundles, kept identical across heads so configs stay comparable."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    class_names: list[str],
) -> dict:
    """Full metric bundle for one trained head.

    ``macro_f1`` is the headline rather than accuracy: with this much class
    imbalance, accuracy is dominated by ``normal`` and ``neptune`` and hides
    everything interesting.
    """
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(sup[i]),
            }
            for i in range(len(class_names))
        },
    }

    out["macro_auc_ovr"] = None
    if proba is not None:
        try:
            present = np.unique(y_true)
            if len(present) > 1:
                sub = proba[:, present]
                out["macro_auc_ovr"] = float(
                    roc_auc_score(
                        y_true,
                        sub / sub.sum(1, keepdims=True),
                        multi_class="ovr",
                        average="macro",
                        labels=present,
                    )
                )
        except ValueError:
            # e.g. a class present in y_true but absent from proba's columns
            pass
    return out


def detection_metrics(
    scores_normal: np.ndarray,
    scores_attack: np.ndarray,
    fa_budgets: tuple[float, ...] = (0.001, 0.01, 0.05),
) -> dict:
    """Threshold-free AUC plus detection rate at each false-alarm budget.

    A fixed 0.5 cutoff is meaningless for intrusion detection: operators pick a
    threshold from a false-alarm budget ("flag the top 1% most suspicious
    traffic"). A detector that ranks an attack as suspicious but below 0.5
    would otherwise score zero.
    """
    y = np.r_[np.zeros(len(scores_normal)), np.ones(len(scores_attack))]
    s = np.r_[scores_normal, scores_attack]
    out = {"auc": float(roc_auc_score(y, s))}
    for fa in fa_budgets:
        thr = np.quantile(scores_normal, 1 - fa)
        out[f"det@{fa:g}"] = float((scores_attack > thr).mean())
    return out
