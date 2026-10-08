"""MNIST-family benchmarks: classification and anomaly detection.

Both task types run the same front-end as the network work -- scale, PCA to
n_modes, tanh phase encoding, fixed reservoir -- so a result here is directly
comparable to the NSL-KDD numbers.

Arms, for both task types:

``quantum``  the C(m,p)-dim reservoir readout
``dials``    the n_modes-dim PCA vector that DROVE the reservoir (the control
             that makes any gain attributable to the reservoir itself)
``raw``      the untouched input, as a floor
plus a strong classical baseline (logistic regression / Isolation Forest),
because beating the control is not the same as being useful.

Run:  python -m elements.experiments.images --task classify --dataset mnist
      python -m elements.experiments.images --task anomaly  --dataset fashion_mnist
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from elements.core.encoding import PhaseEncoder, standardize_inplace
from elements.core.reservoir import Reservoir
from elements.data import images
from elements.eval.metrics import classification_metrics, detection_metrics
from elements.heads.novelty import NoveltyDetector
from elements.heads.linear import SoftmaxHead


def build_views(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_modes: int,
    n_photons: int,
    seed: int,
    device: str = "cpu",
    log=print,
) -> tuple[dict, float]:
    """Produce the raw / dials / quantum views, all fitted on TRAIN only.

    Near-constant pixels are dropped before scaling. On MNIST-family images
    about 160 of the 784 pixels are border pixels that are zero in every
    training row; dividing by their ~0 std amplifies test-time noise to
    |z| > 1500, which swamps the PCA that feeds the reservoir. Tabular data
    has no such columns, which is why this only shows up on images.
    """
    keep = x_train.std(0) > 1e-6
    if not keep.all():
        log(f"  dropping {int((~keep).sum())} near-constant pixels of {len(keep)}")
    x_train, x_test = x_train[:, keep], x_test[:, keep]

    scaler = StandardScaler().fit(x_train)
    A = scaler.transform(x_train).astype(np.float32)
    B = scaler.transform(x_test).astype(np.float32)

    pca = PCA(n_components=n_modes, random_state=seed).fit(A)
    Za, Zb = pca.transform(A), pca.transform(B)
    log(f"  PCA({n_modes}) explains {pca.explained_variance_ratio_.sum():.1%}")

    encoder = PhaseEncoder("tanh").fit(Za)
    reservoir = Reservoir(n_modes=n_modes, n_photons=n_photons, seed=seed, device=device)
    t0 = time.time()
    Qa = reservoir.transform(encoder.transform(Za))
    Qb = reservoir.transform(encoder.transform(Zb))
    t_quantum = time.time() - t0
    log(f"  reservoir {reservoir.tag} -> {Qa.shape[1]} features in {t_quantum:.1f}s")

    q_mean, q_std = standardize_inplace(Qa)
    Qb -= q_mean
    Qb /= q_std

    Ra, Rb = Za.astype(np.float32), Zb.astype(np.float32)
    r_mean, r_std = standardize_inplace(Ra)
    Rb -= r_mean
    Rb /= r_std

    return {"raw": (A, B), "dials": (Ra, Rb), "quantum": (Qa, Qb)}, t_quantum


def run_classification(args, log=print) -> list[dict]:
    data = images.load(args.dataset).subsample(args.n_train, args.n_test, args.seed)
    log(f"[{args.dataset}] train {data.x_train.shape} test {data.x_test.shape}, "
        f"{data.n_classes} classes")

    views, t_quantum = build_views(
        data.x_train, data.x_test, args.n_modes, args.n_photons, args.seed, log=log
    )

    rows = []
    for arm, (Xtr, Xte) in views.items():
        t0 = time.time()
        head = SoftmaxHead(
            data.n_classes, l2=args.l2, max_iter=args.max_iter
        ).fit(Xtr, data.y_train)
        m = classification_metrics(
            data.y_test, head.predict(Xte), head.predict_proba(Xte), data.class_names
        )
        rows.append({
            "dataset": args.dataset, "task": "classify", "arm": arm,
            "dim": int(Xtr.shape[1]), "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"], "fit_s": round(time.time() - t0, 1),
        })
        log(f"  {arm:9s} dim={Xtr.shape[1]:5d} acc={m['accuracy']:.4f} "
            f"macro_F1={m['macro_f1']:.4f} ({time.time() - t0:.1f}s)")

    # A strong classical reference on the raw pixels.
    t0 = time.time()
    clf = LogisticRegression(max_iter=200, n_jobs=-1).fit(data.x_train, data.y_train)
    pred = clf.predict(data.x_test)
    m = classification_metrics(data.y_test, pred, None, data.class_names)
    rows.append({
        "dataset": args.dataset, "task": "classify", "arm": "logreg_raw",
        "dim": int(data.x_train.shape[1]), "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"], "fit_s": round(time.time() - t0, 1),
    })
    log(f"  {'logreg':9s} dim={data.x_train.shape[1]:5d} acc={m['accuracy']:.4f} "
        f"macro_F1={m['macro_f1']:.4f} (classical reference)")
    return rows


def run_anomaly(args, log=print) -> list[dict]:
    data = images.load(args.dataset)
    normal = [int(c) for c in args.normal_classes.split(",")]
    log(f"[{args.dataset}] normal classes {normal}; every other class is unseen")

    x_norm, x_test, is_anom = images.make_novelty_split(data, normal, args.seed)
    if args.n_train:
        x_norm = x_norm[: args.n_train]
    log(f"  train {x_norm.shape} (normal only), test {x_test.shape}, "
        f"anomaly rate {is_anom.mean():.3f}")

    views, _ = build_views(
        x_norm, x_test, args.n_modes, args.n_photons, args.seed, log=log
    )

    rows = []
    for arm, (Xtr, Xte) in views.items():
        t0 = time.time()
        scores = NoveltyDetector(args.n_components, args.seed).fit(Xtr).score(Xte)
        m = detection_metrics(scores[is_anom == 0], scores[is_anom == 1])
        rows.append({
            "dataset": args.dataset, "task": "anomaly", "arm": arm,
            "dim": int(Xtr.shape[1]), **m, "fit_s": round(time.time() - t0, 1),
        })
        log(f"  {arm:9s} dim={Xtr.shape[1]:5d} AUC={m['auc']:.4f} "
            f"det@1%FA={m['det@0.01']:.4f} ({time.time() - t0:.1f}s)")

    # Isolation Forest on raw pixels: the baseline that beat the quantum
    # novelty head operationally on NSL-KDD, so it must appear here too.
    t0 = time.time()
    iso = IsolationForest(random_state=args.seed, n_jobs=-1).fit(x_norm)
    scores = -iso.score_samples(x_test)
    m = detection_metrics(scores[is_anom == 0], scores[is_anom == 1])
    rows.append({
        "dataset": args.dataset, "task": "anomaly", "arm": "iforest_raw",
        "dim": int(x_norm.shape[1]), **m, "fit_s": round(time.time() - t0, 1),
    })
    log(f"  {'iforest':9s} dim={x_norm.shape[1]:5d} AUC={m['auc']:.4f} "
        f"det@1%FA={m['det@0.01']:.4f} (classical reference)")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["classify", "anomaly"], default="classify")
    from elements.data.images import MEDMNIST_SUBSETS

    p.add_argument("--dataset", default="mnist",
                   choices=["mnist", "fashion_mnist", "kmnist", *MEDMNIST_SUBSETS])
    p.add_argument("--n-modes", type=int, default=12)
    p.add_argument("--n-photons", type=int, default=3)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--n-components", type=int, default=32,
                   help="novelty subspace size")
    p.add_argument("--normal-classes", default="0,1,2,3,4",
                   help="anomaly task: which classes count as normal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = (run_classification(args) if args.task == "classify" else run_anomaly(args))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
