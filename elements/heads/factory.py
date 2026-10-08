"""Turn a ``HeadSpec`` into a trained head, with one interface per task.

Supervised heads:  ``fit(X, y, X_val, y_val)``, ``predict``, ``predict_proba``
Novelty detectors: ``fit(X, X_val)``, ``score`` (higher = more anomalous)

Validation rows are used only by heads that early-stop (``linear``, ``mlp``,
``ae``). Every other head -- ``lbfgs``, ``pca``, ``iforest`` and the standalone
classical models -- ignores them and trains on the same fit rows as everything
else, so every head sees identical training data.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from elements.config import HeadSpec
from elements.heads.autoencoder import AEDetector
from elements.heads.linear import SoftmaxHead
from elements.heads.mlp import MLPHead
from elements.heads.novelty import NoveltyDetector


class _LBFGS:
    def __init__(self, spec: HeadSpec, n_classes: int, device):
        self.head = SoftmaxHead(n_classes, l2=spec.l2, max_iter=spec.max_iter, device=device)

    def fit(self, X, y, X_val=None, y_val=None):
        self.head.fit(X, y)
        self.info = {"final_loss": self.head.final_loss}
        return self

    def predict(self, X):
        return self.head.predict(X)

    def predict_proba(self, X):
        return self.head.predict_proba(X)


class _MLP:
    def __init__(self, spec: HeadSpec, n_classes: int, device, seed: int):
        self.head = MLPHead(
            n_classes, hidden=tuple(spec.hidden), dropout=spec.dropout,
            activation=spec.activation, lr=spec.lr, weight_decay=spec.weight_decay,
            batch_size=spec.batch_size, max_epochs=spec.max_epochs,
            patience=spec.patience, device=device, seed=seed,
        )

    def fit(self, X, y, X_val, y_val):
        self.head.fit(X, y, X_val, y_val)
        self.info = {"epochs": self.head.epochs_run,
                     "val_macro_f1": self.head.best_val_macro_f1}
        return self

    def predict(self, X):
        return self.head.predict(X)

    def predict_proba(self, X):
        return self.head.predict_proba(X)


def make_classifier(spec: HeadSpec, n_classes: int, device, seed: int):
    if spec.kind == "lbfgs":
        return _LBFGS(spec, n_classes, device)
    if spec.kind in ("linear", "mlp"):
        return _MLP(spec, n_classes, device, seed)
    return make_classical_classifier(spec, n_classes, seed)


class _PCA:
    def __init__(self, spec: HeadSpec, seed: int):
        self.det = NoveltyDetector(spec.n_components, seed)

    def fit(self, X, X_val=None):
        self.det.fit(X)
        self.info = {"k": int(self.det.pca_.n_components_)}
        return self

    def score(self, X, chunk: int = 8192):
        # NoveltyDetector works in float64; chunking keeps 10k-dim scoring in RAM.
        return np.concatenate([self.det.score(X[i:i + chunk]) for i in range(0, len(X), chunk)])


class _AE:
    def __init__(self, spec: HeadSpec, device, seed: int):
        self.det = AEDetector(
            tuple(spec.hidden), spec.bottleneck, activation=spec.activation, lr=spec.lr,
            weight_decay=spec.weight_decay, batch_size=spec.batch_size,
            max_epochs=spec.max_epochs, patience=spec.patience, device=device, seed=seed,
        )

    def fit(self, X, X_val):
        self.det.fit(X, X_val)
        self.info = {"bottleneck": self.det.bottleneck_, "epochs": self.det.epochs_run}
        return self

    def score(self, X):
        return self.det.score(X)


class _IForest:
    def __init__(self, seed: int):
        self.det = IsolationForest(random_state=seed, n_jobs=-1)

    def fit(self, X, X_val=None):
        self.det.fit(X)
        self.info = {}
        return self

    def score(self, X):
        return -self.det.score_samples(X)


def make_detector(spec: HeadSpec, device, seed: int):
    if spec.kind == "pca":
        return _PCA(spec, seed)
    if spec.kind == "ae":
        return _AE(spec, device, seed)
    if spec.kind == "iforest":
        return _IForest(seed)
    if spec.kind in ("ocsvm", "lof", "knndist", "gmm"):
        return _SklearnDetector(spec, seed)
    raise ValueError(f"{spec.kind!r} is not a novelty detector")


# --------------------------------------------------------- standalone classical models
#
# Plain sklearn / xgboost models, normally run on the raw features as the
# classical baselines the reservoir is compared against. They see exactly the
# same fit rows as every other head; validation rows are not used.


def _subsample(X, y, max_rows: int, seed: int):
    if not max_rows or len(X) <= max_rows:
        return X, y
    idx = np.sort(np.random.default_rng(seed).choice(len(X), max_rows, replace=False))
    return X[idx], (None if y is None else y[idx])


class _SklearnClassifier:
    """Wrap a classifier so predict_proba always has one column per class.

    Rare all23 classes can be missing from a train split; sklearn then drops
    their probability column, and xgboost refuses non-contiguous labels. Labels
    are remapped to the classes present and the columns expanded back.
    """

    def __init__(self, model, n_classes: int, spec: HeadSpec, seed: int,
                 sample_weight: bool = False):
        self.model, self.n_classes, self.spec, self.seed = model, n_classes, spec, seed
        self.sample_weight = sample_weight

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = _subsample(X, y, self.spec.max_train_rows, self.seed)
        self.classes_ = np.unique(y)
        yy = np.searchsorted(self.classes_, y)
        kw = {}
        if self.sample_weight:  # balanced weights for models without class_weight
            counts = np.bincount(yy)
            kw["sample_weight"] = (len(yy) / (len(counts) * counts))[yy]
        self.model.fit(X, yy, **kw)
        self.info = {"n_train_rows": int(len(X))}
        return self

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)
        else:  # linear SVM: softmax of the decision scores, for ranking metrics
            d = self.model.decision_function(X)
            d = d if d.ndim == 2 else np.c_[-d, d]
            e = np.exp(d - d.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
        out = np.zeros((len(X), self.n_classes), dtype=np.float64)
        out[:, self.classes_] = p
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def make_classical_classifier(spec: HeadSpec, n_classes: int, seed: int):
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    k = spec.kind
    if k == "logreg":
        m = LogisticRegression(C=spec.C, max_iter=1000, class_weight="balanced")
    elif k == "rf":
        m = RandomForestClassifier(n_estimators=spec.n_estimators, max_depth=spec.max_depth,
                                   class_weight="balanced", n_jobs=-1, random_state=seed)
    elif k == "extratrees":
        m = ExtraTreesClassifier(n_estimators=spec.n_estimators, max_depth=spec.max_depth,
                                 class_weight="balanced", n_jobs=-1, random_state=seed)
    elif k == "hgb":
        # early_stopping off: its internal stratified split fails on all23
        # classes with a single training row; the round count is fixed instead.
        m = HistGradientBoostingClassifier(max_iter=spec.n_estimators, class_weight="balanced",
                                           early_stopping=False, random_state=seed)
    elif k == "xgb":
        from xgboost import XGBClassifier

        m = XGBClassifier(n_estimators=spec.n_estimators, max_depth=spec.max_depth or 6,
                          tree_method="hist", n_jobs=-1, random_state=seed)
        return _SklearnClassifier(m, n_classes, spec, seed, sample_weight=True)
    elif k == "knn":
        m = KNeighborsClassifier(n_neighbors=spec.n_neighbors, weights="distance", n_jobs=-1)
    elif k == "svm":
        # primal (dual=False): fast when rows >> features; the dual solver took
        # minutes per scheme and did not finish all23 in 20 min.
        m = LinearSVC(C=spec.C, class_weight="balanced", dual=False, max_iter=5000)
    else:
        raise ValueError(f"{k!r} is not a standalone classifier")
    return _SklearnClassifier(m, n_classes, spec, seed)


class _SklearnDetector:
    def __init__(self, spec: HeadSpec, seed: int):
        from sklearn.mixture import GaussianMixture
        from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
        from sklearn.svm import OneClassSVM

        self.spec, self.seed = spec, seed
        self.model = {
            "ocsvm": lambda: OneClassSVM(kernel="rbf", nu=spec.nu, gamma="scale"),
            "lof": lambda: LocalOutlierFactor(n_neighbors=spec.n_neighbors, novelty=True,
                                              n_jobs=-1),
            "knndist": lambda: NearestNeighbors(n_neighbors=spec.n_neighbors, n_jobs=-1),
            # reg_covar 1e-3: one-hot columns that are (almost) constant in
            # normal traffic otherwise collapse a component's variance.
            "gmm": lambda: GaussianMixture(n_components=spec.n_components,
                                           covariance_type="diag", reg_covar=1e-3,
                                           random_state=seed),
        }[spec.kind]()

    def fit(self, X, X_val=None):
        X, _ = _subsample(X, None, self.spec.max_train_rows, self.seed)
        self.model.fit(self._cast(X))
        self.info = {"n_train_rows": int(len(X))}
        return self

    def _cast(self, X):
        # the mixture's covariance estimate needs float64 to stay well-defined
        return X.astype(np.float64) if self.spec.kind == "gmm" else X

    def score(self, X, chunk: int = 8192):
        k = self.spec.kind
        out = []
        for i in range(0, len(X), chunk):
            xb = self._cast(X[i:i + chunk])
            if k == "ocsvm":
                out.append(-self.model.decision_function(xb))
            elif k == "knndist":
                out.append(self.model.kneighbors(xb)[0][:, -1])
            else:  # lof, gmm: higher likelihood = more normal
                out.append(-self.model.score_samples(xb))
        return np.concatenate(out)
