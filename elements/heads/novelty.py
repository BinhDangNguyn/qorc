"""Novelty detection: learn the shape of normal data, score distance from it.

Trained on normal rows only, so nothing in the test set can be "seen" during
training. This is the label-free counterpart to the supervised head, and the
same detector serves network zero-day detection and image anomaly detection.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class NoveltyDetector:
    """Two complementary distances, both fitted on normal data only.

    * **Mahalanobis** -- inside the normal subspace but far from its centre
      (e.g. a ping flood: ordinary shape, extreme volume).
    * **Reconstruction error** -- energy pointing *outside* the normal
      subspace, i.e. a pattern normal data simply never produces.

    The score is their sum, so a row is flagged if it is unusual in either
    sense. No attack or anomaly is ever shown to it during fitting.
    """

    def __init__(self, n_components: int = 32, seed: int = 42) -> None:
        self.n_components = n_components
        self.seed = seed

    def fit(self, X_normal: np.ndarray) -> "NoveltyDetector":
        X_normal = np.asarray(X_normal, dtype=np.float64)
        self.mu_ = X_normal.mean(0)
        Z = X_normal - self.mu_
        k = min(self.n_components, Z.shape[1], Z.shape[0] - 1)
        self.pca_ = PCA(n_components=k, svd_solver="randomized",
                        random_state=self.seed).fit(Z)

        P = self.pca_.transform(Z)
        self.sd_ = P.std(0) + 1e-8
        residual = Z - self.pca_.inverse_transform(P)
        r = np.sqrt((residual**2).sum(1))
        self.r_mu_, self.r_sd_ = r.mean(), r.std() + 1e-8
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher means more anomalous."""
        Z = np.asarray(X, dtype=np.float64) - self.mu_
        P = self.pca_.transform(Z)
        maha = np.sqrt(((P / self.sd_) ** 2).sum(1))
        residual = Z - self.pca_.inverse_transform(P)
        recon = (np.sqrt((residual**2).sum(1)) - self.r_mu_) / self.r_sd_
        return maha + np.maximum(recon, 0)
