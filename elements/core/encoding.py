"""Turning reduced features into reservoir phase angles.

The phase shifters are periodic, so an encoder has to map all of R into a
bounded interval without wrapping -- otherwise two very different inputs land
on the same angle. Two schemes, both fitted on TRAIN rows only:

``tanh``   smooth, monotonic squash into (0, pi). Outliers saturate gracefully
           and stay ordered, which is what novelty detection needs.
``minmax`` the original behaviour: min-max scale then hard clip to [0, 1].
           Kept for reproducing earlier runs; it flattens outliers onto
           exactly 1.0 and destroys the magnitude an anomaly detector reads.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


class PhaseEncoder:
    """Map reduced features to phase angles, using statistics fitted on train.

    Parameters
    ----------
    scheme:
        ``"tanh"`` (default) or ``"minmax"``.
    width:
        tanh only -- how many standard deviations map to the bulk of the
        range. The default of 3 matches the original experiments.
    """

    def __init__(self, scheme: str = "tanh", width: float = 3.0) -> None:
        if scheme not in ("tanh", "minmax"):
            raise ValueError(f"unknown encoding scheme: {scheme!r}")
        self.scheme = scheme
        self.width = width
        self._fitted = False

    def fit(self, Z: np.ndarray) -> "PhaseEncoder":
        if self.scheme == "tanh":
            self.mu_ = Z.mean(0)
            self.sd_ = Z.std(0) + _EPS
        else:
            self.lo_ = float(Z.min())
            self.hi_ = float(Z.max())
        self._fitted = True
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PhaseEncoder.fit must be called before transform")
        if self.scheme == "tanh":
            scaled = (Z - self.mu_) / (self.width * self.sd_)
            return ((np.pi / 2) * (1 + np.tanh(scaled))).astype(np.float32)
        unit = np.clip((Z - self.lo_) / (self.hi_ - self.lo_ + _EPS), 0.0, 1.0)
        return (unit * np.pi).astype(np.float32)

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.fit(Z).transform(Z)


def standardize_inplace(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise ``Q`` in place, returning the (mean, std) to reuse on test.

    In place because at 10,626 dims the obvious ``scaler.transform(Q)`` holds
    two extra full copies at once, which was enough to push the machine into
    swap. Accumulates in float64 for stability, stores float32.
    """
    mean = Q.mean(0, dtype=np.float64).astype(np.float32)
    std = Q.std(0, dtype=np.float64).astype(np.float32)
    std[std < _EPS] = 1.0  # constant columns stay at zero
    Q -= mean
    Q /= std
    return mean, std
