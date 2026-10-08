"""Linear readout and metrics for time-series tasks.

Forecasting readouts are ridge regression, not the L-BFGS softmax used for
classification: the target is continuous, and the closed-form solution removes
optimiser path-sensitivity from the comparison entirely.
"""

from __future__ import annotations

import numpy as np


class RidgeReadout:
    """Closed-form ridge regression, the standard reservoir-computing readout.

    Solved with ``lstsq`` on the normal equations rather than an explicit
    inverse: the reservoir readout is wide and its Gram matrix is often close
    to singular, and lstsq degrades gracefully where ``inv`` returns garbage.
    """

    def __init__(self, alpha: float = 1e-6) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeReadout":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        Xb = np.hstack([Xs, np.ones((len(Xs), 1))])  # bias column
        G = Xb.T @ Xb
        reg = self.alpha * np.eye(G.shape[0])
        reg[-1, -1] = 0.0  # never penalise the bias
        self.W_ = np.linalg.lstsq(G + reg, Xb.T @ y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self.mu_) / self.sd_
        out = np.hstack([Xs, np.ones((len(Xs), 1))]) @ self.W_
        return out[:, 0] if out.shape[1] == 1 else out


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised RMSE -- the standard NARMA/Mackey-Glass figure.

    Divided by the standard deviation of the target, so 1.0 is exactly the
    score of predicting the mean. Anything at or above 1.0 has learnt nothing.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    denom = y_true.std()
    if denom < 1e-12:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)


def memory_capacity(states: np.ndarray, Y_lagged: np.ndarray, alpha: float = 1e-6,
                    train_frac: float = 0.7) -> tuple[float, np.ndarray]:
    """Classic linear memory capacity: sum of squared correlations over lags.

    For each lag k, fit a linear readout reconstructing ``u_{t-k}`` from the
    reservoir state and record the squared correlation between target and
    prediction. MC is the sum over lags; it is bounded above by the state
    dimension, and for a memoryless map it is ~0 for every lag beyond 0.

    Returns ``(total_MC, per_lag_forgetting_curve)``.
    """
    n = len(states)
    cut = int(n * train_frac)
    per_lag = np.zeros(Y_lagged.shape[1])
    for k in range(Y_lagged.shape[1]):
        target = Y_lagged[:, k]
        model = RidgeReadout(alpha).fit(states[:cut], target[:cut])
        pred = model.predict(states[cut:])
        truth = target[cut:]
        if truth.std() < 1e-12 or np.std(pred) < 1e-12:
            continue
        r = np.corrcoef(truth, pred)[0, 1]
        per_lag[k] = 0.0 if not np.isfinite(r) else r**2
    return float(per_lag.sum()), per_lag


class ESN:
    """A classical echo-state network -- the reference every QRC is measured against.

    Included because the whole question is whether a photonic reservoir buys
    anything over the standard classical one. Without this baseline a NARMA
    score is uninterpretable.
    """

    def __init__(
        self,
        n_reservoir: int = 200,
        spectral_radius: float = 0.9,
        input_scale: float = 0.5,
        leak: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leak = leak
        self.seed = seed
        self._built = False

    def _build(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.n_reservoir
        W = rng.standard_normal((n, n))
        # Scaling to a spectral radius below 1 is what gives an ESN the echo
        # state property: the influence of any input decays rather than blows up.
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / radius * self.spectral_radius).astype(np.float32)
        self.W_in = (rng.standard_normal((input_dim, n)) * self.input_scale).astype(np.float32)
        self._built = True

    def run(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float32)
        if u.ndim == 1:
            u = u[:, None]
        if not self._built:
            self._build(u.shape[1])
        n = len(u)
        drive = u @ self.W_in
        H = np.empty((n, self.n_reservoir), dtype=np.float32)
        h = np.zeros(self.n_reservoir, dtype=np.float32)
        for t in range(n):
            h = (1 - self.leak) * h + self.leak * np.tanh(drive[t] + h @ self.W)
            H[t] = h
        return H
