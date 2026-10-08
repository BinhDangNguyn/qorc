"""Standard time-series benchmarks for memory-capable reservoirs.

Three tasks, chosen because each isolates a different demand:

``narma10``     the output depends on the last 10 inputs through a nonlinear
                recurrence. A memoryless model cannot do better than chance on
                it, which makes it the sharpest test of whether memory exists.
``mackey_glass`` a chaotic attractor; prediction needs a short history but the
                dynamics do most of the work.
``memory_capacity`` the linear memory-capacity probe: reconstruct x_{t-k} for
                each lag k. Reports how far back a model can see at all,
                independent of any downstream task.
"""

from __future__ import annotations

import numpy as np


def narma10(n: int, seed: int = 42, warmup: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """NARMA-10: y_t depends nonlinearly on the last 10 inputs.

    y_{t+1} = 0.3 y_t + 0.05 y_t (sum_{i=0..9} y_{t-i}) + 1.5 u_{t-9} u_t + 0.1

    Returns ``(u, y)``, both shape (n,). ``u`` is uniform on [0, 0.5) as in the
    standard formulation -- the system is unstable for larger inputs.
    """
    rng = np.random.default_rng(seed)
    total = n + warmup
    u = rng.uniform(0, 0.5, total)
    y = np.zeros(total)
    for t in range(9, total - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * y[t - 9 : t + 1].sum()
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
        # The recurrence can diverge; standard practice is to clip and reseed.
        if not np.isfinite(y[t + 1]) or abs(y[t + 1]) > 1e3:
            y[t + 1] = 0.0
    return u[warmup:], y[warmup:]


def mackey_glass(
    n: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    power: int = 10,
    dt: float = 1.0,
    seed: int = 42,
    warmup: int = 1000,
) -> np.ndarray:
    """Mackey-Glass delay system, discretised with Euler steps.

    ``tau=17`` is the usual mildly chaotic setting. Returns a single series of
    length ``n``, normalised to roughly zero mean and unit variance.
    """
    rng = np.random.default_rng(seed)
    total = n + warmup
    history = int(tau / dt)
    x = np.zeros(total + history)
    x[:history] = 1.2 + 0.2 * rng.standard_normal(history)
    for t in range(history, total + history - 1):
        xt, xd = x[t], x[t - history]
        x[t + 1] = xt + dt * (beta * xd / (1 + xd**power) - gamma * xt)
    s = x[history + warmup :][:n]
    return (s - s.mean()) / (s.std() + 1e-12)


def memory_capacity_task(
    n: int, max_lag: int = 30, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs plus their own lagged copies as targets.

    Returns ``(u, Y)`` where ``Y[:, k]`` is ``u`` delayed by ``k+1`` steps.
    Fitting a linear readout per lag and summing the squared correlations gives
    the classic memory-capacity number.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, n + max_lag)
    Y = np.column_stack([u[max_lag - k - 1 : n + max_lag - k - 1] for k in range(max_lag)])
    return u[max_lag:], Y


def split(
    *arrays: np.ndarray, train_frac: float = 0.6, washout: int = 100
) -> tuple:
    """Chronological split with a washout period dropped from the train head.

    Time series must never be shuffled: a random split leaks the future into
    training. The washout lets a stateful model forget its zero initial state
    before any of its outputs are scored.
    """
    n = len(arrays[0])
    cut = int(n * train_frac)
    out = []
    for a in arrays:
        out.append((a[washout:cut], a[cut:]))
    return tuple(out)
