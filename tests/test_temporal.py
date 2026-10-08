"""Tests for the temporal strategies and the time-series tasks.

The central property under test is memory: whether an arm's output at step t
depends on what came before it. Reversing a series and comparing per-step
outputs is the cleanest probe -- a memoryless map is invariant under it, a
stateful one is not.
"""

import numpy as np
import pytest

from elements import Reservoir
from elements.core.temporal import (
    DelayEmbedding,
    FeedbackReservoir,
    QuantumMemory,
    sliding_window,
)
from elements.data.timeseries import (
    mackey_glass,
    memory_capacity_task,
    narma10,
)
from elements.heads.forecasting import ESN, RidgeReadout, memory_capacity, nrmse


@pytest.fixture(scope="module")
def reservoir():
    return Reservoir(n_modes=8, n_photons=2, seed=42)


@pytest.fixture(scope="module")
def series():
    return np.random.default_rng(0).standard_normal(60).astype(np.float32)


# --------------------------------------------------------------- windowing
def test_sliding_window_layout():
    W = sliding_window(np.arange(6, dtype=np.float32), 3)
    assert W.shape == (6, 3)
    assert W[4].tolist() == [4, 3, 2]  # [u_t, u_{t-1}, u_{t-2}]
    assert W[0].tolist() == [0, 0, 0]  # left-padded at the start


def test_sliding_window_handles_multivariate_input():
    u = np.arange(12, dtype=np.float32).reshape(6, 2)
    assert sliding_window(u, 3).shape == (6, 6)  # width * n_features


# ------------------------------------------------------- the memory property
def test_bare_reservoir_is_memoryless(reservoir, series):
    """The finding this whole comparison rests on: order does not matter."""
    X = np.tile(series[:, None], (1, 8)).astype(np.float32)
    forward = reservoir.transform(X)
    reversed_ = reservoir.transform(np.ascontiguousarray(X[::-1]))[::-1]
    assert np.allclose(forward, reversed_)


def test_feedback_reservoir_has_memory(reservoir, series):
    """With the loop closed, reversing the series must change the outputs."""
    forward = FeedbackReservoir(reservoir, leak=0.5, seed=1).run(series)
    reversed_ = FeedbackReservoir(reservoir, leak=0.5, seed=1).run(
        np.ascontiguousarray(series[::-1])
    )[::-1]
    assert not np.allclose(forward, reversed_)


def test_zero_leak_reduces_feedback_to_the_memoryless_case(reservoir, series):
    """Confirms the memory comes from the feedback path and nothing else."""
    forward = FeedbackReservoir(reservoir, leak=0.0, seed=1).run(series)
    reversed_ = FeedbackReservoir(reservoir, leak=0.0, seed=1).run(
        np.ascontiguousarray(series[::-1])
    )[::-1]
    assert np.allclose(forward, reversed_)


def test_feedback_is_deterministic(reservoir, series):
    a = FeedbackReservoir(reservoir, leak=0.5, seed=3).run(series)
    b = FeedbackReservoir(reservoir, leak=0.5, seed=3).run(series)
    assert np.array_equal(a, b)


def test_delay_embedding_output_shape(reservoir, series):
    from sklearn.decomposition import PCA

    de = DelayEmbedding(reservoir, window=12, reducer=PCA(n_components=8, random_state=0))
    out = de.fit_transform(series)
    assert out.shape == (len(series), reservoir.dim)


def test_quantum_memory_is_explicitly_unavailable():
    with pytest.raises(NotImplementedError, match="state"):
        QuantumMemory()


# ------------------------------------------------------------------- tasks
def test_narma10_is_finite_and_bounded():
    u, y = narma10(500, seed=0)
    assert u.shape == y.shape == (500,)
    assert np.isfinite(y).all()
    assert 0.0 <= u.min() and u.max() < 0.5


def test_narma10_target_needs_more_than_the_current_input():
    """If y_t correlated strongly with u_t, the task would not test memory."""
    u, y = narma10(2000, seed=0)
    assert abs(np.corrcoef(u, y)[0, 1]) < 0.2


def test_mackey_glass_is_normalised_and_finite():
    s = mackey_glass(500, seed=0)
    assert np.isfinite(s).all()
    assert abs(s.mean()) < 1e-6
    assert abs(s.std() - 1.0) < 1e-6


def test_memory_capacity_targets_are_correctly_lagged():
    u, Y = memory_capacity_task(200, max_lag=10, seed=0)
    assert np.allclose(Y[1:, 0], u[:-1])  # lag 1
    assert np.allclose(Y[5:, 4], u[:-5])  # lag 5


# ----------------------------------------------------------------- readout
def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    y = X @ rng.standard_normal(4)
    model = RidgeReadout(1e-10).fit(X[:200], y[:200])
    assert nrmse(y[200:], model.predict(X[200:])) < 0.05


def test_nrmse_of_the_mean_is_about_one():
    y = np.random.default_rng(0).standard_normal(300)
    assert 0.9 < nrmse(y, np.full(300, y.mean())) < 1.1


def test_esn_reaches_published_narma_performance():
    """Guards the baseline itself: literature puts an ESN at ~0.2-0.4 here."""
    u, y = narma10(2000, seed=42)
    H = ESN(n_reservoir=200, seed=42).run(u)
    cut = 1200
    model = RidgeReadout(1e-6).fit(H[200:cut], y[200:cut])
    assert nrmse(y[cut:], model.predict(H[cut:])) < 0.5


@pytest.mark.slow
def test_memory_capacity_ranks_the_arms_as_measured():
    """The headline result: ESN > delay >> feedback > memoryless.

    Thresholds are loose -- this pins the ordering and the order of magnitude,
    not exact values, so it stays meaningful across library versions.
    """
    u, Y = memory_capacity_task(1500, max_lag=15, seed=42)
    reservoir = Reservoir(n_modes=12, n_photons=3, seed=42)

    from sklearn.decomposition import PCA

    from elements.experiments.temporal import memoryless_states

    mc_memoryless, _ = memory_capacity(memoryless_states(reservoir, u, 12, 42), Y)
    mc_delay, _ = memory_capacity(
        DelayEmbedding(
            reservoir, window=15, reducer=PCA(n_components=12, random_state=42)
        ).fit_transform(u),
        Y,
    )
    mc_feedback, _ = memory_capacity(
        FeedbackReservoir(reservoir, leak=0.5, seed=42).run(u), Y
    )
    mc_esn, _ = memory_capacity(ESN(n_reservoir=200, seed=42).run(u), Y)

    assert mc_memoryless < 0.5, "a memoryless map must have no memory capacity"
    assert mc_feedback > mc_memoryless, "feedback must add some memory"
    assert mc_feedback < 4.0, "feedback memory is short; see FeedbackReservoir docs"
    assert mc_delay > 5.0, "delay embedding should reach most of its window"
    assert mc_esn > mc_delay, "the classical ESN remains the strongest arm"
