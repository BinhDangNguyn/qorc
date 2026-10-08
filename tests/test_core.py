"""Unit tests for the reservoir and the phase encoder.

These are the pieces that are easy to get silently wrong: a reservoir that
quietly became trainable, or an encoder fitted on the test set, would still
produce plausible-looking numbers.
"""

import math

import numpy as np
import pytest

from elements import PhaseEncoder, Reservoir, readout_dim
from elements.core.encoding import standardize_inplace
from elements.core.reservoir import input_state


@pytest.fixture(scope="module")
def reservoir():
    return Reservoir(n_modes=8, n_photons=2, seed=42)


def test_readout_dim_matches_combinatorics():
    assert readout_dim(12, 3) == math.comb(12, 3) == 220
    assert readout_dim(24, 4) == math.comb(24, 4) == 10626
    # the full Fock space is larger: it counts photons sharing a mode
    assert readout_dim(12, 3, no_bunching=False) == math.comb(3 + 12 - 1, 3)


def test_input_state_spreads_photons():
    state = input_state(12, 3)
    assert sum(state) == 3
    assert len(state) == 12
    assert max(state) == 1  # no bunching in the input
    with pytest.raises(ValueError):
        input_state(3, 5)  # more photons than modes


def test_reservoir_output_is_a_probability_distribution(reservoir):
    phases = np.random.default_rng(0).uniform(0, np.pi, (16, 8)).astype(np.float32)
    Q = reservoir.transform(phases)
    assert Q.shape == (16, reservoir.dim)
    assert np.allclose(Q.sum(1), 1.0, atol=1e-5)
    assert (Q >= 0).all()


def test_reservoir_is_deterministic_given_a_seed():
    phases = np.random.default_rng(1).uniform(0, np.pi, (8, 8)).astype(np.float32)
    a = Reservoir(n_modes=8, n_photons=2, seed=7).transform(phases)
    b = Reservoir(n_modes=8, n_photons=2, seed=7).transform(phases)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_reservoirs():
    phases = np.random.default_rng(2).uniform(0, np.pi, (8, 8)).astype(np.float32)
    a = Reservoir(n_modes=8, n_photons=2, seed=1).transform(phases)
    b = Reservoir(n_modes=8, n_photons=2, seed=2).transform(phases)
    assert not np.allclose(a, b)


def test_reservoir_has_no_trainable_parameters(reservoir):
    assert sum(1 for _ in reservoir.layer.parameters()) == 0


def test_batching_does_not_change_the_result(reservoir):
    phases = np.random.default_rng(3).uniform(0, np.pi, (40, 8)).astype(np.float32)
    assert np.allclose(
        reservoir.transform(phases, batch=4), reservoir.transform(phases, batch=4096)
    )


def test_tanh_encoder_stays_inside_the_phase_range():
    Z = np.random.default_rng(4).standard_normal((300, 5)) * 50
    ph = PhaseEncoder("tanh").fit_transform(Z)
    assert ph.min() > 0.0
    assert ph.max() < np.pi


def test_tanh_is_monotonic_where_minmax_clips():
    """The reason tanh replaced minmax: outliers stay ordered instead of
    collapsing onto the same angle."""
    base = np.random.default_rng(5).standard_normal((2000, 1))
    probe = np.array([[2.0], [3.0], [4.0], [5.0], [6.0], [8.0]])

    tanh_vals = PhaseEncoder("tanh").fit(base).transform(probe)[:, 0]
    assert np.all(np.diff(tanh_vals) > 0), "tanh must stay strictly increasing"

    minmax_vals = PhaseEncoder("minmax").fit(base).transform(probe)[:, 0]
    assert (np.diff(minmax_vals) == 0).any(), "minmax is expected to clip here"


def test_encoder_must_be_fitted_before_use():
    with pytest.raises(RuntimeError):
        PhaseEncoder().transform(np.zeros((2, 3)))


def test_encoder_uses_train_statistics_on_test_data():
    """Test rows must be encoded with the TRAIN mu/sd, not their own."""
    rng = np.random.default_rng(6)
    train = rng.standard_normal((500, 3))
    enc = PhaseEncoder("tanh").fit(train)
    shifted = rng.standard_normal((500, 3)) + 10.0
    # a shifted test set must land higher, which only happens if the train
    # statistics were reused
    assert enc.transform(shifted).mean() > enc.transform(train).mean()


def test_standardize_inplace_matches_the_obvious_version():
    Q = np.random.default_rng(7).standard_normal((100, 9)).astype(np.float32) * 3 + 5
    expected = (Q - Q.mean(0)) / Q.std(0)
    mean, std = standardize_inplace(Q)
    assert np.allclose(Q, expected, atol=1e-5)
    assert np.allclose(Q.mean(0), 0, atol=1e-5)
    assert mean.shape == (9,) and std.shape == (9,)


def test_standardize_inplace_survives_constant_columns():
    Q = np.ones((10, 3), dtype=np.float32)
    standardize_inplace(Q)
    assert np.isfinite(Q).all()  # no divide-by-zero
    assert np.allclose(Q, 0.0)
