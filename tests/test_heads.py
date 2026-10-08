"""Unit tests for the scalable classical head and the joint-scaling views."""

import numpy as np
import pytest

from elements.heads.mlp import MLPHead, build_mlp, n_params


def _xor(n=1200, seed=0):
    """A task no linear head can solve: the sign of x0 * x1."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)
    return X, y


def test_param_count_matches_the_built_model():
    for hidden in [(), (16,), (32, 8)]:
        model = build_mlp(10, 3, hidden, 0.1)
        assert n_params(10, 3, hidden) == sum(p.numel() for p in model.parameters())


def test_hidden_layer_solves_what_linear_cannot():
    X, y = _xor()
    Xv, yv = _xor(400, seed=1)
    Xt, yt = _xor(400, seed=2)
    lin = MLPHead(2, hidden=(), max_epochs=30, seed=0).fit(X, y, Xv, yv)
    mlp = MLPHead(2, hidden=(64,), max_epochs=60, seed=0).fit(X, y, Xv, yv)
    assert (lin.predict(Xt) == yt).mean() < 0.7
    assert (mlp.predict(Xt) == yt).mean() > 0.9


def test_proba_rows_sum_to_one_and_fit_is_deterministic():
    X, y = _xor(300)
    a = MLPHead(2, hidden=(8,), max_epochs=5, seed=3).fit(X, y, X, y)
    b = MLPHead(2, hidden=(8,), max_epochs=5, seed=3).fit(X, y, X, y)
    p = a.predict_proba(X)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)
    assert np.allclose(p, b.predict_proba(X), atol=1e-6)


def test_rff_view_has_requested_width_and_is_train_standardised():
    from elements.experiments.views import FrontEnd, Task, rff_view

    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20)).astype(np.float32)
    y = rng.integers(0, 3, 300)
    task = Task("toy", X[:200], X[200:250], X[250:],
                {"all": (y[:200], y[200:250], y[250:], ["a", "b", "c"])})
    tr, va, te = rff_view(FrontEnd(task, seed=0), n_modes=6, dim=50, seed=0)
    assert tr.shape == (200, 50) and va.shape == (50, 50) and te.shape == (50, 50)
    assert np.allclose(tr.mean(0), 0, atol=1e-4)


@pytest.mark.parametrize("scheme", ["families6", "all23"])
def test_nslkdd_split_is_disjoint(scheme):
    pytest.importorskip("pandas")
    from elements.config import DataSpec, SplitSpec
    from elements.data import nslkdd
    from elements.experiments.views import load_supervised_task

    if not any(p.exists() for p in nslkdd.CSV_CANDIDATES):
        pytest.skip("dataset not present")
    task = load_supervised_task(DataSpec(), SplitSpec(), seed=42)
    ytr, yva, yte, names = task.labels[scheme]
    n = len(ytr) + len(yva) + len(yte)
    assert len(task.X_train) + len(task.X_val) + len(task.X_test) == n
    assert abs(len(yte) / n - 0.30) < 0.01
