"""Regression against the pre-package reference run.

These numbers come from ``runs/legacy/qorc_sup_runs/20260725_235053``, produced by
``qorc_supervised_multiclass.ipynb`` before the code was packaged. The package
must reproduce them: that is the whole claim of the refactor.

The tolerance is 5e-3, not exact equality. L-BFGS is path-sensitive and the
reference was produced under a different torch/BLAS build, so the last few
digits legitimately move. For scale, the reference run's own caveats record a
seed-to-seed std of up to 0.17 on one arm -- a 5e-3 drift is far below the
noise this experiment already carries.

Marked ``slow``: run with ``pytest -m slow``.
"""

import pytest

from elements.data import nslkdd
from elements.experiments.reference import SupervisedConfig, run_supervised

TOL = 5e-3

# (scheme, n_photons, n_modes) -> {head: (macro_f1, balanced_acc, accuracy)}
REFERENCE = {
    ("all23", 3, 12): {
        "quantum": (0.5574773375246863, 0.7968333569414088, 0.9539585097375106),
        "dials": (0.4640223994371756, 0.7551529793916353, 0.7132726502963590),
    },
    ("families6", 3, 12): {
        "quantum": (0.9518716125858212, 0.9847807100251961, 0.9792284081287045),
        "dials": (0.7933444887149042, 0.8741262363490924, 0.9009049534292972),
    },
}

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not any(p.exists() for p in nslkdd.CSV_CANDIDATES),
        reason="NSL_KDD_labeled.csv not present",
    ),
]


@pytest.fixture(scope="module")
def data():
    return nslkdd.load()


@pytest.mark.parametrize("key", list(REFERENCE))
def test_matches_reference_run(data, key):
    scheme, n_photons, n_modes = key
    y, names = data.labels(scheme)
    cfg = SupervisedConfig(n_photons=n_photons, n_modes=n_modes, scheme=scheme)
    rec, _ = run_supervised(data.X, y, names, cfg, log=lambda *a, **k: None)

    for head, (f1, bal, acc) in REFERENCE[key].items():
        got = rec["heads"][head]
        assert got["macro_f1"] == pytest.approx(f1, abs=TOL), f"{head} macro_f1"
        assert got["balanced_acc"] == pytest.approx(bal, abs=TOL), f"{head} bal_acc"
        assert got["accuracy"] == pytest.approx(acc, abs=TOL), f"{head} accuracy"


@pytest.mark.parametrize("key", list(REFERENCE))
def test_quantum_head_beats_its_own_input(data, key):
    """The headline claim: the reservoir adds something over the PCA dials
    that drove it. This must hold for the result to mean anything."""
    scheme, n_photons, n_modes = key
    y, names = data.labels(scheme)
    cfg = SupervisedConfig(n_photons=n_photons, n_modes=n_modes, scheme=scheme)
    rec, _ = run_supervised(data.X, y, names, cfg, log=lambda *a, **k: None)
    assert rec["gain_macro_f1"] > 0.05
