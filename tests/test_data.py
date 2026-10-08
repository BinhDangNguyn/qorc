"""Tests for NSL-KDD loading and its label schemes."""

import numpy as np
import pytest

from elements.data import nslkdd

pytestmark = pytest.mark.skipif(
    not any(p.exists() for p in nslkdd.CSV_CANDIDATES),
    reason="NSL_KDD_labeled.csv not present",
)


@pytest.fixture(scope="module")
def data():
    return nslkdd.load()


def test_shape_matches_the_published_dataset(data):
    assert data.n_rows == 125973
    assert data.n_features == 122  # 38 numeric + 84 one-hot
    assert data.X.dtype == np.float32


def test_feature_names_line_up_with_columns(data):
    assert len(data.feature_names) == data.n_features


def test_label_schemes_have_the_expected_class_counts(data):
    assert len(data.labels("families6")[1]) == 7  # normal + 5 named + other
    assert len(data.labels("all23")[1]) == 23
    assert len(data.labels("binary")[1]) == 2


def test_normal_is_always_class_zero(data):
    for scheme in ("families6", "all23", "binary"):
        y, names = data.labels(scheme)
        assert names[0] == "normal"
        assert (y == 0).sum() == 67343


def test_families6_keeps_the_named_attacks_separate(data):
    counts = data.class_counts("families6")
    assert counts["neptune"] == 41214
    assert counts["smurf"] == 2646
    assert counts["other_attack"] > 0


def test_binary_and_families6_agree_on_what_is_an_attack(data):
    y_bin, _ = data.labels("binary")
    y_fam, _ = data.labels("families6")
    assert np.array_equal(y_bin > 0, y_fam > 0)


def test_every_row_gets_a_label(data):
    for scheme in ("families6", "all23", "binary"):
        y, names = data.labels(scheme)
        assert len(y) == data.n_rows
        assert y.min() >= 0 and y.max() < len(names)


def test_unknown_scheme_is_rejected(data):
    with pytest.raises(ValueError):
        data.labels("nonsense")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        nslkdd.load(tmp_path / "nope.csv")
