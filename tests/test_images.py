"""Tests for the MNIST-family datasets and the novelty head.

The dataset tests need a download on first run and are marked ``slow``; the
novelty-head tests are pure numpy and always run.
"""

import numpy as np
import pytest

from elements.heads.novelty import NoveltyDetector

pytest_plugins: list[str] = []


# ----------------------------------------------------------- novelty head
def test_novelty_scores_anomalies_higher_than_normals():
    rng = np.random.default_rng(0)
    normal = rng.standard_normal((400, 8))
    anomalous = rng.standard_normal((100, 8)) + 6.0  # far from the centre
    det = NoveltyDetector(n_components=4, seed=0).fit(normal)
    assert det.score(anomalous).mean() > det.score(normal).mean()


def test_novelty_detects_off_subspace_patterns():
    """Reconstruction error must catch a direction normal data never uses."""
    rng = np.random.default_rng(1)
    # normal data lives in a 2-D plane inside 6-D space
    basis = rng.standard_normal((2, 6))
    normal = rng.standard_normal((400, 2)) @ basis
    det = NoveltyDetector(n_components=2, seed=0).fit(normal)
    off = normal[:50] + rng.standard_normal((50, 6)) * 5.0
    assert det.score(off).mean() > det.score(normal).mean()


def test_novelty_is_deterministic():
    X = np.random.default_rng(2).standard_normal((200, 5))
    a = NoveltyDetector(3, seed=7).fit(X).score(X)
    b = NoveltyDetector(3, seed=7).fit(X).score(X)
    assert np.allclose(a, b)


def test_novelty_never_sees_labels():
    """fit takes only a feature matrix -- there is no place to leak labels."""
    import inspect

    params = inspect.signature(NoveltyDetector.fit).parameters
    assert list(params) == ["self", "X_normal"]


def test_novelty_component_count_is_clamped_to_the_data():
    X = np.random.default_rng(3).standard_normal((10, 4))
    det = NoveltyDetector(n_components=99, seed=0).fit(X)
    assert det.pca_.n_components_ <= 4


# --------------------------------------------------------------- datasets
@pytest.mark.slow
@pytest.mark.parametrize("name", ["mnist", "fashion_mnist", "kmnist"])
def test_image_datasets_load_in_the_expected_shape(name):
    from elements.data import images

    d = images.load(name)
    assert d.x_train.shape == (60000, 784)
    assert d.x_test.shape == (10000, 784)
    assert d.n_classes == 10
    assert d.x_train.dtype == np.float32
    assert 0.0 <= d.x_train.min() and d.x_train.max() <= 1.0


@pytest.mark.slow
def test_subsample_is_stratified():
    from elements.data import images

    s = images.load("mnist").subsample(n_train=1000, n_test=500, seed=0)
    counts = np.bincount(s.y_train)
    assert counts.min() == counts.max() == 100  # every class equally present


@pytest.mark.slow
def test_novelty_split_hides_the_anomalous_classes():
    from elements.data import images

    d = images.load("mnist")
    normal = [0, 1, 2, 3, 4]
    train_x, test_x, is_anom = images.make_novelty_split(d, normal)
    # training must contain only normal rows
    assert len(train_x) == int(np.isin(d.y_train, normal).sum())
    # the flag must match the held-out classes exactly
    assert np.array_equal(is_anom, (~np.isin(d.y_test, normal)).astype(np.int64))
    assert 0.0 < is_anom.mean() < 1.0


@pytest.mark.slow
@pytest.mark.parametrize("name", ["pneumoniamnist", "breastmnist", "retinamnist"])
def test_medmnist_subsets_load_flattened_and_greyscale(name):
    """RetinaMNIST is RGB on disk; every subset must arrive as 784 greyscale."""
    from elements.data import images

    d = images.load(name)
    assert d.x_train.shape[1] == 784
    assert d.x_test.shape[1] == 784
    assert d.x_train.dtype == np.float32
    assert 0.0 <= d.x_train.min() and d.x_train.max() <= 1.0
    assert d.n_classes == images.MEDMNIST_SUBSETS[name][0]
    assert d.y_train.min() >= 0 and d.y_train.max() < d.n_classes


@pytest.mark.slow
def test_medmnist_folds_val_into_train():
    """MedMNIST ships train/val/test; val must not be silently dropped."""
    import numpy as np

    from elements.data import images

    path = images.CACHE / "breastmnist.npz"
    d = images.load("breastmnist")
    with np.load(path) as z:
        expected = len(z["train_images"]) + len(z["val_images"])
    assert len(d.x_train) == expected


def test_greyscale_conversion_collapses_the_channel_axis():
    from elements.data.images import _to_greyscale

    rgb = np.full((4, 28, 28, 3), 255, dtype=np.uint8)
    grey = _to_greyscale(rgb)
    assert grey.shape == (4, 28, 28)
    assert grey.max() <= 255
    # already-greyscale input passes through untouched
    mono = np.zeros((4, 28, 28), dtype=np.uint8)
    assert _to_greyscale(mono).shape == mono.shape


def test_unknown_dataset_is_rejected():
    from elements.data import images

    with pytest.raises(ValueError, match="unknown dataset"):
        images.load("not_a_dataset")
