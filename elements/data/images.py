"""MNIST-family image datasets, in the form the reservoir expects.

Four datasets, all 28x28 greyscale with 10 classes, so they are drop-in
substitutes for each other and differ only in difficulty:

``mnist``          handwritten digits; the original QORC paper's task.
``fashion_mnist``  clothing. Much less linearly separable than digits, which
                   is where the reservoir has the most room to help.
``kmnist``         cursive Kuzushiji characters; the third dataset in the
                   paper. Hardest of the three for classical linear models.
``emnist_letters`` handwritten letters (26 classes, optional download).

Pixels are scaled to [0, 1] by dividing by 255, matching the paper.
"""

from __future__ import annotations

import gzip
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".keras" / "datasets" / "elements"

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

#: Kuzushiji characters, romanised.
KMNIST_CLASSES = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

_KMNIST_URL = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/{}"
_KMNIST_FILES = {
    "x_train": "kmnist-train-imgs.npz",
    "y_train": "kmnist-train-labels.npz",
    "x_test": "kmnist-test-imgs.npz",
    "y_test": "kmnist-test-labels.npz",
}

#: MedMNIST v2, one .npz per subset on Zenodo. All are 28x28; the ones marked
#: RGB carry a trailing channel axis and are converted to greyscale on load.
_MEDMNIST_URL = "https://zenodo.org/records/10519652/files/{}.npz"

MEDMNIST_SUBSETS = {
    # name:          (n_classes, RGB?, description)
    "breastmnist": (2, False, "breast ultrasound, malignant vs normal/benign"),
    "pneumoniamnist": (2, False, "chest X-ray, pneumonia vs normal"),
    "retinamnist": (5, True, "fundus, diabetic retinopathy grade"),
    "dermamnist": (7, True, "dermatoscope, skin lesion type"),
    "bloodmnist": (8, True, "blood cell microscopy"),
    "organamnist": (11, False, "abdominal CT, organ, axial view"),
    "organcmnist": (11, False, "abdominal CT, organ, coronal view"),
    "organsmnist": (11, False, "abdominal CT, organ, sagittal view"),
    "octmnist": (4, False, "retinal OCT"),
    "pathmnist": (9, True, "colon pathology"),
    "tissuemnist": (8, False, "kidney cortex microscopy"),
}

MEDMNIST_CLASS_NAMES = {
    "breastmnist": ["malignant", "normal/benign"],
    "pneumoniamnist": ["normal", "pneumonia"],
}


@dataclass
class ImageData:
    """Flattened images plus labels, ready for the reservoir front-end."""

    x_train: np.ndarray  # (n, 784) float32 in [0, 1]
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    name: str

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]

    def subsample(self, n_train: int = 0, n_test: int = 0, seed: int = 42) -> "ImageData":
        """Take a stratified subset -- useful for smoke tests and sweeps.

        Stratified rather than a head slice: the reservoir sweep compares
        per-class F1, so every class has to survive the cut.
        """
        rng = np.random.default_rng(seed)

        def pick(x, y, n):
            if not n or n >= len(y):
                return x, y
            per = max(1, n // len(np.unique(y)))
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y == c), min(per, int((y == c).sum())),
                           replace=False)
                for c in np.unique(y)
            ])
            rng.shuffle(idx)
            return x[idx], y[idx]

        xtr, ytr = pick(self.x_train, self.y_train, n_train)
        xte, yte = pick(self.x_test, self.y_test, n_test)
        return ImageData(xtr, ytr, xte, yte, self.class_names, self.name)


def _flatten(x: np.ndarray) -> np.ndarray:
    """(n, 28, 28) uint8 -> (n, 784) float32 scaled to [0, 1]."""
    return (x.reshape(len(x), -1).astype(np.float32) / 255.0)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def _load_kmnist() -> tuple:
    """Fetch K-MNIST from the ROIS-CODH .npz mirror.

    The idx/ubyte URLs that older code uses now 404; the .npz files are the
    ones still served.
    """
    out = {}
    for key, fname in _KMNIST_FILES.items():
        path = _download(_KMNIST_URL.format(fname), CACHE / fname)
        with np.load(path) as z:
            out[key] = z["arr_0"]
    return (out["x_train"], out["y_train"]), (out["x_test"], out["y_test"])


def _load_medmnist(name: str) -> tuple:
    """Fetch one MedMNIST v2 subset and merge its val split into train.

    MedMNIST ships a train/val/test triple. The reservoir pipeline fits every
    statistic on train and reports on test, with no model selection in
    between, so the val rows would otherwise go unused -- they are folded into
    train instead.
    """
    path = _download(_MEDMNIST_URL.format(name), CACHE / f"{name}.npz")
    with np.load(path) as z:
        x_train = np.concatenate([z["train_images"], z["val_images"]])
        y_train = np.concatenate([z["train_labels"], z["val_labels"]])
        x_test, y_test = z["test_images"], z["test_labels"]
    return (x_train, y_train), (x_test, y_test)


def _to_greyscale(x: np.ndarray) -> np.ndarray:
    """Collapse an RGB channel axis with the usual luminance weights."""
    if x.ndim == 4 and x.shape[-1] == 3:
        return (x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)
    return x


def _load_emnist_letters() -> tuple:
    """EMNIST letters, 26 classes. Larger download, hence a separate path."""
    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    raise RuntimeError(
        "EMNIST is not bundled: it ships as a single large zip "
        f"({url}). Download and extract it manually, then load the "
        "emnist-letters-*-idx*-ubyte files with _read_idx()."
    )


def _read_idx(path: Path) -> np.ndarray:
    """Read an IDX file (the raw MNIST wire format), gzipped or not."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        data = fh.read()
    n_dims = data[3]
    dims = [int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(n_dims)]
    return np.frombuffer(data[4 + 4 * n_dims :], dtype=np.uint8).reshape(dims)


def load(name: str = "mnist") -> ImageData:
    """Load one MNIST-family dataset, downloading and caching on first use."""
    name = name.lower().replace("-", "_")

    if name in ("mnist", "fashion_mnist"):
        import keras  # imported lazily: only the image path needs tensorflow

        loader = getattr(keras.datasets, name)
        (xtr, ytr), (xte, yte) = loader.load_data()
        classes = (
            [str(i) for i in range(10)] if name == "mnist" else FASHION_CLASSES
        )
    elif name == "kmnist":
        (xtr, ytr), (xte, yte) = _load_kmnist()
        classes = KMNIST_CLASSES
    elif name in MEDMNIST_SUBSETS:
        (xtr, ytr), (xte, yte) = _load_medmnist(name)
        xtr, xte = _to_greyscale(xtr), _to_greyscale(xte)
        n_classes = MEDMNIST_SUBSETS[name][0]
        classes = MEDMNIST_CLASS_NAMES.get(
            name, [f"class{i}" for i in range(n_classes)]
        )
    elif name in ("emnist", "emnist_letters"):
        (xtr, ytr), (xte, yte) = _load_emnist_letters()
        classes = [chr(ord("a") + i) for i in range(26)]
    else:
        known = ", ".join(["mnist", "fashion_mnist", "kmnist", *MEDMNIST_SUBSETS])
        raise ValueError(f"unknown dataset {name!r}; expected one of {known}")

    return ImageData(
        x_train=_flatten(xtr),
        y_train=np.asarray(ytr).astype(np.int64).ravel(),
        x_test=_flatten(xte),
        y_test=np.asarray(yte).astype(np.int64).ravel(),
        class_names=classes,
        name=name,
    )


def make_novelty_split(
    data: ImageData,
    normal_classes: list[int],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a one-class anomaly-detection split from a labelled dataset.

    Returns ``(train_normal, test_x, test_is_anomaly)``. Training keeps ONLY
    the ``normal_classes`` rows, so every held-out class is genuinely unseen --
    the image analogue of the zero-day protocol used on NSL-KDD.
    """
    normal = set(normal_classes)
    train_mask = np.isin(data.y_train, list(normal))
    train_x = data.x_train[train_mask]

    test_x = data.x_test
    test_anom = (~np.isin(data.y_test, list(normal))).astype(np.int64)
    return train_x, test_x, test_anom
