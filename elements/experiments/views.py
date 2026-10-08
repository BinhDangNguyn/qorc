"""Tasks (train / val / test splits) and the feature views every head is trained on.

Views, all fitted on TRAIN rows only:

``quantum``   the C(m, p) reservoir readout
``dials``     the n_modes PCA vector that drove the reservoir -- the tight control
``rff``       random Fourier features of the dials, exactly C(m, p) wide: a
              classical random lift of matched width
``raw``       the scaled input, no bottleneck -- the classical ceiling
``qdials``    [quantum | dials]: the readout sees the reservoir AND its input
``rffdials``  [rff | dials], the classical counterpart of ``qdials``

``dials`` is what keeps a scaled head honest: an MLP on the dials learns its
own nonlinearity, so quantum only earns credit for beating dials under the
SAME head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from elements.config import DataSpec, EncodingSpec, IonTrapSpec, ReservoirSpec, SplitSpec
from elements.core.encoding import PhaseEncoder, standardize_inplace
from elements.core.iontrap import IonTrapReservoir, to_angles
from elements.core.reservoir import Reservoir


@dataclass
class Task:
    """One dataset split three ways. ``labels`` maps scheme -> (y_tr, y_va, y_te, names).

    For novelty the scheme is ``"novelty"`` and only ``y_te`` is set: the raw
    label strings of the scoring rows (``"normal"`` or an attack name).
    """

    name: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    labels: dict


def _split(idx: np.ndarray, strat: np.ndarray | None, frac: float, seed: int):
    if frac <= 0:  # e.g. val_frac 0: no validation rows
        return np.sort(idx), idx[:0]
    try:
        a, b = train_test_split(idx, test_size=frac, random_state=seed, stratify=strat)
    except ValueError:  # a class too small to stratify
        a, b = train_test_split(idx, test_size=frac, random_state=seed)
    return np.sort(a), np.sort(b)


#: Tabular flow datasets: name -> (loader module, fine label scheme to stratify on)
FLOW_DATASETS = {"nslkdd": "all23", "nf_unsw_nb15_v2": "attack10",
                 "nf_cse_cic_ids2018_v2": "attack15"}


def _load_flows(data: DataSpec, seed: int):
    if data.dataset == "nslkdd":
        from elements.data import nslkdd

        ds = nslkdd.load(data.path)
    else:
        from elements.data import netflow

        ds = netflow.load(data.dataset, data.path, dedup=data.dedup)
    return _maybe_subsample(ds, data.max_rows, seed)


def _maybe_subsample(ds, max_rows: int, seed: int):
    """Stratified cap on flow rows, for smoke tests and quick trials."""
    if not max_rows or max_rows >= len(ds.raw_label):
        return ds
    keep, _ = _split(np.arange(len(ds.raw_label)), ds.raw_label,
                     1 - max_rows / len(ds.raw_label), seed)
    ds.X, ds.raw_label = ds.X[keep], ds.raw_label[keep]
    return ds


def load_supervised_task(data: DataSpec, split: SplitSpec, seed: int) -> Task:
    """Train / val / test. Test rows never touch any fitting or model selection."""
    if data.dataset in FLOW_DATASETS:
        ds = _load_flows(data, seed)
        # Stratify on the finest labels so ONE split (and one set of reservoir
        # features) serves every label scheme.
        fine, _ = ds.labels(FLOW_DATASETS[data.dataset])
        tr, te = _split(np.arange(len(fine)), fine, split.test_size, seed)
        tr, va = _split(tr, fine[tr], split.val_frac, seed)
        labels = {}
        for s in data.schemes:
            y, names = ds.labels(s)
            labels[s] = (y[tr], y[va], y[te], names)
        return Task(data.dataset, ds.X[tr], ds.X[va], ds.X[te], labels)

    from elements.data import images

    ds = images.load(data.dataset)
    if data.n_train:
        ds = ds.subsample(data.n_train, 0, seed)
    tr, va = _split(np.arange(len(ds.y_train)), ds.y_train, split.val_frac, seed)
    return Task(data.dataset, ds.x_train[tr], ds.x_train[va], ds.x_test,
                {"all": (ds.y_train[tr], ds.y_train[va], ds.y_test, ds.class_names)})


def load_novelty_task(data: DataSpec, split: SplitSpec, seed: int) -> Task:
    """The team's normal-only zero-day protocol (docs/README_novelty.md).

    train          (1 - test_size) of ``normal``, minus a normal-only val split
    score-normal   test_size of ``normal`` -- sets the false-alarm threshold
    score-attack   EVERY attack row, all types

    No attack row enters fitting, early stopping or model selection, so every
    attack type is unseen.
    """
    if data.dataset not in FLOW_DATASETS:
        raise ValueError(f"the novelty protocol needs a flow dataset {list(FLOW_DATASETS)}")
    ds = _load_flows(data, seed)
    lab = ds.raw_label
    normal = np.flatnonzero(lab == "normal")
    tr, held = _split(normal, None, split.test_size, seed)
    tr, va = _split(tr, None, split.val_frac, seed)
    test = np.r_[held, np.flatnonzero(lab != "normal")]
    assert (lab[tr] == "normal").all() and (lab[va] == "normal").all()
    return Task(f"{data.dataset}_novelty", ds.X[tr], ds.X[va], ds.X[test],
                {"novelty": (None, None, lab[test], None)})


# ------------------------------------------------------------------- feature views


def _rows(fn, x: np.ndarray, width: int) -> np.ndarray:
    """Apply a transform, passing an empty split (val_frac 0) straight through."""
    if len(x) == 0:
        return np.zeros((0, width), dtype=np.float32)
    return fn(x).astype(np.float32)


def _standardize(train: np.ndarray, *others: np.ndarray) -> None:
    mean, std = standardize_inplace(train)
    for o in others:
        o -= mean
        o /= std


class FrontEnd:
    """Scale -> PCA(n_modes), fitted on TRAIN, shared by every view.

    ``drop_constant`` removes columns constant on train (image border pixels,
    whose ~0 std blows up under scaling). Novelty keeps them: a one-hot column
    that is always 0 in normal traffic and fires only in an attack IS the
    zero-day signal, and StandardScaler passes a zero-std column through
    unamplified.
    """

    def __init__(self, task: Task, seed: int, drop_constant: bool = True):
        keep = task.X_train.std(0) > 1e-6
        if not drop_constant:
            keep[:] = True
        self.scaler = StandardScaler().fit(task.X_train[:, keep])
        self.keep = keep
        self.scaled = [_rows(self.scaler.transform, x[:, keep], int(keep.sum()))
                       for x in (task.X_train, task.X_val, task.X_test)]
        self.seed = seed
        self._pca: dict[int, tuple] = {}

    @property
    def n_features(self) -> int:
        return self.scaled[0].shape[1]

    def raw(self) -> list[np.ndarray]:
        return [x.copy() for x in self.scaled]

    def pca(self, n_modes: int):
        """(train, val, test) PCA scores and the variance they explain."""
        if n_modes > self.n_features:
            raise ValueError(f"{n_modes} modes need {n_modes} PCA components, but the "
                             f"input only has {self.n_features} features")
        if n_modes not in self._pca:
            pca = PCA(n_components=n_modes, random_state=self.seed).fit(self.scaled[0])
            Z = [_rows(pca.transform, x, n_modes) for x in self.scaled]
            self._pca[n_modes] = (Z, float(pca.explained_variance_ratio_.sum()))
        return self._pca[n_modes]


def dials_view(fe: FrontEnd, n_modes: int) -> list[np.ndarray]:
    Z, _ = fe.pca(n_modes)
    out = [z.copy() for z in Z]
    _standardize(*out)
    return out


def input_width(fe: FrontEnd, res) -> int:
    """PCA width that drives ``res``. An ion trap can have more encoding slots
    than there are features (the 20-qubit ring has 256 for 122); it then takes
    every component and pads the rest. A photonic reservoir needs exactly one
    component per mode, so ``fe.pca`` rejects an oversized one."""
    return min(res.modes, fe.n_features) if res.kind == "iontrap" else res.modes


def iontrap_view(fe: FrontEnd, res: IonTrapSpec, enc: EncodingSpec, seed: int, device,
                 save_to=None, log=print) -> list[np.ndarray]:
    """Scale -> PCA -> phase-encode -> sliced across steps -> exact simulation."""
    Z, _ = fe.pca(input_width(fe, res))
    encoder = PhaseEncoder(enc.scheme, enc.width).fit(Z[0])
    reservoir = IonTrapReservoir(
        n_in=res.n_in, n_mem=res.n_mem, steps=res.steps, dt=res.dt,
        topology=res.topology, in_sites=res.in_sites, j_scale=res.j_scale,
        h_scale=res.h_scale, seed=seed if res.seed is None else res.seed,
        correlators=res.correlators, memory=res.memory,
        trajectories=res.trajectories, uploads=res.uploads,
        encode_all=res.encode_all, device=device)
    if save_to is not None:
        reservoir.save(save_to)
    out = []
    for k, z in enumerate(Z):
        def run(z, k=k):
            angles = to_angles(encoder.transform(z), reservoir, res.slice_order)
            return reservoir.transform(angles, shots=res.shots, seed=seed + k)
        out.append(_rows(run, z, res.dim))
    _standardize(*out)
    return out


def quantum_view(fe: FrontEnd, res: ReservoirSpec, enc: EncodingSpec, seed: int,
                 device) -> list[np.ndarray]:
    if res.kind == "iontrap":
        return iontrap_view(fe, res, enc, seed, device)
    Z, _ = fe.pca(res.modes)
    encoder = PhaseEncoder(enc.scheme, enc.width).fit(Z[0])
    reservoir = Reservoir(n_modes=res.modes, n_photons=res.photons,
                          seed=seed if res.seed is None else res.seed,
                          no_bunching=res.no_bunching, device=device)
    out = [_rows(lambda z: reservoir.transform(encoder.transform(z)), z, res.dim) for z in Z]
    _standardize(*out)
    return out


def rff_view(fe: FrontEnd, n_modes: int, dim: int, seed: int,
             chunk: int = 8192) -> list[np.ndarray]:
    """Random Fourier features of the standardised dials, ``dim`` wide.

    Bandwidth by the median heuristic on train pairwise distances -- untuned,
    as the reservoir is untuned.
    """
    Zs = dials_view(fe, n_modes)
    rng = np.random.default_rng(seed + 10_000)
    sample = Zs[0][rng.choice(len(Zs[0]), size=min(2000, len(Zs[0])), replace=False)]
    d2 = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1)
    sigma = float(np.sqrt(np.median(d2[d2 > 0])))
    W = (rng.standard_normal((n_modes, dim)) / sigma).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, dim).astype(np.float32)
    scale = np.float32(np.sqrt(2.0 / dim))
    out = []
    for z in Zs:
        f = np.empty((len(z), dim), dtype=np.float32)
        for i in range(0, len(z), chunk):
            f[i:i + chunk] = scale * np.cos(z[i:i + chunk] @ W + b)
        out.append(f)
    _standardize(*out)
    return out


def append_dials(views: list, dials: list) -> None:
    """Concatenate dials onto each split in place, one split at a time, so the
    peak is one extra split rather than a second copy of all three."""
    for i, d in enumerate(dials):
        views[i] = np.concatenate([views[i], d], axis=1)
