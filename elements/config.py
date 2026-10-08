"""Experiment configuration: what to run, as data rather than code.

One YAML (or JSON) file describes a whole experiment. The reservoir and the
classical head are both specified freely, and a *scaling schedule* says how
they grow together. Three schedules are supported; pick exactly one:

``ladder``   an explicit list of steps, each a reservoir plus the head(s) to
             train on it. Quantum and classical grow together, step by step.

             ladder:
               - reservoir: {photons: 3, modes: 12}
                 head: {kind: mlp, hidden: [128]}
               - reservoir: {photons: 3, modes: 20}
                 head: {kind: mlp, hidden: [256]}

``grid``     every reservoir crossed with every head.

             grid:
               reservoirs: [{photons: 3, modes: 12}, {photons: 4, modes: 24}]
               heads: [{kind: linear}, {kind: mlp, hidden: [512, 512]}]

``rule``     a list of reservoirs, and a rule that derives the head from the
             reservoir's readout dimension:

             rule:
               reservoirs: [...]
               head: {kind: mlp, ratio: 4, min_width: 128, max_width: 2048,
                      depth: 1, dropout: 0.1}

``baseline_heads`` are trained at every step in addition, e.g. the original
L-BFGS linear head, so one run shows before and after scaling the head.

Every field has a default, so a config only states what it changes.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

#: The repository root; relative ``out`` paths resolve against it, so a run
#: lands in the same place whether started from a notebook or the shell.
REPO_ROOT = Path(__file__).resolve().parents[1]

TASKS = ("supervised", "novelty")
SUPERVISED_KINDS = ("lbfgs", "linear", "mlp",
                    # standalone classical models (usually on the raw features)
                    "logreg", "rf", "extratrees", "hgb", "xgb", "knn", "svm")
NOVELTY_KINDS = ("pca", "ae", "iforest",
                 "ocsvm", "lof", "knndist", "gmm")
#: Kinds without gradient-trained weights; n_params reports 0 for them.
_NON_NEURAL = {"pca", "iforest", "logreg", "rf", "extratrees", "hgb", "xgb", "knn", "svm",
               "ocsvm", "lof", "knndist", "gmm"}
ARMS = ("quantum", "dials", "rff", "raw", "qdials", "rffdials")


def _build(cls, data: dict | None):
    """Dataclass from a dict, rejecting unknown keys (a typo is an error)."""
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown field(s) {sorted(unknown)}; "
                         f"allowed: {sorted(known)}")
    return cls(**data)


# ---------------------------------------------------------------------- reservoir


@dataclass
class ReservoirSpec:
    """One photonic reservoir. ``seed`` None means: use the experiment seed."""

    photons: int
    modes: int
    no_bunching: bool = True
    seed: int | None = None

    def __post_init__(self):
        if self.photons < 1 or self.modes < 2:
            raise ValueError(f"need photons >= 1 and modes >= 2, got {self}")
        if self.no_bunching and self.photons > self.modes:
            raise ValueError(f"{self.photons} photons cannot sit in {self.modes} modes "
                             "without bunching")

    @property
    def tag(self) -> str:
        return f"{self.photons}p{self.modes}m" + ("" if self.no_bunching else "_fock")

    @property
    def dim(self) -> int:
        if self.no_bunching:
            return math.comb(self.modes, self.photons)
        return math.comb(self.photons + self.modes - 1, self.photons)

    kind = "photonic"


@dataclass
class IonTrapSpec:
    """A trapped-ion reservoir with quantum memory (``elements.core.iontrap``).

    ``modes`` is the PCA width fed in: 0 means one component per encoding slot
    (``2 * n_in * steps``), capped at the number of input features. It plays the
    role ``modes`` plays for a photonic reservoir, so ``dials`` and ``rff`` are
    built at that width. ``in_sites`` ``spread`` places the inputs evenly round
    a ring; None uses sites 0..n_in-1. ``memory`` False resets the memory qubits
    every step (the no-memory control). ``shots`` 0 gives exact expectations.
    ``trajectories`` 0 simulates the density matrix exactly (<= 12 qubits); K > 0
    averages K quantum trajectories instead, required above 12 qubits.
    ``uploads`` R re-uploads data R times per measurement round (each upload
    followed by one Trotter step); ``encode_all`` puts data on the memory
    qubits too. Either packs more features per round, so fewer rounds (and
    mid-circuit measurements) carry the same input; exact, <= 10 qubits.
    ``seed`` None means: use the experiment seed.
    """

    n_in: int
    n_mem: int
    steps: int
    kind: str = "iontrap"
    dt: float = 0.3
    topology: str = "all_to_all"
    in_sites: str | list[int] | None = "spread"
    j_scale: float = 1.0
    h_scale: float = 1.0
    correlators: bool = True
    memory: bool = True
    slice_order: str = "interleave"
    modes: int = 0
    shots: int = 0
    trajectories: int = 0
    uploads: int = 1
    encode_all: bool = False
    seed: int | None = None

    photons = 0  # the runner's records carry n_photons; an ion trap has none

    def __post_init__(self):
        from elements.core.iontrap import SLICE_ORDERS, TOPOLOGIES, spread_sites

        if self.kind != "iontrap":
            raise ValueError(f"IonTrapSpec kind must be 'iontrap', got {self.kind!r}")
        if self.n_in < 1 or self.n_mem < 0 or self.steps < 1:
            raise ValueError(f"need n_in >= 1, n_mem >= 0, steps >= 1, got {self}")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}")
        if self.slice_order not in SLICE_ORDERS:
            raise ValueError(f"slice_order must be one of {SLICE_ORDERS}")
        if self.n_qubits > 12 and not self.trajectories:
            raise ValueError(f"{self.n_qubits} qubits is beyond exact density-matrix "
                             "simulation (<= 12); set trajectories: K")
        if (self.uploads > 1 or self.encode_all) and (self.trajectories
                                                       or self.n_qubits > 10):
            raise ValueError("uploads > 1 / encode_all need exact simulation of <= 10 "
                             "qubits")
        if self.n_qubits > 24:
            raise ValueError(f"{self.n_qubits} qubits: a 2^N state per trajectory "
                             "no longer fits on the GPU")
        if not self.modes:
            self.modes = self.n_slots
        if self.in_sites == "spread":
            self.in_sites = (spread_sites(self.n_qubits, self.n_in)
                             if self.topology == "ring" else None)

    @property
    def n_qubits(self) -> int:
        return self.n_in + self.n_mem

    @property
    def n_enc(self) -> int:
        return self.n_qubits if self.encode_all else self.n_in

    @property
    def n_slots(self) -> int:
        return 2 * self.n_enc * self.uploads * self.steps

    @property
    def tag(self) -> str:
        topo = {"all_to_all": "a2a", "ring": "ring"}[self.topology]
        t = f"it{self.n_in}+{self.n_mem}_{topo}_L{self.steps}_dt{self.dt:g}"
        if self.uploads > 1:
            t += f"_R{self.uploads}"
        if self.encode_all:
            t += "_encall"
        if self.j_scale != 1.0:
            t += f"_J{self.j_scale:g}"
        if self.h_scale != 1.0:
            t += f"_h{self.h_scale:g}"
        if self.modes != self.n_slots:
            t += f"_w{self.modes}"
        if self.slice_order != "interleave":
            t += f"_{self.slice_order}"
        if not self.correlators:
            t += "_nocorr"
        if not self.memory:
            t += "_nomem"
        if self.trajectories:
            t += f"_traj{self.trajectories}"
        if self.shots:
            t += f"_{self.shots}sh"
        return t

    @property
    def dim(self) -> int:
        from elements.core.iontrap import readout_dim

        return readout_dim(self.n_in, self.n_mem, self.steps, self.correlators)


def reservoir_from_dict(d: dict):
    """``kind: iontrap`` builds an ``IonTrapSpec``; anything else is photonic."""
    if dict(d).get("kind", "photonic") == "iontrap":
        return _build(IonTrapSpec, d)
    d = {k: v for k, v in d.items() if k != "kind"}
    return _build(ReservoirSpec, d)


# --------------------------------------------------------------------------- heads


@dataclass
class HeadSpec:
    """A classical head (supervised) or detector (novelty).

    kind      supervised: ``lbfgs`` (the original full-batch linear softmax),
              ``linear`` (softmax trained like the MLPs), ``mlp``, and the
              standalone classical models ``logreg``, ``rf`` (random forest),
              ``extratrees``, ``hgb`` (histogram gradient boosting), ``xgb``,
              ``knn``, ``svm`` (linear SVM);
              novelty: ``pca`` (Mahalanobis + residual), ``ae`` (autoencoder),
              ``iforest``, ``ocsvm`` (one-class SVM), ``lof`` (local outlier
              factor), ``knndist`` (distance to the k-th normal neighbour),
              ``gmm`` (Gaussian mixture likelihood).
    hidden    hidden-layer widths for ``mlp`` / ``ae`` (encoder side).
    arms      train this head only on these feature views, e.g. ``[raw]`` for an
              IsolationForest baseline. None trains it on every arm.
    """

    kind: str
    hidden: list[int] = field(default_factory=list)
    activation: str = "gelu"
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 12
    l2: float = 1e-4  # lbfgs
    max_iter: int = 200  # lbfgs
    bottleneck: int = 32  # ae (capped at a third of the input width)
    n_components: int = 32  # pca; gmm mixture components
    n_estimators: int = 300  # rf / extratrees / xgb; hgb boosting rounds (iforest keeps 100)
    max_depth: int | None = None  # rf / extratrees / xgb (None = model default)
    n_neighbors: int = 20  # knn / lof / knndist
    C: float = 1.0  # logreg / svm (inverse regularisation)
    nu: float = 0.05  # ocsvm: bound on the training outlier fraction
    max_train_rows: int = 0  # subsample train for O(n^2) models such as ocsvm (0 = all)
    arms: list[str] | None = None  # restrict to these feature views (None = all)
    name: str | None = None

    def __post_init__(self):
        if self.kind not in SUPERVISED_KINDS + NOVELTY_KINDS:
            raise ValueError(f"unknown head kind {self.kind!r}")
        if self.activation not in ("gelu", "relu", "tanh"):
            raise ValueError(f"unknown activation {self.activation!r}")
        self.hidden = [int(h) for h in self.hidden]
        if self.kind not in ("mlp", "ae") and self.hidden:
            raise ValueError(f"head kind {self.kind!r} takes no hidden layers")
        if self.arms is not None and set(self.arms) - set(ARMS):
            raise ValueError(f"head {self.label}: unknown arm(s) in {self.arms}")

    def applies_to(self, arm: str) -> bool:
        return self.arms is None or arm in self.arms

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        width = "x".join(str(h) for h in self.hidden)
        return {
            "lbfgs": "lbfgs",
            "linear": "linear",
            "mlp": f"mlp{width}",
            "ae": f"ae{width}-b{self.bottleneck}",
            "pca": f"pca{self.n_components}",
            "iforest": "iforest",
            "logreg": "logreg",
            "rf": f"rf{self.n_estimators}",
            "extratrees": f"extratrees{self.n_estimators}",
            "hgb": "hgb",
            "xgb": f"xgb{self.n_estimators}",
            "knn": f"knn{self.n_neighbors}",
            "svm": "linsvm",
            "ocsvm": f"ocsvm-nu{self.nu:g}",
            "lof": f"lof{self.n_neighbors}",
            "knndist": f"knndist{self.n_neighbors}",
            "gmm": f"gmm{self.n_components}",
        }[self.kind]

    @property
    def task(self) -> str:
        return "supervised" if self.kind in SUPERVISED_KINDS else "novelty"

    def n_params(self, d_in: int, n_out: int) -> int:
        """Trainable parameters for a given input width (0 for non-neural models)."""
        if self.kind in _NON_NEURAL:
            return 0
        if self.kind == "ae":
            b = max(2, min(self.bottleneck, d_in // 3))
            widths = [d_in, *self.hidden, b, *reversed(self.hidden), d_in]
        else:
            widths = [d_in, *self.hidden, n_out]
        return sum(a * b + b for a, b in zip(widths, widths[1:]))


def head_from_rule(rule: dict, readout_dim: int) -> HeadSpec:
    """``rule`` head: width = readout_dim / ratio, rounded to a power of 2, clipped."""
    rule = dict(rule)
    ratio = rule.pop("ratio", 4)
    lo, hi = rule.pop("min_width", 128), rule.pop("max_width", 2048)
    depth = rule.pop("depth", 1)
    width = 2 ** int(round(math.log2(max(readout_dim / ratio, 1))))
    width = int(min(max(width, lo), hi))
    return _build(HeadSpec, {**rule, "hidden": [width] * depth})


# ------------------------------------------------------------------------ the rest


@dataclass
class DataSpec:
    dataset: str = "nslkdd"  # nslkdd | nf_unsw_nb15_v2 | nf_cse_cic_ids2018_v2 | mnist | kmnist | fashion_mnist | <medmnist>
    path: str | None = None  # flow csv; None searches the usual places
    schemes: list[str] = field(default_factory=lambda: ["families6", "all23"])
    n_train: int = 0  # images only: cap the training set (0 = all)
    max_rows: int = 0  # flow data: stratified subsample before splitting (0 = all); smoke tests
    dedup: bool = False  # NetFlow datasets: drop exact duplicate rows before splitting


@dataclass
class SplitSpec:
    test_size: float = 0.30  # supervised; novelty holds out this share of normal
    val_frac: float = 0.10  # carved from TRAIN, for early stopping only; 0 = none
    # (allowed only when no head early-stops: lbfgs / pca / iforest)


@dataclass
class EncodingSpec:
    scheme: str = "tanh"  # tanh | minmax
    width: float = 3.0  # tanh: standard deviations mapped to the bulk of (0, pi)


@dataclass
class Step:
    reservoir: ReservoirSpec
    heads: list[HeadSpec]
    role: dict = field(default_factory=dict)  # head label -> "scaled" | "baseline"


@dataclass
class ExperimentConfig:
    name: str
    task: str = "supervised"
    data: DataSpec = field(default_factory=DataSpec)
    seed: int = 42
    split: SplitSpec = field(default_factory=SplitSpec)
    encoding: EncodingSpec = field(default_factory=EncodingSpec)
    arms: list[str] = field(default_factory=lambda: ["quantum", "dials", "rff"])
    ladder: list | None = None
    grid: dict | None = None
    rule: dict | None = None
    models: list | None = None
    baseline_heads: list = field(default_factory=list)
    device: str = "auto"
    out: str = "runs"
    notes: str = ""

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        bad = set(self.arms) - set(ARMS)
        if bad:
            raise ValueError(f"unknown arm(s) {sorted(bad)}; allowed {ARMS}")
        given = [k for k in ("ladder", "grid", "rule", "models") if getattr(self, k)]
        if len(given) != 1:
            raise ValueError("give exactly one of ladder / grid / rule / models, got "
                             f"{given or 'none'}")
        if self.models and self.arms != ["raw"]:
            raise ValueError("a `models` config trains standalone classical models on the "
                             "raw features; set arms: [raw]")
        early_stopping = {"linear", "mlp", "ae"}
        for h in self.all_heads():
            if h.kind in early_stopping and self.split.val_frac <= 0:
                raise ValueError(f"head {h.label} early-stops on validation rows; "
                                 "set split.val_frac > 0")
            if h.task != self.task:
                raise ValueError(f"head {h.label} ({h.kind}) is a {h.task} head, "
                                 f"but this is a {self.task} experiment")

    # ---- the schedule, resolved to concrete steps -----------------------------
    def all_heads(self) -> list[HeadSpec]:
        """Every distinct head the run trains, in first-use order."""
        heads = {h.label: h for st in self.steps() for h in st.heads}
        for h in self.model_heads():
            heads.setdefault(h.label, h)
        return list(heads.values())

    def model_heads(self) -> list[HeadSpec]:
        """``models`` schedule: standalone classical models, no reservoir at all."""
        return [_build(HeadSpec, h) for h in (self.models or [])]

    def steps(self) -> list[Step]:
        baselines = [_build(HeadSpec, h) for h in self.baseline_heads]
        out: list[Step] = []
        if self.models:
            return out  # no reservoir steps; see model_heads()
        if self.ladder:
            for item in self.ladder:
                item = dict(item)
                res = reservoir_from_dict(item.pop("reservoir"))
                heads = item.pop("heads", None) or [item.pop("head")]
                if item:
                    raise ValueError(f"ladder step: unknown field(s) {sorted(item)}")
                out.append(self._step(res, [_build(HeadSpec, h) for h in heads], baselines))
        elif self.grid:
            heads = [_build(HeadSpec, h) for h in self.grid["heads"]]
            for r in self.grid["reservoirs"]:
                out.append(self._step(reservoir_from_dict(r), heads, baselines))
        else:
            for r in self.rule["reservoirs"]:
                res = reservoir_from_dict(r)
                out.append(self._step(res, [head_from_rule(self.rule["head"], res.dim)],
                                      baselines))
        return out

    @staticmethod
    def _step(res, scaled, baselines) -> Step:
        heads, role = [], {}
        for h, r in [(h, "scaled") for h in scaled] + [(h, "baseline") for h in baselines]:
            if h.label not in role:
                heads.append(h)
                role[h.label] = r
        return Step(res, heads, role)

    @property
    def out_dir(self) -> Path:
        out = Path(self.out)
        return (out if out.is_absolute() else REPO_ROOT / out) / self.name

    def to_dict(self) -> dict:
        return asdict(self)


def from_dict(d: dict) -> ExperimentConfig:
    d = copy.deepcopy(d)
    for key, cls in (("data", DataSpec), ("split", SplitSpec), ("encoding", EncodingSpec)):
        d[key] = _build(cls, d.get(key))
    return _build(ExperimentConfig, d)


def load(path: str | Path, overrides: dict | None = None) -> ExperimentConfig:
    """Read a YAML / JSON config, applying dotted overrides like ``{"seed": 1}``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    d = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    for key, value in (overrides or {}).items():
        node = d
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return from_dict(d)
