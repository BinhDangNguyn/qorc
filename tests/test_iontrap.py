"""The ion-trap reservoir: exact simulator vs two independent references."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from elements import config as C
from elements.core.iontrap import IonTrapReservoir, readout_dim, spread_sites, to_angles

HW_MODULE = Path("D:/Dev/ai-ml/reservoir-computing/ion_traps/qrc_iontrap_core.py")


def _angles(res, n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, (n, res.steps, res.uploads, res.n_enc, 2))
    a[..., 1] *= 2
    return a


# ------------------------------------------------- reference 1: full density matrix


def _op(N, q, U):
    """2x2 gate on qubit q, with qubit q = bit q of the index (LSB first)."""
    out = np.ones((1, 1))
    for k in reversed(range(N)):
        out = np.kron(out, U if k == q else np.eye(2))
    return out


def _brute(res, a):
    """Gate by gate on the whole 2^N register, measure-reset as Kraus maps."""
    N = res.n_qubits
    dim = 1 << N
    bits = (np.arange(dim)[:, None] >> np.arange(N)) & 1
    z = 1 - 2 * bits
    ry = lambda t: np.array([[np.cos(np.pi * t / 2), -np.sin(np.pi * t / 2)],
                             [np.sin(np.pi * t / 2), np.cos(np.pi * t / 2)]])
    rz = lambda p: np.diag([np.exp(-1j * np.pi * p / 2), np.exp(1j * np.pi * p / 2)])
    rx = lambda h: np.array([[np.cos(h), -1j * np.sin(h)], [-1j * np.sin(h), np.cos(h)]])
    phase = sum(res.J[min(i, j), max(i, j)] * res.dt * z[:, i] * z[:, j] for i, j in res.edges)
    ZZ = np.diag(np.exp(-1j * phase))
    X = np.eye(dim)
    for q in range(N):
        X = _op(N, q, rx(res.h[q] * res.dt)) @ X
    k0 = np.array([[1, 0], [0, 0]])
    k1 = np.array([[0, 1], [0, 0]])

    psi = np.zeros(dim, complex)
    psi[0] = 1
    rho = np.outer(psi, psi)
    for k, q in enumerate(res.mem_sites):
        G = _op(N, q, ry(res.mem_init[k]))
        rho = G @ rho @ G.conj().T
    rho0_mem = rho.copy()
    feats_single, feats_corr = [], []
    enc_sites = (res.in_sites + res.mem_sites)[:res.n_enc]
    U = X @ ZZ
    for t in range(res.steps):
        if not res.memory and t:  # memory register back to its initial state
            rho = rho0_mem
        for r in range(res.uploads):
            for k, q in enumerate(enc_sites):
                G = _op(N, q, rz(a[t, r, k, 1])) @ _op(N, q, ry(a[t, r, k, 0]))
                rho = G @ rho @ G.conj().T
            rho = U @ rho @ U.conj().T
        p = np.real(np.diag(rho))
        zin = z[:, res.in_sites]
        feats_single.append(p @ zin)
        feats_corr.append([p @ (zin[:, i] * zin[:, j]) for i in range(res.n_in)
                           for j in range(i + 1, res.n_in)])
        for q in res.in_sites:
            A0, A1 = _op(N, q, k0), _op(N, q, k1)
            rho = A0 @ rho @ A0.conj().T + A1 @ rho @ A1.conj().T
    p = np.real(np.diag(rho))
    final = p @ z[:, res.mem_sites]
    return np.concatenate([np.ravel(feats_single), final, np.ravel(feats_corr)])


@pytest.mark.parametrize("topology,n_in,n_mem,steps,extra", [
    ("all_to_all", 2, 2, 3, {}),
    ("ring", 2, 3, 4, {}),   # inputs spread, includes the wraparound bond
    ("all_to_all", 3, 0, 2, {}),  # no memory register at all
    ("all_to_all", 2, 2, 3, {"uploads": 3}),  # re-uploading inside a round
    ("ring", 2, 3, 2, {"encode_all": True}),  # data on the memory qubits too
    ("all_to_all", 2, 2, 2, {"uploads": 2, "encode_all": True}),
    ("all_to_all", 2, 2, 3, {"uploads": 2, "memory": False}),
])
def test_exact_matches_full_density_matrix(topology, n_in, n_mem, steps, extra):
    sites = spread_sites(n_in + n_mem, n_in) if topology == "ring" else None
    res = IonTrapReservoir(n_in, n_mem, steps, dt=0.4, topology=topology,
                           in_sites=sites, seed=3, **extra)
    a = _angles(res, 3)
    got = res.transform(a)
    want = np.stack([_brute(res, a[i]) for i in range(len(a))])
    assert got.shape == (3, res.dim)
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_ring_closes():
    """The wraparound bond (n-1, 0) must couple; J is stored upper-triangular."""
    res = IonTrapReservoir(2, 3, 2, topology="ring", seed=1)
    assert (4, 0) in res.edges and res.J[0, 4] != 0


# --------------------------------------- reference 2: the hardware module's sampler


@pytest.mark.skipif(not HW_MODULE.exists(), reason="ion_traps module not present")
def test_exact_matches_hardware_module_trajectories():
    spec_ = importlib.util.spec_from_file_location("hw", HW_MODULE)
    hw = importlib.util.module_from_spec(spec_)
    sys.modules["hw"] = hw  # dataclasses resolve annotations through sys.modules
    spec_.loader.exec_module(hw)

    spec = hw.ReservoirSpec(tag="t", n_in=2, n_mem=2, n_steps=3, dt=0.3,
                            topology="all_to_all", seed=1234)
    rng = np.random.default_rng(1234)  # same draw order as IonTrapReservoir
    N = spec.n_qubits
    spec.J = np.triu(rng.uniform(-1, 1, (N, N)), 1)
    spec.h = rng.uniform(-1, 1, N)
    spec.mem_init = rng.uniform(0.2, 0.8, spec.n_mem)

    res = IonTrapReservoir(2, 2, 3, dt=0.3, topology="all_to_all", seed=1234)
    np.testing.assert_allclose(res.J, spec.J)
    a = _angles(res, 2, seed=5)[:, :, 0]  # the hardware module has no uploads axis
    shots = 20000
    bits = hw.run_numpy(spec, a, shots, seed=0, verbose=False)
    ref = hw.features_from_bits(bits, spec, correlators=True)
    # hw layout: L*n_in mids, N finals (inputs first), then correlators
    L, n_in = 3, 2
    ref = np.hstack([ref[:, :L * n_in], ref[:, L * n_in + n_in:L * n_in + N],
                     ref[:, L * n_in + N:]])
    got = res.transform(a)
    assert np.abs(got - ref).max() < 5 / math.sqrt(shots)


# -------------------------------------------------------------- shots, memory, config


def test_shots_are_unbiased_and_noisy():
    res = IonTrapReservoir(2, 2, 3, seed=0)
    a = _angles(res, 4)
    exact = res.transform(a)
    noisy = res.transform(a, shots=50, seed=1)
    many = res.transform(a, shots=200_000, seed=2)
    assert np.abs(noisy - exact).max() > 1e-3
    assert np.abs(many - exact).max() < 0.02


def test_memory_carries_the_past():
    """Changing step 0 must move the final memory readout -- unless memory is off."""
    for memory, moves in ((True, True), (False, False)):
        res = IonTrapReservoir(2, 2, 4, seed=0, memory=memory)
        a = _angles(res, 1)
        b = a.copy()
        b[0, 0] = 1 - b[0, 0]
        mem = slice(res.steps * res.n_in, res.steps * res.n_in + res.n_mem)
        delta = np.abs(res.transform(a)[0, mem] - res.transform(b)[0, mem]).max()
        assert (delta > 1e-4) == moves


def test_angles_cover_every_slot_and_pad():
    res = IonTrapReservoir(2, 1, 3, seed=0)
    ph = np.full((5, 4), np.pi / 4)  # fewer components than the 12 slots
    a = to_angles(ph, res)
    assert a.shape == (5, 3, 1, 2, 2)
    assert np.isclose(a[..., 0], 0.25).sum() + np.isclose(a[..., 0], 0.5).sum() == 5 * 6


def test_config_parses_iontrap_and_photonic():
    cfg = C.from_dict({
        "name": "x", "task": "supervised", "arms": ["quantum", "dials"],
        "grid": {"reservoirs": [
            {"photons": 3, "modes": 12},
            {"kind": "iontrap", "n_in": 4, "n_mem": 4, "steps": 8},
            {"kind": "iontrap", "n_in": 4, "n_mem": 4, "steps": 8, "memory": False},
        ], "heads": [{"kind": "lbfgs"}]},
    })
    r = [st.reservoir for st in cfg.steps()]
    assert r[0].kind == "photonic" and r[0].tag == "3p12m"
    assert r[1].kind == "iontrap" and r[1].modes == 64
    assert r[1].dim == readout_dim(4, 4, 8) == 8 * 4 + 4 + 8 * 6
    assert r[1].tag != r[2].tag and r[2].tag.endswith("_nomem")
    # a saved config round-trips to the same tag
    again = C.reservoir_from_dict({k: v for k, v in vars(r[1]).items()})
    assert again.tag == r[1].tag
    with pytest.raises(ValueError):
        C.reservoir_from_dict({"kind": "iontrap", "n_in": 8, "n_mem": 12, "steps": 16})


@pytest.mark.parametrize("memory", [True, False])
def test_trajectories_converge_to_exact(memory):
    kw = dict(dt=0.3, topology="ring", in_sites=spread_sites(5, 2), seed=4, memory=memory)
    exact = IonTrapReservoir(2, 3, 4, **kw)
    traj = IonTrapReservoir(2, 3, 4, trajectories=3000, **kw)
    a = _angles(exact, 3)
    d = np.abs(exact.transform(a) - traj.transform(a, seed=0))
    assert d.mean() < 3e-3 and d.max() < 0.05
    if not memory:  # one trajectory is exact when nothing is carried forward
        np.testing.assert_allclose(d, 0, atol=1e-8)


def test_twenty_qubits_need_trajectories():
    ring = {"kind": "iontrap", "n_in": 8, "n_mem": 12, "steps": 16, "topology": "ring"}
    with pytest.raises(ValueError):
        C.reservoir_from_dict(ring)
    spec = C.reservoir_from_dict({**ring, "trajectories": 8})
    assert spec.in_sites == [0, 2, 5, 7, 10, 12, 15, 17]
    assert spec.dim == 16 * 8 + 12 + 16 * 28 and spec.tag.endswith("_traj8")


def test_roundrobin_deals_components_evenly():
    from elements.core.iontrap import slot_order

    idx = slot_order(128, 16, "roundrobin").reshape(16, 8)
    assert sorted(idx[:, 0]) == list(range(16))  # components 0..15 hit every step
    assert sorted(idx.ravel()) == list(range(128))


def test_series_matches_batched_rounds():
    """series() on one sequence == laws() on one sample with steps = T."""
    for extra in ({}, {"uploads": 2}, {"memory": False}):
        T = 5
        res = IonTrapReservoir(2, 2, T, seed=9, **extra)
        a = _angles(res, 1)
        f = res.series(a[0])
        full = res.transform(a)[0]
        L, k = T, res.n_in
        np.testing.assert_allclose(f[:, :k].ravel(), full[:L * k], atol=1e-6)
        np.testing.assert_allclose(f[:, k:].ravel(), full[L * k + res.n_mem:], atol=1e-6)
        oracle = res.series(a[0], mem_readout=True)
        assert oracle.shape == (T, k + 1 + res.n_mem)


def test_config_packs_more_per_round():
    base = {"kind": "iontrap", "n_in": 4, "n_mem": 4}
    r2 = C.reservoir_from_dict({**base, "steps": 8, "uploads": 2})
    ea = C.reservoir_from_dict({**base, "steps": 8, "encode_all": True})
    assert r2.n_slots == ea.n_slots == 128 and r2.modes == 128
    assert r2.tag.endswith("_R2") and ea.tag.endswith("_encall")
    assert r2.dim == ea.dim == readout_dim(4, 4, 8)
    with pytest.raises(ValueError):
        C.reservoir_from_dict({"kind": "iontrap", "n_in": 4, "n_mem": 8, "steps": 4,
                               "uploads": 2})
