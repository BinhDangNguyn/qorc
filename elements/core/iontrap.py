"""A trapped-ion quantum reservoir with genuine quantum memory (NISQRC).

Ported from ``reservoir-computing/ion_traps/qrc_iontrap_core.py``, which keeps
the pytket / Quantinuum submission path. This module is the simulation side,
built to run on a whole dataset rather than a few hundred rows.

The register splits into INPUT and MEMORY qubits. Each of ``steps`` timesteps:

1. encode a slice of the sample on the input qubits (dense angle encoding:
   Ry = latitude, Rz = longitude, two features per qubit);
2. evolve the whole register under one Trotter step of
   ``H = sum_<ij> J_ij Z_i Z_j + sum_i h_i X_i`` (frozen random J, h);
3. measure and reset ONLY the input qubits.

The memory qubits are never measured, so they carry correlations from one step
to the next -- the thing the photonic reservoir cannot do (see
``core/temporal.py:QuantumMemory``). Nothing here is trained.

Exact simulation
----------------
Measure-then-reset with no feed-forward is a CPTP channel: "trace out the input
qubits, put them back in |0>". So the memory register evolves deterministically
as a density matrix, and every readout feature is an exact expectation value:

    rho_out = V rho_mem V^+,     V = U_step (|psi_in(x_t)> (x) I_mem)
    P_t(x)  = <x| Tr_mem rho_out |x>        joint outcome law of the inputs
    rho_mem <- Tr_in rho_out

Cost per sample and step is O(A^2 M^2) to form V and O(A M^3) for the partial
trace (A = 2^n_in, M = 2^n_mem), so this is practical up to ~12 qubits. Larger
registers (the 20-qubit ring) need trajectory or hardware runs.

``shots > 0`` draws a finite shot table from the exact per-step laws, so each
feature carries the 1/sqrt(shots) noise a hardware run would. Within a step the
outcome law is exact; correlations ACROSS steps are not reproduced, which no
readout feature here depends on.

Conventions (pytket half-turns, as in the hardware path):
``ZZPhase(2 J dt / pi)`` realises exp(-i J dt Z Z); ``PhasedX(2 h dt / pi, 0)``
realises exp(-i h dt X); the encoding is ``PhasedX(theta, 0.5)`` (= Ry(pi
theta)) then ``Rz(phi)``, with theta in (0, 1) and phi in (0, 2) half-turns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

TOPOLOGIES = ("all_to_all", "ring")
SLICE_ORDERS = ("interleave", "roundrobin", "variance_first", "variance_last")


def make_edges(topology: str, n: int) -> list[tuple[int, int]]:
    """Coupled pairs: every pair (all-to-all) or nearest neighbours (ring)."""
    if topology == "all_to_all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if topology == "ring":
        if n < 3:
            raise ValueError("a ring needs at least 3 qubits")
        return [(i, (i + 1) % n) for i in range(n)]
    raise ValueError(f"unknown topology {topology!r}")


def spread_sites(n_total: int, n_in: int) -> list[int]:
    """Input sites spread evenly round a ring, so every memory qubit is a few
    bonds from an injection site. ``spread_sites(20, 8) -> [0, 2, 5, 7, ...]``."""
    if n_in > n_total:
        raise ValueError("more input qubits than qubits")
    sites = sorted({(i * n_total) // n_in for i in range(n_in)})
    if len(sites) != n_in:
        sites = sorted(set(range(0, n_total, max(1, n_total // n_in))))[:n_in]
    return sites


def readout_dim(n_in: int, n_mem: int, steps: int, correlators: bool = True) -> int:
    """<Z> per (step, input) + final <Z> per memory qubit + same-step <Z_i Z_j>.

    The input qubits' final <Z> is left out: they were just reset, so it is
    identically +1 and carries nothing.
    """
    return steps * n_in + n_mem + (steps * math.comb(n_in, 2) if correlators else 0)


# ------------------------------------------------------------------ single gates


def _ry(theta_ht):
    """Ry(pi * theta) as (..., 2, 2); theta in half-turns."""
    a = np.pi * np.asarray(theta_ht, dtype=np.float64) / 2
    c, s = np.cos(a), np.sin(a)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(np.complex128)


def _rx(alpha_ht: float) -> np.ndarray:
    a = np.pi * alpha_ht / 2
    return np.array([[np.cos(a), -1j * np.sin(a)], [-1j * np.sin(a), np.cos(a)]])


def _kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = np.ones((1, 1), dtype=np.complex128)
    for m in mats:
        out = np.kron(out, m)
    return out


@dataclass
class IonTrapReservoir:
    """Frozen random couplings plus the exact simulator that reads them.

    Two reservoirs with the same (n_in, n_mem, topology, dt, scales, seed) are
    identical. ``trajectories`` 0 simulates the density matrix exactly (up to
    ~12 qubits); K > 0 averages K quantum trajectories instead (``_laws_traj``),
    which is what makes the 20-qubit ring reachable.

    ``uploads`` R re-uploads data R times per measurement round, each upload
    followed by one Trotter step, so a round carries ``2 * n_enc * R`` features
    with ONE mid-circuit measurement. ``encode_all`` also rotates the memory
    qubits by data (n_enc = all qubits instead of the inputs). Both use the
    general dense-density-matrix round, practical up to ~10 qubits.
    ``leak`` gamma pulls the memory back to its initial state after each round,
    rho <- (1 - gamma) rho + gamma rho0 -- engineered dissipation, i.e. fading
    memory (on hardware: reset the memory register with probability gamma);
    gamma = 1 is ``memory=False``, which re-initialises the memory register at every
    step: same Hilbert space and same per-step dynamics, but nothing carried
    forward -- the control that isolates what the memory contributes.
    """

    n_in: int
    n_mem: int
    steps: int
    dt: float = 0.3
    topology: str = "all_to_all"
    in_sites: list[int] | None = None
    j_scale: float = 1.0
    h_scale: float = 1.0
    seed: int = 1234
    correlators: bool = True
    memory: bool = True
    trajectories: int = 0
    uploads: int = 1
    encode_all: bool = False
    leak: float = 0.0
    device: torch.device | str = "cpu"
    J: np.ndarray = field(init=False, repr=False)
    h: np.ndarray = field(init=False, repr=False)
    mem_init: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology {self.topology!r}")
        N = self.n_qubits
        if self.in_sites is None:
            self.in_sites = list(range(self.n_in))
        self.in_sites = [int(s) for s in self.in_sites]
        if len(set(self.in_sites)) != self.n_in or max(self.in_sites) >= N:
            raise ValueError(f"in_sites {self.in_sites} must be {self.n_in} distinct "
                             f"sites below {N}")
        if self.uploads < 1:
            raise ValueError("uploads must be >= 1")
        if self.leak and self.trajectories:
            raise ValueError("leak is implemented for exact simulation only")
        if self.general and (self.trajectories or N > 10):
            raise ValueError("uploads > 1 / encode_all need exact simulation of <= 10 "
                             "qubits")
        self.device = torch.device(self.device)
        # Same draw order as the hardware module, so a seed means the same
        # reservoir in both places.
        rng = np.random.default_rng(self.seed)
        self.J = np.triu(rng.uniform(-self.j_scale, self.j_scale, (N, N)), 1)
        self.h = rng.uniform(-self.h_scale, self.h_scale, N)
        # |0> is a Z eigenstate and inert under the ZZ layer; a fixed random Ry
        # makes the memory live from step 0.
        self.mem_init = rng.uniform(0.2, 0.8, self.n_mem)
        self._build()

    # ---- geometry -------------------------------------------------------------
    @property
    def n_qubits(self) -> int:
        return self.n_in + self.n_mem

    @property
    def mem_sites(self) -> list[int]:
        inp = set(self.in_sites)
        return [q for q in range(self.n_qubits) if q not in inp]

    @property
    def edges(self) -> list[tuple[int, int]]:
        return make_edges(self.topology, self.n_qubits)

    @property
    def n_enc(self) -> int:
        """Qubits that receive data: the inputs, or every qubit (encode_all)."""
        return self.n_qubits if self.encode_all else self.n_in

    @property
    def general(self) -> bool:
        """Needs the dense round (re-uploading, or data on the memory qubits)."""
        return self.uploads > 1 or self.encode_all

    @property
    def n_slots(self) -> int:
        """Features one sample can carry: 2 per encoded qubit per upload per step."""
        return 2 * self.n_enc * self.uploads * self.steps

    @property
    def dim(self) -> int:
        return readout_dim(self.n_in, self.n_mem, self.steps, self.correlators)

    # ---- the frozen step unitary ----------------------------------------------
    def _build(self) -> None:
        """U_step = X-layer . ZZ-layer, in LOGICAL order [inputs..., memory...].

        Logical qubit l is the l-th kron factor (most significant first), so a
        basis index splits as ``x * M + y`` with x over inputs, y over memory.
        """
        N, A, M = self.n_qubits, 1 << self.n_in, 1 << self.n_mem
        order = self.in_sites + self.mem_sites  # logical -> physical site
        pos = {site: l for l, site in enumerate(order)}

        idx = np.arange(1 << N, dtype=np.int64)
        phase = np.zeros(1 << N)
        for i, j in self.edges:
            zi = 1 - 2 * ((idx >> (N - 1 - pos[i])) & 1)
            zj = 1 - 2 * ((idx >> (N - 1 - pos[j])) & 1)
            phase += self.J[min(i, j), max(i, j)] * self.dt * zi * zj
        del idx
        zz = np.exp(-1j * phase)
        del phase
        # exp(-i h dt X) == PhasedX(2 h dt / pi, 0)
        rx = [_rx(2 * self.h[site] * self.dt / np.pi) for site in order]

        cdt = torch.complex64 if self.device.type == "cuda" else torch.complex128
        self._cdt = cdt
        m0 = _kron_all([_ry(a) @ np.array([[1], [0]]) for a in self.mem_init]) \
            if self.n_mem else np.ones((1, 1))
        if self.trajectories:
            # Never form U: the ZZ layer is a diagonal and the X layer factors
            # into kron blocks of <= 4 qubits, applied one tensor axis at a time.
            self._zz = torch.as_tensor(zz, dtype=cdt, device=self.device)
            groups = [list(range(a, min(a + 4, hi)))
                      for lo, hi in ((0, self.n_in), (self.n_in, N))
                      for a in range(lo, hi, 4)]
            self._xblocks = [torch.as_tensor(_kron_all([rx[l] for l in g]), dtype=cdt,
                                             device=self.device) for g in groups]
            self._phi0 = torch.as_tensor(m0[:, 0], dtype=cdt, device=self.device)
        else:
            U = _kron_all(rx) * zz[None, :]  # X . diag(zz)
            self._U = torch.as_tensor(U.reshape(A, M, A, M), dtype=cdt, device=self.device)
            if self.general:
                self._Ufull = self._U.reshape(A * M, A * M)
            self._rho0 = torch.as_tensor(m0 @ m0.conj().T, dtype=cdt, device=self.device)

        # +-1 eigenvalue tables over the 2^n_in input outcomes and 2^n_mem
        # memory outcomes, for turning laws into <Z> and <Z_i Z_j>.
        za = np.array([[1 - 2 * ((x >> (self.n_in - 1 - k)) & 1) for k in range(self.n_in)]
                       for x in range(A)], dtype=np.float64)
        pairs = [(i, j) for i in range(self.n_in) for j in range(i + 1, self.n_in)]
        zza = (np.stack([za[:, i] * za[:, j] for i, j in pairs], 1)
               if pairs and self.correlators else np.zeros((A, 0)))
        zm = np.array([[1 - 2 * ((y >> (self.n_mem - 1 - k)) & 1) for k in range(self.n_mem)]
                       for y in range(M)], dtype=np.float64).reshape(M, self.n_mem)
        rdt = torch.float32 if cdt == torch.complex64 else torch.float64
        self._Zin = torch.as_tensor(np.hstack([za, zza]), dtype=rdt, device=self.device)
        self._Zmem = torch.as_tensor(zm, dtype=rdt, device=self.device)

    # ---- simulation -----------------------------------------------------------
    def _rotations(self, angles: torch.Tensor) -> torch.Tensor:
        """(B, k, 2) half-turn angles -> (B, k, 2, 2) Rz(phi) . Ry(pi theta)."""
        th, ph = np.pi * angles[..., 0] / 2, np.pi * angles[..., 1] / 2
        c, s = torch.cos(th).to(self._cdt), torch.sin(th).to(self._cdt)
        em, ep = torch.exp(-1j * ph).to(self._cdt), torch.exp(1j * ph).to(self._cdt)
        return torch.stack([torch.stack([em * c, -em * s], -1),
                            torch.stack([ep * s, ep * c], -1)], -2)

    def _encoder(self, angles: torch.Tensor) -> torch.Tensor:
        """(B, n_enc, 2) -> (B, D, D): data rotations on the encoded qubits and
        identity elsewhere, in logical order [inputs..., memory...]."""
        R = self._rotations(angles)
        B = R.shape[0]
        eye = torch.eye(2, dtype=self._cdt, device=self.device).expand(B, 2, 2)
        E = None
        for l in range(self.n_qubits):
            g = R[:, l] if l < self.n_enc else eye
            E = g if E is None else torch.einsum("bij,bkl->bikjl", E, g).reshape(
                B, E.shape[1] * 2, E.shape[2] * 2)
        return E

    def _round(self, rho: torch.Tensor, a: torch.Tensor):
        """One measurement round starting from memory state ``rho`` (B, M, M).

        ``a`` (B, uploads, n_enc, 2). Returns the joint law of the input
        measurements P (B, A) and the memory state after measure-and-reset.
        """
        A, M = 1 << self.n_in, 1 << self.n_mem
        if not self.general:
            psi = self._input_state(a[:, 0])
            V = torch.einsum("xyim,bi->bxym", self._U, psi)  # (B, A, M, M_in)
            W = torch.einsum("bxym,bmn->bxyn", V, rho)
            # P(x) = sum_y (V rho V^+)[xy, xy]
            P = torch.einsum("bxyn,bxyn->bx", W, V.conj()).real.clamp_min(0)
            # rho_mem <- Tr_in (V rho V^+)
            rho = torch.einsum("bxyn,bxzn->byz", W, V.conj())
        else:
            B, D = rho.shape[0], A * M
            full = torch.zeros(B, A, M, A, M, dtype=self._cdt, device=self.device)
            full[:, 0, :, 0, :] = rho  # inputs start the round in |0>
            full = full.reshape(B, D, D)
            for r in range(self.uploads):
                G = self._Ufull @ self._encoder(a[:, r])  # encode, then evolve
                full = G @ full @ G.conj().transpose(1, 2)
            full = full.reshape(B, A, M, A, M)
            blocks = torch.diagonal(full, dim1=1, dim2=3)  # (B, M, M, A): x == x'
            P = torch.diagonal(blocks, dim1=1, dim2=2).real.sum(-1).clamp_min(0)
            rho = blocks.sum(-1)  # Tr_in
        rho = 0.5 * (rho + rho.transpose(1, 2).conj())  # keep it Hermitian
        if self.leak:
            rho = (1 - self.leak) * rho + self.leak * self._rho0
        return P, rho

    def series(self, angles: np.ndarray, mem_readout: bool = False,
               rho_start: np.ndarray | None = None) -> np.ndarray:
        """Run ONE long sequence as one circuit: round t injects ``angles[t]``.

        angles (T, uploads, n_enc, 2), or (T, n_in, 2) when uploads == 1 and
        only the inputs are encoded. Returns per-round features (T, d): input
        <Z> and same-round <Z_i Z_j> -- what hardware measures. ``mem_readout``
        appends the memory <Z> after each round: a simulation-only ORACLE
        (hardware would have to rerun the circuit up to t to read it), useful
        to see how much memory the measured inputs fail to expose.
        ``rho_start`` overrides the initial memory state (echo-state tests).
        """
        if self.trajectories:
            raise ValueError("series() needs exact simulation (trajectories=0)")
        a = torch.as_tensor(np.asarray(angles), device=self.device,
                            dtype=torch.float32 if self._cdt == torch.complex64
                            else torch.float64)
        if a.ndim == 3:
            a = a[:, None]
        rho = (self._rho0 if rho_start is None else
               torch.as_tensor(rho_start, dtype=self._cdt, device=self.device))[None].clone()
        out = []
        with torch.no_grad():
            for t in range(len(a)):
                if not self.memory and t:
                    rho = self._rho0[None].clone()
                P, rho = self._round(rho, a[t:t + 1])
                f = [P @ self._Zin]
                if mem_readout:
                    f.append(torch.diagonal(rho, dim1=1, dim2=2).real @ self._Zmem)
                out.append(torch.cat(f, 1))
        return torch.cat(out, 0).float().cpu().numpy()

    def _input_state(self, angles: torch.Tensor) -> torch.Tensor:
        """(B, n_in, 2) half-turn angles -> (B, 2^n_in) product state."""
        th, ph = np.pi * angles[..., 0] / 2, np.pi * angles[..., 1] / 2
        q = torch.stack([torch.cos(th) * torch.exp(-1j * ph),
                         torch.sin(th) * torch.exp(1j * ph)], -1).to(self._cdt)
        psi = q[:, 0]
        for k in range(1, self.n_in):
            psi = (psi[:, :, None] * q[:, k, None, :]).reshape(len(q), -1)
        return psi

    def _laws_traj(self, a: torch.Tensor, gen) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantum-trajectory estimate of the same laws as ``laws``.

        After measure-and-reset the input qubits are back in |0>, so between
        steps a trajectory is just a PURE memory state phi (2^n_mem). Each step:
        Psi = U (psi_in (x) phi); record this trajectory's exact outcome law
        P(x) = sum_y |Psi[x, y]|^2; draw x from it; continue with
        phi = Psi[x, :] / norm. Averaging the recorded laws (not sampled bits)
        over K trajectories is unbiased for the exact laws, and far less noisy
        than averaging K shots.
        """
        # Without memory every trajectory restarts from phi0 each step, so they
        # are all identical and one is exact.
        n, K = a.shape[0], (self.trajectories if self.memory else 1)
        A, M = 1 << self.n_in, 1 << self.n_mem
        a = a.repeat_interleave(K, 0)
        B = a.shape[0]
        rows = torch.arange(B, device=self.device)
        dims = [blk.shape[0] for blk in self._xblocks]
        phi = self._phi0.expand(B, M).clone()
        P = []
        for t in range(self.steps):
            if not self.memory and t:
                phi = self._phi0.expand(B, M).clone()
            psi = (self._input_state(a[:, t, 0])[:, :, None] * phi[:, None, :]).reshape(B, -1)
            psi *= self._zz
            psi = psi.reshape(B, *dims)
            for g, X in enumerate(self._xblocks):
                psi = torch.movedim(torch.movedim(psi, g + 1, -1) @ X.T, -1, g + 1)
            psi = psi.reshape(B, A, M)
            prob = psi.real ** 2 + psi.imag ** 2
            p = prob.sum(-1)
            P.append(p)
            if t == self.steps - 1:
                # final memory readout: the marginal over x, i.e. averaged over
                # the last measurement exactly rather than one sampled outcome
                P_mem = prob.sum(1).reshape(n, K, M).mean(1)
                break
            x = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=gen)[:, 0]
            phi = psi[rows, x] / p[rows, x].sqrt()[:, None]
            del psi, prob
        P_in = torch.stack(P, 1).reshape(n, K, self.steps, A).mean(1)
        return P_in, P_mem

    def laws(self, angles: np.ndarray, gen=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact outcome laws. angles (B, steps, uploads, n_enc, 2) in
        half-turns; (B, steps, n_in, 2) is accepted when uploads == 1.

        Returns ``P_in`` (B, steps, 2^n_in), the joint law of the input
        measurements at each step, and ``P_mem`` (B, 2^n_mem), the law of the
        final memory readout.
        """
        a = torch.as_tensor(angles, device=self.device,
                            dtype=torch.float32 if self._cdt == torch.complex64
                            else torch.float64)
        if a.ndim == 4:
            a = a[:, :, None]
        if self.trajectories:
            return self._laws_traj(a, gen)
        B = a.shape[0]
        rho = self._rho0.expand(B, -1, -1).clone()
        P = []
        for t in range(self.steps):
            if not self.memory and t:
                rho = self._rho0.expand(B, -1, -1).clone()
            p, rho = self._round(rho, a[:, t])
            P.append(p)
        P_in = torch.stack(P, 1)
        P_mem = torch.diagonal(rho, dim1=1, dim2=2).real.clamp_min(0)
        return P_in, P_mem

    def _features(self, P_in: torch.Tensor, P_mem: torch.Tensor) -> torch.Tensor:
        """Laws (or empirical frequencies) -> readout features."""
        B, L = P_in.shape[:2]
        E = P_in @ self._Zin  # (B, L, n_in + C(n_in, 2))
        single, corr = E[..., :self.n_in], E[..., self.n_in:]
        return torch.cat([single.reshape(B, -1), P_mem @ self._Zmem,
                          corr.reshape(B, -1)], 1)

    def transform(self, angles: np.ndarray, shots: int = 0, batch: int | None = None,
                  seed: int = 0) -> np.ndarray:
        """(n, steps, n_in, 2) angles -> (n, dim) features. ``shots`` 0 = exact.

        ``batch`` None sizes batches so the (B, A, M, M) intermediate stays
        near 512 MB.
        """
        if batch is None:
            if self.trajectories:  # ~4 live copies of the 2^N state per trajectory
                k = self.trajectories if self.memory else 1
                per_row = k * (1 << self.n_qubits) * 8 * 4
                batch = int(max(1, min(4096, 2.5e9 // per_row)))
            elif self.general:  # a few dense D x D matrices per row
                per_row = (1 << self.n_qubits) ** 2 * 8 * 6
                batch = int(max(1, min(4096, 1.5e9 // per_row)))
            else:
                per_row = (1 << self.n_in) * (1 << self.n_mem) ** 2 * 16
                batch = int(max(1, min(4096, 5e8 // per_row)))
        gen = (torch.Generator(device=self.device).manual_seed(seed)
               if shots or self.trajectories else None)
        out = []
        with torch.no_grad():
            for i in range(0, len(angles), batch):
                P_in, P_mem = self.laws(angles[i:i + batch], gen)
                if shots:
                    P_in = _sample(P_in, shots, gen)
                    P_mem = _sample(P_mem, shots, gen)
                out.append(self._features(P_in, P_mem).float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, self.dim), np.float32)

    # ---- the deliverable: the frozen parameters --------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the frozen reservoir. With the hardware module's spec this file
        fully determines the circuit that would be submitted."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, J=self.J, h=self.h, mem_init=self.mem_init, dt=self.dt,
                 seed=self.seed, in_sites=np.asarray(self.in_sites),
                 topology=self.topology, n_in=self.n_in, n_mem=self.n_mem,
                 steps=self.steps, uploads=self.uploads, encode_all=self.encode_all)
        return path

    def budget(self) -> dict:
        """Gate counts for ONE sample on hardware (as in the hardware module)."""
        L, R, n_in, N = self.steps, self.uploads, self.n_in, self.n_qubits
        return {
            "n_2q": L * R * len(self.edges),
            "n_1q": self.n_mem + L * R * (2 * self.n_enc + N),
            "n_meas": L * n_in + N,
            "depth_2q": L * R * (2 if self.topology == "ring" else N - 1),
        }


def _sample(P: torch.Tensor, shots: int, gen) -> torch.Tensor:
    """Empirical frequencies of ``shots`` draws from each row's law (last axis)."""
    flat = P.reshape(-1, P.shape[-1]).float()
    flat = flat / flat.sum(-1, keepdim=True)
    draws = torch.multinomial(flat, shots, replacement=True, generator=gen)
    freq = torch.zeros_like(flat).scatter_add_(1, draws, torch.ones_like(draws, dtype=flat.dtype))
    return (freq / shots).reshape(P.shape).to(P.dtype)


# ------------------------------------------------------------------- encoding


def slot_order(n_slots: int, steps: int, order: str = "interleave") -> np.ndarray:
    """Which feature component lands in which (step, qubit, angle) slot.

    PCA orders components by variance. ``roundrobin`` deals component c to
    step c mod steps, so every step gets an equal share of the strong ones.
    ``interleave`` is the original ion_traps ordering, kept so earlier runs
    reproduce: it equals ``roundrobin`` only when a step holds exactly
    ``steps`` slots (e.g. L = 8 with 4 inputs); at L = 16 it puts the strongest
    components on even steps only. The last two put them all first or last.
    """
    idx = np.arange(n_slots)
    if order == "roundrobin":
        return idx.reshape(-1, steps).T.ravel()
    if order == "variance_first":
        return idx
    if order == "variance_last":
        return idx[::-1].copy()
    if order == "interleave":
        return idx.reshape(steps, -1).T.ravel()
    raise ValueError(f"unknown slice_order {order!r}")


def to_angles(phases: np.ndarray, res: IonTrapReservoir,
              order: str = "interleave") -> np.ndarray:
    """PhaseEncoder output in (0, pi) -> (n, steps, uploads, n_enc, 2) angles.

    Missing components (fewer features than slots) are padded at mid-range,
    pi/2: a fixed rotation that carries no data.
    """
    u = np.asarray(phases, dtype=np.float64) / np.pi  # (0, 1)
    n, d = u.shape
    need = res.n_slots
    if d < need:
        u = np.hstack([u, np.full((n, need - d), 0.5)])
    u = u[:, :need][:, slot_order(need, res.steps, order)]
    a = u.reshape(n, res.steps, res.uploads, res.n_enc, 2).copy()
    a[..., 1] *= 2.0  # longitude spans (0, 2 pi)
    return a
