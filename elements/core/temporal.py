"""Three ways to give the photonic reservoir a sense of time.

The reservoir itself is memoryless: measurement collapses the state and the
photons leave the system, so ``p_t = f(x_t)`` with no dependence on ``x_{t-1}``.
Anything temporal has to be added around it. Three strategies, in increasing
order of how much memory they actually create:

``DelayEmbedding``
    No memory at all -- stack a sliding window ``[x_t, ..., x_{t-k}]`` into one
    input vector. Fully parallel, but the horizon is hard-capped at the window.

``FeedbackReservoir``
    Real recurrence, closed through classical measurement: the previous
    readout is mixed back into the phases at step t, so
    ``p_t = f(x_t, p_{t-1})``. Must run sequentially over time.

``QuantumMemory``
    Not implementable on this stack -- see the class docstring. Kept as an
    explicit record of what would be required, rather than a silent omission.
"""

from __future__ import annotations

import numpy as np

from elements.core.encoding import PhaseEncoder
from elements.core.reservoir import Reservoir


def sliding_window(u: np.ndarray, width: int) -> np.ndarray:
    """Stack ``width`` consecutive steps into each row.

    Row t holds ``[u_t, u_{t-1}, ..., u_{t-width+1}]``. The first rows are
    left-padded with the earliest value so the output keeps the input length.
    """
    u = np.asarray(u, dtype=np.float32)
    if u.ndim == 1:
        u = u[:, None]
    n, d = u.shape
    padded = np.vstack([np.repeat(u[:1], width - 1, axis=0), u])
    cols = [padded[i : i + n] for i in range(width - 1, -1, -1)]
    return np.stack(cols, axis=1).reshape(n, width * d)


class DelayEmbedding:
    """Strategy 1: widen the input instead of adding state.

    The reservoir stays exactly as it is; only what is fed to it changes. This
    is the cheap option -- it parallelises over the whole series -- but it can
    only see ``window`` steps back, by construction.
    """

    def __init__(
        self,
        reservoir: Reservoir,
        window: int = 10,
        encoding: str = "tanh",
        reducer=None,
    ) -> None:
        self.reservoir = reservoir
        self.window = window
        self.encoder = PhaseEncoder(encoding)
        self.reducer = reducer  # e.g. a PCA when window*d > n_modes

    def _reduce(self, W: np.ndarray, fit: bool) -> np.ndarray:
        if self.reducer is None:
            return W
        return self.reducer.fit_transform(W) if fit else self.reducer.transform(W)

    def fit_transform(self, u: np.ndarray) -> np.ndarray:
        Z = self._reduce(sliding_window(u, self.window), fit=True)
        return self.reservoir.transform(self.encoder.fit_transform(Z))

    def transform(self, u: np.ndarray) -> np.ndarray:
        Z = self._reduce(sliding_window(u, self.window), fit=False)
        return self.reservoir.transform(self.encoder.transform(Z))


class FeedbackReservoir:
    """Strategy 2: close the loop through the classical measurement result.

    At each step the phases are driven by the current input *and* by a
    projection of the previous Fock-state readout::

        phase_t = squash(W_in . x_t  +  leak * W_fb . p_{t-1})
        p_t     = reservoir(phase_t)

    This is genuine recurrence -- ``p_t`` depends on the whole history -- and it
    is what most work labelled "quantum reservoir computing" actually does. The
    memory lives in the classical feedback path, not in the quantum state: the
    photons are still measured and lost at every step.

    Cost: the loop is sequential, so this is n_steps reservoir calls instead of
    one batched call.

    Measured limitation -- the memory this creates is SHORT. A perturbation
    decays by roughly 30x per step and is gone within 2-3 steps, giving a
    linear memory capacity of only ~2 against ~12 for a comparable ESN. The
    cause is structural, not a tuning failure: the readout is a probability
    distribution over C(m,p) Fock states, so it sums to 1 and each entry is
    O(1/dim) with a still smaller variance. The feedback signal is therefore
    tiny next to the input drive. Rescaling it to O(1) does not help -- it
    pushes the reservoir into a chaotic regime and memory capacity falls
    further (measured: MC 1.45 -> 0.13 as the rescaled leak goes 0.3 -> 1.0).
    Raising ``leak`` alone saturates at MC ~2 for the same reason.

    Treat this arm as "short-memory", suitable where 2-3 steps of history are
    enough, and prefer ``DelayEmbedding`` when a longer horizon is needed.

    Parameters
    ----------
    leak:
        How strongly the previous readout drives the next step. 0 reduces this
        to the memoryless reservoir; large values saturate it.
    fb_dim:
        Width of the random feedback projection. The readout is C(m,p)-wide,
        far wider than the n_modes phases it has to drive.
    """

    def __init__(
        self,
        reservoir: Reservoir,
        leak: float = 0.3,
        input_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.reservoir = reservoir
        self.leak = leak
        self.input_scale = input_scale
        self.seed = seed
        self._built = False

    def _build(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        m = self.reservoir.n_modes
        # Fixed random maps, exactly as untrained as the reservoir itself.
        self.W_in = rng.standard_normal((input_dim, m)).astype(np.float32)
        self.W_fb = rng.standard_normal((self.reservoir.dim, m)).astype(np.float32)
        self.W_fb /= np.sqrt(self.reservoir.dim)  # keep the feedback O(1)
        self._built = True

    def run(self, u: np.ndarray, report=None) -> np.ndarray:
        """Run the series through the loop, one readout per step."""
        u = np.asarray(u, dtype=np.float32)
        if u.ndim == 1:
            u = u[:, None]
        if not self._built:
            self._build(u.shape[1])

        n = len(u)
        drive = u @ self.W_in * self.input_scale  # (n, n_modes), precomputed
        out = np.empty((n, self.reservoir.dim), dtype=np.float32)
        p_prev = np.zeros(self.reservoir.dim, dtype=np.float32)

        for t in range(n):
            pre = drive[t] + self.leak * (p_prev @ self.W_fb)
            # tanh squash into (0, pi): same reasoning as PhaseEncoder, applied
            # per step since there is no full-series pass to fit statistics on.
            phase = ((np.pi / 2) * (1 + np.tanh(pre))).astype(np.float32)
            p = self.reservoir.transform(phase[None, :])[0]
            out[t] = p
            p_prev = p
            if report is not None and (t + 1) % 500 == 0:
                report(t + 1, n)
        return out


class QuantumMemory:
    """Strategy 3: keep the quantum state coherent across time steps.

    NOT IMPLEMENTED, and not implementable on this stack. Recorded here so the
    gap stays explicit rather than silently missing from the comparison.

    Real quantum memory means carrying the state itself from t to t+1, which
    requires all of:

    1. No projective measurement each step. ``MeasurementStrategy.probs()``
       collapses the state and discards every amplitude and phase -- the very
       thing that would carry information forward.
    2. Photons that survive the step. After detection they are absorbed; the
       next step reloads a fresh ``input_state``, so nothing persists.
    3. A circuit mapping state -> state rather than state -> probabilities,
       plus ancilla modes to hold the retained subsystem.

    MerLin's ``QuantumLayer`` exposes measurement outcomes, not the output
    state, so (1) and (3) are outside what this API can express. Genuine
    photonic memory also needs delay lines or cavities whose coherence time
    exceeds the step interval -- a hardware property, not a code change.

    The practical consequence: ``FeedbackReservoir`` is the closest achievable
    approximation, and its memory is classical.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Quantum memory needs state-preserving evolution across steps: no "
            "projective measurement, retained photons, and a state->state "
            "circuit with ancilla modes. MerLin's QuantumLayer returns "
            "measurement probabilities, so the state cannot be carried "
            "forward. See the class docstring; use FeedbackReservoir instead."
        )
