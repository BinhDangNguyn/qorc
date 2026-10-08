"""The fixed photonic reservoir.

One implementation, replacing the copies previously carried by
``qorc_zeroday_v2.py``, ``qorc_supervised_multiclass.ipynb`` and the other
experiment scripts. The circuit structure is unchanged from those:

    precircuit (Haar-random unitary) -> phase shifters (the data) -> reservoir

where ``reservoir`` is a copy of ``precircuit``, following the QORC paper.
Nothing here is trainable: the reservoir is a fixed random feature map and
only the head that reads it out ever learns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import perceval as pcvl
import torch

import merlin as ML

try:
    MeasurementStrategy = ML.MeasurementStrategy
    ComputationSpace = ML.ComputationSpace
except AttributeError:  # older merlin layouts
    from merlin.measurement.strategies import (  # type: ignore[no-redef]
        ComputationSpace,
        MeasurementStrategy,
    )


def readout_dim(n_modes: int, n_photons: int, no_bunching: bool = True) -> int:
    """Number of Fock states the reservoir reports for this configuration.

    Unbunched (at most one photon per mode) gives C(m, n); the full Fock space
    additionally counts states where photons share a mode.
    """
    if no_bunching:
        return math.comb(n_modes, n_photons)
    return math.comb(n_photons + n_modes - 1, n_photons)


def input_state(n_modes: int, n_photons: int) -> list[int]:
    """Spread ``n_photons`` as evenly as possible across ``n_modes``."""
    if n_photons > n_modes:
        raise ValueError(
            f"cannot place {n_photons} photons in {n_modes} modes without bunching"
        )
    step = (n_modes - 1) / (n_photons - 1) if n_photons > 1 else 0
    state = [0] * n_modes
    for k in range(n_photons):
        state[int(round(k * step))] = 1
    return state


@dataclass
class Reservoir:
    """A fixed photonic feature map: phases in, Fock-state probabilities out.

    The unitary is drawn from ``seed``, so two Reservoirs built with the same
    (n_modes, n_photons, seed) are identical -- the sweeps depend on this.
    """

    n_modes: int
    n_photons: int
    seed: int = 42
    no_bunching: bool = True
    device: torch.device | str = "cpu"
    dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.dim = readout_dim(self.n_modes, self.n_photons, self.no_bunching)
        self.input_state = input_state(self.n_modes, self.n_photons)
        self.layer = self._build()

    def _build(self):
        # perceval's random_unitary draws from the global numpy RNG, so the
        # seed has to be set here rather than passed in.
        np.random.seed(self.seed)
        precircuit = pcvl.Unitary(pcvl.Matrix.random_unitary(self.n_modes))
        reservoir = precircuit.copy()

        encoder = pcvl.Circuit(self.n_modes)
        for i in range(self.n_modes):
            encoder.add(i, pcvl.PS(pcvl.P(f"px{i + 1}")))

        space = ComputationSpace.UNBUNCHED if self.no_bunching else ComputationSpace.FOCK
        layer = ML.QuantumLayer(
            input_size=self.n_modes,
            circuit=precircuit // encoder // reservoir,
            trainable_parameters=[],  # the reservoir never learns
            input_parameters=["px"],
            input_state=self.input_state,
            measurement_strategy=MeasurementStrategy.probs(computation_space=space),
            device=self.device,
        )
        layer.eval()

        n_params = sum(1 for _ in layer.parameters())
        if n_params:
            raise AssertionError(f"reservoir must be untrained, got {n_params} params")
        return layer

    def transform(self, phases: np.ndarray, batch: int = 4096) -> np.ndarray:
        """Push encoded phases through the reservoir, batched, without grad."""
        outs = []
        with torch.no_grad():
            for i in range(0, len(phases), batch):
                xb = torch.as_tensor(
                    phases[i : i + batch], dtype=torch.float32, device=self.device
                )
                outs.append(self.layer(xb).cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)

    @property
    def tag(self) -> str:
        """Short config label, e.g. ``3p12m`` -- matches the run directories."""
        return f"{self.n_photons}p{self.n_modes}m"

    def __repr__(self) -> str:
        return (
            f"Reservoir({self.tag}, dim={self.dim}, seed={self.seed}, "
            f"device={self.device})"
        )
