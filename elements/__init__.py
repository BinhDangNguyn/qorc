"""Quantum Reservoir Computing -- a photonic reservoir with pluggable heads.

The reservoir is a fixed random feature map; only the head learns. That split
is what lets one reservoir serve supervised classification, novelty detection
and other domains without ever retraining the quantum part.
"""

__version__ = "0.1.0"

from elements.core.encoding import PhaseEncoder, standardize_inplace
from elements.core.reservoir import Reservoir, readout_dim

__all__ = [
    "Reservoir",
    "readout_dim",
    "PhaseEncoder",
    "standardize_inplace",
    "__version__",
]
