"""NF-UNSW-NB15-v2; kept as an alias of :mod:`elements.data.netflow`."""

from __future__ import annotations

from pathlib import Path

from elements.data.netflow import IDENTIFIERS, LEAKY, NetFlow, load as _load  # noqa: F401


def load(path: str | Path | None = None, dedup: bool = False) -> NetFlow:
    return _load("nf_unsw_nb15_v2", path, dedup=dedup)
