"""Crash-safe, resumable sweeps over reservoir configurations.

Every experiment is appended to ``results.jsonl`` and fsync-ed the moment it
finishes. If a long run dies overnight, rerunning it skips what is already
recorded and resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from elements.experiments.reference import SupervisedConfig, run_supervised

#: The configurations the project sweeps, smallest reservoir first.
DEFAULT_CONFIGS = [(3, 12), (3, 20), (3, 24), (4, 12), (4, 20), (4, 24)]


class RunDir:
    """A timestamped output directory with a log and a results file."""

    def __init__(self, root: str | Path = "runs/reference_sweeps", run_id: str | None = None):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True, parents=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.root / self.run_id
        self.path.mkdir(exist_ok=True)
        (self.path / "artifacts").mkdir(exist_ok=True)
        self.results = self.path / "results.jsonl"
        self.log_path = self.path / "run.log"

    def log(self, msg: str, level: str = "INFO") -> None:
        """Timestamped log to stdout AND disk, flushed immediately, so a run
        that dies still leaves a complete trace."""
        line = f"{datetime.now().strftime('%H:%M:%S')} [{level:5s}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def append(self, rec: dict) -> None:
        with open(self.results, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def done_keys(self) -> set[tuple[str, str, int]]:
        """(config, scheme, seed) triples already recorded."""
        if not self.results.exists():
            return set()
        keys = set()
        with open(self.results, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    keys.add((r["config"], r["scheme"], r.get("seed", 42)))
        return keys


def load_results(path: str | Path) -> pd.DataFrame:
    """Flatten a ``results.jsonl`` into one row per (experiment, head)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            for head, m in r["heads"].items():
                rows.append(
                    {
                        "config": r["config"],
                        "scheme": r["scheme"],
                        "seed": r.get("seed", 42),
                        "head": head,
                        "readout_dim": r["readout_dim"],
                        # the width this head actually saw: the full readout
                        # for `quantum`, but only n_modes for the `dials`
                        # control -- reporting readout_dim for both would
                        # misdescribe the control.
                        "input_dim": m["input_dim"],
                        "n_photons": r["n_photons"],
                        "n_modes": r["n_modes"],
                        "n_classes": r["n_classes"],
                        "accuracy": m["accuracy"],
                        "balanced_acc": m["balanced_acc"],
                        "macro_f1": m["macro_f1"],
                        "weighted_f1": m["weighted_f1"],
                        "macro_auc": m.get("macro_auc_ovr"),
                        "fit_time_s": m["fit_time_s"],
                        "t_quantum_s": r["t_quantum_s"],
                    }
                )
    return pd.DataFrame(rows)


def sweep_supervised(
    X: np.ndarray,
    labels: dict[str, tuple[np.ndarray, list[str]]],
    configs: list[tuple[int, int]] | None = None,
    seeds: tuple[int, ...] = (42,),
    run: RunDir | None = None,
    resume: bool = True,
    **cfg_kwargs,
) -> RunDir:
    """Run every (config x scheme x seed) combination, checkpointing each one.

    ``labels`` maps a scheme name to the ``(y, class_names)`` it produces, as
    returned by ``NSLKDD.labels``.
    """
    configs = configs or DEFAULT_CONFIGS
    run = run or RunDir()
    done = run.done_keys() if resume else set()
    if done:
        run.log(f"resuming: {len(done)} experiments already recorded")

    total = len(configs) * len(labels) * len(seeds)
    run.log(f"sweep: {total} experiments ({len(configs)} configs x "
            f"{len(labels)} schemes x {len(seeds)} seeds)")

    n_done = n_fail = 0
    t_start = time.time()
    for seed in seeds:
        for n_photons, n_modes in configs:
            for scheme, (y, names) in labels.items():
                cfg = SupervisedConfig(
                    n_photons=n_photons, n_modes=n_modes,
                    scheme=scheme, seed=seed, **cfg_kwargs,
                )
                if (cfg.tag, scheme, seed) in done:
                    run.log(f"skip {cfg.tag}/{scheme}/seed{seed} (already done)")
                    continue
                try:
                    rec, confusions = run_supervised(X, y, names, cfg, log=run.log)
                    for head, cm in confusions.items():
                        np.save(
                            run.path / "artifacts"
                            / f"cm_{cfg.tag}_{scheme}_s{seed}_{head}.npy",
                            cm,
                        )
                    run.append(rec)
                    n_done += 1
                except (MemoryError, RuntimeError, ValueError) as exc:
                    n_fail += 1
                    run.log(f"FAILED {cfg.tag}/{scheme}/seed{seed}: {exc}", "ERROR")
                    run.log(traceback.format_exc(), "ERROR")

    run.log(
        f"sweep finished: {n_done} ok, {n_fail} failed, "
        f"{time.time() - t_start:.0f}s total -> {run.results}"
    )
    return run
