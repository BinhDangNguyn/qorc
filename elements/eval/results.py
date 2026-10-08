"""Append-only, fsync-ed results store with resume keys, plus a run log.

Every trained head appends one JSON line the moment it finishes. Rerunning an
experiment skips every (dataset, scheme, seed, feature, head) already on disk,
so an interrupted overnight run resumes where it stopped.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


class Results:
    def __init__(self, out: str | Path):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "results.jsonl"
        self.log_path = self.out / "run.log"
        self.done: set = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.done.add(self.key(**json.loads(line)))

    @staticmethod
    def key(dataset, scheme, seed, feature_id, head, **_):
        return (dataset, scheme, int(seed), feature_id, head)

    def append(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.done.add(self.key(**rec))

    def log(self, msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def load_results(*paths: str | Path, keep: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read one or more run dirs (or results.jsonl files) into one frame.

    Nested per-class / per-type dicts are dropped unless named in ``keep``.
    Duplicate (dataset, scheme, seed, feature, head) rows keep the first.
    """
    rows = []
    for p in paths:
        p = Path(p)
        f = p / "results.jsonl" if p.is_dir() or not p.suffix else p
        if not f.exists():
            raise FileNotFoundError(f"no results at {f} -- run the experiment first "
                                    "(RUN = True in the notebook, or python -m elements.run)")
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    drop = [c for c in ("per_class_f1", "per_type", "per_class") if c in df and c not in keep]
    df = df.drop(columns=drop)
    return df.drop_duplicates(["dataset", "scheme", "seed", "feature_id", "head"])
