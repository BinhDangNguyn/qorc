"""End-to-end runner on a small NSL-KDD subsample: both tasks, resume, protocol."""

import numpy as np
import pytest

from elements import config as C
from elements.data import nslkdd
from elements.eval.results import load_results
from elements.experiments.runner import plan, run
from elements.experiments.views import load_novelty_task

pytestmark = pytest.mark.skipif(not any(p.exists() for p in nslkdd.CSV_CANDIDATES),
                                reason="NSL-KDD csv not present")

STEP = {"reservoir": {"photons": 2, "modes": 8}}


def _supervised(tmp_path):
    return C.from_dict({
        "name": "sup", "task": "supervised", "out": str(tmp_path),
        "data": {"max_rows": 3000, "schemes": ["families6"]},
        "arms": ["quantum", "dials", "raw"],
        "baseline_heads": [{"kind": "lbfgs", "max_iter": 30}],
        "ladder": [{**STEP, "head": {"kind": "mlp", "hidden": [16], "max_epochs": 3}}],
        "device": "cpu",
    })


def test_supervised_run_records_every_head_and_resumes(tmp_path):
    cfg = _supervised(tmp_path)
    assert "fits" in plan(cfg)
    run(cfg)
    df = load_results(cfg.out_dir)
    # 3 arms x 2 heads x 1 scheme
    assert len(df) == 6
    assert set(df["head_role"]) == {"scaled", "baseline"}
    assert df["macro_f1"].between(0, 1).all()
    assert {"false_alarm_rate", "miss_rate"} <= set(df.columns)

    n = len(df)
    run(cfg)  # everything is on disk: nothing is retrained
    assert len(load_results(cfg.out_dir)) == n


def test_novelty_protocol_trains_on_normal_only(tmp_path):
    cfg = C.from_dict({
        "name": "nov", "task": "novelty", "out": str(tmp_path),
        "data": {"max_rows": 4000}, "arms": ["quantum", "raw"],
        "baseline_heads": [{"kind": "iforest", "arms": ["raw"]}],
        "ladder": [{**STEP, "head": {"kind": "pca", "n_components": 4}}],
        "split": {"val_frac": 0.0}, "device": "cpu",
    })
    task = load_novelty_task(cfg.data, cfg.split, cfg.seed)
    y = task.labels["novelty"][2]
    assert (y == "normal").any() and (y != "normal").any()
    assert len(task.X_val) == 0

    run(cfg)
    df = load_results(cfg.out_dir)
    assert set(zip(df.arm, df["head"])) == {("quantum", "pca4"), ("raw", "pca4"),
                                             ("raw", "iforest")}
    assert np.isfinite(df["macro_auc"]).all()
