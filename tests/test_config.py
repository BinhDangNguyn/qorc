"""The experiment config: schedules resolve to the right steps, mistakes are caught."""

from pathlib import Path

import pytest

from elements import config as C

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(**over):
    base = {"name": "t", "task": "supervised",
            "ladder": [{"reservoir": {"photons": 3, "modes": 12},
                        "head": {"kind": "mlp", "hidden": [128]}}]}
    base.update(over)
    return C.from_dict(base)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_shipped_config_loads(path):
    cfg = C.load(path)
    assert cfg.all_heads(), "a config must train at least one head"


def test_ladder_grows_both_sides():
    cfg = C.load(CONFIGS / "supervised_ladder.yaml")
    steps = cfg.steps()
    dims = [s.reservoir.dim for s in steps]
    widths = [next(h for h in s.heads if s.role[h.label] == "scaled").hidden[0] for s in steps]
    assert dims == sorted(dims) and widths == sorted(widths)
    assert all("lbfgs" in s.role and s.role["lbfgs"] == "baseline" for s in steps)


def test_rule_derives_head_from_readout():
    cfg = C.load(CONFIGS / "supervised_rule.yaml")
    got = {s.reservoir.tag: s.heads[0].hidden for s in cfg.steps()}
    assert got["3p12m"] == [128]      # 220 / 4 = 55 -> clipped up to 128
    assert got["3p24m"] == [512]      # 2024 / 4 = 506 -> 512
    assert got["4p24m"] == [2048]     # 10626 / 4 -> clipped down to 2048


def test_head_labels_and_params():
    h = C.HeadSpec(kind="mlp", hidden=[512, 256])
    assert h.label == "mlp512x256"
    assert h.n_params(10, 3) == 10 * 512 + 512 + 512 * 256 + 256 + 256 * 3 + 3
    assert C.HeadSpec(kind="ae", hidden=[64], bottleneck=8).label == "ae64-b8"
    assert C.HeadSpec(kind="lbfgs", name="orig").label == "orig"


def test_reservoir_dims():
    assert C.ReservoirSpec(3, 12).dim == 220
    assert C.ReservoirSpec(4, 24).dim == 10626
    assert C.ReservoirSpec(2, 6, no_bunching=False).dim == 21


@pytest.mark.parametrize("bad, match", [
    ({"task": "clustering"}, "task"),
    ({"arms": ["quantum", "psychic"]}, "arm"),
    ({"grid": {"reservoirs": [], "heads": []}}, "exactly one"),
    ({"ladder": [{"reservoir": {"photons": 3, "modes": 12},
                  "head": {"kind": "ae", "hidden": [8]}}]}, "novelty head"),
    ({"ladder": [{"reservoir": {"photons": 3, "modes": 12},
                  "head": {"kind": "mlp", "hiden": [8]}}]}, "unknown field"),
    ({"ladder": [{"reservoir": {"photons": 13, "modes": 12},
                  "head": {"kind": "mlp"}}]}, "bunching"),
    ({"split": {"val_frac": 0.0}}, "early-stops"),
    ({"ladder": None, "models": [{"kind": "rf"}]}, r"arms: \[raw\]"),
])
def test_mistakes_are_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**bad)


def test_dotted_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text((CONFIGS / "novelty_ladder.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    cfg = C.load(p, {"seed": 7, "data.max_rows": 500})
    assert cfg.seed == 7 and cfg.data.max_rows == 500


def test_models_schedule_has_no_reservoir():
    cfg = C.load(CONFIGS / "classical_supervised.yaml")
    assert cfg.steps() == []
    labels = [h.label for h in cfg.all_heads()]
    assert {"rf300", "xgb300", "logreg", "mlp512x512"} <= set(labels)
    assert all(h.n_params(121, 7) == 0 for h in cfg.model_heads() if h.kind != "mlp")
