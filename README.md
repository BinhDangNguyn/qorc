# QORC — quantum optical reservoir computing for network intrusion detection

A fixed photonic reservoir (boson sampling in a random interferometer, simulated
with MerLin / Perceval) turns each input into Fock-state probabilities; a
classical head reads them out. The reservoir never learns. This repo studies
when that helps, on NSL-KDD attack classification, zero-day (novelty)
detection, and image / forecasting benchmarks.

```
X --> scale --> PCA(n_modes) --> tanh phases --> [ U --> phase dials --> U ] --> Fock probs --> head
                                                    fixed, never trained         C(m, p) dims   trained
```

## Layout

```
elements/          all code (import it; notebooks hold no logic)
  config.py        experiment configs: reservoir, head, scaling schedule
  run.py           CLI: python -m elements.run <config> [--plan] [--set k=v]
  core/            reservoir, phase encoding, temporal wrappers
  heads/           lbfgs / linear / MLP classifiers, PCA / autoencoder detectors, factory
  data/            NSL-KDD, MNIST family + MedMNIST, time series
  experiments/     runner (supervised + novelty), feature views, images, temporal,
                   reference.py (the historical protocol, pinned by a regression test)
  eval/            metrics, resumable results store
  viz/             every figure and table the notebooks show
configs/           one YAML per experiment
notebooks/         thin: config + plan + run switch + figures
runs/              results.jsonl + run.log + config.yaml per experiment (figures gitignored)
docs/              findings, protocol notes, literature
reference/QORC/    reproduction code of the QORC paper (external)
_archive/          superseded scripts and notebooks, kept verbatim
data/              datasets (not committed; see data/README.md)
tests/
```

## Setup

Uses the existing conda env (torch + CUDA, perceval, merlin):

```bash
conda activate py3.10_pytorch
pip install -e . --no-deps      # optional: notebooks also work without it
pytest -m "not slow"            # ~15 s
pytest -m slow                  # full-data regression against the historical run
```

## Running an experiment

Every experiment is a config. Always look at the plan first:

```bash
python -m elements.run configs/supervised_ladder.yaml --plan
python -m elements.run configs/supervised_ladder.yaml
```

`--plan` prints each step (reservoir, readout width, heads, parameter counts),
how many fits are already on disk, peak memory and a rough time estimate, and
exits. Runs are resumable: every trained head is appended to
`runs/<name>/results.jsonl` the moment it finishes.

Override anything without editing the file:

```bash
python -m elements.run configs/novelty_ladder.yaml --set seed=7 --set data.max_rows=5000
```

### Configs

| config | what it asks |
|---|---|
| `supervised_reference.yaml` | the original design: reservoir grows, head stays linear |
| `supervised_ladder.yaml` | reservoir and MLP head grow together, step by step |
| `supervised_rule.yaml` | same, with the head derived from the readout width by a rule |
| `supervised_grid.yaml` | every reservoir x every head, all controls and hybrids |
| `novelty_reference.yaml` | original zero-day design: linear PCA detector at every size |
| `novelty_ladder.yaml` | reservoir and autoencoder detector grow together |
| `kmnist_modes.yaml` | more modes with few photons, on K-MNIST |

### Writing a config

```yaml
name: my_run
task: supervised            # or novelty
data: {dataset: nslkdd, path: D:/.../NSL_KDD_labeled.csv, schemes: [families6, all23]}
arms: [quantum, dials, rff, raw]      # also qdials, rffdials
baseline_heads: [{kind: lbfgs}]       # trained at every step as well
ladder:                               # or grid: {reservoirs, heads} / rule: {reservoirs, head}
  - reservoir: {photons: 3, modes: 12}
    head: {kind: mlp, hidden: [128]}
  - reservoir: {photons: 4, modes: 24, no_bunching: true}
    head: {kind: mlp, hidden: [1024, 256], activation: relu, dropout: 0.2}
```

Heads: `lbfgs`, `linear`, `mlp` (supervised); `pca`, `ae`, `iforest` (novelty).
Every field and its default is in `elements/config.py`; unknown fields are an
error, so a typo cannot silently fall back to a default.

## Notebooks

| notebook | |
|---|---|
| `01_data_overview` | class balance; variance kept per mode count |
| `10_supervised_multiclass` | original supervised design (linear head) |
| `11_supervised_scaled_head` | reservoir + head scaled together |
| `20_zeroday_novelty` | original zero-day design (linear detector) |
| `21_zeroday_novelty_scaled` | reservoir + detector scaled together |
| `30_joint_scaling_report` | the full reservoir x head grid, NSL-KDD and K-MNIST |
| `40_images_and_forecasting` | images, image anomalies, forecasting; the PCA rule |

Each run notebook has `RUN = False`; flip it (or use the CLI) to train.

## Honesty rules this code enforces

* Every statistic (scaler, PCA, phase encoder, standardisation) is fitted on
  train rows only; test rows never reach fitting, early stopping or selection.
* Novelty trains on normal traffic only -- asserted in code.
* The reservoir gets credit only for beating its own `dials` **under the same
  head**, and `rff` checks it against a classical lift of equal width.
* Results so far and their caveats: `docs/findings_joint_scaling.md`.
