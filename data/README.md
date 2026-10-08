# Data

Nothing in this folder is committed (see `.gitignore`); only this file is.

## NSL-KDD (`NSL_KDD_labeled.csv`, 14 MB)

125,973 rows, 41 raw columns + `label` (23 distinct labels: `normal` + 22 attack
types). `elements.data.nslkdd` one-hot encodes `protocol_type`, `service` and
`flag`, giving 122 features.

The loader looks for it, in order:

1. the `path` field of the experiment config (`data.path`)
2. the `QORC_NSLKDD` environment variable
3. `data/NSL_KDD_labeled.csv` in this repo
4. `D:/Dev/ai-ml/reservoir-computing/QRC/data/NSL_KDD_labeled.csv` (the team copy)

The shipped configs point at (4) explicitly. To make the repo self-contained,
copy the file here.

## Other datasets

| data | where it lives | used by |
|---|---|---|
| MNIST / Fashion-MNIST / K-MNIST / MedMNIST | downloaded on first use to `~/.keras/datasets/qrcpkg/` | `elements.data.images`, `configs/kmnist_modes.yaml` |
| zero-day held-out-attack splits (70 MB) | `reservoir-computing/artifacts/zeroday/` (built by `_archive/scripts/build_zeroday_splits.py`) | only the archived Gap-1 study; the current novelty protocol trains on normal traffic only and needs no splits |
| UNSW-NB15 (605 MB), network_behaviour (73 MB) | `reservoir-computing/QRC/data/` | not used by any current experiment |
