# Archive

Copied verbatim from `D:\Dev\ai-ml\reservoir-computing` (root and `QRC/team/`)
when this repo was created. Nothing here is imported by `elements`; the
originals are still in place in the old folder and in the team's QRC repo.

| here | what it was | superseded by |
|---|---|---|
| `scripts/qorc_nslkdd*.py`, `qorc_zeroday_v2.py`, `gap1_zeroday_qrc.py` | standalone experiment scripts, each with its own copy of the reservoir | `elements.experiments.runner` + `configs/` |
| `scripts/qorc_ae_nslkdd_*.py`, `qorc_quandela_run.py`, `quandela_run_simple.py` | autoencoder front-end variant; Quandela cloud / hardware runs | not ported yet -- the hardware path is the next thing worth porting |
| `scripts/classical_baselines.py` (+ `.json`) | RF / XGB / LogReg / SVM / IsolationForest baselines (script, hard-coded paths) | results still valid; IsolationForest is now a head kind |
| `scripts/build_zeroday_splits.py` | held-out-attack splits for the Gap-1 study | normal-only novelty protocol (no splits needed) |
| `scripts/qrcpkg_results.py`, `joint_scaling_report.py` | plotting helpers | `elements.viz` |
| `notebooks/01_data_investigation`, `02_data_encoding` | data exploration and encoding study | `notebooks/01_data_overview` |
| `notebooks/qrcpkg_results` | image / forecasting results | `notebooks/40_images_and_forecasting` |
| `notebooks/reservoirpy_tutorial` | classical reservoir-computing tutorial | -- |
| `team_notebooks/qorc_supervised_multiclass` | original supervised sweep | `notebooks/10_*`, `11_*` |
| `team_notebooks/qorc_zeroday_sweep_novelty`, `qorc_zeroday_analysis` | original zero-day sweeps | `notebooks/20_*`, `21_*` |
| `team_notebooks/qorc_pitch_figures`, `business_market`, `quantum_investment`, `dataset_overview`, `multiple_config_graph` | pitch deck, market analysis, early dataset look | not ported (presentation material) |
| `figures/` | pitch and market figures | -- |
| `circuit_spec.json`, `encoding_pipeline.joblib` | artefacts of the encoding study | -- |
