# QORC — zero-day novelty detection (`qorc_zeroday_sweep_novelty.ipynb`)

The novelty half of the two-head QORC system, swept over reservoir capacity.
Companion to `qorc_supervised_multiclass.ipynb`, built to the same engineering
standard (smoke test, crash-safe checkpointing, auto-generated summary).

This notebook **does not replace** `qorc_zeroday_analysis.ipynb`; that one is
still the record of the original per-attack, multi-seed experiment. This is a
different experiment — see "How this differs" below.

## The architecture this notebook implements

```
X (122 feat) --> scaler --> PCA(n_modes) --> tanh phase encode
                                                 |
                              precircuit -> phase dials -> reservoir     [FIXED]
                                                 |
                                          readout: C(m,p) probs
                                                 |
                        +------------------------+------------------------+
                        |                                                 |
                 NOVELTY HEAD                                   SUPERVISED HEAD
              (normal rows only)                              (attack labels)
              -- THIS NOTEBOOK --                             -- handled elsewhere --
```

The quantum stage is a direct port of the supervised notebook, generalised over
`(n_modes, n_photons)` and never trained (`trainable_parameters=[]`).

## The protocol — normal-only training

This is the load-bearing discipline of the whole notebook:

| set | rows | contents |
|---|---|---|
| train | 47,140 | 70% of `normal`. **No attacks of any kind.** |
| score-normal | 20,203 | the other 30% of `normal` — sets the false-alarm threshold |
| score-attack | 58,630 | **every** attack row, all 22 types |

Because no attack ever enters training, every attack type is simultaneously
"unseen" — there is nothing to hold out. The smoke test asserts this directly
(`training set is 100% normal`), since it is the one property whose failure
would invalidate every number in the notebook.

A practical consequence: the training set no longer depends on which attack is
being scored, so the reservoir and detectors are fitted **once per config** and
score all 22 types. The whole sweep takes ~1.2 minutes.

## How this differs from `qorc_zeroday_analysis.ipynb`

| | original | this notebook |
|---|---|---|
| training set | normal + all *other* attacks | **normal only** |
| attacks evaluated | 6 hand-picked | **all 22 in the dataset** |
| supervised arm | `SUP_raw`, `SUP_q` included | **removed** — novelty only |
| varies | 3 random seeds | **4 reservoir capacities**, 1 seed |
| reservoir | fixed 3p20m, CPU | 4 configs, GPU |
| refits | once per (attack × seed) | once per config |

The original's supervised arm answered a question the team has since moved off
(see `README_supervised.md` — a binary classifier judging an attack type it
never saw is not a meaningful zero-day test). This notebook drops it entirely.

## What the sweep varies

| config | photons | modes | readout dim | feature matrix (full data) |
|---|---|---|---|---|
| `3p12m` | 3 | 12 | 220 | 0.11 GB |
| `3p20m` | 3 | 20 | 1140 | 0.57 GB |
| `3p24m` | 3 | 24 | 2024 | 1.02 GB |
| `4p24m` | 4 | 24 | 10626 | 5.35 GB |

Same four configs as the supervised sweep, so the two are comparable config for
config. `3p20m` is the original zero-day notebook's setting.

## Detectors — all trained on normal only

| tag | input | dim | role |
|---|---|---|---|
| `NOV_q` | reservoir readout | C(m,p) | the quantum arm |
| `NOV_dials` | PCA input to the reservoir | n_modes | **the tight control** |
| `NOV_raw` | full scaled feature vector | 122 | classical reference |
| `IForest` | full scaled feature vector | 122 | standard baseline |

`NOV_dials` sees exactly what entered the reservoir, through the same
front-end, so `gain = AUC(NOV_q) − AUC(NOV_dials)` isolates the reservoir's own
contribution. Without it a high AUC says nothing about whether the quantum layer
did any work.

The detector itself is Mahalanobis distance (within the normal subspace) plus
positive reconstruction error (outside it). Its PCA, its scaling statistics, and
the detection threshold are all fitted on normal-train alone.

## Metrics — and which one to quote

* **`auc_macro_big`** — AUC averaged over attack *types* with ≥20 rows.
  **This is the headline number.**
* `auc_all` — pooled over attack rows. Dominated by `neptune` (41,214 of 58,630
  rows), so it mostly measures one attack family. Reported, not quoted.
* **`det@0.001 / 0.01 / 0.05`** — detection rate with the threshold fixed on
  normal traffic at that false-alarm budget. These are the deployable numbers;
  AUC is threshold-free and flatters detectors that would be unusable in
  practice. At 67,343 normal rows a 1% FA rate is ~670 alerts to triage.

## Running it

1. Open the notebook, **Restart & Run All**.
2. Watch for `SMOKE TEST PASSED` (25 checks, ~5 s). The sweep cell refuses to
   start otherwise.
3. The sweep runs 4 configs in about **1.2 minutes** on an RTX 2070 Max-Q.
4. Everything lands in `qorc_nov_runs/<timestamp>/`.

### Crash safety

Results are appended to `results.jsonl` the moment each config finishes,
flushed and `fsync`-ed. Rerunning the sweep cell skips what is already on disk.
To resume into a previous run directory, set `RESUME_FROM`:

```python
RESUME_FROM = Path("qorc_nov_runs/20260726_093611")
```

A failing config is logged to `errors.log` with its traceback; the sweep
continues.

### Output layout

```
qorc_nov_runs/<timestamp>/
    run.log                    every log line, fsync-ed as written
    errors.log                 tracebacks, only if something failed
    results.jsonl              one JSON object per config
    summary.txt                auto-generated conclusions + caveats
    metrics_flat.csv           one row per (config, detector)
    metrics_per_attack.csv     per-attack AUC + detection rates
    metrics_operational.csv    detection at each false-alarm budget
    artifacts/scores_*.npz     raw score arrays (normal + attack)
```

Raw scores are saved so the ROC and distribution plots can be redrawn without
recomputing the reservoir.

## Analysis (Part II)

Section 8 onwards reads only from `results.jsonl` and the score artifacts, so it
runs in a fresh kernel. Point `INSPECT_PATH` at the run you want.

- **8.1** headline table — quantum vs both controls, per config
- **8.2** does capacity help? (the core question)
- **8.3** quantum gain per config
- **8.4** per-attack AUC + gain, and heatmaps per detector
- **8.5** operational view — detection at fixed false-alarm budgets
- **8.6** score distributions, normal vs attack
- **8.7** ROC curves, per attack and per config
- **8.8** cost scaling
- **9** auto-generated summary with per-attack verdicts and caveats
- **9.1** CSV export

## Results from the included example run (`20260726_093611`)

Kept in the repo as a worked example. Headline macro AUC:

| config | NOV_q | NOV_dials | NOV_raw | IForest | gain vs dials |
|---|---|---|---|---|---|
| `3p12m` | **0.9120** | 0.8653 | 0.8985 | 0.9074 | **+0.0467** |
| `3p20m` | 0.9067 | 0.8939 | 0.8985 | 0.9074 | +0.0128 |
| `3p24m` | 0.9044 | 0.8946 | 0.8985 | 0.9074 | +0.0099 |
| `4p24m` | 0.8937 | 0.8946 | 0.8985 | 0.9074 | −0.0009 |

Three things worth noting, none of them yet established (single seed):

* **The quantum arm beats its own input in 3 of 4 configs**, and the smallest
  reservoir shows the largest gain (+0.047).
* **Capacity trends *down*, not up** (−0.018 macro AUC from 220 to 10,626 dims).
  This is the opposite of the supervised expectation and is mechanistically
  plausible: a linear classifier benefits from more dimensions, but a
  Mahalanobis-based novelty detector suffers from distance concentration as
  dimension grows. If it survives multiple seeds it is an interesting result in
  its own right — for the novelty head, **smaller is better**.
* **Isolation Forest is competitive** (0.9074 macro AUC) and much better at a
  1% false-alarm budget (0.798 vs 0.233 for `NOV_q`). The reservoir's advantage
  is in ranking, not in operating-point behaviour. Any write-up that quotes AUC
  without the det@FA columns is telling half the story.

**`back` is the honest failure.** Every PCA-based novelty detector scores it
*below chance* — `NOV_q` 0.35–0.50, `NOV_dials` 0.24, `NOV_raw` 0.26 — meaning
they systematically rank it as *more normal than normal*. 859 of its 956 rows
are plain `tcp/http/SF`, i.e. it mimics ordinary web traffic closely enough that
a normal-only linear-subspace model cannot flag it. Only Isolation Forest clears
chance (0.736), which is a point in favour of keeping a non-linear baseline in
the ensemble. Worth stating plainly rather than burying in a mean.

## Reading the results honestly

* **Single seed (42).** The reservoir unitary and the normal split are both
  random. The original zero-day notebook saw std up to 0.17 across seeds on one
  arm — larger than every effect in the table above. **The capacity trend and
  the quantum gain are both smaller than known seed noise and are therefore not
  established.** Rerun with 2–3 seeds before claiming either. The sweep loop
  already supports it: add a seed dimension to `CONFIGS` and include it in the
  `results.jsonl` key.
* **These are not real zero-days.** They are known NSL-KDD attacks the detector
  was not shown. That is the right experiment for the question, but it does not
  demonstrate performance against genuinely novel traffic.
* **Tail classes.** 9 of 22 attack types have fewer than 20 rows (`spy` has 2).
  They are reported but excluded from aggregates via `MIN_SUPPORT`; their
  individual AUCs are one row's luck.
* **Quote `auc_macro_big`, not `auc_all`.** Pooled AUC is a `neptune` detector.

## Memory notes

Same hardware constraints as the supervised notebook (RTX 2070 Max-Q, 8 GB) —
see `README_supervised.md`. Two apply here:

**Standardise in place.** At `4p24m` the three feature matrices total ~5.4 GB.
`scaler.transform(Q).astype(np.float32)` would allocate two extra full copies
and push RSS past 11 GB into swap. `apply_stats_inplace` keeps the peak at one
matrix.

**Chunked scoring.** `NoveltyDetector.score` processes 20,000 rows at a time so
a 10,626-dim score pass does not allocate a second full copy. Verified against
unchunked: **bit-identical** (max diff 0.0).

Unlike the supervised head there is no L-BFGS and no GPU-resident design matrix,
so VRAM is not a constraint here — the reservoir forward pass is the only GPU
work (12.3 s for all 125,973 rows at `4p24m`).
