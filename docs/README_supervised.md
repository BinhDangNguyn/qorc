# QORC — supervised attack classification (`qorc_supervised_multiclass.ipynb`)

The supervised half of the two-head QORC system. The novelty / zero-day head is
maintained separately by another team member.

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
              -- handled elsewhere --                          -- THIS NOTEBOOK --
```

The pre-circuit, encoding and reservoir are a direct port of
`qorc_zeroday_analysis.ipynb`, generalised over `(n_modes, n_photons)`. The
quantum stage is never trained (`trainable_parameters=[]`). Only the classical
layer after the readout changes, and the two heads are trained **completely
separately** — no shared classical weights, no joint objective.

## What the sweep varies

Reservoir capacity, not random seeds:

| config | photons | modes | readout dim | feature matrix (full data) |
|---|---|---|---|---|
| `3p12m` | 3 | 12 | 220 | 0.11 GB |
| `3p20m` | 3 | 20 | 1140 | 0.57 GB |
| `3p24m` | 3 | 24 | 2024 | 1.02 GB |
| `4p24m` | 4 | 24 | 10626 | 5.35 GB |

`3p20m` is the configuration used in the original zero-day notebook, so it is
the point of comparison with existing results.

## Two label schemes, both run for every config

| scheme | classes | notes |
|---|---|---|
| `families6` | 7 | `normal` + neptune, satan, ipsweep, portsweep, smurf + `other_attack`. Every class >2,600 rows. **Quote this for headline numbers.** |
| `all23` | 23 | Every label in the file. Includes classes with 2–11 rows whose F1 is noise. Useful for fine-grained detail, not for claims. |

## The classical control

Every experiment trains **two** heads on the same split:

* `quantum` — the `C(m,p)`-dim reservoir readout
* `dials` — the `n_modes`-dim PCA vector that was fed *into* the reservoir

`dials` is a deliberately tight control: it sees exactly the information that
entered the reservoir, through the same preprocessing. Any difference is
therefore attributable to the reservoir itself rather than to the front-end.
`gain = quantum − dials` is the number that matters.

## Head implementation

Multinomial logistic regression with L2, implemented in torch and trained with
L-BFGS on the GPU. This is not a modelling choice so much as a practical one: at
10,626 dims scikit-learn's `LogisticRegression` takes many minutes per fit,
while this takes under a second. Using **one** implementation for every config
keeps the comparison honest — no config gets a different optimiser.

`class_weight="balanced"` is on by default. Under `all23` the largest class is
20,000× the smallest; without weighting the head simply never predicts the tail.

## Running it overnight

1. Open the notebook, **Restart & Run All**.
2. Watch for `SMOKE TEST PASSED` (~1 min). The sweep cell refuses to start if
   the smoke test failed, so a bad environment cannot silently waste a night.
3. The sweep runs 8 experiments (4 configs × 2 schemes). Expect roughly 20–40
   minutes total; the `4p24m` pair dominates.
4. Everything lands in `qorc_sup_runs/<timestamp>/`.

### Crash safety

Results are appended to `results.jsonl` **the moment each experiment finishes**,
flushed and `fsync`-ed. If the machine dies, rerun the sweep cell — it reads
what is already on disk and runs only what is missing.

To resume into a *previous* run directory, set `RESUME_FROM` in the sweep cell:

```python
RESUME_FROM = Path("qorc_sup_runs/20260726_013000")
```

A failing experiment is logged to `errors.log` with its traceback and the sweep
continues; one bad config cannot take down the whole night.

### Output layout

```
qorc_sup_runs/<timestamp>/
    run.log                    every log line, fsync-ed as it is written
    errors.log                 tracebacks, only if something failed
    results.jsonl              one JSON object per experiment
    summary.txt                auto-generated conclusions + caveats
    metrics_flat.csv           one row per (config, scheme, head)
    metrics_per_class.csv      per-class precision/recall/F1/support
    metrics_operational.csv    false-alarm / miss / attack-type accuracy
    artifacts/cm_*.npy         confusion matrices
```

## Analysis (Part II)

Section 8 onwards reads only from `results.jsonl`, so it runs in a **fresh
kernel** the next morning without recomputing anything. Point `INSPECT_PATH` at
the run you want and execute from section 8.

- **8.1** headline table, quantum vs control per config
- **8.2** does more reservoir capacity help? (the core question)
- **8.3** quantum gain per config
- **8.4** per-class F1 + gain heatmap — which attack families benefit
- **8.5** confusion matrices
- **8.6** operational view: false-alarm rate, miss rate, attack-type accuracy
- **8.7** cost — accuracy bought per second of reservoir time
- **8.8** auto-generated summary with caveats
- **8.9** CSV export for the write-up

## Reading the results honestly

* **Single seed.** Each config runs at `SEED=42` only. The reservoir unitary is
  random, and the original zero-day notebook saw std up to 0.17 across seeds on
  one arm. A capacity trend smaller than that is **not established**. If the
  sweep shows an interesting trend, rerun with 2–3 seeds per config before
  claiming it — that is the natural follow-up, and the sweep loop already
  supports it (add a seed dimension to `CONFIGS`).
* **Closed-set.** This is classification over *known* attack families. It says
  nothing about zero-day performance — that is the novelty head's job, by
  design. Do not quote these numbers as zero-day results.
* **`all23` tail.** Classes with 2–11 rows produce meaningless F1. They are
  reported for completeness and excluded from the per-class plots via a support
  threshold.
* **Higher readout dim is not automatically better.** 10,626 features on ~88k
  training rows is a regime where a linear head can overfit; the `dials` control
  and the balanced-accuracy column are what keep that visible.

## Memory notes

Both of these were found by actually running `4p24m` on the full dataset, and
both will silently hang an overnight run if reintroduced.

**System RAM — standardise in place.** At `4p24m` the train+test feature
matrices are ~5.4 GB. The notebook standardises them **in place**; the obvious
`scaler.transform(Q).astype(np.float32)` allocates two extra full copies and
pushes RSS past 11 GB, which sends the machine into swap. Measured: 11.25 GB
naive vs 5.39 GB in place, numerically identical to `StandardScaler` to 3.6e-07.

**VRAM — the head always streams.** The 88,181 × 10,626 training matrix is
3.75 GB. Holding it on an 8 GB card alongside the L-BFGS history exhausts VRAM
(observed: 7,880 / 8,192 MiB, then a stall). `GPUSoftmaxHead` therefore leaves
the design matrix in system RAM and accumulates the gradient over `chunk`-sized
slices. Measured at `4p24m`: **0.72 GB peak VRAM to fit, 1.41 GB to predict.**

The chunked gradient was verified against the full-batch one at fixed weights:
relative error **3.2e-07**, and bit-identical at `chunk=32768`. It is the same
objective, correctly computed.

`GPU_RESIDENT_GB` defaults to `0.0`, meaning *always chunk* — including the
small configs that would fit on the GPU. That is deliberate. L-BFGS is
path-sensitive: a 1e-7 gradient difference is enough to stop at a slightly
different point within `max_iter`, which showed up as a ~4% loss difference
between the two paths on a 600-dim problem. Since the whole purpose of this
notebook is comparing four reservoir sizes, every config must go through an
identical optimiser. Raise the threshold only if you want speed over
comparability.

**A note on `final_loss`.** It is re-evaluated after fitting rather than taken
from `LBFGS.step()`, which returns the loss from the *first* closure call — at
`W=0`, i.e. always `ln(n_classes)`. If you see `final_loss ≈ 1.9459` for a
7-class run, that is the bug returning, not a converged model.

`MAX_FEATURE_GB = 6.0` guards against adding a config that would not fit; raise
it deliberately, not by accident.
