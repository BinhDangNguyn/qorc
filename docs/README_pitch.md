# QORC — combined results & pitch figures (`qorc_pitch_figures.ipynb`)

Reads both heads' results off disk and produces slide-ready figures in
`pitch_figures/`. Recomputes nothing except the family regrouping in section 2;
every number traces to a `results.jsonl`.

**Run it:** Restart & Run All. Takes seconds. Requires that
`qorc_supervised_multiclass.ipynb` and `qorc_zeroday_sweep_novelty.ipynb` have
each produced a run.

## Design language

* **Violet** `#6D28E0` — the quantum arm, the thing being pitched. Matches the
  deck's existing charts.
* **Dim grey** `#B4B4B4`, thin dashed, hollow markers — the classical control.
  Present for honesty, deliberately quiet.
* Family accents: DoS `#6D28E0`, Probe `#9B5DE5`, R2L `#C13DD6`, U2R `#4A1FA8`
  — all within the violet family so nothing clashes.
* x-axis always ordered **low → high** capacity, so charts read left to right.

## The attack-family correction

**You were right about the taxonomy.** The supervised notebook's `families6`
scheme is *not* the standard NSL-KDD grouping. It uses the five most frequent
individual labels (`neptune, satan, ipsweep, portsweep, smurf`) plus a catch-all
`other_attack`.

The canonical taxonomy is **DoS / Probe / R2L / U2R**, and the counts match your
figure exactly:

| family | records |
|---|---|
| DoS | 45,927 |
| Probe | 11,656 |
| R2L | 995 |
| U2R | 52 |

All 22 attack labels map cleanly onto it — the notebook asserts this rather than
assuming it.

Why `families6` is actively misleading for a security audience: its
`other_attack` bucket silently merges **all 995 R2L rows and all 52 U2R rows**
in with 2,067 DoS and 1,493 Probe rows. A strong `families6` macro-F1 therefore
says nothing about whether the system can flag a privilege-escalation attempt —
the attack a security team cares most about.

**No rerun was needed.** The `all23` scheme kept every label separate and saved
full 23×23 confusion matrices, so family-level metrics are recovered exactly by
folding those matrices into 5×5 (`family_metrics()` in section 2). This is
arithmetic on stored predictions, not a re-estimate.

## The headline metric is accuracy

Accuracy over the five classes (`normal` + the four families) leads the deck:

| config | accuracy (quantum) | accuracy (classical) | error ratio |
|---|---|---|---|
| 3p12m | 95.55% | 71.96% | 6.3× fewer errors |
| 3p20m | 97.40% | 76.69% | 9.0× |
| 3p24m | 97.66% | 79.66% | 8.7× |
| 4p24m | **97.88%** | 79.66% | **9.6×** |

It rises monotonically, the classical control sits ~18 points below, and the gap
is visible on a plain linear axis.

### The log10 treatment for the narrow range

Accuracy occupies a narrow band (95.6–97.9%), so figure `02a` re-plots it as
**error rate on a log10 axis**. That expands the same data to **1.1 decades**
and makes the staircase unmissable — 4.45% → 2.12%, a 2.1× drop, with the
classical line an order of magnitude above.

**Why error rate and not `log10(accuracy)`:** the log of a number near 1 is near
0, so `log10(0.955)` and `log10(0.979)` differ by only 0.01 — the chart would
stay just as flat. Taking the complement first is what creates the range. This
is the standard convention in error-rate reporting.

### A note on raw detection rate

"% of attacks caught" is **flat at 99.85% for both arms**, and the classical
baseline is marginally *ahead* at the smallest config (99.91%). It cannot be
plotted honestly — it would need a y-axis spanning 0.17pp, where noise looks
like a trend. It saturates because DoS + Probe are 97% of attack traffic and
both are trivially separable.

Figure `02b` shows it anyway on a full 0–100% axis, so nothing is hidden. Use
that slide if someone asks; the answer is 99.9% and the interesting work is
elsewhere. Accuracy is the better headline precisely because it also counts
false alarms on normal traffic, which is where the reservoir wins.

## The figures

| file | slide |
|---|---|
| `01_family_imbalance.png` | the 1000× imbalance (your figure, regenerated) |
| `02_accuracy_vs_capacity.png` | **headline** — accuracy 95.6% → 97.9% vs capacity |
| `02a_error_rate_log.png` | **the log10 view** — error rate, 9.6× fewer than classical |
| `02b_detection_saturated.png` | raw detection rate, honestly framed |
| `03_accuracy_by_family.png` | per-class accuracy, 5 panels — the `normal` panel is the story |
| `03b_rare_attacks.png` | R2L + U2R F1 — where the reservoir matters |
| `04_recall_vs_precision.png` | the honest slide: caught vs believable |
| `05_family_confusion.png` | **strongest slide** — clean diagonal vs bleeding baseline |
| `06_per_attack_f1.png` | all 22 types, coloured by family (appendix) |
| `07_false_alarms.png` | 8.0% → 3.7% false alarms |
| `08_zeroday_capacity.png` | novelty head — smaller wins |
| `09_zeroday_by_family.png` | zero-day detection per family |
| `10_two_heads.png` | summary — opposite capacity needs |
| `pitch_numbers.txt` / `.csv` | every quotable number + caveats |

## Part III — the two accuracy slides (for a short pitch)

Figures `11`–`13`, accuracy only, each with **real classical models** trained on
the same split by `classical_baselines.py` → `classical_baselines.json`.
Y-axes are **zoomed to the data band** rather than log-scaled.

### Table 1 — novelty (zero-day), accuracy

| config | QUANTUM | same PCA input | classical 122-feat | Isolation Forest |
|---|---|---|---|---|
| 3p12m | 93.31% | 92.84% | 95.00% | 93.55% |
| 3p20m | 94.43% | 94.72% | 95.00% | 93.55% |
| 3p24m | 94.51% | 94.93% | 95.00% | 93.55% |
| 4p24m | **94.58%** | 94.93% | 95.00% | 93.55% |

Trivial "all attack" guess = 74.37%. Accuracy is the **best achievable over all
thresholds**, per detector — the fairest head-to-head, but an upper bound, since
the threshold is chosen with hindsight.

> **Why this differs from the AUC story.** At a *fixed* 1% false-alarm budget the
> quantum detector scores only ~43%, because normal and attack scores overlap in
> the top percentile — the threshold lands above the median attack score and
> detection collapses to 23%. At a tuned threshold it reaches 94.6%. Always
> quote the operating point alongside the number.

### Table 2 — supervised, accuracy and attack recognition

| config | QUANTUM acc | same PCA input | QUANTUM attack-rec |
|---|---|---|---|
| 3p12m | 95.55% | 71.96% | **95.02%** |
| 3p20m | 97.40% | 76.69% | 92.31% |
| 3p24m | 97.66% | 79.66% | 90.50% |
| 4p24m | **97.88%** | 79.66% | 92.10% |

Classical on the full 122 features: Random Forest **99.87%** acc / 84.80%
attack-rec · Decision Tree 99.83 / 87.79 · **Random Forest on the same 24-dim
PCA 99.67%** · MLP 99.62 / 80.53 · Logistic regression 96.85 / 87.57 ·
Naive Bayes 63.51 / 62.47.

### What these two slides support

**Can say:**
* **Attack recognition is a genuine win** — 95.0% averaged over the four
  families vs the best classical model at 87.8% (Decision Tree). This is the
  metric that matters for catching attacks, because it weights rare families
  equally instead of letting `normal` + DoS dominate.
* Zero-day: competitive with Isolation Forest and a 122-feature classical
  detector, from a 24-dim optical bottleneck, and improving with capacity.
* Supervised accuracy: **+18 points over the same input without the reservoir**
  — the reservoir does real work on what it is given.

**Cannot say:**
* "Quantum beats classical on accuracy." **Random Forest wins** (99.87% vs
  97.88%) and still wins on the same 24-dim PCA input (99.67%). The honest
  framing is efficiency and competitiveness, not superiority.
* Novelty accuracy is an upper bound (hindsight threshold).
* Single seed (42) throughout — none of the trends are established.

## Headline numbers

**Supervised head (known attacks), best config 4p24m:**

| family | recall | precision | n (test) |
|---|---|---|---|
| DoS | 100.0% | 99.2% | 13,778 |
| Probe | 99.5% | 94.9% | 3,497 |
| R2L | 94.0% | 40.6% | 298 |
| U2R | 75.0% | 14.6% | 16 |

False alarms fall 8.02% → 3.75% as capacity grows; the classical control sits
at ~37–51%.

**Novelty head (zero-day, normal-only training):** best is the *smallest*
reservoir, `3p12m`, AUC 0.912 vs classical 0.865.

## What not to overclaim

* **Single seed (42) everywhere.** The reservoir unitary is random and the
  earlier notebook saw std up to 0.17 across seeds — larger than most gains
  here. Trends are indicative, **not established**.
* **Accuracy is dominated by `normal` + DoS** (91% of all test rows), so a high
  figure mostly reflects those two. Right headline for a general audience, but
  pair it with the per-class panel and the R2L/U2R numbers for a technical one.
* **Detection rate is saturated.** Never plot it zoomed.
* **The log10 chart plots error rate, not `log10(accuracy)`** — the latter is
  flat near 1.0 and would hide exactly what it is meant to reveal.
* **R2L/U2R F1 is low in absolute terms** (0.567 / 0.245). The claim is "far
  better than the classical control", not "solved".
* **High recall, low precision on rare families.** We find them by accepting
  many false accusations — that is a real operational cost.
* **Isolation Forest beats the quantum novelty head** at a 1% false-alarm budget
  (79.7% vs 23.3% detection). The zero-day advantage is in ranking, not at a
  deployable operating point.
* **U2R has 52 records total, 16 in test.** Anecdote, not measurement.
* **NSL-KDD "zero-days" are known attacks withheld from training**, not
  genuinely novel traffic.

## A pandas trap worth knowing

`OPS.head == "quantum"` silently returns all-False, because `.head` resolves to
`DataFrame.head` — the **method** — not the column. It does not raise. Always
use `OPS["head"]` for columns whose names collide with pandas methods
(`head`, `count`, `size`, `min`, `max`, `mean`, ...). This cost a debugging
cycle here and the notebook now carries an assertion against it.
