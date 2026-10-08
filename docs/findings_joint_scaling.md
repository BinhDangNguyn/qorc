# Joint scaling: findings so far (seed 42)

Question: the original sweeps grew the reservoir (220 -> 10,626 Fock states)
under a fixed linear readout, and macro-F1 flattened below expectations. Was the
plateau set by the reservoir or by the linear head?

Method: grow both. Every head is trained on the same split for every feature
view -- `quantum`, its `dials` (the PCA input), `rff` (classical random Fourier
lift of the same width), `raw` (no bottleneck) and the hybrids `qdials` /
`rffdials` -- so a bigger head helps the controls exactly as much as it helps
the reservoir. Data: `runs/joint_scaling`, `runs/joint_scaling_modes`,
`runs/joint_novelty_partial`. Figures: `notebooks/30_joint_scaling_report`.

**One seed.** Differences under ~0.01-0.02 are not established (on NSL-KDD
`all23`, whose tail classes have 2-11 rows, under ~0.05). Treat the numbers
below as direction, not as final.

## NSL-KDD, supervised (macro-F1)

families6 (7 classes):

| features | linear | MLP 128 | MLP 512 | MLP 2048 | MLP 512x2 |
|---|---|---|---|---|---|
| quantum 3p12m (220) | 0.941 | 0.972 | 0.970 | 0.974 | 0.978 |
| quantum 3p20m (1,140) | 0.964 | 0.973 | 0.980 | 0.981 | **0.984** |
| quantum 4p24m (10,626) | 0.972 | 0.973 | 0.976 | 0.979 | 0.983 |
| dials 24 | 0.823 | 0.951 | 0.957 | 0.961 | 0.969 |
| RFF 1,140 (of dials 20) | 0.964 | 0.973 | 0.974 | 0.977 | **0.984** |
| raw 122 features | 0.908 | 0.972 | 0.977 | 0.975 | **0.984** |

1. **A bigger head does lift the plateau**: best quantum goes from 0.972
   (linear, 4p24m) to 0.984 -- and it gets there with a *smaller* reservoir
   (3p20m) once the head has a hidden layer.
2. **0.984 is the task's ceiling, not the reservoir's**: raw features and RFF
   reach the same number with the same head. The plateau seen before was the
   linear head; there is little left to win on families6.
3. **The quantum-vs-dials gap shrinks from ~+0.15 to ~+0.015** once both get the
   same MLP. It stays positive (the lift is still useful), but most of the old
   gap was the linear head's inability to build nonlinearity from the dials.
4. all23 moves between 0.63 and 0.72 across heads with no stable ordering --
   single-seed noise on 2-11-row classes. Raw + MLP 2048 reaches 0.717, RFF
   0.719, quantum 0.709.

## K-MNIST, supervised (macro-F1, 54k train / 10k test)

| features | linear | MLP 512 | MLP 512x2 |
|---|---|---|---|
| quantum 3p12m | 0.705 | 0.806 | 0.815 |
| quantum 3p24m | 0.866 | 0.898 | 0.902 |
| quantum 4p24m | 0.882 | 0.903 | 0.896 |
| dials 24 | 0.625 | 0.881 | 0.912 |
| dials 64 | 0.677 | 0.916 | **0.932** |
| RFF 10,626 | 0.888 | 0.911 | 0.918 |
| raw 784 px | 0.698 | 0.910 | 0.921 |

5. With a linear head the reservoir helps a lot (+0.26 over its dials at 24
   modes). **With a large head, the reservoir falls *below* its own input**
   (0.896-0.902 vs 0.912 at 24 modes): the Fock readout loses information the
   MLP could have used.
6. **Photons multiply features, not information.** 3 -> 4 photons at 24 modes
   (2,024 -> 10,626 features) changes nothing once the head is an MLP.
7. **More modes feed in more information, but the quantum readout does not pass
   it on**: dials go 0.912 -> 0.932 from 24 to 64 modes (beating the 784 raw
   pixels), while quantum 2p64m / 2p96m stay at ~0.905. Concatenating the dials
   back (`qdials`) does not recover the dials' score either.
8. RFF of the same width beats quantum by 0.01-0.02 at every size.

## NSL-KDD, zero-day novelty (partial: steps up to 20 modes)

Normal-only training; macro AUC over attack types with >= 20 rows, and
detection at a 1% false-alarm budget (pooled):

| features | PCA-32 (original) | AE 512 | AE 512x2 |
|---|---|---|---|
| quantum 3p12m | 0.917 / 0.42 | 0.973 / 0.92 | 0.974 / 0.93 |
| dials 12 | 0.869 / 0.63 | 0.949 / 0.84 | 0.965 / 0.85 |
| RFF 220 | 0.910 / 0.32 | 0.972 / 0.91 | 0.975 / 0.92 |
| raw 122 | 0.899 / 0.10 | 0.979 / 0.91 | 0.979 / 0.94 |
| IsolationForest, raw | 0.900 / 0.80 | | |

9. **The detector is the bottleneck for zero-day detection, by far**: an
   autoencoder lifts macro AUC from ~0.90 to ~0.97-0.98 and detection at 1%
   false alarms from 0.10-0.42 to 0.91-0.94, on every feature view.
10. With the linear detector the reservoir is clearly ahead of its dials
    (0.917 vs 0.869); with an autoencoder, quantum, RFF and raw are within
    ~0.01 of each other.

## NSL-KDD, zero-day novelty ladder (complete: `runs/novelty_ladder`)

Reservoir and autoencoder grow together; the original PCA-32 detector runs at
every step. Pooled detection at a 1% false-alarm budget:

| step | reservoir | AE | quantum | RFF | dials | raw | quantum + PCA-32 |
|---|---|---|---|---|---|---|---|
| 1 | 3p12m (220) | 128-b16 | 0.854 | 0.851 | 0.702 | 0.876 | 0.236 |
| 2 | 3p20m (1,140) | 256-b32 | 0.863 | 0.918 | 0.918 | 0.937 | 0.058 |
| 3 | 3p24m (2,024) | 512-b32 | 0.929 | 0.925 | 0.803 | 0.942 | 0.066 |
| 4 | 4p20m (4,845) | 1024-b64 | 0.935 | 0.934 | 0.906 | 0.920 | 0.062 |
| 5 | 4p24m (10,626) | 2048-b64 | **0.939** | 0.937 | 0.798 | 0.903 | 0.026 |

IsolationForest on raw: 0.798. Macro AUC climbs 0.962 -> 0.977 for quantum,
with RFF within 0.005 at every step and raw at 0.969-0.980.

11. **Scaling both sides together works for zero-day detection**: quantum
    detection at 1% false alarms rises at every step (0.854 -> 0.939), and
    attacks the small setup misses become catchable -- ipsweep 0.18 -> 0.90,
    nmap 0.18 -> 0.82, satan 0.66 -> 0.90, teardrop 0.11 -> 0.60.
12. **But a classical random lift of the same width does exactly the same**
    (RFF within 0.003 at steps 1 and 3-5). The quantum-over-dials margin
    (+0.13-0.14 at steps 3 and 5) is a lift effect, not a quantum one.
13. **The linear detector is unusable at low false-alarm budgets on the Fock
    readout** (0.03-0.24), although its AUC looks fine (~0.90). The original
    design's headline AUC hid this.
14. raw + AE peaks at step 3 (0.942) and falls as the AE outgrows its 122
    inputs (0.903 at AE 2048): the quantum readout keeps a wide AE stable,
    which is a practical, not a fundamental, advantage.
15. Never caught by any view: `back` (~0.10) and `warezclient` (~0.13) --
    they look like normal traffic in these features.

## Against standalone classical models (`runs/classical_*`, `runs/comparison/`)

Classical models trained on the raw features only, same seed, split and test
rows as the quantum runs. The quantum side is fixed in advance (largest
reservoir), not picked on the test set.

Zero-day, detection at 1% false alarms (pooled) / mean over attack types:

| detector | pooled | mean per type | macro AUC |
|---|---|---|---|
| autoencoder 512, raw | **0.942** | **0.769** | **0.979** |
| quantum 4p24m + AE 2048 | 0.939 | 0.722 | 0.977 |
| Gaussian mixture (8) | 0.867 | 0.619 | 0.919 |
| kNN distance (5) | 0.853 | 0.530 | 0.964 |
| IsolationForest | 0.798 | 0.234 | 0.917 |
| one-class SVM | 0.784 | 0.487 | 0.911 |
| PCA-32, raw | 0.309 | 0.201 | 0.899 |
| LOF (20) | 0.202 | 0.227 | 0.937 |
| quantum 4p24m + PCA-32 (original) | 0.026 | 0.121 | 0.898 |

Supervised macro-F1 (families6 / all23): XGBoost 0.997 / 0.839,
HistGradientBoosting 0.997 / 0.811, random forest 0.995 / 0.779, extra trees
0.995 / 0.822, MLP 512x2 0.986 / 0.678, kNN 0.982 / 0.734, logistic regression
0.926 / 0.669, linear SVM 0.941 / 0.604 -- against quantum 4p24m 0.983 / 0.626
(MLP 512x2) and 0.972 / 0.684 (linear head).

16. **Zero-day: the scaled quantum detector ties the best classical detector**
    and beats six of the seven classical families tried, including the
    IsolationForest baseline by 14 points. By attack family it is strongest on
    Probe (ipsweep 90% vs <=84% for any classical model, satan 90% vs <=84%,
    nmap 82% -- tied with the raw autoencoder); GMM and the raw autoencoder are
    far better on small DoS attacks (teardrop, pod).
17. **Supervised: tree ensembles win outright.** The reservoir beats linear
    classical models, not gradient boosting or random forests. For closed-set
    attack classification on NSL-KDD there is no case for the reservoir.
18. AUC alone would have ranked LOF (0.937) above GMM (0.919) and made the
    original quantum + PCA design look usable (0.898); at a fixed 1% false-alarm
    budget they catch 20% and 3%. Report detection at a fixed budget.

## What this means

* The earlier plateau was mostly the classical side. Scaling the head is the
  single biggest lever on both tasks.
* Once the head is strong, the photonic reservoir is competitive with -- but not
  better than -- a classical random lift of the same width, and on images it
  discards information its own input still carries.
* Where the reservoir still pays: **cheap heads**. With a linear readout or a
  linear detector, the reservoir is worth +0.10-0.26 over its input. If the
  deployment constraint is a tiny classical head (edge hardware, a
  hardware-readout budget), that is the defensible story.

## Next

1. Run the ladders (`configs/supervised_ladder.yaml`, `configs/novelty_ladder.yaml`)
   to confirm the coupled-scaling picture end to end; then 2-3 seeds on the
   points that carry a claim.
2. Why the Fock readout loses information at high modes: the phase-encoding
   width (`encoding.width`) and bunched / PNR readout (`no_bunching: false`) are
   config fields now -- each is one run. Independent pre/post unitaries (the
   reservoir is currently a copy of the pre-circuit, as in the paper) would need
   one more `ReservoirSpec` field.
3. Fairness check for "cheap head" claims: compare at equal *total* parameter
   count (head params are in every record as `n_params`).
