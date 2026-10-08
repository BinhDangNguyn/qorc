#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QORC on NSL-KDD with Autoencoder reduction
==========================================
One self-contained script that produces the PCA-free variant of the
hackathon experiment.

The original paper used PCA for image-style inputs. For tabular network
traffic such as NSL-KDD, this variant uses an unsupervised autoencoder (AE)
to compress the scaled raw features into the 20 numbers consumed by the
photonic phase-shifter dials.

  PHASE 1 -- Known-attack comparison (70/30 split of all data)
     * Classical baselines: RandomForest (+ XGBoost if installed)      [GAP 2]
     * Ablation -- the SAME linear readout trained on four inputs:     [GAP 3]
         A) RAW features only        (122 columns)
         B) QUANTUM features only    (1140 columns from the reservoir)
         C) RAW + QUANTUM together   (Minh's original setup)
         D) AE-20 only               (fair control: exactly the 20
                                      numbers the quantum layer receives)

  PHASE 2 -- Zero-day demo                                             [GAP 1]
     * Every row of one chosen attack type is removed from training.
     * All models are retrained without it, then asked to judge the
       attack they have never seen.
     * Detection rate is reported NEXT TO the false-alarm rate, so a
       model that just flags everything cannot fake "detection".

The photonic part (pre-circuit -> phase-shifter dials -> reservoir) is
structurally IDENTICAL to Minh's working notebook. The readout is
scikit-learn LogisticRegression -- the same kind of model as his
nn.Linear (one trained linear layer), minus the epoch machinery.

Run:
    python qorc_ae_nslkdd_experiment.py
Dependencies:
    pip install perceval-quandela merlinquantum torch scikit-learn pandas
    (optional) pip install xgboost
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import perceval as pcvl
import merlin as ML

# Merlin has moved these names between versions -- accept either location.
try:
    MeasurementStrategy = ML.MeasurementStrategy
    ComputationSpace = ML.ComputationSpace
except AttributeError:
    from merlin.measurement.strategies import MeasurementStrategy, ComputationSpace

# XGBoost is optional -- if missing we simply skip it (RandomForest stays).
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ----------------------------------------------------------------------
# 0) Settings
# ----------------------------------------------------------------------
N_MODES     = 20         # photonic modes = AE latent dimensions = encoding dials
N_PHOTONS   = 3          # photons injected into the chip
TEST_SIZE   = 0.30       # team decision: 70% train / 30% test
SEED        = 42
NO_BUNCHING = True       # same as Minh's b_no_bunching=True -> C(20,3)=1140 features

AE_EPOCHS       = 50
AE_BATCH_SIZE   = 512
AE_LR           = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_DEVICE       = torch.device("cpu")

ZERO_DAY_ATTACK = ["smurf"]   # attack type(s) hidden from training in Phase 2
# whole-family variant:  ["satan", "ipsweep", "portsweep", "nmap"]   (Probe)

# The CSV is searched for in these places -- first existing path wins.
CSV_CANDIDATES = [
    r"D:\claude things\QRC\data\NSL_KDD_labeled.csv", 
    r"D:\claude things\NSL_KDD_labeled.csv",
    r"/home/minhdtr/Documents/quantum/QRC/data/NSL_KDD_labeled.csv",
    "data/NSL_KDD_labeled.csv",
    "network data/NSL_KDD_labeled.csv",
    "NSL_KDD_labeled.csv",
]

np.random.seed(SEED)
torch.manual_seed(SEED)

# Which input each model consumes (used to route the zero-day rows later).
LINEAR_ARMS = [
    ("A raw only",         "raw"),
    ("B quantum only",     "quantum"),
    ("C raw + quantum",    "both"),
    ("D ae20 only (fair)", "ae20"),
]
MODEL_INPUT = {**dict(LINEAR_ARMS), "RF baseline": "raw", "XGB baseline": "raw"}

# ----------------------------------------------------------------------
# 1) Load the labeled data
# ----------------------------------------------------------------------
csv_path = next((p for p in CSV_CANDIDATES if Path(p).exists()), None)
if csv_path is None:
    raise FileNotFoundError(
        "NSL_KDD_labeled.csv not found. Put your copy's full path first in "
        "CSV_CANDIDATES at the top of this script.")
print(f"[data] loading {csv_path}")
df = pd.read_csv(csv_path)

categorical_cols = ["protocol_type", "service", "flag"]
target_col = "label"
numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]

y_all = np.where(df[target_col] == "normal", 0, 1).astype(np.int64)  # 0=normal 1=attack
attack_type = df[target_col].to_numpy()                              # kept for zero-day

# One-hot the three text columns. Fitted on the whole file on purpose:
# protocol/service/flag form a FIXED vocabulary (domain knowledge, no label
# info), and this keeps the feature layout identical across both phases.
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[categorical_cols]).astype(np.float32)
X_num = df[numeric_cols].to_numpy(dtype=np.float32)
X_all = np.concatenate([X_num, X_cat], axis=1)
print(f"[data] {X_all.shape[0]:,} rows, {X_all.shape[1]} raw features "
      f"({int((y_all == 0).sum()):,} normal / {int((y_all == 1).sum()):,} attack)")

# ----------------------------------------------------------------------
# 2) Build the photonic reservoir ONCE (identical structure to Minh's)
#    pre-circuit -> 20 data dials -> reservoir.   trainable_parameters=[]
#    means NOTHING inside the circuit ever learns -- reservoir computing.
#    One fixed circuit is reused by both phases, so results are comparable.
# ----------------------------------------------------------------------
U = pcvl.Matrix.random_unitary(N_MODES)          # random but fixed (numpy seeded)
precircuit = pcvl.Unitary(U)
reservoir  = precircuit.copy()                   # same scrambler twice, like QORC

c_var = pcvl.Circuit(N_MODES)
for i in range(N_MODES):
    c_var.add(i, pcvl.PS(pcvl.P(f"px{i + 1}")))  # the encoding dials px1..px20

qorc_circuit = precircuit // c_var // reservoir

step = (N_MODES - 1) / (N_PHOTONS - 1) if N_PHOTONS > 1 else 0
input_state = [0] * N_MODES
for k in range(N_PHOTONS):
    input_state[int(round(k * step))] = 1        # photons spread evenly across modes

if NO_BUNCHING:
    n_qfeat = math.comb(N_MODES, N_PHOTONS)                  # 1140
    space = ComputationSpace.UNBUNCHED
else:
    n_qfeat = math.comb(N_PHOTONS + N_MODES - 1, N_PHOTONS)
    space = ComputationSpace.FOCK

quantum_layer = ML.QuantumLayer(
    input_size=N_MODES,
    circuit=qorc_circuit,
    trainable_parameters=[],           # the reservoir is FIXED
    input_parameters=["px"],           # our data drives the dials
    input_state=input_state,
    measurement_strategy=MeasurementStrategy.probs(computation_space=space),
    device=torch.device("cpu"),
)
quantum_layer.eval()
print(f"[quantum] modes={N_MODES} photons={N_PHOTONS} -> {n_qfeat} quantum features/row")


def quantum_features(phases: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Push rows of dial settings through the circuit in chunks, so laptop
    RAM is never overwhelmed. Returns one probability vector per row."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(phases), batch):
            t = torch.tensor(phases[i:i + batch], dtype=torch.float32)
            outs.append(quantum_layer(t).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def fit_autoencoder(X_tr: np.ndarray) -> TabularAutoencoder:
    torch.manual_seed(SEED)
    model = TabularAutoencoder(input_dim=X_tr.shape[1], latent_dim=N_MODES).to(AE_DEVICE)
    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    model.train()
    final_loss = float("nan")
    for _ in range(AE_EPOCHS):
        running_loss = 0.0
        n_seen = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(AE_DEVICE)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_x)
            n_seen += len(batch_x)
        final_loss = running_loss / max(n_seen, 1)

    model.eval()
    print(f"  [ae] trained {AE_EPOCHS} epochs, final reconstruction mse = {final_loss:.6f}")
    return model


def encode_with_autoencoder(model: TabularAutoencoder, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            t = torch.tensor(X[i:i + batch], dtype=torch.float32, device=AE_DEVICE)
            outs.append(model.encoder(t).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


# ----------------------------------------------------------------------
# 3) One experiment = split -> scale -> AE -> phases -> quantum -> models.
#    Used twice: Phase 1 (all data) and Phase 2 (zero-day attack removed).
# ----------------------------------------------------------------------
def run_pipeline(X, y, title):
    print(f"\n=== {title} ===")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"  train {len(X_tr):,} rows / test {len(X_te):,} rows")

    # scale raw features -- statistics come from TRAIN only (no leakage)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # shrink raw features -> 20 (AE latent), then squeeze into the [0,1] dial range
    autoencoder = fit_autoencoder(X_tr)
    Z_tr = encode_with_autoencoder(autoencoder, X_tr)
    Z_te = encode_with_autoencoder(autoencoder, X_te)
    lo, hi = Z_tr.min(), Z_tr.max()

    def to_phase(a):
        # test rows can fall slightly outside the train range -> clip to [0,1]
        return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

    t0 = time.time()
    Q_tr = quantum_features(to_phase(Z_tr))
    Q_te = quantum_features(to_phase(Z_te))
    print(f"  [quantum] features computed in {time.time() - t0:.1f}s")

    qscaler = StandardScaler().fit(Q_tr)          # tidy the quantum features
    Q_tr = qscaler.transform(Q_tr).astype(np.float32)
    Q_te = qscaler.transform(Q_te).astype(np.float32)

    aescaler = StandardScaler().fit(Z_tr)         # tidy the AE-20 control
    AE20_tr = aescaler.transform(Z_tr).astype(np.float32)
    AE20_te = aescaler.transform(Z_te).astype(np.float32)

    # every input representation, train and test side
    reps_tr = {"raw": X_tr, "quantum": Q_tr,
               "both": np.hstack([X_tr, Q_tr]), "ae20": AE20_tr}
    reps_te = {"raw": X_te, "quantum": Q_te,
               "both": np.hstack([X_te, Q_te]), "ae20": AE20_te}

    models, accs = {}, {}

    for name, rep in LINEAR_ARMS:                 # the ablation (gap 3)
        clf = LogisticRegression(max_iter=5000)
        clf.fit(reps_tr[rep], y_tr)
        models[name] = clf
        accs[name] = accuracy_score(y_te, clf.predict(reps_te[rep]))

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(reps_tr["raw"], y_tr)                  # baseline (gap 2)
    models["RF baseline"] = rf
    accs["RF baseline"] = accuracy_score(y_te, rf.predict(reps_te["raw"]))

    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.2,
                            eval_metric="logloss", random_state=SEED, n_jobs=-1)
        xgb.fit(reps_tr["raw"], y_tr)
        models["XGB baseline"] = xgb
        accs["XGB baseline"] = accuracy_score(y_te, xgb.predict(reps_te["raw"]))

    for k, v in accs.items():
        print(f"  {k:<22s} test accuracy = {v:.4f}")

    transforms = dict(scaler=scaler, autoencoder=autoencoder, lo=lo, hi=hi,
                      qscaler=qscaler, aescaler=aescaler)
    return models, accs, transforms, reps_te, y_te


def reps_for_new_rows(tf, X_new):
    """Run brand-new rows (the zero-day attacks) through a phase's fitted
    transforms, producing every input representation the models expect."""
    Xs = tf["scaler"].transform(X_new).astype(np.float32)
    Z = encode_with_autoencoder(tf["autoencoder"], Xs)
    ph = np.clip((Z - tf["lo"]) / (tf["hi"] - tf["lo"] + 1e-8),
                 0.0, 1.0).astype(np.float32)
    Q = tf["qscaler"].transform(quantum_features(ph)).astype(np.float32)
    AE20 = tf["aescaler"].transform(Z).astype(np.float32)
    return {"raw": Xs, "quantum": Q, "both": np.hstack([Xs, Q]), "ae20": AE20}


# ----------------------------------------------------------------------
# 4) PHASE 1 -- every attack type is available to learn from (gaps 2 & 3)
# ----------------------------------------------------------------------
models1, accs1, tf1, reps_te1, y_te1 = run_pipeline(
    X_all, y_all, "PHASE 1: known-attack comparison with AE reduction (70/30)")

pred_both = models1["C raw + quantum"].predict(reps_te1["both"])
print("\n[phase 1] detailed report -- C raw + quantum:")
print(classification_report(y_te1, pred_both, target_names=["normal", "attack"]))

# ----------------------------------------------------------------------
# 5) PHASE 2 -- the zero-day demo (gap 1)
# ----------------------------------------------------------------------
held = np.isin(attack_type, ZERO_DAY_ATTACK)
X_seen, y_seen = X_all[~held], y_all[~held]
X_zero = X_all[held]                              # every one of these is an attack
print(f"\nZero-day setup: '{', '.join(ZERO_DAY_ATTACK)}' fully removed from "
      f"training ({int(held.sum()):,} rows held back, {len(X_seen):,} remain)")

models2, accs2, tf2, reps_te2, y_te2 = run_pipeline(
    X_seen, y_seen, "PHASE 2: retrained WITHOUT the zero-day attack, AE reduction (70/30)")

reps_zero = reps_for_new_rows(tf2, X_zero)
normal_mask = (y_te2 == 0)

print(f"\n=== ZERO-DAY RESULTS: '{', '.join(ZERO_DAY_ATTACK)}' "
      f"(never seen in training) ===")
print(f"  {'model':<22s} {'seen-test acc':>13s} {'zero-day detection':>19s} "
      f"{'false alarms':>13s}")
for name, model in models2.items():
    rep = MODEL_INPUT[name]
    detection = model.predict(reps_zero[rep]).mean()               # flagged as attack
    false_alarm = model.predict(reps_te2[rep])[normal_mask].mean() # normal flagged
    print(f"  {name:<22s} {accs2[name]:>13.4f} {detection:>18.1%} "
          f"{false_alarm:>12.1%}")

# ----------------------------------------------------------------------
# 6) How to read the numbers (printed so it sits next to the results)
# ----------------------------------------------------------------------
print("""
==================== HOW TO READ THE RESULTS ====================
GAP 2 (baselines)  RF/XGB = today's best classical practice. We do
  NOT expect to beat them on accuracy; the claim is viability plus
  zero-day generalisation, not raw accuracy supremacy.
GAP 3 (ablation)   Compare A, B, C, D:
    B well above D        -> the reservoir added value beyond the 20
                             AE latent numbers it was given. Strongest evidence.
    C above A             -> quantum features helped the combined model.
    A ~ C with B lower    -> raw features carried it; say so honestly
                             and lean on the zero-day story.
GAP 1 (zero-day)   'Detection' = share of the never-seen attack rows
  flagged as attacks. Only meaningful NEXT TO 'false alarms': a model
  that flags everything would score high on both. High detection with
  LOW false alarms is the demo moment.
=================================================================""")