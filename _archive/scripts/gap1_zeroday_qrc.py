"""
gap1_zeroday_qrc.py
===================
Gap 1 (+ Gap 2 baseline): evaluate the photonic QRC on the zero-day held-out-attack
splits, and compare against classical baselines on the SAME splits.

Pipeline per scenario (leakage-free — everything fit on TRAIN only):
  raw NSL-KDD  -> NSLKDDPreprocessor (target-enc + log1p + MinMax)  -> X01 in [0,1]^122
              -> PCA(n_modes) -> global min-max to [0,1]            -> Z in [0,1]^20
              -> MerLin QuantumLayer (U // PS(pi-ish phases) // U)   -> 1140 Fock probs
              -> StandardScaler -> Logistic Regression readout       -> is_attack

Circuit matches the team's QORC repo (create_qorc_quantum_layer):
  n_photons=3, n_modes=20, no_bunching -> C(20,3)=1140 features, scheme A.

Baselines on the SAME preprocessed features (X01, 122-dim):
  RandomForest, XGBoost, and Logistic Regression.

Metrics (test is imbalanced, so NOT accuracy-only):
  zero-day recall (on held-out attack), ROC-AUC, PR-AUC, detection rate @ FPR=1%.

Outputs under artifacts/zeroday/<scenario>/:
  results.json  (all metrics, all models)
Plus artifacts/zeroday/RESULTS.md  (summary table across scenarios).
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import merlin as ML
import perceval as pcvl
import xgboost as xgb

# ---------------------------------------------------------------------------- config
ZERODAY_ROOT = r"D:\Python\AI\Code\ReservoirComputing\artifacts\zeroday"
RNG_SEED = 42
N_PHOTONS = 3
N_MODES = 20          # == PCA components == QuantumLayer input_size
NO_BUNCHING = True    # C(20,3)=1140 features, matches team's QORC report
PHASE_SCALE = 2 * np.pi  # z in [0,1] -> phi in [0, 2pi] (QORC uses full circle)
TARGET_FPR = 0.01     # detection-rate operating point

CAT_COLS = ["protocol_type", "service", "flag"]
LOG_COLS = ["duration", "src_bytes", "dst_bytes", "count", "srv_count",
            "dst_host_count", "dst_host_srv_count", "hot", "num_compromised", "num_root"]
DROP_COLS = ["num_outbound_cmds"]
SCENARIOS = ["smurf", "probe_all", "r2l_all", "neptune"]


# ---------------------------------------------------------- preprocessor (from nb 02)
class NSLKDDPreprocessor(BaseEstimator, TransformerMixin):
    """raw NSL-KDD frame -> dense [0,1] matrix. Fit on train only (no leakage)."""

    def fit(self, X: pd.DataFrame, y=None):
        y = np.asarray(y)
        self.te_maps_, self.global_rate_ = {}, float(np.mean(y))
        for c in CAT_COLS:
            self.te_maps_[c] = pd.Series(y, index=X.index).groupby(X[c]).mean()
        Z = self._raw(X)
        self.columns_ = Z.columns.tolist()
        self.scaler_ = MinMaxScaler().fit(Z.values)
        return self

    def _raw(self, X):
        Z = X.drop(columns=[c for c in DROP_COLS if c in X.columns]).copy()
        for c in CAT_COLS:
            Z[c] = Z[c].map(self.te_maps_[c]).fillna(self.global_rate_)
        for c in LOG_COLS:
            Z[c] = np.log1p(Z[c])
        return Z.select_dtypes(include=np.number)

    def transform(self, X):
        Z = self._raw(X)[self.columns_]
        return np.clip(self.scaler_.transform(Z.values), 0.0, 1.0)


# ---------------------------------------------------------------- quantum reservoir
def build_quantum_layer(seed: int):
    """scheme A reservoir: U_random // data phase column // U_random (== QORC)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    U = pcvl.Matrix.random_unitary(N_MODES)
    inter1 = pcvl.Unitary(U)
    inter2 = inter1.copy()
    c_var = pcvl.Circuit(N_MODES)
    for i in range(N_MODES):
        c_var.add(i, pcvl.PS(pcvl.P(f"px{i + 1}")))
    circuit = inter1 // c_var // inter2

    step = (N_MODES - 1) / (N_PHOTONS - 1) if N_PHOTONS > 1 else 0
    state = [0] * N_MODES
    for k in range(N_PHOTONS):
        state[int(round(k * step))] = 1

    out_size = (math.comb(N_MODES, N_PHOTONS) if NO_BUNCHING
                else math.comb(N_PHOTONS + N_MODES - 1, N_PHOTONS))
    strategy = (ML.MeasurementStrategy.probs() if NO_BUNCHING
                else ML.MeasurementStrategy.probs(
                    computation_space=ML.ComputationSpace.FOCK))
    layer = ML.QuantumLayer(
        input_size=N_MODES, circuit=circuit, trainable_parameters=[],
        input_parameters=["px"], input_state=state,
        measurement_strategy=strategy, device=torch.device("cpu"))
    layer.eval()
    return layer, out_size


def encode(layer, Z, batch=1024):
    """Z in [0,1]^n_modes -> Fock probabilities. Phases = PHASE_SCALE * z."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            x = torch.tensor(Z[i:i + batch] * PHASE_SCALE, dtype=torch.float32)
            outs.append(layer(x).cpu().numpy())
    return np.concatenate(outs, axis=0)


# ------------------------------------------------------------------------- metrics
def metrics(y_true, scores):
    """Threshold-independent detection metrics (higher score = more attack).

    On a zero-day test the score distribution shifts, so a fixed 0.5 threshold is
    meaningless — we report AUCs and recall at fixed low false-positive rates instead.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall_at_fpr1%": recall_at_fpr(y_true, scores, 0.01),
        "recall_at_fpr5%": recall_at_fpr(y_true, scores, 0.05),
    }


def recall_at_fpr(y_true, scores, target_fpr):
    """Recall on attacks at the strictest threshold keeping FPR <= target_fpr."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


# --------------------------------------------------------------------- per-scenario
def run_scenario(name: str, layer, out_size: int) -> dict:
    d = os.path.join(ZERODAY_ROOT, name)
    train = pd.read_csv(os.path.join(d, "train.csv"))
    test = pd.read_csv(os.path.join(d, "test.csv"))
    drop = ["label", "category", "is_attack"]
    Xtr_raw, Xte_raw = train.drop(columns=drop), test.drop(columns=drop)
    ytr, yte = train["is_attack"].values, test["is_attack"].values

    # classical preprocess (fit on train only)
    prep = NSLKDDPreprocessor().fit(Xtr_raw, ytr)
    X01_tr, X01_te = prep.transform(Xtr_raw), prep.transform(Xte_raw)

    # PCA -> [0,1]^n_modes (phase-ready)
    pca = PCA(n_components=N_MODES, random_state=RNG_SEED).fit(X01_tr)
    Zp_tr, Zp_te = pca.transform(X01_tr), pca.transform(X01_te)
    gmin, gmax = Zp_tr.min(), Zp_tr.max()
    Z_tr = np.clip((Zp_tr - gmin) / (gmax - gmin + 1e-8), 0, 1)
    Z_te = np.clip((Zp_te - gmin) / (gmax - gmin + 1e-8), 0, 1)

    # quantum features
    t0 = time.time()
    Q_tr, Q_te = encode(layer, Z_tr), encode(layer, Z_te)
    enc_s = time.time() - t0

    # standardise quantum features (mandatory — see nb 02 section 5)
    ss = StandardScaler().fit(Q_tr)
    Q_tr_s, Q_te_s = ss.transform(Q_tr), ss.transform(Q_te)

    res = {"scenario": name,
           "counts": {"train": int(len(ytr)), "test": int(len(yte)),
                      "test_attack": int(yte.sum()), "test_normal": int((yte == 0).sum())},
           "config": {"n_photons": N_PHOTONS, "n_modes": N_MODES,
                      "no_bunching": NO_BUNCHING, "quantum_features": out_size,
                      "phase_scale": "2pi", "encode_seconds": round(enc_s, 2)},
           "models": {}}

    # ---- QRC readout: linear on standardised quantum features ----
    # 1140 features with tiny variance -> needs stronger regularisation (C small).
    qrc = LogisticRegression(max_iter=5000, C=0.1)
    qrc.fit(Q_tr_s, ytr)
    res["models"]["QRC (quantum only)"] = metrics(yte, qrc.predict_proba(Q_te_s)[:, 1])

    # ---- classical baselines on the 122-dim preprocessed features (Gap 2) ----
    lr = LogisticRegression(max_iter=5000, C=1.0)
    lr.fit(X01_tr, ytr)
    res["models"]["LogReg (raw 122)"] = metrics(yte, lr.predict_proba(X01_te)[:, 1])

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG_SEED)
    rf.fit(X01_tr, ytr)
    res["models"]["RandomForest (raw 122)"] = metrics(yte, rf.predict_proba(X01_te)[:, 1])

    xgc = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                            eval_metric="logloss", random_state=RNG_SEED)
    xgc.fit(X01_tr, ytr)
    res["models"]["XGBoost (raw 122)"] = metrics(yte, xgc.predict_proba(X01_te)[:, 1])

    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[{name}]  test={len(yte)} (attack={int(yte.sum())})  "
          f"quantum_features={out_size}  encode={enc_s:.1f}s")
    for m, v in res["models"].items():
        print(f"    {m:26s} ROC-AUC={v['roc_auc']:.3f}  PR-AUC={v['pr_auc']:.3f}  "
              f"rec@FPR1%={v['recall_at_fpr1%']:.3f}  "
              f"rec@FPR5%={v['recall_at_fpr5%']:.3f}")
    return res


def write_summary(all_res):
    lines = ["# Gap 1 — Zero-day QRC results\n",
             f"Circuit: {N_PHOTONS} photons / {N_MODES} modes, "
             f"no-bunching -> {all_res[0]['config']['quantum_features']} features "
             f"(scheme A, MerLin). Phase scale 2pi.\n",
             "Task: binary novelty detection (is_attack). Test = held-out attack + 25% normal.\n",
             "Metric focus: **recall on held-out attack** and ROC/PR-AUC "
             "(accuracy is misleading on imbalanced test).\n"]
    for r in all_res:
        lines.append(f"\n## {r['scenario']}  "
                     f"(test attack={r['counts']['test_attack']}, "
                     f"normal={r['counts']['test_normal']})\n")
        lines.append("| Model | ROC-AUC | PR-AUC | recall@FPR1% | recall@FPR5% |")
        lines.append("|---|---|---|---|---|")
        for m, v in r["models"].items():
            lines.append(f"| {m} | {v['roc_auc']:.3f} | {v['pr_auc']:.3f} | "
                         f"{v['recall_at_fpr1%']:.3f} | {v['recall_at_fpr5%']:.3f} |")
    with open(os.path.join(ZERODAY_ROOT, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    print(f"building quantum reservoir ({N_PHOTONS}p/{N_MODES}m, "
          f"no_bunching={NO_BUNCHING})...")
    layer, out_size = build_quantum_layer(RNG_SEED)
    print(f"quantum feature dim = {out_size}")
    all_res = [run_scenario(s, layer, out_size) for s in SCENARIOS]
    write_summary(all_res)
    print(f"\nwrote results.json per scenario + RESULTS.md under\n  {ZERODAY_ROOT}")


if __name__ == "__main__":
    main()
