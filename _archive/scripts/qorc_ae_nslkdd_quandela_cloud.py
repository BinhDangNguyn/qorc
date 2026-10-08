#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QORC on NSL-KDD with Autoencoder reduction and Quandela Cloud execution.

This script keeps the classification experiment from
qorc_ae_nslkdd_experiment.py, but can compute the QORC reservoir features on
Quandela Cloud through MerlinProcessor. Classifier training remains local.

Run a local smoke test:
    python3 qorc_ae_nslkdd_quandela_cloud.py --local-quantum --limit 200

Run on Quandela Cloud simulator:
    export QUANDELA_TOKEN=_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mjg3MiwiZXhwIjoxNzg2OTgyODkxLjA2MTI0NH0.hxaFfc_BDS5euV9NOUuerUyFAg5GmpW7xw87zLKvSPzqJlwDyV1rBMopB0MVPXCjDtn5Uy6JlXSh6ivl31Eghg
    python3 qorc_ae_nslkdd_quandela_cloud.py --qpu-device sim:slos
"""


from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path


def bootstrap_workspace_venv() -> None:
    try:
        import numpy  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    if os.environ.get("QORC_VENV_BOOTSTRAPPED") == "1":
        return

    workspace_root = Path(__file__).resolve().parents[2]
    venv_python = workspace_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    os.environ["QORC_VENV_BOOTSTRAPPED"] = "1"
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


bootstrap_workspace_venv()

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import perceval as pcvl
import merlin as ML


SCRIPT_DIR = Path(__file__).resolve().parent
QRC_DIR = SCRIPT_DIR.parent
QORC_DIR = QRC_DIR / "QORC"
if str(QORC_DIR) not in sys.path:
    sys.path.insert(0, str(QORC_DIR))

REMOTE_HELPER_PATH = QORC_DIR / "lib" / "lib_remote_qorc.py"
remote_spec = importlib.util.spec_from_file_location("qorc_lib_remote_qorc", REMOTE_HELPER_PATH)
if remote_spec is None or remote_spec.loader is None:
    forward_remote_qorc_quantum_layer = None
    REMOTE_IMPORT_ERROR = f"Could not load {REMOTE_HELPER_PATH}"
else:
    try:
        remote_module = importlib.util.module_from_spec(remote_spec)
        remote_spec.loader.exec_module(remote_module)
        forward_remote_qorc_quantum_layer = remote_module.forward_remote_qorc_quantum_layer
    except Exception as exc:
        forward_remote_qorc_quantum_layer = None
        REMOTE_IMPORT_ERROR = exc
    else:
        REMOTE_IMPORT_ERROR = None

try:
    MeasurementStrategy = ML.MeasurementStrategy
    ComputationSpace = ML.ComputationSpace
except AttributeError:
    from merlin.measurement.strategies import MeasurementStrategy, ComputationSpace

try:
    XGBClassifier = importlib.import_module("xgboost").XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False


N_MODES = 12
N_PHOTONS = 3
TEST_SIZE = 0.30
SEED = 42
NO_BUNCHING = True

AE_EPOCHS = 50
AE_BATCH_SIZE = 512
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_DEVICE = torch.device("cpu")

DEFAULT_ZERO_DAY_ATTACK = ["smurf"]

CSV_CANDIDATES = [
    str(QRC_DIR / "data" / "NSL_KDD_labeled.csv"),
    str(QRC_DIR / "network data" / "NSL_KDD_labeled.csv"),
    r"D:\claude things\QRC\data\NSL_KDD_labeled.csv",
    r"D:\claude things\NSL_KDD_labeled.csv",
    r"/home/minhdtr/Documents/quantum/QRC/data/NSL_KDD_labeled.csv",
    "data/NSL_KDD_labeled.csv",
    "network data/NSL_KDD_labeled.csv",
    "NSL_KDD_labeled.csv",
]

LINEAR_ARMS = [
    ("A raw only", "raw"),
    ("B quantum only", "quantum"),
    ("C raw + quantum", "both"),
    ("D ae20 only (fair)", "ae20"),
]
MODEL_INPUT = {**dict(LINEAR_ARMS), "RF baseline": "raw", "XGB baseline": "raw"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NSL-KDD AE-QORC classification experiment with Quandela Cloud quantum features."
    )
    parser.add_argument("--csv", type=Path, default=None, help="Path to NSL_KDD_labeled.csv.")
    parser.add_argument("--qpu-device", default="sim:slos", help="Quandela backend name, default: sim:slos.")
    parser.add_argument("--nsample", type=int, default=1000, help="Remote samples per input row, default: 1000.")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N rows for smoke tests. Default uses all rows.")
    parser.add_argument("--zero-day", nargs="+", default=DEFAULT_ZERO_DAY_ATTACK, help="Attack label(s) hidden from Phase 2 training.")
    parser.add_argument("--local-quantum", action="store_true", help="Compute quantum features locally instead of using Quandela Cloud.")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "out" / "qorc_ae_quandela_cloud", help="Base output directory.")
    parser.add_argument("--ae-epochs", type=int, default=AE_EPOCHS, help="Autoencoder training epochs.")
    return parser.parse_args()


def configure_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qorc_ae_quandela_cloud")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_csv_path(csv_path: Path | None) -> Path:
    if csv_path is not None:
        path = csv_path.expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"CSV file not found: {path}")

    for candidate in CSV_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "NSL_KDD_labeled.csv not found. Pass --csv or place it in QRC/data."
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def load_nsl_kdd(csv_path: Path, limit: int, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("[data] loading %s", csv_path)
    df = pd.read_csv(csv_path)
    if limit > 0:
        df = df.head(limit).copy()
        logger.info("[data] limited to first %s rows", limit)

    categorical_cols = ["protocol_type", "service", "flag"]
    target_col = "label"
    numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]

    y_all = np.where(df[target_col] == "normal", 0, 1).astype(np.int64)
    attack_type = df[target_col].to_numpy()

    encoder = make_one_hot_encoder()
    x_cat = encoder.fit_transform(df[categorical_cols]).astype(np.float32)
    x_num = df[numeric_cols].to_numpy(dtype=np.float32)
    x_all = np.concatenate([x_num, x_cat], axis=1)

    logger.info(
        "[data] %s rows, %s raw features (%s normal / %s attack)",
        f"{x_all.shape[0]:,}",
        x_all.shape[1],
        f"{int((y_all == 0).sum()):,}",
        f"{int((y_all == 1).sum()):,}",
    )
    return x_all, y_all, attack_type


def build_qorc_quantum_layer(logger: logging.Logger) -> tuple[ML.QuantumLayer, int]:
    unitary = pcvl.Matrix.random_unitary(N_MODES)
    precircuit = pcvl.Unitary(unitary)
    reservoir = precircuit.copy()

    c_var = pcvl.Circuit(N_MODES)
    for mode in range(N_MODES):
        c_var.add(mode, pcvl.PS(pcvl.P(f"px{mode + 1}")))

    qorc_circuit = precircuit // c_var // reservoir

    step = (N_MODES - 1) / (N_PHOTONS - 1) if N_PHOTONS > 1 else 0
    input_state = [0] * N_MODES
    for photon in range(N_PHOTONS):
        input_state[int(round(photon * step))] = 1

    if NO_BUNCHING:
        n_qfeat = math.comb(N_MODES, N_PHOTONS)
        space = ComputationSpace.UNBUNCHED
    else:
        n_qfeat = math.comb(N_PHOTONS + N_MODES - 1, N_PHOTONS)
        space = ComputationSpace.FOCK

    quantum_layer = ML.QuantumLayer(
        input_size=N_MODES,
        circuit=qorc_circuit,
        trainable_parameters=[],
        input_parameters=["px"],
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.probs(computation_space=space),
        device=torch.device("cpu"),
    )
    quantum_layer.eval()
    logger.info("[quantum] modes=%s photons=%s -> %s quantum features/row", N_MODES, N_PHOTONS, n_qfeat)
    return quantum_layer, n_qfeat


def quantum_features_local(quantum_layer: ML.QuantumLayer, phases: np.ndarray, batch: int = 4096) -> np.ndarray:
    if len(phases) == 0:
        return np.empty((0, math.comb(N_MODES, N_PHOTONS)), dtype=np.float32)

    outs = []
    with torch.no_grad():
        for start in range(0, len(phases), batch):
            tensor = torch.tensor(phases[start : start + batch], dtype=torch.float32)
            outs.append(quantum_layer(tensor).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def quantum_features_train_test(
    quantum_layer: ML.QuantumLayer,
    train_phases: np.ndarray,
    test_phases: np.ndarray,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    if args.local_quantum:
        return (
            quantum_features_local(quantum_layer, train_phases),
            quantum_features_local(quantum_layer, test_phases),
        )

    if forward_remote_qorc_quantum_layer is None:
        raise ImportError(f"Could not import remote QORC helper: {REMOTE_IMPORT_ERROR}")
    if not os.environ.get("QUANDELA_TOKEN", "").strip():
        raise RuntimeError("QUANDELA_TOKEN is required unless --local-quantum is used.")

    train_tensor = torch.tensor(train_phases, dtype=torch.float32)
    val_tensor = torch.empty((0, train_phases.shape[1]), dtype=torch.float32)
    test_tensor = torch.tensor(test_phases, dtype=torch.float32)

    train_q, _, test_q = forward_remote_qorc_quantum_layer(
        train_tensor,
        val_tensor,
        test_tensor,
        quantum_layer,
        args.qpu_device,
        args.nsample,
        logger,
    )
    return train_q.detach().cpu().numpy().astype(np.float32), test_q.detach().cpu().numpy().astype(np.float32)


def quantum_features_new_rows(
    quantum_layer: ML.QuantumLayer,
    phases: np.ndarray,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> np.ndarray:
    if args.local_quantum:
        return quantum_features_local(quantum_layer, phases)

    if forward_remote_qorc_quantum_layer is None:
        raise ImportError(f"Could not import remote QORC helper: {REMOTE_IMPORT_ERROR}")
    if not os.environ.get("QUANDELA_TOKEN", "").strip():
        raise RuntimeError("QUANDELA_TOKEN is required unless --local-quantum is used.")

    empty_tensor = torch.empty((0, phases.shape[1]), dtype=torch.float32)
    phase_tensor = torch.tensor(phases, dtype=torch.float32)
    _, _, new_q = forward_remote_qorc_quantum_layer(
        empty_tensor,
        empty_tensor,
        phase_tensor,
        quantum_layer,
        args.qpu_device,
        args.nsample,
        logger,
    )
    return new_q.detach().cpu().numpy().astype(np.float32)


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


def fit_autoencoder(x_tr: np.ndarray, epochs: int, logger: logging.Logger) -> TabularAutoencoder:
    torch.manual_seed(SEED)
    model = TabularAutoencoder(input_dim=x_tr.shape[1], latent_dim=N_MODES).to(AE_DEVICE)
    dataset = TensorDataset(torch.tensor(x_tr, dtype=torch.float32))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    model.train()
    final_loss = float("nan")
    for _ in range(epochs):
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
    logger.info("  [ae] trained %s epochs, final reconstruction mse = %.6f", epochs, final_loss)
    return model


def encode_with_autoencoder(model: TabularAutoencoder, x_data: np.ndarray, batch: int = 4096) -> np.ndarray:
    if len(x_data) == 0:
        return np.empty((0, N_MODES), dtype=np.float32)

    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_data), batch):
            tensor = torch.tensor(x_data[start : start + batch], dtype=torch.float32, device=AE_DEVICE)
            outs.append(model.encoder(tensor).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def make_phase_transform(z_train: np.ndarray):
    lo = z_train.min()
    hi = z_train.max()

    def to_phase(array: np.ndarray) -> np.ndarray:
        return np.clip((array - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

    return lo, hi, to_phase


def run_pipeline(
    x_data: np.ndarray,
    y_data: np.ndarray,
    title: str,
    quantum_layer: ML.QuantumLayer,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[dict, dict, dict, dict, np.ndarray]:
    logger.info("\n=== %s ===", title)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_data, y_data, test_size=TEST_SIZE, random_state=SEED, stratify=y_data
    )
    logger.info("  train %s rows / test %s rows", f"{len(x_tr):,}", f"{len(x_te):,}")

    scaler = StandardScaler().fit(x_tr)
    x_tr = scaler.transform(x_tr).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)

    autoencoder = fit_autoencoder(x_tr, args.ae_epochs, logger)
    z_tr = encode_with_autoencoder(autoencoder, x_tr)
    z_te = encode_with_autoencoder(autoencoder, x_te)
    lo, hi, to_phase = make_phase_transform(z_tr)

    start = time.time()
    q_tr, q_te = quantum_features_train_test(
        quantum_layer, to_phase(z_tr), to_phase(z_te), args, logger
    )
    logger.info("  [quantum] features computed in %.1fs", time.time() - start)

    expected_qfeat = math.comb(N_MODES, N_PHOTONS) if NO_BUNCHING else math.comb(N_PHOTONS + N_MODES - 1, N_PHOTONS)
    if q_tr.shape[1] != expected_qfeat:
        raise ValueError(f"Expected {expected_qfeat} quantum features, got {q_tr.shape[1]}")

    qscaler = StandardScaler().fit(q_tr)
    q_tr = qscaler.transform(q_tr).astype(np.float32)
    q_te = qscaler.transform(q_te).astype(np.float32)

    aescaler = StandardScaler().fit(z_tr)
    ae20_tr = aescaler.transform(z_tr).astype(np.float32)
    ae20_te = aescaler.transform(z_te).astype(np.float32)

    reps_tr = {"raw": x_tr, "quantum": q_tr, "both": np.hstack([x_tr, q_tr]), "ae20": ae20_tr}
    reps_te = {"raw": x_te, "quantum": q_te, "both": np.hstack([x_te, q_te]), "ae20": ae20_te}

    models = {}
    accs = {}
    for name, rep in LINEAR_ARMS:
        clf = LogisticRegression(max_iter=5000)
        clf.fit(reps_tr[rep], y_tr)
        models[name] = clf
        accs[name] = accuracy_score(y_te, clf.predict(reps_te[rep]))

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(reps_tr["raw"], y_tr)
    models["RF baseline"] = rf
    accs["RF baseline"] = accuracy_score(y_te, rf.predict(reps_te["raw"]))

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.2,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )
        xgb.fit(reps_tr["raw"], y_tr)
        models["XGB baseline"] = xgb
        accs["XGB baseline"] = accuracy_score(y_te, xgb.predict(reps_te["raw"]))

    for name, value in accs.items():
        logger.info("  %-22s test accuracy = %.4f", name, value)

    transforms = dict(
        scaler=scaler,
        autoencoder=autoencoder,
        lo=lo,
        hi=hi,
        qscaler=qscaler,
        aescaler=aescaler,
    )
    return models, accs, transforms, reps_te, y_te


def reps_for_new_rows(
    transforms: dict,
    x_new: np.ndarray,
    quantum_layer: ML.QuantumLayer,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict:
    x_scaled = transforms["scaler"].transform(x_new).astype(np.float32)
    z_new = encode_with_autoencoder(transforms["autoencoder"], x_scaled)
    phases = np.clip(
        (z_new - transforms["lo"]) / (transforms["hi"] - transforms["lo"] + 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    q_new = quantum_features_new_rows(quantum_layer, phases, args, logger)
    q_new = transforms["qscaler"].transform(q_new).astype(np.float32)
    ae20 = transforms["aescaler"].transform(z_new).astype(np.float32)
    return {"raw": x_scaled, "quantum": q_new, "both": np.hstack([x_scaled, q_new]), "ae20": ae20}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / timestamp
    logger = configure_logging(run_dir)

    if not args.local_quantum and not os.environ.get("QUANDELA_TOKEN", "").strip():
        raise RuntimeError("QUANDELA_TOKEN is required for cloud mode. Use --local-quantum for a local smoke test.")

    csv_path = resolve_csv_path(args.csv)
    x_all, y_all, attack_type = load_nsl_kdd(csv_path, args.limit, logger)
    if len(np.unique(y_all)) < 2:
        raise ValueError("The selected rows contain only one class. Increase --limit or use the full dataset.")

    quantum_layer, n_qfeat = build_qorc_quantum_layer(logger)
    metrics = {
        "config": {
            "csv_path": str(csv_path),
            "limit": args.limit,
            "qpu_device": args.qpu_device,
            "nsample": args.nsample,
            "local_quantum": args.local_quantum,
            "zero_day": args.zero_day,
            "ae_epochs": args.ae_epochs,
            "n_modes": N_MODES,
            "n_photons": N_PHOTONS,
            "no_bunching": NO_BUNCHING,
            "n_qfeat": n_qfeat,
        }
    }

    models1, accs1, transforms1, reps_te1, y_te1 = run_pipeline(
        x_all,
        y_all,
        "PHASE 1: known-attack comparison with AE reduction (70/30)",
        quantum_layer,
        args,
        logger,
    )
    metrics["phase1_accuracy"] = {name: float(value) for name, value in accs1.items()}

    pred_both = models1["C raw + quantum"].predict(reps_te1["both"])
    phase1_report = classification_report(
        y_te1, pred_both, target_names=["normal", "attack"], output_dict=True
    )
    metrics["phase1_report_raw_plus_quantum"] = phase1_report
    logger.info("\n[phase 1] detailed report -- C raw + quantum:\n%s", classification_report(y_te1, pred_both, target_names=["normal", "attack"]))

    held = np.isin(attack_type, args.zero_day)
    if int(held.sum()) == 0:
        raise ValueError(f"No zero-day rows found for labels: {args.zero_day}")

    x_seen = x_all[~held]
    y_seen = y_all[~held]
    x_zero = x_all[held]
    logger.info(
        "\nZero-day setup: '%s' fully removed from training (%s rows held back, %s remain)",
        ", ".join(args.zero_day),
        f"{int(held.sum()):,}",
        f"{len(x_seen):,}",
    )

    models2, accs2, transforms2, reps_te2, y_te2 = run_pipeline(
        x_seen,
        y_seen,
        "PHASE 2: retrained WITHOUT the zero-day attack, AE reduction (70/30)",
        quantum_layer,
        args,
        logger,
    )
    metrics["phase2_seen_accuracy"] = {name: float(value) for name, value in accs2.items()}

    reps_zero = reps_for_new_rows(transforms2, x_zero, quantum_layer, args, logger)
    normal_mask = y_te2 == 0

    zero_day_metrics = {}
    logger.info(
        "\n=== ZERO-DAY RESULTS: '%s' (never seen in training) ===",
        ", ".join(args.zero_day),
    )
    logger.info("  %-22s %13s %19s %13s", "model", "seen-test acc", "zero-day detection", "false alarms")
    for name, model in models2.items():
        rep = MODEL_INPUT[name]
        detection = float(model.predict(reps_zero[rep]).mean())
        false_alarm = float(model.predict(reps_te2[rep])[normal_mask].mean())
        zero_day_metrics[name] = {
            "seen_test_accuracy": float(accs2[name]),
            "zero_day_detection": detection,
            "false_alarm_rate": false_alarm,
        }
        logger.info("  %-22s %13.4f %18.1f%% %12.1f%%", name, accs2[name], detection * 100, false_alarm * 100)

    metrics["zero_day"] = {
        "labels": args.zero_day,
        "held_rows": int(held.sum()),
        "metrics": zero_day_metrics,
    }
    write_json(run_dir / "metrics.json", metrics)

    logger.info("\nResults written to %s", run_dir)
    logger.info(
        "\n==================== HOW TO READ THE RESULTS ====================\n"
        "GAP 2 (baselines)  RF/XGB = today's best classical practice. We do not expect\n"
        "  to beat them on accuracy; the claim is viability plus zero-day generalisation.\n"
        "GAP 3 (ablation)   Compare A, B, C, D. B above D means the reservoir added\n"
        "  value beyond the 20 AE latent numbers it was given. C above A means quantum\n"
        "  features helped the combined model.\n"
        "GAP 1 (zero-day)   Detection is only meaningful next to false alarms. High\n"
        "  detection with low false alarms is the demo moment.\n"
        "================================================================"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)