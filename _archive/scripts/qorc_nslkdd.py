#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone QORC applied to NSL-KDD intrusion detection."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
for path in (REPO_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lib.lib_datasets import get_dataloader, split_fold_numpy, tensor_dataset
from lib.lib_learning import get_device, model_eval, model_fit
from lib.lib_qorc_encoding_and_linear_training import create_qorc_quantum_layer
from runtime_lib.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone QORC experiment on the NSL-KDD intrusion dataset."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "network data" / "NSL_KDD_labeled.csv",
        help="Path to NSL_KDD_labeled.csv. Relative paths are resolved from papers/QORC.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outdir",
        help="Base output directory. A timestamped run directory is created inside it.",
    )
    parser.add_argument("--n-photons", type=int, default=3, help="Number of photons.")
    parser.add_argument("--n-modes", type=int, default=20, help="Number of optical modes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed; use -1 to skip seeding.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction.")
    parser.add_argument("--fold-index", type=int, default=0, help="Validation fold index.")
    parser.add_argument("--n-fold", type=int, default=5, help="Number of train/validation folds.")
    parser.add_argument(
        "--dataset-truncate",
        type=int,
        default=0,
        help="Limit train, validation, and test sizes after splitting. Use 0 for the full dataset.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Adagrad learning rate.")
    parser.add_argument("--reduce-lr-patience", type=int, default=10, help="ReduceLROnPlateau patience.")
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--f-out-weights", default="f_weights_out.pth", help="Checkpoint filename.")
    parser.add_argument("--device", default="cpu", help="Torch device name, for example cpu or cuda:0.")
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DataLoader pinned memory.",
    )
    parser.add_argument(
        "--binary-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use binary normal-vs-attack labels. Disable for multiclass labels.",
    )
    parser.add_argument(
        "--b-no-bunching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use unbunched output states. Disable to include full Fock space.",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path = PROJECT_ROOT) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return (base / path).resolve()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def set_seed(seed: int) -> None:
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=False)


def normalize_global_min_max(data: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    epsilon = 1e-8
    return (data - global_min) / (global_max - global_min + epsilon)


def normalize_standard_scaler(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8
    return (data - mean) / (std + epsilon)


def load_nslkdd_data(
    csv_path: Path,
    binary_task: bool,
    test_size: float,
    n_fold: int,
    fold_index: int,
    dataset_truncate: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    logger.info("Loading NSL-KDD data from %s", csv_path)
    df = pd.read_csv(csv_path)

    categorical_cols = ["protocol_type", "service", "flag"]
    target_col = "label"
    missing_cols = [col for col in [*categorical_cols, target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required NSL-KDD columns: {missing_cols}")

    feature_cols = [col for col in df.columns if col != target_col]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    if binary_task:
        y_raw = np.where(df[target_col] == "normal", 0, 1)
        class_names = ["normal", "attack"]
    else:
        class_names = [str(label) for label in sorted(df[target_col].unique())]
        label_to_idx = {label: index for index, label in enumerate(class_names)}
        y_raw = df[target_col].map(label_to_idx).to_numpy()

    encoder = make_one_hot_encoder()
    x_cat = encoder.fit_transform(df[categorical_cols])
    x_num = df[numeric_cols].to_numpy(dtype=np.float32)
    x_all = np.concatenate([x_num, x_cat], axis=1).astype(np.float32)

    train_val_data, test_data, train_val_label, test_label = train_test_split(
        x_all,
        y_raw,
        test_size=test_size,
        random_state=seed if seed >= 0 else None,
        stratify=y_raw,
    )
    val_label, val_data, train_label, train_data = split_fold_numpy(
        train_val_label,
        train_val_data,
        n_fold,
        fold_index,
        split_seed=seed,
    )

    if dataset_truncate > 0:
        train_data = train_data[:dataset_truncate]
        train_label = train_label[:dataset_truncate]
        val_data = val_data[:dataset_truncate]
        val_label = val_label[:dataset_truncate]
        test_data = test_data[:dataset_truncate]
        test_label = test_label[:dataset_truncate]

    raw_scaler = StandardScaler()
    train_data = raw_scaler.fit_transform(train_data).astype(np.float32)
    val_data = raw_scaler.transform(val_data).astype(np.float32)
    test_data = raw_scaler.transform(test_data).astype(np.float32)

    logger.info("Dataset sizes:")
    logger.info("train: %s, val: %s, test: %s", train_data.shape, val_data.shape, test_data.shape)

    return train_data, val_data, test_data, train_label, val_label, test_label, class_names


def compute_qorc_features(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    n_photons: int,
    n_modes: int,
    b_no_bunching: bool,
    device_name: str,
    compute_device: torch.device,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
    pca = PCA(n_components=n_modes)
    train_data_pca = pca.fit_transform(train_data)
    val_data_pca = pca.transform(val_data)
    test_data_pca = pca.transform(test_data)

    pca_train_global_min = train_data_pca.min()
    pca_train_global_max = train_data_pca.max()

    train_data_pca_norm = normalize_global_min_max(train_data_pca, pca_train_global_min, pca_train_global_max)
    val_data_pca_norm = normalize_global_min_max(val_data_pca, pca_train_global_min, pca_train_global_max)
    test_data_pca_norm = normalize_global_min_max(test_data_pca, pca_train_global_min, pca_train_global_max)

    time_layer_start = time.time()
    qorc_quantum_layer, qorc_output_size = create_qorc_quantum_layer(
        n_photons,
        n_modes,
        b_no_bunching,
        device_name,
        logger,
    )
    time_layer_end = time.time()

    logger.info("Quantum features size: %s", qorc_output_size)
    logger.info("Computing quantum features")
    time_features_start = time.time()
    train_data_qorc = qorc_quantum_layer(
        torch.tensor(train_data_pca_norm, dtype=torch.float32, device=compute_device)
    )
    val_data_qorc = qorc_quantum_layer(
        torch.tensor(val_data_pca_norm, dtype=torch.float32, device=compute_device)
    )
    test_data_qorc = qorc_quantum_layer(
        torch.tensor(test_data_pca_norm, dtype=torch.float32, device=compute_device)
    )
    time_features_end = time.time()

    qorc_train_mean = train_data_qorc.detach().mean(dim=0)
    qorc_train_std = train_data_qorc.detach().std(dim=0)

    train_data_qorc_norm = normalize_standard_scaler(train_data_qorc, qorc_train_mean, qorc_train_std)
    val_data_qorc_norm = normalize_standard_scaler(val_data_qorc, qorc_train_mean, qorc_train_std)
    test_data_qorc_norm = normalize_standard_scaler(test_data_qorc, qorc_train_mean, qorc_train_std)

    return (
        train_data_qorc_norm,
        val_data_qorc_norm,
        test_data_qorc_norm,
        qorc_output_size,
        time_layer_end - time_layer_start,
        time_features_end - time_features_start,
    )


def collect_predictions(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    compute_device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(compute_device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_true.extend(labels.numpy().tolist())
    return all_true, all_preds


def tensor_accuracy_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def main() -> None:
    args = parse_args()
    csv_path = resolve_path(args.csv)
    outdir = resolve_path(args.outdir)
    if not csv_path.exists():
        raise FileNotFoundError(f"NSL-KDD CSV not found: {csv_path}")
    if args.fold_index < 0 or args.fold_index >= args.n_fold:
        raise ValueError("--fold-index must satisfy 0 <= fold_index < n_fold")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = outdir / f"run_nslkdd_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging("info", run_dir / "run.log")
    logger = logging.getLogger(__name__)
    logger.info("Run directory: %s", run_dir)

    set_seed(args.seed)
    storage_device = torch.device("cpu")
    compute_device = get_device(args.device)

    time_t1 = time.time()
    train_data, val_data, test_data, train_label, val_label, test_label, class_names = load_nslkdd_data(
        csv_path=csv_path,
        binary_task=args.binary_task,
        test_size=args.test_size,
        n_fold=args.n_fold,
        fold_index=args.fold_index,
        dataset_truncate=args.dataset_truncate,
        seed=args.seed,
        logger=logger,
    )

    (
        train_data_qorc_norm,
        val_data_qorc_norm,
        test_data_qorc_norm,
        qorc_output_size,
        quantum_layer_duration,
        quantum_features_duration,
    ) = compute_qorc_features(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        n_photons=args.n_photons,
        n_modes=args.n_modes,
        b_no_bunching=args.b_no_bunching,
        device_name=args.device,
        compute_device=compute_device,
        logger=logger,
    )

    dtype = torch.float32
    all_train_data = torch.cat(
        (torch.tensor(train_data, dtype=dtype, device=compute_device), train_data_qorc_norm),
        dim=1,
    )
    all_val_data = torch.cat(
        (torch.tensor(val_data, dtype=dtype, device=compute_device), val_data_qorc_norm),
        dim=1,
    )
    all_test_data = torch.cat(
        (torch.tensor(test_data, dtype=dtype, device=compute_device), test_data_qorc_norm),
        dim=1,
    )

    ds_train = tensor_dataset(all_train_data, train_label, storage_device, dtype=torch.float32)
    ds_val = tensor_dataset(all_val_data, val_label, storage_device, dtype=torch.float32)
    ds_test = tensor_dataset(all_test_data, test_label, storage_device, dtype=torch.float32)

    train_loader = get_dataloader(ds_train, args.batch_size, True, args.num_workers, args.pin_memory, args.seed)
    val_loader = get_dataloader(ds_val, args.batch_size, False, args.num_workers, args.pin_memory, args.seed)
    test_loader = get_dataloader(ds_test, args.batch_size, False, args.num_workers, args.pin_memory, args.seed)

    n_features_raw = train_data.shape[1]
    n_classes = len(class_names)
    n_model_input_features = n_features_raw + qorc_output_size
    logger.info("n_model_input_features: %s", n_model_input_features)

    model = nn.Linear(n_model_input_features, n_classes, bias=True, device=compute_device)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    model.to(compute_device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    logger.info("Evaluation before training (on test set)")
    model_eval(model, test_loader, criterion, compute_device, logger, True, True)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, eps=1e-7)
    weights_path = run_dir / args.f_out_weights

    time_t3 = time.time()
    (
        train_loss_history,
        train_accuracy_history,
        val_loss_history,
        val_accuracy_history,
        fit_duration,
        best_val_epoch,
    ) = model_fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.epochs,
        str(weights_path),
        early_stop_patience=args.epochs,
        early_stop_min_delta=0.000001,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor,
        device=compute_device,
        logger=logger,
        b_use_cosine_scheduler=False,
        calc_accuracy=True,
    )
    time_t4 = time.time()

    best_state_dict = torch.load(weights_path, map_location=compute_device)
    model.load_state_dict(best_state_dict)

    _, train_acc, _ = model_eval(model, train_loader, criterion, compute_device, logger, True, True)
    _, val_acc, _ = model_eval(model, val_loader, criterion, compute_device, logger, True, True)
    _, test_acc, _ = model_eval(model, test_loader, criterion, compute_device, logger, True, True)
    time_t5 = time.time()

    train_acc_float = tensor_accuracy_to_float(train_acc)
    val_acc_float = tensor_accuracy_to_float(val_acc)
    test_acc_float = tensor_accuracy_to_float(test_acc)

    logger.info("Duration - Quantum layer creation: %.2fs", quantum_layer_duration)
    logger.info("Duration - Quantum features encoding: %.2fs", quantum_features_duration)
    logger.info("Duration - training: %.2fs", time_t4 - time_t3)
    logger.info("Duration - total: %.2fs", time_t5 - time_t1)
    logger.info("Best val epoch: %s", best_val_epoch)
    logger.info(
        "Accuracies - train: %.4f, val: %.4f, test: %.4f",
        train_acc_float,
        val_acc_float,
        test_acc_float,
    )

    all_true, all_preds = collect_predictions(model, test_loader, compute_device)
    labels = list(range(len(class_names)))
    report = classification_report(
        all_true,
        all_preds,
        labels=labels,
        target_names=[str(class_name) for class_name in class_names],
        zero_division=0,
    )
    print(report)
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(all_true, all_preds, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    confusion_matrix_path = run_dir / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    summary = {
        "csv_path": str(csv_path),
        "run_dir": str(run_dir),
        "binary_task": args.binary_task,
        "class_names": class_names,
        "n_photons": args.n_photons,
        "n_modes": args.n_modes,
        "seed": args.seed,
        "test_size": args.test_size,
        "fold_index": args.fold_index,
        "n_fold": args.n_fold,
        "dataset_truncate": args.dataset_truncate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "b_no_bunching": args.b_no_bunching,
        "device": args.device,
        "n_features_raw": int(n_features_raw),
        "qorc_output_size": int(qorc_output_size),
        "n_model_input_features": int(n_model_input_features),
        "best_val_epoch": int(best_val_epoch),
        "train_accuracy": train_acc_float,
        "val_accuracy": val_acc_float,
        "test_accuracy": test_acc_float,
        "quantum_layer_duration_seconds": quantum_layer_duration,
        "quantum_features_duration_seconds": quantum_features_duration,
        "model_fit_duration_seconds": fit_duration,
        "training_wall_time_seconds": time_t4 - time_t3,
        "total_duration_seconds": time_t5 - time_t1,
        "weights_path": str(weights_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "train_loss_history": [float(value) for value in train_loss_history],
        "train_accuracy_history": [tensor_accuracy_to_float(value) for value in train_accuracy_history],
        "val_loss_history": [float(value) for value in val_loss_history],
        "val_accuracy_history": [tensor_accuracy_to_float(value) for value in val_accuracy_history],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()