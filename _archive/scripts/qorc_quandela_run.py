#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the QORC quantum layer from qorc_nslkdd.py on Quandela Cloud.

This is a quantum-only demo: it builds the exact QORC photonic circuit used in
based-on-merlin_repo/papers/QORC/qorc_nslkdd.py (via
create_qorc_quantum_layer), prepares a small batch of real NSL-KDD rows
(PCA-reduced to phase inputs), and submits them to Quandela Cloud to obtain
quantum (Fock-probability) features. No classifier is trained here -- see
qorc_nslkdd.py for the full experiment.

The demo's PCA + normalization are fit on the small demo batch itself (not on
a proper train split like qorc_nslkdd.py), so the resulting numbers are not a
real experiment result -- this script only exercises the QORC circuit against
Quandela Cloud.

The Quandela API token is never hardcoded: set it via the QUANDELA_TOKEN
environment variable before running.

Local smoke test (no token / no cloud call needed):
    python3 qorc_quandela_run.py --local-quantum

Run on Quandela Cloud:
    export QUANDELA_TOKEN=...   # never hardcode this
    _T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mjg3MiwiZXhwIjoxNzg2OTgyODkxLjA2MTI0NH0.hxaFfc_BDS5euV9NOUuerUyFAg5GmpW7xw87zLKvSPzqJlwDyV1rBMopB0MVPXCjDtn5Uy6JlXSh6ivl31Eghg
    python3 qorc_quandela_run.py --qpu-device sim:ascella --limit 50

Resume a previously submitted raw job by ID:
    python3 qorc_quandela_run.py --resume-job-id <job-id>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# 1. Make the real QORC circuit builder importable (papers/QORC/lib package).
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # QRC/team
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]  # .../quantum
REPO_ROOT = WORKSPACE_ROOT / "based-on-merlin_repo"  # provides the runtime_lib package
QORC_ROOT = REPO_ROOT / "papers" / "QORC"  # provides the lib package
for _path in (REPO_ROOT, QORC_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from lib.lib_qorc_encoding_and_linear_training import create_qorc_quantum_layer  # noqa: E402

import perceval as pcvl  # noqa: E402
from perceval.runtime import RemoteConfig  # noqa: E402
from merlin.core.merlin_processor import MerlinProcessor  # noqa: E402


DEFAULT_CSV = WORKSPACE_ROOT / "QRC" / "data" / "NSL_KDD_labeled.csv"
VALID_QPU_DEVICES = ["sim:slos", "sim:ascella", "sim:belenos", "qpu:ascella", "qpu:belenos"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the QORC quantum layer (from qorc_nslkdd.py) on Quandela Cloud."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to NSL_KDD_labeled.csv.")
    parser.add_argument("--n-photons", type=int, default=3, help="Number of photons.")
    parser.add_argument("--n-modes", type=int, default=20, help="Number of optical modes.")
    parser.add_argument(
        "--b-no-bunching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use unbunched output states (matches qorc_nslkdd.py's default).",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of NSL-KDD rows to send as a demo batch.")
    parser.add_argument(
        "--qpu-device",
        default="sim:ascella",
        choices=VALID_QPU_DEVICES,
        help="Quandela Cloud backend name.",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=1000,
        help="Shots per row (ignored by backends that return exact probabilities).",
    )
    parser.add_argument(
        "--local-quantum",
        action="store_true",
        help="Compute quantum features locally instead of submitting to Quandela Cloud. "
        "Useful for testing without a QUANDELA_TOKEN.",
    )
    parser.add_argument(
        "--resume-job-id",
        default="",
        help="Resume a previously submitted raw Quandela job by ID and print its raw "
        "results, instead of submitting a new batch.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to a timestamped folder under QRC/team/out/qorc_quandela_demo/.",
    )
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def load_demo_batch(csv_path: Path, limit: int, n_modes: int, logger: logging.Logger) -> np.ndarray:
    """Load a handful of NSL-KDD rows and reduce them to n_modes phase inputs.

    NOTE: this is a demo-only data prep path. PCA and min-max normalization are
    fit on the small demo batch itself, unlike qorc_nslkdd.py's full pipeline
    which fits them on a proper train split. Do not treat these numbers as a
    real experiment result.
    """
    logger.info("Loading demo batch from %s", csv_path)
    df = pd.read_csv(csv_path)
    if limit > 0:
        df = df.head(limit).copy()

    categorical_cols = ["protocol_type", "service", "flag"]
    target_col = "label"
    numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]

    encoder = make_one_hot_encoder()
    x_cat = encoder.fit_transform(df[categorical_cols])
    x_num = df[numeric_cols].to_numpy(dtype=np.float32)
    x_all = np.concatenate([x_num, x_cat], axis=1).astype(np.float32)

    n_components = min(n_modes, x_all.shape[0], x_all.shape[1])
    if n_components < n_modes:
        logger.warning(
            "Demo batch too small (%s rows, %s features) for a full %s-component PCA; "
            "using %s components and zero-padding the rest. Pass a larger --limit to avoid this.",
            x_all.shape[0], x_all.shape[1], n_modes, n_components,
        )
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x_all)
    if n_components < n_modes:
        pad = np.zeros((x_pca.shape[0], n_modes - n_components), dtype=np.float32)
        x_pca = np.concatenate([x_pca, pad], axis=1)

    epsilon = 1e-8
    x_min, x_max = x_pca.min(), x_pca.max()
    x_norm = (x_pca - x_min) / (x_max - x_min + epsilon)
    return x_norm.astype(np.float32)


def build_remote_processor(qpu_device: str) -> pcvl.RemoteProcessor:
    # 2. Configure Quandela Cloud auth. The token is read from the environment
    #    only -- never hardcode it in source (see quandele_run.py's leaked
    #    token for why this matters).
    token = os.environ.get("QUANDELA_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "QUANDELA_TOKEN environment variable is not set. Export your Quandela "
            "Cloud API token before running (never hardcode it in source), or pass "
            "--local-quantum to run fully offline without a token."
        )
    RemoteConfig.set_token(token)
    return pcvl.RemoteProcessor(qpu_device)


def submit_and_wait(
    quantum_layer,
    data_tensor: torch.Tensor,
    qpu_device: str,
    nsample: int,
    logger: logging.Logger,
) -> torch.Tensor:
    remote_processor = build_remote_processor(qpu_device)
    proc = MerlinProcessor(remote_processor, chunk_concurrency=4)
    quantum_layer.eval()

    logger.info("Submitting %s row(s) to Quandela Cloud backend '%s' ...", data_tensor.shape[0], qpu_device)
    fut = proc.forward_async(quantum_layer, data_tensor, nsample=nsample)

    seen_job_ids: set[str] = set()
    while not fut.done():
        new_ids = set(fut.job_ids) - seen_job_ids
        if new_ids:
            seen_job_ids |= new_ids
            logger.info("Job id(s) received: %s", sorted(new_ids))
        status = fut.status()
        logger.info(
            "Waiting... state=%s progress=%.0f%% chunks=%s/%s",
            status.get("state"),
            (status.get("progress") or 0.0) * 100,
            status.get("chunks_done"),
            status.get("chunks_total"),
        )
        time.sleep(2)

    result = fut.wait()
    logger.info("Job(s) complete: %s", sorted(seen_job_ids | set(fut.job_ids)))
    return result


def resume_remote_job(qpu_device: str, job_id: str, logger: logging.Logger) -> None:
    """Resume a previously submitted raw Quandela job and print its raw results.

    This talks directly to a single perceval RemoteJob (mirroring
    quandele_run.py's resume-by-job-id pattern) rather than going through
    MerlinProcessor, which may split a batch into several remote jobs.
    """
    remote_processor = build_remote_processor(qpu_device)
    job = remote_processor.resume_job(job_id)
    logger.info("Resuming job %s on backend %s ...", job_id, qpu_device)
    while not job.is_complete:
        time.sleep(1)
    results = job.get_results()
    logger.info("Resumed job raw results: %s", results.get("results", results))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("qorc_quandela_run")

    # 3. Build the exact QORC quantum circuit used by qorc_nslkdd.py.
    quantum_layer, qorc_output_size = create_qorc_quantum_layer(
        args.n_photons, args.n_modes, args.b_no_bunching, "cpu", logger,
    )
    logger.info(
        "QORC circuit ready: %s modes, %s photons -> %s quantum features/row",
        args.n_modes, args.n_photons, qorc_output_size,
    )

    if args.resume_job_id:
        resume_remote_job(args.qpu_device, args.resume_job_id, logger)
        return

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"NSL-KDD CSV not found: {csv_path}")

    out_dir = args.out or (SCRIPT_DIR / "out" / "qorc_quandela_demo" / dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Prepare a small demo batch of real NSL-KDD rows as phase inputs.
    phases = load_demo_batch(csv_path, args.limit, args.n_modes, logger)
    data_tensor = torch.tensor(phases, dtype=torch.float32)
    logger.info("Prepared demo batch: %s", tuple(data_tensor.shape))

    # 5. Compute quantum features, locally or via Quandela Cloud.
    if args.local_quantum:
        quantum_layer.eval()
        with torch.no_grad():
            result = quantum_layer(data_tensor)
        logger.info("Computed quantum features locally (no cloud submission).")
        backend_used = "local"
    else:
        result = submit_and_wait(quantum_layer, data_tensor, args.qpu_device, args.nsample, logger)
        backend_used = args.qpu_device

    # 6. Report and save results.
    result_np = result.detach().cpu().numpy()
    logger.info("Quantum feature tensor shape: %s", result_np.shape)
    logger.info("First row preview: %s", np.array2string(result_np[0], precision=4, max_line_width=120))

    features_path = out_dir / "quantum_features.npy"
    np.save(features_path, result_np)
    summary = {
        "csv_path": str(csv_path),
        "timestamp": dt.datetime.now().isoformat(),
        "n_photons": args.n_photons,
        "n_modes": args.n_modes,
        "b_no_bunching": args.b_no_bunching,
        "limit": args.limit,
        "backend": backend_used,
        "nsample": args.nsample,
        "qorc_output_size": qorc_output_size,
        "result_shape": list(result_np.shape),
        "features_path": str(features_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
