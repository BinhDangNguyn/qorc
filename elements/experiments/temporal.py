"""Compare the three temporal strategies on standard time-series tasks.

Arms compared, all reading out through the same ridge regression so the only
difference is the state each produces:

``memoryless``  the current reservoir, fed one step at a time -- the control.
                If an arm cannot beat this, its "memory" is doing nothing.
``delay``       sliding-window input, reservoir unchanged.
``feedback``    previous readout mixed back into the phases.
``esn``         classical echo-state network, the reference point.
``linear``      ridge on the raw window, no reservoir at all. Catches the case
                where a task is simply linear and the reservoir is decorative.

Run:  python -m elements.experiments.temporal
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from elements.core.reservoir import Reservoir
from elements.core.temporal import DelayEmbedding, FeedbackReservoir, sliding_window
from elements.data.timeseries import mackey_glass, memory_capacity_task, narma10
from elements.heads.forecasting import ESN, RidgeReadout, memory_capacity, nrmse


def memoryless_states(
    reservoir: Reservoir, u: np.ndarray, n_modes: int, seed: int
) -> np.ndarray:
    """The current architecture: each step encoded independently.

    The scalar input is spread across the modes by a fixed random projection,
    NOT by repeating it into every mode. Repeating it drives all phase
    shifters identically and collapses the readout to rank 1, which would make
    the control look far weaker than the architecture really is -- the
    comparison has to be lost on the merits, not on a rigged baseline.
    """
    from elements.core.encoding import PhaseEncoder

    u = np.asarray(u, dtype=np.float32).reshape(len(u), -1)
    rng = np.random.default_rng(seed)
    W_in = rng.standard_normal((u.shape[1], n_modes)).astype(np.float32)
    return reservoir.transform(PhaseEncoder("tanh").fit_transform(u @ W_in))


def evaluate_states(
    states: np.ndarray,
    target: np.ndarray,
    washout: int,
    train_frac: float,
    alpha: float,
) -> float:
    """Fit a ridge readout on the train segment, score NRMSE on the test tail."""
    n = len(states)
    cut = int(n * train_frac)
    model = RidgeReadout(alpha).fit(states[washout:cut], target[washout:cut])
    return nrmse(target[cut:], model.predict(states[cut:]))


def run_task(
    name: str,
    u: np.ndarray,
    y: np.ndarray,
    n_modes: int,
    n_photons: int,
    window: int,
    leak: float,
    esn_size: int,
    alpha: float,
    washout: int,
    train_frac: float,
    seed: int,
    log=print,
) -> list[dict]:
    """Run every arm on one task and return one record per arm."""
    reservoir = Reservoir(n_modes=n_modes, n_photons=n_photons, seed=seed)
    rows: list[dict] = []

    def record(arm: str, states: np.ndarray, t_build: float, note: str = "") -> None:
        score = evaluate_states(states, y, washout, train_frac, alpha)
        rows.append(
            {
                "task": name,
                "arm": arm,
                "nrmse": score,
                "state_dim": int(states.shape[1]),
                "build_s": round(t_build, 2),
                "note": note,
            }
        )
        log(f"  {arm:12s} dim={states.shape[1]:5d} NRMSE={score:.4f} ({t_build:.1f}s) {note}")

    # -- control: the reservoir exactly as it is today, one step at a time ----
    t0 = time.time()
    states = memoryless_states(reservoir, u, n_modes, seed)
    record("memoryless", states, time.time() - t0, "current architecture")

    # -- strategy 1: delay embedding -----------------------------------------
    t0 = time.time()
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=n_modes, random_state=seed) if window > n_modes else None
    de = DelayEmbedding(reservoir, window=window, reducer=reducer)
    record("delay", de.fit_transform(u), time.time() - t0, f"window={window}")

    # -- strategy 2: classical feedback loop ---------------------------------
    t0 = time.time()
    fb = FeedbackReservoir(reservoir, leak=leak, seed=seed)
    record("feedback", fb.run(u), time.time() - t0, f"leak={leak}")

    # -- reference: classical ESN --------------------------------------------
    t0 = time.time()
    record("esn", ESN(n_reservoir=esn_size, seed=seed).run(u), time.time() - t0,
           f"classical, N={esn_size}")

    # -- floor: ridge on the raw window, no reservoir -------------------------
    t0 = time.time()
    record("linear", sliding_window(u, window), time.time() - t0, "no reservoir")

    return rows


def run_memory_capacity(
    n: int, max_lag: int, n_modes: int, n_photons: int, window: int,
    leak: float, esn_size: int, alpha: float, seed: int, log=print,
) -> list[dict]:
    """Measure how far back each arm can actually see."""
    u, Y = memory_capacity_task(n, max_lag=max_lag, seed=seed)
    reservoir = Reservoir(n_modes=n_modes, n_photons=n_photons, seed=seed)
    from sklearn.decomposition import PCA

    arms = {
        "memoryless": memoryless_states(reservoir, u, n_modes, seed),
        "delay": DelayEmbedding(
            reservoir, window=window,
            reducer=PCA(n_components=n_modes, random_state=seed) if window > n_modes else None,
        ).fit_transform(u),
        "feedback": FeedbackReservoir(reservoir, leak=leak, seed=seed).run(u),
        "esn": ESN(n_reservoir=esn_size, seed=seed).run(u),
    }

    rows = []
    for arm, states in arms.items():
        mc, curve = memory_capacity(states, Y, alpha=alpha)
        rows.append(
            {
                "task": "memory_capacity",
                "arm": arm,
                "memory_capacity": round(mc, 3),
                "state_dim": int(states.shape[1]),
                "curve": [round(float(c), 4) for c in curve],
            }
        )
        log(f"  {arm:12s} dim={states.shape[1]:5d} MC={mc:6.2f}  "
            f"lags1-5={np.round(curve[:5], 2)}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3000, help="series length")
    p.add_argument("--n-modes", type=int, default=12)
    p.add_argument("--n-photons", type=int, default=3)
    p.add_argument("--window", type=int, default=15, help="delay-embedding width")
    p.add_argument("--leak", type=float, default=0.3, help="feedback strength")
    p.add_argument("--esn-size", type=int, default=200)
    p.add_argument("--alpha", type=float, default=1e-6, help="ridge penalty")
    p.add_argument("--washout", type=int, default=100)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--max-lag", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("runs/benchmarks/temporal_benchmark.json"))
    args = p.parse_args()

    print(f"config: {args.n_photons}p{args.n_modes}m readout dim "
          f"{Reservoir(n_modes=args.n_modes, n_photons=args.n_photons).dim}, "
          f"n={args.n}, window={args.window}, leak={args.leak}")

    rows: list[dict] = []

    print("\n=== NARMA-10 (needs 10 steps of memory; the decisive task) ===")
    u, y = narma10(args.n, seed=args.seed)
    rows += run_task("narma10", u, y, args.n_modes, args.n_photons, args.window,
                     args.leak, args.esn_size, args.alpha, args.washout,
                     args.train_frac, args.seed)

    print("\n=== Mackey-Glass (predict 1 step ahead) ===")
    mg = mackey_glass(args.n + 1, seed=args.seed)
    rows += run_task("mackey_glass", mg[:-1], mg[1:], args.n_modes, args.n_photons,
                     args.window, args.leak, args.esn_size, args.alpha,
                     args.washout, args.train_frac, args.seed)

    print(f"\n=== Linear memory capacity (max_lag={args.max_lag}) ===")
    rows += run_memory_capacity(args.n, args.max_lag, args.n_modes, args.n_photons,
                                args.window, args.leak, args.esn_size, args.alpha,
                                args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
