"""Does the trapped-ion reservoir's quantum memory carry the past? Time series.

NSL-KDD rows have no time axis, so they cannot answer this. Here a scalar
series u_t is injected one round at a time into ONE long circuit: round t
encodes u_t on the input qubits, evolves, then measures and resets the inputs
only. The memory qubits are never measured, so if anything of u_{t-k} survives
to round t, it survives in them.

Arms, all read out by the same ridge regression:

``ion``        4 input + 4 memory ions, all-to-all; state = what hardware measures
               each round (input <Z> + same-round <Z_i Z_j>, 10 numbers)
``ion_nomem``  the same with the memory register reset every round -- the control.
               ion - ion_nomem is what the quantum memory is worth.
``ion_V4``     each input held for V = 4 rounds (temporal multiplexing), state
               = the 4 rounds' readouts, 40 numbers
``ion62``      6 input + 2 memory ions: more of the register is read every round
``ion_oracle`` ``ion`` plus the memory qubits' <Z> after each round. Simulation
               ONLY (hardware would rerun the circuit up to t to read it); shows
               how much memory exists but is not exposed through the inputs
``ion12``      4 input + 8 memory ions on a ring (exact, 12 qubits)
``esn<N>``     classical echo-state networks: N = 10 / 40 match the ion state
               widths, N = 200 is the usual reference
``linear``     ridge on the raw window of the last 15 inputs

Every arm's knobs (ion: dt, input scale and memory leak; ESN: spectral radius and input
scale; all: ridge alpha) are chosen on a validation block inside the training
segment, then refit and scored once on the held-out tail. Tasks and data are
the ones in ``elements.experiments.temporal``, so the photonic numbers in
``runs/benchmarks/temporal_benchmark.json`` are directly comparable.

Run:  python -m elements.experiments.iontrap_temporal
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from elements.core.iontrap import IonTrapReservoir, spread_sites
from elements.core.temporal import sliding_window
from elements.data.timeseries import mackey_glass, memory_capacity_task, narma10
from elements.heads.forecasting import ESN, RidgeReadout, memory_capacity, nrmse

ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2)
ION_DT = (0.2, 0.8)
ION_SCALE = (0.25, 0.5, 1.0)
# Engineered dissipation of the memory (IonTrapReservoir.leak). Without it the
# memory "heats up": fresh random input kicks every round drive it toward the
# maximally mixed state, where it no longer responds to recent inputs, and the
# measured memory capacity is ~0 for every dt and scale (first run of this
# benchmark). 0 stays in the grid so the tuner can still pick "no leak".
ION_LEAK = (0.0, 0.1, 0.3, 0.6)
ESN_RHO = (0.5, 0.9, 1.0, 1.2)
ESN_SCALE = (0.1, 0.5, 1.0)


# ------------------------------------------------------------------- the states


def ion_angles(u: np.ndarray, res: IonTrapReservoir, scale: float, seed: int,
               fit_upto: int) -> np.ndarray:
    """Scalar series -> (T, uploads, n_enc, 2) half-turn angles.

    A fixed random projection spreads u_t over every encoded angle (not the
    same angle on every qubit, which would collapse the readout), then the same
    tanh squash as the static encoder. Standardised with statistics from the
    training segment only.
    """
    u = np.asarray(u, dtype=np.float64).reshape(len(u), -1)
    mu, sd = u[:fit_upto].mean(0), u[:fit_upto].std(0) + 1e-12
    z = (u - mu) / sd
    rng = np.random.default_rng(seed + 777)
    shape = (res.uploads, res.n_enc, 2)
    W = rng.standard_normal((z.shape[1], int(np.prod(shape)))) * scale
    b = rng.uniform(-0.5, 0.5, int(np.prod(shape)))
    a = 0.5 * (1 + np.tanh(z @ W + b))  # (0, 1)
    a = a.reshape(len(u), *shape)
    a[..., 1] *= 2.0  # longitude spans (0, 2 pi)
    return a


def ion_states(u, *, n_in, n_mem, topology, dt, scale, memory, substeps, oracle,
               seed, fit_upto, device, leak=0.0) -> np.ndarray:
    sites = spread_sites(n_in + n_mem, n_in) if topology == "ring" else None
    res = IonTrapReservoir(n_in, n_mem, 1, dt=dt, topology=topology, in_sites=sites,
                           seed=seed, memory=memory, leak=leak, device=device)
    a = ion_angles(u, res, scale, seed, fit_upto)
    a = np.repeat(a, substeps, axis=0)  # hold each input for `substeps` rounds
    f = res.series(a, mem_readout=oracle)
    return f.reshape(len(u), -1)


# ------------------------------------------------------------------ evaluation


def _val_score(states, target, washout, cut, kind):
    """Score on a validation block: the last quarter of the training segment."""
    vcut = washout + int((cut - washout) * 0.75)
    if kind == "mc":
        mc, _ = memory_capacity(states[washout:cut], target[washout:cut])
        return -mc
    best = np.inf
    for alpha in ALPHAS:
        m = RidgeReadout(alpha).fit(states[washout:vcut], target[washout:vcut])
        best = min(best, nrmse(target[vcut:cut], m.predict(states[vcut:cut])))
    return best


def _test_score(states, target, washout, cut, kind):
    if kind == "mc":
        mc, curve = memory_capacity(states[washout:], target[washout:])
        return {"memory_capacity": round(mc, 3), "curve": [round(float(c), 4) for c in curve]}
    vcut = washout + int((cut - washout) * 0.75)
    alpha = min(ALPHAS, key=lambda a: nrmse(
        target[vcut:cut],
        RidgeReadout(a).fit(states[washout:vcut], target[washout:vcut])
        .predict(states[vcut:cut])))
    m = RidgeReadout(alpha).fit(states[washout:cut], target[washout:cut])
    return {"nrmse": nrmse(target[cut:], m.predict(states[cut:])), "alpha": alpha}


def tuned(name, build, grid, u, target, washout, cut, kind, log):
    """Pick the grid point with the best validation score, report its test score."""
    t0 = time.time()
    best = None
    for params in grid:
        states = build(**params)
        v = _val_score(states, target, washout, cut, kind)
        if best is None or v < best[0]:
            best = (v, params, states)
    _, params, states = best
    rec = {"arm": name, "state_dim": int(states.shape[1]), "params": params,
           **_test_score(states, target, washout, cut, kind),
           "tune_s": round(time.time() - t0, 1)}
    score = (f"MC={rec['memory_capacity']:6.2f}" if kind == "mc"
             else f"NRMSE={rec['nrmse']:.4f}")
    log(f"  {name:12s} dim={rec['state_dim']:4d} {score}  {params}  "
        f"({rec['tune_s']:.0f}s)")
    return rec


def run_task(task, u, target, kind, args, log=print) -> list[dict]:
    cut, washout, seed, dev = int(len(u) * args.train_frac), args.washout, args.seed, args.device
    nomem_grid = [dict(dt=d, scale=s) for d, s in itertools.product(ION_DT, ION_SCALE)]
    ion_grid = [dict(dt=d, scale=s, leak=g)
                for d, s, g in itertools.product(ION_DT, ION_SCALE, ION_LEAK)]

    def ion(n_in=4, n_mem=4, topology="all_to_all", memory=True, substeps=1, oracle=False):
        # one sequence at a time: 8 ions are kernel-launch bound, faster on CPU
        d = "cpu" if dev == "auto" and n_in + n_mem <= 8 else (
            "cuda" if dev == "auto" else dev)
        return lambda dt, scale, leak=0.0: ion_states(
            u, n_in=n_in, n_mem=n_mem, topology=topology, dt=dt, scale=scale,
            memory=memory, substeps=substeps, oracle=oracle, seed=seed, fit_upto=cut,
            device=d, leak=leak)

    arms = [
        ("ion", ion(), ion_grid),
        ("ion_nomem", ion(memory=False), nomem_grid),
        ("ion_V4", ion(substeps=4), ion_grid),
        ("ion_oracle", ion(oracle=True), ion_grid),
        ("ion62", ion(n_in=6, n_mem=2), ion_grid),
        ("ion62_nomem", ion(n_in=6, n_mem=2, memory=False), nomem_grid),
        ("ion62_V4", ion(n_in=6, n_mem=2, substeps=4), ion_grid),
        ("ion62_V4_nomem", ion(n_in=6, n_mem=2, substeps=4, memory=False), nomem_grid),
    ]
    if args.ion12:
        arms += [("ion12", ion(n_mem=8, topology="ring"), ion_grid),
                 ("ion12_nomem", ion(n_mem=8, topology="ring", memory=False), nomem_grid)]
    esn_grid = [dict(rho=r, scale=s) for r, s in itertools.product(ESN_RHO, ESN_SCALE)]
    for n in (10, 40, 200):
        arms.append((f"esn{n}", lambda rho, scale, n=n: ESN(
            n_reservoir=n, spectral_radius=rho, input_scale=scale, seed=seed).run(u),
            esn_grid))
    arms.append(("linear", lambda: sliding_window(u, 15), [{}]))

    rows = []
    for name, build, grid in arms:
        rows.append({"task": task, **tuned(name, build, grid, u, target, washout, cut,
                                            kind, log)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--washout", type=int, default=100)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--max-lag", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ion12", action="store_true", help="also run the 12-ion ring")
    p.add_argument("--device", default="auto" if torch.cuda.is_available() else "cpu",
                   help="auto: CPU for 8 ions, GPU for the 12-ion ring")
    p.add_argument("--out", type=Path, default=Path("runs/benchmarks/iontrap_temporal.json"))
    args = p.parse_args()

    rows: list[dict] = []
    print("=== NARMA-10 ===")
    u, y = narma10(args.n, seed=args.seed)
    rows += run_task("narma10", u, y, "nrmse", args)

    print("\n=== Mackey-Glass, 1 step ahead ===")
    mg = mackey_glass(args.n + 1, seed=args.seed)
    rows += run_task("mackey_glass", mg[:-1], mg[1:], "nrmse", args)

    print(f"\n=== Linear memory capacity (max_lag={args.max_lag}) ===")
    u, Y = memory_capacity_task(args.n, max_lag=args.max_lag, seed=args.seed)
    rows += run_task("memory_capacity", u, Y, "mc", args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
