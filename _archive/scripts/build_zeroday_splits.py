"""
build_zeroday_splits.py
=======================
Gap 1 — Zero-day (held-out-attack) train/test splits of NSL-KDD.

Keeps the RAW data untouched (all 41 original features + label). No preprocessing,
no PCA — downstream users apply their own encoding. This only decides WHICH rows go
to train vs test so that each scenario holds one attack type entirely out of training.

Rules (leakage-free by construction):
  * The held-out attack rows appear ONLY in test.
  * Normal traffic is split 75/25 (train/test) so test has a benign baseline too.
  * All other ("seen") attacks go to train.
  * Adds a `category` column (5-class family) and `is_attack` (0/1) for convenience.

Outputs, per scenario, under artifacts/zeroday/<scenario>/:
  train.csv   test.csv     (raw features + label, category, is_attack)
Plus artifacts/zeroday/INDEX.json summarising every scenario.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_PATH = r"D:\Python\AI\Code\ReservoirComputing\NSL_KDD_labeled.csv"
OUT_ROOT = r"D:\Python\AI\Code\ReservoirComputing\artifacts\zeroday"
RNG_SEED = 42
NORMAL_TEST_FRAC = 0.25

ATTACK_MAP = {
    "normal": "normal",
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

SCENARIOS = {
    "smurf": {
        "held_out_labels": ["smurf"],
        "desc": "DoS variant 'smurf' held out of train; recognise a NEW variant of a "
                "family the model has otherwise seen.",
    },
    "probe_all": {
        "held_out_labels": [k for k, v in ATTACK_MAP.items() if v == "Probe"],
        "desc": "Entire Probe category held out; detect an attack CONCEPT (port-scanning) "
                "never seen in training. Headline scenario.",
    },
    "r2l_all": {
        "held_out_labels": [k for k, v in ATTACK_MAP.items() if v == "R2L"],
        "desc": "Entire R2L category held out; hard application-layer attacks, typically "
                "low recall — probes the limits.",
    },
    "neptune": {
        "held_out_labels": ["neptune"],
        "desc": "'neptune' (SYN flood, ~90% of DoS) held out; extreme generalisation test "
                "— train keeps almost no DoS.",
    },
}


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["category"] = df["label"].map(ATTACK_MAP)
    assert df["category"].isna().sum() == 0, "unmapped attack labels present"
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df.reset_index(drop=True)


def make_zeroday_indices(df, held_out_labels, rng):
    held = set(held_out_labels)
    is_normal = df["category"].eq("normal").values
    is_held = df["label"].isin(held).values
    is_seen_attack = (~is_normal) & (~is_held)

    normal_idx = np.where(is_normal)[0]
    n_test = int(round(len(normal_idx) * NORMAL_TEST_FRAC))
    perm = rng.permutation(normal_idx)
    normal_test_idx, normal_train_idx = perm[:n_test], perm[n_test:]

    train_idx = np.sort(np.concatenate([normal_train_idx, np.where(is_seen_attack)[0]]))
    test_idx = np.sort(np.concatenate([normal_test_idx, np.where(is_held)[0]]))
    return train_idx, test_idx


def build_scenario(name, spec, df):
    rng = np.random.default_rng(RNG_SEED)
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    train_idx, test_idx = make_zeroday_indices(df, spec["held_out_labels"], rng)

    # leakage guard: held-out labels must not appear in train
    leaked = set(df["label"].iloc[train_idx]) & set(spec["held_out_labels"])
    assert not leaked, f"LEAKAGE: held-out labels in train: {leaked}"

    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    held_in_test = int(test["label"].isin(spec["held_out_labels"]).sum())
    manifest = {
        "scenario": name,
        "description": spec["desc"],
        "task": "binary is_attack (novelty detection)",
        "held_out_labels": spec["held_out_labels"],
        "rng_seed": RNG_SEED,
        "format": "raw NSL-KDD features (41) + label, category, is_attack. No preprocessing.",
        "counts": {
            "train_total": int(len(train)),
            "train_normal": int((train["is_attack"] == 0).sum()),
            "train_attack": int((train["is_attack"] == 1).sum()),
            "test_total": int(len(test)),
            "test_normal": int((test["is_attack"] == 0).sum()),
            "test_attack_held_out": held_in_test,
        },
        "files": {"train": "train.csv", "test": "test.csv"},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{name}] train={len(train)} "
          f"(normal={manifest['counts']['train_normal']}, "
          f"attack={manifest['counts']['train_attack']}) | "
          f"test={len(test)} (held-out attack={held_in_test}, "
          f"normal={manifest['counts']['test_normal']})")
    return manifest


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    df = load_frame()
    print(f"loaded {len(df)} rows; building {len(SCENARIOS)} zero-day scenarios\n")

    index = {"seed": RNG_SEED, "source_csv": os.path.basename(DATA_PATH),
             "normal_test_frac": NORMAL_TEST_FRAC,
             "format": "raw features only (no preprocessing / no PCA)",
             "scenarios": {}}
    for name, spec in SCENARIOS.items():
        index["scenarios"][name] = build_scenario(name, spec, df)

    with open(os.path.join(OUT_ROOT, "INDEX.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote INDEX.json + {len(SCENARIOS)} scenario folders under\n  {OUT_ROOT}")


if __name__ == "__main__":
    main()
