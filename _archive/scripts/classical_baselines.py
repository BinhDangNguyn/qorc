"""Classical baselines for both QORC heads, on exactly the same splits.

Novelty head  : normal-only training, threshold calibrated on normal only.
Supervised head: closed-set family classification, same stratified split.

Writes classical_baselines.json next to the notebooks.
"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
SEED, TEST_SIZE = 42, 0.30
OUT = Path(r"D:\Python\AI\Code\ReservoirComputing\QRC\team\classical_baselines.json")

# ----------------------------------------------------------------- data
CSV = r"D:\Python\AI\Code\ReservoirComputing\NSL_KDD_labeled.csv"
df = pd.read_csv(CSV)
cat = ["protocol_type", "service", "flag"]
num = [c for c in df.columns if c not in cat + ["label"]]
X = np.concatenate([df[num].to_numpy(np.float32),
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    .fit_transform(df[cat]).astype(np.float32)], axis=1)
raw = df.label.to_numpy()
is_norm = raw == "normal"

FAMILY_OF = {}
for fam, ls in {
    "DoS": ["neptune", "smurf", "back", "teardrop", "pod", "land"],
    "Probe": ["satan", "ipsweep", "portsweep", "nmap"],
    "R2L": ["warezclient", "guess_passwd", "warezmaster", "imap", "ftp_write",
            "multihop", "phf", "spy"],
    "U2R": ["buffer_overflow", "rootkit", "loadmodule", "perl"],
}.items():
    for l in ls:
        FAMILY_OF[l] = fam
GROUP = ["normal", "DoS", "Probe", "R2L", "U2R"]
y_fam = np.array([GROUP.index("normal" if n == "normal" else FAMILY_OF[n])
                  for n in raw], dtype=np.int64)

results = {"novelty": {}, "supervised": {}}

# ================================================================= NOVELTY
# Same protocol as qorc_zeroday_sweep_novelty.ipynb: train on 70% of normal,
# score the rest of normal + every attack row. Threshold set at the 99th
# percentile of TRAIN-normal scores -- calibrated without ever seeing an attack.
print("=" * 70, "\nNOVELTY BASELINES (normal-only training)\n", "=" * 70)
nidx = np.where(is_norm)[0]
aidx = np.where(~is_norm)[0]
tr, te = train_test_split(nidx, test_size=TEST_SIZE, random_state=SEED)

sc = StandardScaler().fit(X[tr])
A, B, C = sc.transform(X[tr]), sc.transform(X[te]), sc.transform(X[aidx])
y_true = np.r_[np.zeros(len(B)), np.ones(len(C))]      # 0 normal, 1 attack

FA = 0.01     # 1% false-alarm budget, fixed on train-normal


def nov_acc(name, s_tr, s_te, s_atk, elapsed):
    """Accuracy at a threshold calibrated on TRAIN-normal only."""
    thr = np.quantile(s_tr, 1 - FA)
    pred = np.r_[(s_te > thr).astype(int), (s_atk > thr).astype(int)]
    acc = accuracy_score(y_true, pred)
    bal = balanced_accuracy_score(y_true, pred)
    det = float((s_atk > thr).mean())
    far = float((s_te > thr).mean())
    results["novelty"][name] = {"accuracy": float(acc), "balanced_acc": float(bal),
                                "detection": det, "false_alarm": far,
                                "time_s": elapsed}
    print(f"  {name:22s} acc={acc:.4f} bal={bal:.4f} det={det:.4f} "
          f"fa={far:.4f} ({elapsed:.1f}s)")


t = time.time()
iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(A)
nov_acc("Isolation Forest", -iso.score_samples(A), -iso.score_samples(B),
        -iso.score_samples(C), time.time() - t)

t = time.time()
mu = A.mean(0)
cov = EmpiricalCovariance().fit(A)
nov_acc("Mahalanobis (full cov)", cov.mahalanobis(A), cov.mahalanobis(B),
        cov.mahalanobis(C), time.time() - t)

t = time.time()
p = PCA(n_components=32, random_state=SEED).fit(A)


def pca_recon(M):
    return np.sqrt(((M - p.inverse_transform(p.transform(M))) ** 2).sum(1))


nov_acc("PCA reconstruction", pca_recon(A), pca_recon(B), pca_recon(C),
        time.time() - t)

# One-Class SVM on a subsample -- full 47k x 122 is too slow to be useful
t = time.time()
sub = np.random.default_rng(SEED).choice(len(A), 6000, replace=False)
oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(A[sub])
nov_acc("One-Class SVM (6k sub)", -oc.decision_function(A),
        -oc.decision_function(B), -oc.decision_function(C), time.time() - t)

# ============================================================== SUPERVISED
# Closed-set family classification on the same stratified split.
print("\n" + "=" * 70, "\nSUPERVISED BASELINES (5-class family)\n", "=" * 70)
itr, ite = train_test_split(np.arange(len(y_fam)), test_size=TEST_SIZE,
                            random_state=SEED, stratify=y_fam)
sc2 = StandardScaler().fit(X[itr])
Xtr, Xte = sc2.transform(X[itr]), sc2.transform(X[ite])
ytr, yte = y_fam[itr], y_fam[ite]


def sup_acc(name, model, elapsed=None):
    t0 = time.time()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    el = time.time() - t0 if elapsed is None else elapsed
    acc = accuracy_score(yte, pred)
    bal = balanced_accuracy_score(yte, pred)
    # mean per-ATTACK-family accuracy (recall), excluding normal -- this is
    # "average accuracy of the attacks it got right" for the 4 attack families
    recs = [float((pred[yte == k] == k).mean()) for k in range(1, 5)]
    results["supervised"][name] = {
        "accuracy": float(acc), "balanced_acc": float(bal),
        "attack_family_recall": dict(zip(GROUP[1:], recs)),
        "mean_attack_recall": float(np.mean(recs)), "time_s": el}
    print(f"  {name:22s} acc={acc:.4f} bal={bal:.4f} "
          f"mean_attack_rec={np.mean(recs):.4f} ({el:.1f}s)")


sup_acc("Logistic Regression", LogisticRegression(max_iter=1000,
                                                  class_weight="balanced"))
sup_acc("Decision Tree", DecisionTreeClassifier(random_state=SEED,
                                                class_weight="balanced"))
sup_acc("Random Forest", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                random_state=SEED,
                                                class_weight="balanced"))
sup_acc("Linear SVM", LinearSVC(class_weight="balanced", max_iter=3000))
sup_acc("k-NN (k=5)", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
sup_acc("Gaussian Naive Bayes", GaussianNB())
sup_acc("MLP (100,)", MLPClassifier(hidden_layer_sizes=(100,), max_iter=120,
                                    random_state=SEED))

OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nwritten -> {OUT}")
