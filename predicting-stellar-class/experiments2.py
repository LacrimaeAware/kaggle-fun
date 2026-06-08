"""Experiment 02 for Predicting Stellar Class: levers beyond features.

Experiment 01 found no feature transform of the raw columns beats the baseline
above the fold noise. This batch tests the levers 01 did not:

1. A multi-seed baseline, to estimate the seed-to-seed noise floor (what change
   counts as real).
2. Model capacity (more trees, more leaves).
3. Leak-free kNN neighbor features in standardized color-redshift space (local
   density and local class fractions from training-fold labels only). This adds
   information from the joint distribution, not a row-wise function of the inputs.
4. Per-class threshold tuning aligned to balanced accuracy, fit on one out-of-fold
   split and evaluated on a disjoint one, so the reported gain is not optimistic.

Run from the predicting-stellar-class/ folder, in the project .venv:
    python experiments2.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TARGET = "class"
CATEG = ["spectral_type", "galaxy_population"]
BANDS = ["u", "g", "r", "i", "z"]
BASE_NUM = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
SEED = 42
SUBSAMPLE = 200_000
N_FOLDS = 5

PARAMS = dict(
    objective="multiclass", n_estimators=350, learning_rate=0.05, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
    class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1,
)
PARAMS_BIG = dict(PARAMS, n_estimators=1500, learning_rate=0.03, num_leaves=255)


def load_subsample():
    train = pd.read_csv(DATA_DIR / "train.csv")
    if len(train) > SUBSAMPLE:
        train, _ = train_test_split(
            train, train_size=SUBSAMPLE, stratify=train[TARGET], random_state=SEED)
    train = train.reset_index(drop=True)
    classes = sorted(train[TARGET].unique())
    y = train[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = train[BASE_NUM + CATEG].copy()
    for c in CATEG:
        X[c] = X[c].astype("category")
    return X, y, classes


def lgbm_oof(X, y, folds, params):
    """Return out-of-fold class probabilities (n, 3)."""
    proba = np.zeros((len(X), 3))
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
        proba[va] = m.predict_proba(X.iloc[va])
    return proba


def knn_oof(X, y, folds, k=50):
    """Baseline features plus leak-free kNN neighbor features."""
    nn_cols = np.column_stack([
        X["u"] - X["g"], X["g"] - X["r"], X["r"] - X["i"], X["i"] - X["z"],
        X["redshift"].to_numpy(),
    ])
    proba = np.zeros((len(X), 3))
    for tr, va in folds:
        mu, sd = nn_cols[tr].mean(0), nn_cols[tr].std(0) + 1e-9
        Ztr, Zva = (nn_cols[tr] - mu) / sd, (nn_cols[va] - mu) / sd
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(Ztr)
        # training rows: drop self (first neighbor)
        d_tr, idx_tr = nn.kneighbors(Ztr)
        d_tr, idx_tr = d_tr[:, 1:], idx_tr[:, 1:]
        d_va, idx_va = nn.kneighbors(Zva, n_neighbors=k)
        ytr = y[tr]

        def feats(dist, idx):
            lab = ytr[idx]  # (n, k)
            fr = np.stack([(lab == c).mean(1) for c in range(3)], axis=1)
            return np.column_stack([dist.mean(1), dist[:, 0], fr])

        cols = ["knn_meand", "knn_d1", "knn_f0", "knn_f1", "knn_f2"]
        Xtr = pd.concat([X.iloc[tr].reset_index(drop=True),
                         pd.DataFrame(feats(d_tr, idx_tr), columns=cols)], axis=1)
        Xva = pd.concat([X.iloc[va].reset_index(drop=True),
                         pd.DataFrame(feats(d_va, idx_va), columns=cols)], axis=1)
        m = lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr)
        proba[va] = m.predict_proba(Xva)
    return proba


def optimize_thresholds(proba, y, seed=0):
    """Fit per-class weights on one stratified half of the OOF rows, evaluate on
    the disjoint half. Returns (eval baseline, eval optimized)."""
    idx = np.arange(len(y))
    fit_idx, ev_idx = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)

    def neg_balacc(p):
        w = np.exp(np.array([p[0], p[1], 0.0]))
        return -balanced_accuracy_score(y[fit_idx], (proba[fit_idx] * w).argmax(1))

    res = minimize(neg_balacc, x0=[0.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 500})
    w = np.exp(np.array([res.x[0], res.x[1], 0.0]))
    base_ev = balanced_accuracy_score(y[ev_idx], proba[ev_idx].argmax(1))
    opt_ev = balanced_accuracy_score(y[ev_idx], (proba[ev_idx] * w).argmax(1))
    return base_ev, opt_ev, w


def main():
    if not (DATA_DIR / "train.csv").exists():
        sys.exit(f"missing data in {DATA_DIR}. Download it first.")
    X, y, classes = load_subsample()
    print(f"subsample {len(X)} rows, classes {classes}")

    # 1. multi-seed baseline -> noise floor
    seed_scores = []
    for s in (42, 1, 7):
        folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(X, y))
        ba = balanced_accuracy_score(y, lgbm_oof(X, y, folds, PARAMS).argmax(1))
        seed_scores.append(ba)
        print(f"baseline seed {s}: bal_acc {ba:.4f}")
    base_mean, base_std = float(np.mean(seed_scores)), float(np.std(seed_scores))
    print(f"baseline across seeds: mean {base_mean:.4f}, std {base_std:.4f} (noise floor)")

    canon = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    base_proba = lgbm_oof(X, y, canon, PARAMS)
    base_canon = balanced_accuracy_score(y, base_proba.argmax(1))

    rows = [{"lever": "baseline (canonical folds)", "bal_acc": base_canon,
             "delta_vs_base": 0.0}]

    # 2. capacity
    t0 = time.time()
    big = balanced_accuracy_score(y, lgbm_oof(X, y, canon, PARAMS_BIG).argmax(1))
    rows.append({"lever": "capacity (1500 trees, lr0.03, 255 leaves)",
                 "bal_acc": big, "delta_vs_base": big - base_canon})
    print(f"capacity: bal_acc {big:.4f} ({time.time()-t0:.0f}s)")

    # 3. kNN neighbor features
    t0 = time.time()
    knn = balanced_accuracy_score(y, knn_oof(X, y, canon, k=50).argmax(1))
    rows.append({"lever": "knn neighbor features (k=50, leak-free)",
                 "bal_acc": knn, "delta_vs_base": knn - base_canon})
    print(f"knn: bal_acc {knn:.4f} ({time.time()-t0:.0f}s)")

    # 4. threshold tuning on baseline OOF (fit/eval disjoint)
    base_ev, opt_ev, w = optimize_thresholds(base_proba, y, seed=0)
    rows.append({"lever": "per-class thresholds (eval split, vs argmax on same split)",
                 "bal_acc": opt_ev, "delta_vs_base": opt_ev - base_ev})
    print(f"thresholds: eval argmax {base_ev:.4f} -> optimized {opt_ev:.4f} "
          f"(delta {opt_ev-base_ev:+.4f}), weights {np.round(w,3)}")

    res = pd.DataFrame(rows)
    res["bal_acc"] = res["bal_acc"].round(5)
    res["delta_vs_base"] = res["delta_vs_base"].round(5)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "experiments2_levers.csv", index=False)
    print("\n=== levers ===")
    print(res.to_string(index=False))
    print(f"\nnoise floor (baseline seed std): {base_std:.4f}. "
          f"A delta below this is within noise.")


if __name__ == "__main__":
    main()
