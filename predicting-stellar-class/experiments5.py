"""Experiment 05 for Predicting Stellar Class: full-scale tuning and seed bagging.

Experiments 01-04 ran on a subsample with a fast model and tested added features
and levers. This one drops that shortcut: it trains on the full training set and
asks whether full-scale effects the screen could not see (more trees with early
stopping, seed bagging to cut variance, one alternative hyperparameter set) beat
the first submission's out-of-fold balanced accuracy of 0.9640.

Comparison is apples-to-apples with baseline.py: same 5-fold StratifiedKFold
(seed 42), same early-stopping-on-fold setup, out-of-fold balanced accuracy.

Run from the predicting-stellar-class/ folder, in the project .venv:
    python experiments5.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TARGET = "class"
ID = "id"
CATEG = ["spectral_type", "galaxy_population"]
BASE_NUM = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
N_FOLDS = 5

CFG_A = dict(objective="multiclass", n_estimators=3000, learning_rate=0.05, num_leaves=63,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
             class_weight="balanced", n_jobs=-1, verbosity=-1)
CFG_B = dict(objective="multiclass", n_estimators=5000, learning_rate=0.02, num_leaves=31,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_child_samples=100,
             reg_lambda=2.0, class_weight="balanced", n_jobs=-1, verbosity=-1)


def load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    classes = sorted(train[TARGET].unique())
    y = train[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = train[BASE_NUM + CATEG].copy()
    Xt = test[BASE_NUM + CATEG].copy()
    for c in CATEG:
        cats = pd.Index(pd.concat([X[c], Xt[c]]).dropna().unique())
        X[c] = pd.Categorical(X[c], categories=cats)
        Xt[c] = pd.Categorical(Xt[c], categories=cats)
    return X, y, Xt, test[ID], classes


def cv_oof(X, y, Xt, folds, params, seeds):
    """Seed-bagged OOF and test probabilities: per fold, average over seeds."""
    oof = np.zeros((len(X), 3))
    test_p = np.zeros((len(Xt), 3))
    for tr, va in folds:
        for s in seeds:
            m = lgb.LGBMClassifier(**params, random_state=s)
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                  eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof[va] += m.predict_proba(X.iloc[va]) / len(seeds)
            test_p += m.predict_proba(Xt) / (len(seeds) * N_FOLDS)
    return oof, test_p


def main():
    if not (DATA_DIR / "train.csv").exists():
        sys.exit(f"missing data in {DATA_DIR}. Download it first.")
    X, y, Xt, test_id, classes = load()
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(X, y))
    print(f"full train {len(X)} rows, test {len(Xt)} rows, classes {classes}")

    results = {}

    t0 = time.time()
    oof_a1, _ = cv_oof(X, y, Xt, folds, CFG_A, seeds=[42])
    results["A single seed (cfg A)"] = balanced_accuracy_score(y, oof_a1.argmax(1))
    print(f"A single: {results['A single seed (cfg A)']:.5f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    oof_a3, test_a3 = cv_oof(X, y, Xt, folds, CFG_A, seeds=[42, 1, 7])
    results["A 3-seed bag (cfg A)"] = balanced_accuracy_score(y, oof_a3.argmax(1))
    print(f"A 3-seed bag: {results['A 3-seed bag (cfg A)']:.5f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    oof_b1, _ = cv_oof(X, y, Xt, folds, CFG_B, seeds=[42])
    results["B single seed (cfg B, tuned)"] = balanced_accuracy_score(y, oof_b1.argmax(1))
    print(f"B single: {results['B single seed (cfg B, tuned)']:.5f} ({time.time()-t0:.0f}s)")

    # best test predictions -> a submission file (not submitted)
    best_name = max(results, key=results.get)
    print(f"\nbest config: {best_name} at {results[best_name]:.5f}")
    int_to_class = {i: c for i, c in enumerate(classes)}
    sub = pd.DataFrame({ID: test_id, TARGET: [int_to_class[i] for i in test_a3.argmax(1)]})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(RESULTS_DIR / "submission_seedbag.csv", index=False)

    print("\n=== full-scale OOF balanced accuracy ===")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {v:.5f}   (delta vs first attempt 0.9640: {v-0.9640:+.5f})")
    print("\nfirst submission was OOF 0.9640, public LB 0.96523.")
    print(f"wrote results/submission_seedbag.csv from the 3-seed bag (NOT submitted).")


if __name__ == "__main__":
    main()
