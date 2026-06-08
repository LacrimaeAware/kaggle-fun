"""Experiment 04 for Predicting Stellar Class: model diversity.

The last standard lever not yet tested. Trains a CatBoost model alongside the
LightGBM baseline on the same folds, and blends their out-of-fold class
probabilities, to see whether a second model type adds anything over the
LightGBM baseline on balanced accuracy.

Run from the predicting-stellar-class/ folder, in the project .venv:
    python experiments4.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TARGET = "class"
CATEG = ["spectral_type", "galaxy_population"]
BASE_NUM = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
SEED = 42
SUBSAMPLE = 200_000
N_FOLDS = 5

LGB_PARAMS = dict(
    objective="multiclass", n_estimators=350, learning_rate=0.05, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
    class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1,
)
CAT_PARAMS = dict(
    iterations=400, learning_rate=0.05, depth=8, loss_function="MultiClass",
    auto_class_weights="Balanced", random_seed=SEED, verbose=False, thread_count=-1,
)


def load_subsample():
    train = pd.read_csv(DATA_DIR / "train.csv")
    if len(train) > SUBSAMPLE:
        train, _ = train_test_split(
            train, train_size=SUBSAMPLE, stratify=train[TARGET], random_state=SEED)
    train = train.reset_index(drop=True)
    classes = sorted(train[TARGET].unique())
    y = train[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = train[BASE_NUM + CATEG].copy()
    return X, y, classes


def main():
    if not (DATA_DIR / "train.csv").exists():
        sys.exit(f"missing data in {DATA_DIR}. Download it first.")
    X, y, classes = load_subsample()
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    print(f"subsample {len(X)} rows, classes {classes}")

    # views: LightGBM uses category dtype; CatBoost uses string categoricals + cat_features
    X_lgb = X.copy()
    for c in CATEG:
        X_lgb[c] = X_lgb[c].astype("category")
    X_cat = X.copy()
    for c in CATEG:
        X_cat[c] = X_cat[c].astype(str)
    cat_idx = [X_cat.columns.get_loc(c) for c in CATEG]

    oof_lgb = np.zeros((len(X), 3))
    oof_cat = np.zeros((len(X), 3))

    t0 = time.time()
    for tr, va in folds:
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(X_lgb.iloc[tr], y[tr])
        oof_lgb[va] = m.predict_proba(X_lgb.iloc[va])
    print(f"lightgbm done ({time.time()-t0:.0f}s)")

    t0 = time.time()
    for tr, va in folds:
        m = CatBoostClassifier(**CAT_PARAMS)
        m.fit(X_cat.iloc[tr], y[tr], cat_features=cat_idx)
        oof_cat[va] = m.predict_proba(X_cat.iloc[va])
    print(f"catboost done ({time.time()-t0:.0f}s)")

    ba_lgb = balanced_accuracy_score(y, oof_lgb.argmax(1))
    ba_cat = balanced_accuracy_score(y, oof_cat.argmax(1))

    # search the blend weight on a grid, report the best
    best = (0.5, -1.0)
    for w in np.linspace(0, 1, 21):
        ba = balanced_accuracy_score(y, (w * oof_lgb + (1 - w) * oof_cat).argmax(1))
        if ba > best[1]:
            best = (round(float(w), 2), ba)
    w_best, ba_blend = best
    ba_blend_even = balanced_accuracy_score(y, (0.5 * oof_lgb + 0.5 * oof_cat).argmax(1))

    rows = [
        {"model": "lightgbm (baseline)", "bal_acc": round(ba_lgb, 5), "delta_vs_lgb": 0.0},
        {"model": "catboost", "bal_acc": round(ba_cat, 5), "delta_vs_lgb": round(ba_cat - ba_lgb, 5)},
        {"model": "blend 0.5/0.5", "bal_acc": round(ba_blend_even, 5), "delta_vs_lgb": round(ba_blend_even - ba_lgb, 5)},
        {"model": f"blend best (lgb weight {w_best})", "bal_acc": round(ba_blend, 5), "delta_vs_lgb": round(ba_blend - ba_lgb, 5)},
    ]
    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "experiments4_blend.csv", index=False)
    print("\n=== model diversity ===")
    print(res.to_string(index=False))
    print("\nnoise floor from experiment 02: 0.0002. A delta below it is within noise.")


if __name__ == "__main__":
    main()
