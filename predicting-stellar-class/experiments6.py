"""Experiment 06: stacked ensemble with threshold calibration.

The public top-of-leaderboard recipe (per the forum variance analysis): a diverse
base-model zoo, a balanced-logistic meta-model stacked on the out-of-fold class
log-probabilities, then per-class threshold calibration by differential evolution,
all evaluated under honest nested cross-validation (the meta-model and the
thresholds never see the rows they are scored on).

This builds a GBDT/tree zoo (two LightGBMs, XGBoost, CatBoost,
HistGradientBoosting). It omits the neural-tabular models (RealMLP, TabM) the very
top uses, which need a GPU, so it is expected to land somewhat below the ~0.9697
stacked ceiling but well above the single-LightGBM first attempt (OOF 0.9640).

Run from predicting-stellar-class/ in the project .venv:
    python experiments6.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TARGET = "class"
ID = "id"
CATEG = ["spectral_type", "galaxy_population"]
BASE_NUM = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CLASSES = ["GALAXY", "QSO", "STAR"]
SEED = 42
NF = 5


def load():
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    classes = sorted(tr[TARGET].unique())
    y = tr[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    # ordinal-encode the two low-cardinality categoricals into a single numeric matrix
    feats = BASE_NUM + CATEG
    X = tr[feats].copy()
    Xt = te[feats].copy()
    for c in CATEG:
        cats = pd.Index(pd.concat([X[c], Xt[c]]).dropna().unique())
        X[c] = pd.Categorical(X[c], categories=cats).codes
        Xt[c] = pd.Categorical(Xt[c], categories=cats).codes
    return X.to_numpy(float), y, Xt.to_numpy(float), te[ID], classes


def base_models():
    sw = None  # set per-fit for xgboost
    return {
        "lgb_bal": lambda: lgb.LGBMClassifier(
            objective="multiclass", n_estimators=500, learning_rate=0.05, num_leaves=63,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
            class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1),
        "lgb_unbal": lambda: lgb.LGBMClassifier(
            objective="multiclass", n_estimators=700, learning_rate=0.03, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_child_samples=100,
            reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbosity=-1),
        "xgb": lambda: xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=500, learning_rate=0.05,
            max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=SEED, n_jobs=-1, verbosity=0),
        "cat": lambda: CatBoostClassifier(
            iterations=400, depth=8, learning_rate=0.05, loss_function="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED, verbose=False, thread_count=-1),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0,
            class_weight="balanced", random_state=SEED),
    }


def fit_base(name, mk, Xtr, ytr, Xva, Xte):
    m = mk()
    if name == "xgb":
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    else:
        m.fit(Xtr, ytr)
    return m.predict_proba(Xva), m.predict_proba(Xte)


def de_thresholds(proba, y):
    def neg(w):
        return -balanced_accuracy_score(y, (proba * w).argmax(1))
    res = differential_evolution(neg, bounds=[(0.1, 5.0)] * 3, seed=SEED,
                                 maxiter=30, popsize=15, tol=1e-4, polish=False)
    return res.x


def main():
    if not (DATA_DIR / "train.csv").exists():
        sys.exit(f"missing data in {DATA_DIR}. Download it first.")
    X, y, Xte, test_id, classes = load()
    n, m = len(X), len(Xte)
    print(f"train {n}, test {m}, classes {classes}")
    folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(X, y))
    models = base_models()

    # 1. base out-of-fold and test probabilities
    base_oof = {k: np.zeros((n, 3)) for k in models}
    base_test = {k: np.zeros((m, 3)) for k in models}
    for k, mk in models.items():
        t0 = time.time()
        for tr, va in folds:
            pv, pt = fit_base(k, mk, X[tr], y[tr], X[va], Xte)
            base_oof[k][va] = pv
            base_test[k] += pt / NF
        ba = balanced_accuracy_score(y, base_oof[k].argmax(1))
        print(f"base {k:10s} OOF bal_acc {ba:.5f}  ({time.time()-t0:.0f}s)")

    # 2. stacking features = concatenated log-probabilities
    def stack(d):
        return np.hstack([np.log(np.clip(d[k], 1e-7, 1.0)) for k in models])
    Soof, Ste = stack(base_oof), stack(base_test)

    # 3. meta-model (balanced logreg), honest OOF via CV over the stack features
    meta_oof = np.zeros((n, 3))
    for tr, va in folds:
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
        lr.fit(Soof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(Soof[va])
    ba_meta = balanced_accuracy_score(y, meta_oof.argmax(1))
    print(f"meta (balanced logreg) OOF bal_acc {ba_meta:.5f}")

    # 4. threshold calibration, nested: fit weights on disjoint OOF rows
    thr_folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED + 1).split(meta_oof, y))
    cal_pred = np.zeros(n, dtype=int)
    for tr, va in thr_folds:
        w = de_thresholds(meta_oof[tr], y[tr])
        cal_pred[va] = (meta_oof[va] * w).argmax(1)
    ba_cal = balanced_accuracy_score(y, cal_pred)
    print(f"meta + threshold calibration OOF bal_acc {ba_cal:.5f}")

    # 5. final test submission: fit meta + thresholds on all OOF, apply to test
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0).fit(Soof, y)
    meta_test = lr.predict_proba(Ste)
    w_final = de_thresholds(meta_oof, y)
    int_to_class = {i: c for i, c in enumerate(classes)}
    sub = pd.DataFrame({ID: test_id,
                        TARGET: [int_to_class[i] for i in (meta_test * w_final).argmax(1)]})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(RESULTS_DIR / "submission_stack.csv", index=False)

    print("\n=== summary (OOF balanced accuracy) ===")
    for k in models:
        print(f"  base {k:10s} {balanced_accuracy_score(y, base_oof[k].argmax(1)):.5f}")
    print(f"  meta (stack)         {ba_meta:.5f}")
    print(f"  meta + thresholds    {ba_cal:.5f}")
    print(f"  first attempt was    0.96400 (single LightGBM)")
    print(f"  delta vs first       {ba_cal-0.9640:+.5f}")
    print(f"final threshold weights {np.round(w_final,3)}")
    print("wrote results/submission_stack.csv (NOT submitted)")


if __name__ == "__main__":
    main()
