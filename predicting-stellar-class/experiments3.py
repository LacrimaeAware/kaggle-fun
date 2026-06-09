"""Experiment 03 for Predicting Stellar Class: physics-motivated features and diagnostics.

From astronomy literature on optical colors: the quasar class often limits
balanced accuracy, the optical star/quasar degeneracy is the obstacle, and the
stellar locus is a 1D curve u-g = 2.15 (g-r) + 0.26. This batch:

1. Tests the signed perpendicular distance from that stellar-locus line, and a
   UV-excess region indicator (u-g < 0.6 and g-r > 0), against the baseline on
   fixed folds.
2. Rechecks per-class threshold tuning with a grid (experiment 02's optimizer may
   have stalled on a piecewise-constant objective).
3. Reports the diagnostics that explain the ceiling: the out-of-fold confusion
   matrix and per-class recall, and the baseline feature importances.

Run from the predicting-stellar-class/ folder, in the project .venv:
    python experiments3.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def f_locus(df):
    ug, gr = df["u"] - df["g"], df["g"] - df["r"]
    out = pd.DataFrame(index=df.index)
    out["dperp"] = (2.15 * gr - ug + 0.26) / np.sqrt(2.15 ** 2 + 1.0)  # signed locus distance
    return out


def f_uvx(df):
    ug, gr = df["u"] - df["g"], df["g"] - df["r"]
    out = pd.DataFrame(index=df.index)
    out["uvx"] = ((ug < 0.6) & (gr > 0)).astype(float)
    return out


def combine(*funcs):
    return lambda df: pd.concat([fn(df) for fn in funcs], axis=1)


EXPERIMENTS = [
    ("baseline", None),
    ("+dperp (stellar-locus distance)", f_locus),
    ("+uvx indicator", f_uvx),
    ("+dperp+uvx (locus pack)", combine(f_locus, f_uvx)),
]


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


def oof_predict(X, y, folds, want_proba=False, want_importance=False):
    proba = np.zeros((len(X), 3))
    imp = np.zeros(X.shape[1])
    for tr, va in folds:
        m = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y[tr])
        proba[va] = m.predict_proba(X.iloc[va])
        if want_importance:
            imp += m.booster_.feature_importance(importance_type="gain")
    if want_importance:
        return proba, dict(zip(X.columns, imp / len(folds)))
    return proba


def grid_thresholds(proba, y, seed=0):
    idx = np.arange(len(y))
    fit_idx, ev_idx = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    grid = np.linspace(0.4, 2.5, 22)
    best_w, best_fit = (1.0, 1.0), -1.0
    for w0 in grid:
        for w1 in grid:
            w = np.array([w0, w1, 1.0])
            s = balanced_accuracy_score(y[fit_idx], (proba[fit_idx] * w).argmax(1))
            if s > best_fit:
                best_fit, best_w = s, (w0, w1)
    w = np.array([best_w[0], best_w[1], 1.0])
    base_ev = balanced_accuracy_score(y[ev_idx], proba[ev_idx].argmax(1))
    opt_ev = balanced_accuracy_score(y[ev_idx], (proba[ev_idx] * w).argmax(1))
    return base_ev, opt_ev, best_w


def main():
    if not (DATA_DIR / "train.csv").exists():
        sys.exit(f"missing data in {DATA_DIR}. Download it first.")
    X, y, classes = load_subsample()
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    print(f"subsample {len(X)} rows, classes {classes}")

    rows = []
    base_proba = None
    base_imp = None
    for name, fn in EXPERIMENTS:
        Xe = X if fn is None else pd.concat([X, fn(X)], axis=1)
        if name == "baseline":
            proba, base_imp = oof_predict(Xe, y, folds, want_importance=True)
            base_proba = proba
        else:
            proba = oof_predict(Xe, y, folds)
        ba = balanced_accuracy_score(y, proba.argmax(1))
        rows.append({"experiment": name, "bal_acc": round(ba, 5), "n_features": Xe.shape[1]})
        print(f"{name:34s} bal_acc {ba:.5f}  feats {Xe.shape[1]}")

    base_ba = rows[0]["bal_acc"]
    for r in rows:
        r["delta_vs_baseline"] = round(r["bal_acc"] - base_ba, 5)

    # diagnostics: confusion matrix and per-class recall on baseline OOF
    pred = base_proba.argmax(1)
    cm = confusion_matrix(y, pred)
    recalls = cm.diagonal() / cm.sum(1)
    print("\n=== baseline OOF confusion matrix (rows true, cols predicted) ===")
    print(pd.DataFrame(cm, index=classes, columns=classes))
    print("per-class recall:", {classes[i]: round(float(recalls[i]), 4) for i in range(3)})
    print(f"balanced accuracy (mean recall): {recalls.mean():.5f}")

    print("\n=== baseline feature importance (gain, mean over folds) ===")
    imp = pd.Series(base_imp).sort_values(ascending=False)
    imp_norm = (imp / imp.sum()).round(4)
    print(imp_norm.to_string())

    # grid threshold recheck
    base_ev, opt_ev, w = grid_thresholds(base_proba, y)
    print(f"\n=== grid per-class thresholds (fit 60pct, eval 40pct) ===")
    print(f"eval argmax {base_ev:.5f} -> optimized {opt_ev:.5f} "
          f"(delta {opt_ev-base_ev:+.5f}), weights (QSO/GALAXY scan) {np.round(w,3)}")

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "experiments3_locus.csv", index=False)
    print("\n=== features ===")
    print(res.to_string(index=False))
    print("\nnoise floor from experiment 02: 0.0002. A delta below it is within noise.")


if __name__ == "__main__":
    main()
