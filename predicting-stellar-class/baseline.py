"""LightGBM baseline for Predicting Stellar Class (playground-series-s6e6).

Trains a multiclass LightGBM model with StratifiedKFold cross-validation and
balanced class weights. The score is out-of-fold balanced accuracy, which
mirrors the competition metric. The submission uses the mean test-set class
probabilities across folds.

Run from the predicting-stellar-class/ folder, in the project .venv:
    python baseline.py
"""

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


@dataclass
class Config:
    data_dir: Path = Path(__file__).resolve().parent / "data"
    results_dir: Path = Path(__file__).resolve().parent / "results"
    target: str = "class"
    id_col: str = "id"
    categorical: tuple = ("spectral_type", "galaxy_population")
    n_folds: int = 5
    seed: int = 42
    params: dict = field(default_factory=lambda: {
        "objective": "multiclass",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    })


def load_data(cfg):
    train = pd.read_csv(cfg.data_dir / "train.csv")
    test = pd.read_csv(cfg.data_dir / "test.csv")
    return train, test


def prepare(cfg, train, test):
    """Encode the target to integers and set consistent category dtypes."""
    classes = sorted(train[cfg.target].unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = train[cfg.target].map(class_to_int).to_numpy()

    feature_cols = [c for c in train.columns if c not in (cfg.id_col, cfg.target)]
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    for col in cfg.categorical:
        cats = pd.Index(pd.concat([X[col], X_test[col]]).dropna().unique())
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)

    return X, y, X_test, feature_cols, classes


def run_cv(cfg, X, y, X_test, classes):
    """StratifiedKFold CV. Returns OOF probabilities, mean test probabilities,
    and the out-of-fold balanced accuracy."""
    n_class = len(classes)
    oof = np.zeros((len(X), n_class))
    test_pred = np.zeros((len(X_test), n_class))

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMClassifier(**cfg.params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(X_va)
        test_pred += model.predict_proba(X_test) / cfg.n_folds
        fold_score = balanced_accuracy_score(y_va, oof[va_idx].argmax(1))
        print(f"fold {fold}: balanced accuracy {fold_score:.4f} "
              f"(best_iter {model.best_iteration_})")

    oof_score = balanced_accuracy_score(y, oof.argmax(1))
    print(f"OOF balanced accuracy: {oof_score:.4f}")
    return oof, test_pred, oof_score


def make_submission(cfg, test, test_pred, classes):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    int_to_class = {i: c for i, c in enumerate(classes)}
    pred_label = [int_to_class[i] for i in test_pred.argmax(1)]
    sub = pd.DataFrame({cfg.id_col: test[cfg.id_col], cfg.target: pred_label})
    out = cfg.results_dir / "submission_lgbm_baseline.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out} ({len(sub)} rows)")
    print("predicted class distribution:")
    print(sub[cfg.target].value_counts(normalize=True).round(4))
    return out


def main():
    cfg = Config()
    if not (cfg.data_dir / "train.csv").exists():
        sys.exit(f"missing data in {cfg.data_dir}. Download it first.")
    train, test = load_data(cfg)
    X, y, X_test, feature_cols, classes = prepare(cfg, train, test)
    print(f"features: {feature_cols}")
    print(f"classes: {classes}")
    oof, test_pred, oof_score = run_cv(cfg, X, y, X_test, classes)
    make_submission(cfg, test, test_pred, classes)


if __name__ == "__main__":
    main()
