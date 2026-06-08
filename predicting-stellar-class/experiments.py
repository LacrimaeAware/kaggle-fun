"""Mini-experiment harness for Predicting Stellar Class (playground-series-s6e6).

Each experiment is the raw-feature baseline plus one feature idea. All experiments
share one fixed set of StratifiedKFold folds, so the comparison is paired and the
signal is the change in out-of-fold balanced accuracy (the competition metric)
relative to the baseline. A Gaussian-noise experiment is included as a null
control: a useful idea should beat it.

This is a screen. It runs on a stratified subsample with a fast LightGBM to rank
ideas. Ideas that clear the null are confirmed later at full scale with the
baseline.py configuration.

Run from the predicting-stellar-class/ folder, in the project .venv:
    python experiments.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TARGET = "class"
ID = "id"
CATEG = ["spectral_type", "galaxy_population"]
BANDS = ["u", "g", "r", "i", "z"]
BASE_NUM = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
SEED = 42
SUBSAMPLE = 250_000
N_FOLDS = 5

PARAMS = dict(
    objective="multiclass",
    n_estimators=350,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=50,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)


# --- static, row-wise feature builders (no leakage, computed once) ---

def f_colors_adjacent(df):
    out = pd.DataFrame(index=df.index)
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        out[f"col_{a}{b}"] = df[a] - df[b]
    return out


def f_colors_all(df):
    out = pd.DataFrame(index=df.index)
    for i in range(len(BANDS)):
        for j in range(i + 1, len(BANDS)):
            out[f"col_{BANDS[i]}{BANDS[j]}"] = df[BANDS[i]] - df[BANDS[j]]
    return out


def f_sed_moments(df):
    m = df[BANDS].to_numpy()
    out = pd.DataFrame(index=df.index)
    out["sed_mean"] = m.mean(1)
    out["sed_std"] = m.std(1)
    out["sed_skew"] = skew(m, axis=1)
    out["sed_kurt"] = kurtosis(m, axis=1)
    return out


def f_sed_curvature(df):
    # second differences along the wavelength-ordered band sequence
    out = pd.DataFrame(index=df.index)
    out["curv_ugr"] = df["u"] - 2 * df["g"] + df["r"]
    out["curv_gri"] = df["g"] - 2 * df["r"] + df["i"]
    out["curv_riz"] = df["r"] - 2 * df["i"] + df["z"]
    return out


def f_centered_bands(df):
    m = df[BANDS].to_numpy()
    centered = m - m.mean(1, keepdims=True)
    return pd.DataFrame(centered, index=df.index, columns=[f"cb_{b}" for b in BANDS])


def f_redshift(df):
    out = pd.DataFrame(index=df.index)
    out["z_log1p"] = np.log1p(df["redshift"].clip(lower=0))
    out["z_sq"] = df["redshift"] ** 2
    return out


def f_redshift_color(df):
    out = pd.DataFrame(index=df.index)
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        out[f"zx_{a}{b}"] = df["redshift"] * (df[a] - df[b])
    return out


def f_null_gaussian(df):
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((len(df), 4))
    return pd.DataFrame(noise, index=df.index, columns=[f"noise{i}" for i in range(4)])


def combine(*funcs):
    def _f(df):
        return pd.concat([fn(df) for fn in funcs], axis=1)
    return _f


# --- per-fold transformers (fit on the training fold only, no leakage) ---

class PCABands:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_tr):
        self.p = PCA(n_components=self.k, random_state=SEED).fit(X_tr[BANDS].to_numpy())
        return self

    def transform(self, X):
        z = self.p.transform(X[BANDS].to_numpy())
        return pd.DataFrame(z, index=X.index, columns=[f"pca{i}" for i in range(self.k)])


class GMMColor:
    """Gaussian-mixture density and responsibilities in colour-redshift space.
    Captures multi-modal (non-Gaussian) structure such as the stellar locus."""

    def __init__(self, n_components=8):
        self.n_components = n_components

    @staticmethod
    def _raw(X):
        return np.column_stack([
            X["u"] - X["g"], X["g"] - X["r"], X["r"] - X["i"], X["i"] - X["z"],
            X["redshift"].to_numpy(),
        ])

    def fit(self, X_tr):
        self.g = GaussianMixture(
            n_components=self.n_components, covariance_type="full",
            max_iter=30, random_state=SEED,
        ).fit(self._raw(X_tr))
        return self

    def transform(self, X):
        feats = self._raw(X)
        out = pd.DataFrame(index=X.index)
        out["gmm_logdens"] = self.g.score_samples(feats)
        resp = self.g.predict_proba(feats)
        for i in range(resp.shape[1]):
            out[f"gmm_r{i}"] = resp[:, i]
        return out


# name, static feature function (or None), per-fold transformer factory (or None)
EXPERIMENTS = [
    ("baseline", None, None),
    ("null_gaussian", f_null_gaussian, None),
    ("colors_adjacent", f_colors_adjacent, None),
    ("colors_all_pairs", f_colors_all, None),
    ("sed_moments", f_sed_moments, None),
    ("sed_curvature", f_sed_curvature, None),
    ("centered_bands", f_centered_bands, None),
    ("redshift_transforms", f_redshift, None),
    ("redshift_color_interactions", f_redshift_color, None),
    ("colors_adjacent+sed_moments", combine(f_colors_adjacent, f_sed_moments), None),
    ("colors_all+curvature+moments", combine(f_colors_all, f_sed_curvature, f_sed_moments), None),
    ("pca_bands_foldfit", None, PCABands),
    ("gmm_density_foldfit", None, GMMColor),
    ("colors_adjacent+gmm", f_colors_adjacent, GMMColor),
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


def run_experiment(X_base, y, folds, static_fn, transformer_factory):
    X = X_base.copy()
    if static_fn is not None:
        X = pd.concat([X, static_fn(X_base)], axis=1)
    oof = np.zeros(len(X), dtype=int)
    fold_scores = []
    for tr_idx, va_idx in folds:
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        if transformer_factory is not None:
            t = transformer_factory().fit(X_tr)
            X_tr = pd.concat([X_tr, t.transform(X_tr)], axis=1)
            X_va = pd.concat([X_va, t.transform(X_va)], axis=1)
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X_tr, y[tr_idx])
        pred = model.predict(X_va)
        oof[va_idx] = pred
        fold_scores.append(balanced_accuracy_score(y[va_idx], pred))
    return balanced_accuracy_score(y, oof), float(np.std(fold_scores)), X.shape[1]


def main():
    if not (DATA_DIR / "train.csv").exists():
        sys.exit(f"missing data in {DATA_DIR}. Download it first.")
    X_base, y, classes = load_subsample()
    print(f"subsample rows: {len(X_base)}, classes: {classes}, base features: {X_base.shape[1]}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_base, y))

    rows = []
    for name, static_fn, factory in EXPERIMENTS:
        t0 = time.time()
        score, fold_std, n_feat = run_experiment(X_base, y, folds, static_fn, factory)
        rows.append({"experiment": name, "bal_acc": score, "fold_std": fold_std,
                     "n_features": n_feat, "seconds": round(time.time() - t0, 1)})
        print(f"{name:32s} bal_acc {score:.4f}  fold_std {fold_std:.4f}  "
              f"feats {n_feat:3d}  {rows[-1]['seconds']}s")

    res = pd.DataFrame(rows)
    base = res.loc[res.experiment == "baseline", "bal_acc"].iloc[0]
    res["delta_vs_baseline"] = (res["bal_acc"] - base).round(5)
    res = res.sort_values("bal_acc", ascending=False).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "experiments_screen.csv"
    res.to_csv(out, index=False)
    print("\n=== ranked by out-of-fold balanced accuracy ===")
    print(res[["experiment", "bal_acc", "delta_vs_baseline", "fold_std", "n_features"]].to_string(index=False))
    null = res.loc[res.experiment == "null_gaussian", "delta_vs_baseline"].iloc[0]
    print(f"\nbaseline bal_acc {base:.4f}; null_gaussian delta {null:+.5f} "
          f"(the bar a real idea must clear)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
