"""Train the L2 value model: P(win) from the L1 feature vector, as a gradient-boosted tree.

Why a tree, not logistic (measured): a linear logit cannot represent the prize/KO threshold that
dominates a TCG position, and its partial weights sign-flip on correlated features (it scored a
leaf with the opponent HEALTHIER as better). A gradient-boosted tree expresses thresholds and
interactions and does not sign-flip; on the same data it lifted held-out AUC 0.675 -> ~0.72.

The fitted trees are exported as raw arrays (feature/threshold/children/leaf-value per node) plus
the learning rate and init score, so inference is pure numpy in value_model.py -- no sklearn in
the agent. The export is VERIFIED to reproduce sklearn's predict_proba within 1e-6 before saving.

Evaluation discipline: GAME-WISE train/val split (rows within a game share the label and are
near-duplicates; a row-wise split leaks them and reports a falsely tight metric).

    python tools/train_value.py --data data/selfplay/v2.jsonl --out agent/value_weights.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import value_model as VM   # reuse the exact inference traversal for verification


def load(data_path: Path, keep_all: bool = False):
    keys = json.loads((data_path.parent / (data_path.name + ".keys.json")).read_text())
    X, y, turn, gid = [], [], [], []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not keep_all and r["y"] == 0.5:     # classification: drop draws (ambiguous label)
            continue
        X.append(r["feat"]); y.append(r["y"]); turn.append(r.get("turn", 0)); gid.append(r.get("gid", -1))
    return np.array(X, float), np.array(y, float), np.array(turn, int), np.array(gid, int), keys


def export_gbm(clf, keys):
    """Export a fitted GradientBoostingClassifier to plain arrays for pure-numpy inference.
    decision_function(x) = init + lr * sum_t leaf_value_t(x); proba = sigmoid(decision_function)."""
    lr = float(clf.learning_rate)
    trees = []
    for est in clf.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "feature": t.feature.astype(int).tolist(),       # -2 at leaves
            "threshold": t.threshold.astype(float).tolist(),
            "left": t.children_left.astype(int).tolist(),
            "right": t.children_right.astype(int).tolist(),
            "value": t.value[:, 0, 0].astype(float).tolist(),  # leaf raw step
        })
    return {"kind": "gbm", "keys": keys, "lr": lr, "trees": trees}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/selfplay/v2.jsonl")
    ap.add_argument("--out", default="agent/value_weights.json")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimators", type=int, default=300)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--target", choices=["outcome", "value"], default="outcome",
                    help="outcome = classify win/loss (MC); value = regress on the search-bootstrapped logit")
    args = ap.parse_args()

    data_path = (ROOT / args.data).resolve()
    X, y, turn, gid, keys = load(data_path, keep_all=(args.target == "value"))

    rng = np.random.default_rng(args.seed)
    games = np.unique(gid)
    rng.shuffle(games)
    n_val_games = max(1, int(len(games) * args.val_frac))
    val_games = set(games[:n_val_games].tolist())
    is_val = np.array([g in val_games for g in gid])
    Xtr, Xva, ytr, yva, tva = X[~is_val], X[is_val], y[~is_val], y[is_val], turn[is_val]

    if args.target == "value":
        # REGRESS on the search-bootstrapped value LOGIT; value_model applies the sigmoid, so the
        # regressor predicts a logit and inference recovers a probability in (0,1).
        # CLIP the logit: terminal lines give |logit|~500 (hand eval +/-1e6 / scale) which would
        # saturate the value to 0/1 and erase the mid-game gradient local ranking needs. +/-6 maps
        # terminals to ~0.998/0.002 while leaving non-terminal positions' fine gradient intact.
        CLIP = 6.0
        ytr = np.clip(ytr, -CLIP, CLIP)
        yva = np.clip(yva, -CLIP, CLIP)
        clf = GradientBoostingRegressor(
            n_estimators=args.estimators, max_depth=args.depth, learning_rate=args.lr,
            subsample=0.8, min_samples_leaf=20, random_state=args.seed,
        )
        clf.fit(Xtr, ytr)
        export = export_gbm(clf, keys)
        raw = clf.predict(Xva)                          # the predicted logit
        tree_sum = np.zeros(len(Xva))
        for est in clf.estimators_[:, 0]:
            tree_sum += est.predict(Xva)
        init_arr = raw - export["lr"] * tree_sum
        export["init"] = float(np.mean(init_arr))
        assert float(np.std(init_arr)) < 1e-6, f"init not constant (std {np.std(init_arr):.2e})"
        p_numpy = np.array([VM.proba_from_export(export, dict(zip(keys, row))) for row in Xva])
        max_diff = float(np.max(np.abs(1.0 / (1.0 + np.exp(-raw)) - p_numpy)))
        assert max_diff < 1e-6, f"export mismatch: max |sklearn-numpy| = {max_diff:.2e}"
        print(f"export verified: numpy inference matches sklearn within {max_diff:.1e}")
        corr = float(np.corrcoef(raw, yva)[0, 1]) if np.std(raw) > 1e-9 else 0.0
        mse = float(np.mean((raw - yva) ** 2))
        print(f"data: {len(X)} rows / {len(games)} games ({len(Xtr)} train / {len(Xva)} val rows) [search-bootstrapped value target]")
        print(f"val Pearson(predicted, target) {corr:.3f}   val MSE {mse:.4f}   target std {float(np.std(yva)):.3f}")
        imp = clf.feature_importances_
        print("top features by importance:")
        for i in np.argsort(-imp)[:12]:
            print(f"  {keys[i]:22s} {imp[i]:.3f}")
        export["val_corr"] = corr; export["n_rows"] = int(len(X)); export["target"] = "value"
        (ROOT / args.out).write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {args.out} ({(ROOT / args.out).stat().st_size // 1024} KB, {len(export['trees'])} trees)")
        return

    clf = GradientBoostingClassifier(
        n_estimators=args.estimators, max_depth=args.depth, learning_rate=args.lr,
        subsample=0.8, min_samples_leaf=20, random_state=args.seed,
    )
    clf.fit(Xtr, ytr)

    # export, then compute init empirically so numpy inference matches sklearn, then VERIFY
    export = export_gbm(clf, keys)
    df = clf.decision_function(Xva).ravel()
    tree_sum = np.zeros(len(Xva))
    for est in clf.estimators_[:, 0]:
        tree_sum += est.predict(Xva)
    init_arr = df - export["lr"] * tree_sum
    export["init"] = float(np.mean(init_arr))
    assert float(np.std(init_arr)) < 1e-6, f"init not constant (std {np.std(init_arr):.2e})"

    p_sklearn = clf.predict_proba(Xva)[:, 1]
    p_numpy = np.array([VM.proba_from_export(export, dict(zip(keys, row))) for row in Xva])
    max_diff = float(np.max(np.abs(p_sklearn - p_numpy)))
    assert max_diff < 1e-6, f"export mismatch: max |sklearn-numpy| = {max_diff:.2e}"
    print(f"export verified: numpy inference matches sklearn within {max_diff:.1e}")

    p = p_numpy
    acc = float(((p >= 0.5) == (yva >= 0.5)).mean())
    base = float(max(yva.mean(), 1 - yva.mean()))
    auc = float(roc_auc_score(yva, p)) if len(set(yva)) > 1 else float("nan")
    brier = float(brier_score_loss(yva, p))
    print(f"data: {len(X)} rows / {len(games)} games ({len(Xtr)} train / {len(Xva)} val rows), win-rate {y.mean():.3f}")
    print(f"val accuracy {acc:.3f}  (majority-class baseline {base:.3f})")
    print(f"val AUC      {auc:.3f}   val Brier {brier:.3f}  (0.25 = no-skill)")
    for lo, hi, label in [(0, 6, "early t<6"), (6, 14, "mid 6-13"), (14, 999, "late t>=14")]:
        m = (tva >= lo) & (tva < hi)
        if m.sum() > 10:
            a = float(((p[m] >= 0.5) == (yva[m] >= 0.5)).mean())
            print(f"  {label:11s} n={int(m.sum()):5d}  acc {a:.3f}")
    imp = clf.feature_importances_
    print("top features by importance:")
    for i in np.argsort(-imp)[:12]:
        print(f"  {keys[i]:22s} {imp[i]:.3f}")

    export["val_acc"] = acc; export["val_auc"] = auc; export["n_rows"] = int(len(X))
    (ROOT / args.out).write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.out} ({(ROOT / args.out).stat().st_size // 1024} KB, {len(export['trees'])} trees)")


if __name__ == "__main__":
    main()
