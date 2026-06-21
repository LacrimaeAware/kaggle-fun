"""Search Metadata Dominance Audit -- is R1 a real deployable signal, or label leakage?

R1's strong fields (B's train_continuous_terrain_v1.py) are `spread` (= dataset value_spread, derived from
hand_norm_advantage = the STRONGER N=32 eval) and `value_se` (= sqrt(stronger_variance/4)). The high_regret
TARGET is also computed from those same N=32 stronger runs. So R1 predicting high_regret partly reads the
label off its own computation. At inference the live agent has only the N=8 search; it does NOT have the N=32
stats (computing them IS the expensive thing the risk flag was meant to trigger). So stronger-derived inputs
are non-deployable leakage; only criticality + live N=8 stats are deployable.

This audit retrains the same balanced-logistic probe on B's held-out test games for high_regret, with three
input sets: FULL (B's R1), LEAK-ONLY (spread, value_se), LIVE-ONLY (criticality + live entropy/modal/variance/
margin + is_selected). If FULL/LEAK ~ 0.99 AUROC but LIVE collapses, the dominance was leakage.

    python tools/audit_search_metadata_dominance.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "data" / "manifests"
B_EVAL = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/.claude/worktrees/robust-learner-v2/"
              "pokemon-tcg-ai-battle/docs/workstreams/continuous_terrain_representation_v1_eval.json")


def recall_at_fpr(y, s, fpr_target):
    if len(set(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    ok = [t for f, t in zip(fpr, tpr) if f <= fpr_target]
    return round(max(ok), 3) if ok else 0.0


def feats(rec):
    """Per-option feature dict for one decision (broadcast decision-level fields)."""
    crit = float((rec.get("criticality") or {}).get("score", 0.0) or 0.0)
    ent = float(rec.get("live_action_entropy", 0.0) or 0.0)
    modal = float(rec.get("modal_action_stability", 0.0) or 0.0)
    spread = float(rec.get("value_spread", 0.0) or 0.0)                       # STRONGER-derived (leak)
    sel = rec.get("search_selected_option")
    live_vals = [float(o.get("mean_live_value") or 0.0) for o in rec["options"]]
    sv = sorted(live_vals, reverse=True)
    live_margin = (sv[0] - sv[1]) if len(sv) > 1 else 0.0                     # LIVE top-2 margin
    rows = []
    for o in rec["options"]:
        rows.append({
            "spread": spread,                                                 # leak
            "value_se": float(o.get("value_se") or 0.0),                      # leak (stronger var)
            "entropy": ent, "modal": modal, "crit": crit,
            "is_selected": 1.0 if o["index"] == sel else 0.0,
            "live_var": float(o.get("live_value_variance") or 0.0),           # live
            "live_margin": live_margin,                                       # live
            "y": 1.0 if float(o.get("high_regret_prob", 0.0) or 0.0) >= 0.5 else 0.0,
            "group": rec["group_id"],
        })
    return rows


SETS = {
    "R1_FULL (B's R1: leak+live)": ["spread", "value_se", "entropy", "modal", "crit", "is_selected", "live_var", "live_margin"],
    "LEAK_ONLY (spread, value_se)": ["spread", "value_se"],
    "LIVE_ONLY (crit+entropy+modal+var+margin+sel)": ["entropy", "modal", "crit", "is_selected", "live_var", "live_margin"],
    "LIVE_MINIMAL (entropy, modal, crit)": ["entropy", "modal", "crit"],
    "JUST value_se (single leak field)": ["value_se"],
    "JUST spread (single leak field)": ["spread"],
}


def main():
    recs = [json.loads(l) for l in open(MAN / "continuous_terrain_v1.jsonl", encoding="utf-8")]
    split = json.load(open(B_EVAL, encoding="utf-8"))["split"]
    train_g, test_g = set(split["train"]), set(split["test"])
    rows = [r for rec in recs for r in feats(rec) if not rec.get("eval_only")]
    tr = [r for r in rows if r["group"] in train_g]
    te = [r for r in rows if r["group"] in test_g]
    ytr = np.array([r["y"] for r in tr])
    yte = np.array([r["y"] for r in te])
    print(f"train options {len(tr)} (pos {int(ytr.sum())}) | test options {len(te)} (pos {int(yte.sum())})")
    print(f"{'feature set':52s} {'AP':>6s} {'AUROC':>6s} {'rec@5':>6s} {'rec@10':>7s}")
    out = {}
    for name, cols in SETS.items():
        Xtr = np.array([[r[c] for c in cols] for r in tr], dtype=float)
        Xte = np.array([[r[c] for c in cols] for r in te], dtype=float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        s = clf.predict_proba(Xte)[:, 1]
        ap = round(average_precision_score(yte, s), 3)
        au = round(roc_auc_score(yte, s), 3)
        r5 = recall_at_fpr(yte, s, 0.05)
        r10 = recall_at_fpr(yte, s, 0.10)
        out[name] = {"AP": ap, "AUROC": au, "recall@FPR5": r5, "recall@FPR10": r10,
                     "coef": {c: round(float(w), 3) for c, w in zip(cols, clf.coef_[0])}}
        print(f"{name:52s} {ap:6.3f} {au:6.3f} {str(r5):>6s} {str(r10):>7s}")
    json.dump(out, open(MAN / "audit_search_metadata_dominance.json", "w", encoding="utf-8"), indent=1)
    print("\ncoefficients (standardized):")
    for name in ("R1_FULL (B's R1: leak+live)",):
        print(f"  {name}: {out[name]['coef']}")


if __name__ == "__main__":
    main()
