"""H024-v2 Phase 1 (sim-free): does a DECISION-CENTERED target rank sibling options better than the
absolute leaf-value regression we have been doing? Reuses the existing E013 data (data/selfplay/
actions.jsonl: per row feat[47], y=hand-search leaf value, gid=decision). One isolated variable:
raw absolute y vs centered advantage A_{g,i}=y_{g,i}-mean_j(y_{g,j}). This is TEACHER IMITATION (the
target is the hand eval, so it cannot beat hand search); its only purpose is the centering question.

Metrics are WITHIN-DECISION (the methodology, not global AUC): top-1 agreement with the teacher's best
option, pairwise order accuracy, and regret (teacher-best value minus the value of the model's pick).
Stratified by decision criticality = spread max(y)-min(y). Baselines: random and chose-option-0.

    python tools/rank_phase1.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROWS = [json.loads(l) for l in open(ROOT / "data" / "selfplay" / "actions.jsonl", encoding="utf-8")]


def groups():
    g = defaultdict(list)
    for r in ROWS:
        g[r["gid"]].append((r["feat"], r["y"]))
    return {k: v for k, v in g.items() if len(v) >= 2}   # real decisions only


def within_metrics(pred_by_gid, gtruth, subset):
    top1 = pair_hit = pair_tot = 0
    regret = []
    n = 0
    for gid in subset:
        ys = np.array([y for _, y in gtruth[gid]])
        p = pred_by_gid[gid]
        n += 1
        tb = int(np.argmax(ys))
        mp = int(np.argmax(p))
        top1 += (mp == tb)
        regret.append(float(ys[tb] - ys[mp]))
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                if ys[i] == ys[j]:
                    continue
                pair_tot += 1
                pair_hit += ((p[i] - p[j]) * (ys[i] - ys[j]) > 0)
    return {"top1": top1 / n if n else 0, "pair": pair_hit / pair_tot if pair_tot else 0,
            "regret": statistics.mean(regret) if regret else 0, "n": n}


def main():
    from sklearn.ensemble import GradientBoostingRegressor
    G = groups()
    gids = sorted(G)
    rng = np.random.default_rng(0)
    arr = np.array(gids); rng.shuffle(arr)
    cut = int(0.7 * len(arr))
    train_g, test_g = set(arr[:cut].tolist()), set(arr[cut:].tolist())
    print(f"{len(gids)} decisions ({len(ROWS)} options), 70/30 split by decision")

    # build train matrices for raw and centered targets
    Xtr, ytr_raw, ytr_cen = [], [], []
    for gid in train_g:
        ys = [y for _, y in G[gid]]; m = statistics.mean(ys)
        for (f, y) in G[gid]:
            Xtr.append(f); ytr_raw.append(y); ytr_cen.append(y - m)
    Xtr = np.array(Xtr)
    truth = G

    def train_pred(target):
        clf = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
        clf.fit(Xtr, np.array(target))
        return {gid: clf.predict(np.array([f for f, _ in G[gid]])) for gid in test_g}

    pred_raw = train_pred(ytr_raw)
    pred_cen = train_pred(ytr_cen)

    # criticality strata by spread on test decisions
    spread = {gid: (max(y for _, y in G[gid]) - min(y for _, y in G[gid])) for gid in test_g}
    hi_cut = np.quantile(list(spread.values()), 2 / 3)
    hi = {gid for gid in test_g if spread[gid] >= hi_cut}
    lo = {gid for gid in test_g if spread[gid] < hi_cut}

    # baselines
    rand_top1 = statistics.mean(1 / len(G[gid]) for gid in test_g)
    opt0_top1 = statistics.mean(1.0 if int(np.argmax([y for _, y in G[gid]])) == 0 else 0.0 for gid in test_g)
    print(f"\nBASELINE within-decision top-1: random {rand_top1:.3f} | chose-option-0 {opt0_top1:.3f}")

    for name, sub in [("ALL test", test_g), ("HIGH-criticality (top third spread)", hi), ("low-criticality", lo)]:
        mr = within_metrics(pred_raw, truth, sub)
        mc = within_metrics(pred_cen, truth, sub)
        print(f"\n{name} (n={mr['n']}):")
        print(f"  RAW abs-value : top1 {mr['top1']:.3f} | pair {mr['pair']:.3f} | regret {mr['regret']:.4f}")
        print(f"  CENTERED adv  : top1 {mc['top1']:.3f} | pair {mc['pair']:.3f} | regret {mc['regret']:.4f}")
    print("\nRead: if CENTERED beats RAW on within-decision top-1/pair/regret (esp. high-criticality), a")
    print("decision-relative target helps local ranking -> carry it into the real action-conditioned model.")
    print("NOTE: teacher-imitation only (target = hand eval); this cannot beat hand search, it isolates centering.")


if __name__ == "__main__":
    main()
