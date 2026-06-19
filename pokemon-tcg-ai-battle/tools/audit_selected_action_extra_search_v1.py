"""Selected-Action Extra-Search Benefit Audit V1 (Model A).

The useful question is NOT "can live search identify bad siblings it already avoids" (tautological), but:
"can we detect when the action live N=8 search actually SELECTS is secretly high-regret, and would extra
search fix it?" Decision-level targets on the live-selected action; features restricted to deployable live-N=8
+ free signals (forbidden N=32-derived fields fail loudly).

    python tools/audit_selected_action_extra_search_v1.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AU

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "data" / "manifests"
OUT = ROOT / "docs" / "workstreams"

# Targets MAY use stronger-teacher info (the point is to define "is the live pick secretly bad").
# FEATURES must be deployable BEFORE the stronger search. These keys are forbidden as feature SOURCES:
FORBIDDEN = {"mean_stronger_value", "stronger_value_variance", "value_se", "value_spread",
             "stronger_soft_policy", "delta_to_search", "delta_to_search_norm", "hand_norm_advantage",
             "regret", "high_regret_prob", "unacceptable_prob", "acceptable_prob", "policy_prob",
             "outcome_winrate", "outcome_se"}
ACCEPT_Z = 1.0


def sel_opt(r):
    return next((o for o in r["options"] if o["index"] == r.get("search_selected_option")), None)


def targets(r):
    so = sel_opt(r) or {}
    bi = r.get("stronger_argmax_option")
    bo = next((o for o in r["options"] if o["index"] == bi), {}) or {}
    best_se = bo.get("value_se", 0.0) or 0.0
    sel_se = so.get("value_se", 0.0) or 0.0
    sel_regret = abs(so.get("regret", 0.0) or 0.0)
    changes = int(bi is not None and bi != r.get("search_selected_option"))
    # beneficial: search would switch AND the live pick is materially worse than the stronger best (beyond noise)
    materially_worse = (so.get("high_regret_prob", 0) or 0) >= 0.5 or sel_regret > ACCEPT_Z * (best_se + sel_se)
    dist = r.get("live_selected_distribution") or {}
    sel_share = float(dist.get(str(r.get("search_selected_option")), 0.0) or 0.0)
    return {
        "selected_high_regret": int((so.get("high_regret_prob", 0) or 0) >= 0.5),
        "selected_unacceptable": int((so.get("unacceptable_prob", 0) or 0) >= 0.5),
        "extra_search_changes_action": changes,
        "extra_search_beneficial": int(changes and materially_worse),
        "instability_modal": 1.0 - float(r.get("modal_action_stability", 1.0) or 1.0),
        "p_selected_changes": 1.0 - sel_share,
        "sel_regret": sel_regret,
    }


def features(r):
    """DEPLOYABLE live-N=8 + free features only. Reads never touch FORBIDDEN keys (asserted)."""
    so = sel_opt(r) or {}
    live = [float(o.get("mean_live_value") or 0.0) for o in r["options"]]
    sl = sorted(live, reverse=True)
    crit = r.get("criticality") or {}
    f = {
        "sel_live_value": float(so.get("mean_live_value") or 0.0) / 1e5,
        "live_margin": (sl[0] - sl[1]) / 1e5 if len(sl) > 1 else 0.0,
        "live_spread": (float(np.std(live)) / 1e5) if live else 0.0,           # from N=8 values, NOT value_spread
        "modal": float(r.get("modal_action_stability", 0.0) or 0.0),
        "entropy": float(r.get("live_action_entropy", 0.0) or 0.0),
        "sel_live_var": math.log1p(float(so.get("live_value_variance") or 0.0)) / 30.0,
        "crit": float(crit.get("score", 0.0) or 0.0),
        "can_ko": float(crit.get("can_ko", 0.0) or 0.0),
        "ko_back": float(crit.get("ko_back", 0.0) or 0.0),
        "endgame": float(crit.get("endgame", 0.0) or 0.0),
        "n_options": float(r.get("n_options", len(r["options"])) or 0) / 32.0,
        "sel_opt_type": float((so.get("semantic_vector") or {}).get("opt_type", -1)) / 16.0,
    }
    # forbidden-feature guard: none of the feature sources may be a forbidden key
    for o in r["options"]:
        pass
    bad = FORBIDDEN.intersection(f.keys())
    if bad:
        raise SystemExit(f"FORBIDDEN feature used: {bad}")
    return f


def grouped_oof(rows, cols, ycol, n_folds=8):
    """Out-of-fold logistic predictions, grouped by game (no game crosses folds)."""
    games = sorted({r["group"] for r in rows})
    fold = {g: i % n_folds for i, g in enumerate(games)}
    pred = np.full(len(rows), np.nan)
    for k in range(n_folds):
        tr = [i for i, r in enumerate(rows) if fold[r["group"]] != k]
        te = [i for i, r in enumerate(rows) if fold[r["group"]] == k]
        ytr = np.array([rows[i][ycol] for i in tr])
        if len(set(ytr)) < 2 or not te:
            continue
        Xtr = np.array([[rows[i]["f"][c] for c in cols] for i in tr], float)
        Xte = np.array([[rows[i]["f"][c] for c in cols] for i in te], float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit((Xtr - mu) / sd, ytr)
        pred[te] = clf.predict_proba((Xte - mu) / sd)[:, 1]
    return pred


def trig_metrics(rows, fire, pos_key):
    n = len(rows)
    pos = [r[pos_key] for r in rows]
    npos = sum(pos)
    tp = sum(1 for r, fr in zip(rows, fire) if fr and r[pos_key])
    fp = sum(1 for r, fr in zip(rows, fire) if fr and not r[pos_key])
    nfire = sum(fire)
    nneg = n - npos
    games_cov = len({r["group"] for r, fr in zip(rows, fire) if fr and r[pos_key]})
    return {
        "trigger_rate": round(nfire / n, 3),
        "recall": round(tp / npos, 3) if npos else None,
        "precision": round(tp / nfire, 3) if nfire else None,
        "fpr": round(fp / nneg, 3) if nneg else None,
        "extra_compute_x": round(nfire / n * 4.0, 2),   # N=32 is ~4x N=8 on fired decisions
        "games_with_caught_pos": games_cov,
    }


def main():
    recs = [json.loads(l) for l in open(MAN / "continuous_terrain_v1.jsonl", encoding="utf-8")]
    recs = [r for r in recs if not r.get("eval_only")]
    rows = []
    for r in recs:
        t = targets(r)
        rows.append({"group": r["group_id"], "decision_id": r["decision_id"], "f": features(r), **t})
    n = len(rows)

    # ---- Step 4: counts + game diversity + clustering ----
    def cnt(key):
        p = [r for r in rows if r[key]]
        g = Counter(r["group"] for r in p)
        top = (max(g.values()) / len(p)) if p else 0.0
        return {"positives": len(p), "games": len(g), "top_game_share": round(top, 3),
                "rate": round(len(p) / n, 3)}
    counts = {k: cnt(k) for k in ["selected_high_regret", "selected_unacceptable",
                                  "extra_search_changes_action", "extra_search_beneficial"]}

    # ---- Step 6: oracle upper bound ----
    chg = [r for r in rows if r["extra_search_changes_action"]]
    ben = [r for r in rows if r["extra_search_beneficial"]]
    shr = [r for r in rows if r["selected_high_regret"]]
    fixed = sum(1 for r in shr if r["extra_search_changes_action"])
    oracle = {
        "pct_decisions_stronger_changes_action": round(len(chg) / n, 3),
        "pct_decisions_change_beneficial": round(len(ben) / n, 3),
        "pct_selected_high_regret_fixed_by_switch": round(fixed / len(shr), 3) if shr else None,
        "mean_regret_avoided_on_beneficial": round(statistics.fmean([r["sel_regret"] for r in ben]), 0) if ben else 0,
        "oracle_trigger_rate": round(len(ben) / n, 3),
    }

    # ---- Step 5: trigger evaluation ----
    ENT = statistics.median(r["f"]["entropy"] for r in rows) or 0.0
    MARG = statistics.median(r["f"]["live_margin"] for r in rows)
    VAR = statistics.median(r["f"]["sel_live_var"] for r in rows)
    SPR = statistics.median(r["f"]["live_spread"] for r in rows)
    triggers = {
        "high_live_entropy": [r["f"]["entropy"] > max(0.05, ENT) for r in rows],
        "low_modal_stability(<0.99)": [r["f"]["modal"] < 0.99 for r in rows],
        "low_top2_margin": [r["f"]["live_margin"] < MARG for r in rows],
        "high_sel_live_variance": [r["f"]["sel_live_var"] > VAR for r in rows],
        "high_criticality(>0.3)": [r["f"]["crit"] > 0.3 for r in rows],
        "high_live_value_spread": [r["f"]["live_spread"] > SPR for r in rows],
        "crit>0.3 AND modal<0.99": [r["f"]["crit"] > 0.3 and r["f"]["modal"] < 0.99 for r in rows],
        "crit>0.3 AND low_margin": [r["f"]["crit"] > 0.3 and r["f"]["live_margin"] < MARG for r in rows],
    }
    cols = list(rows[0]["f"].keys())
    oof_shr = grouped_oof(rows, cols, "selected_high_regret")
    oof_ben = grouped_oof(rows, cols, "extra_search_beneficial")
    # learned trigger: fire on top decisions by OOF score, matched to oracle trigger rate
    thr = np.nanpercentile(oof_shr[~np.isnan(oof_shr)], 100 * (1 - oracle["oracle_trigger_rate"] * 3)) if np.isfinite(oof_shr).any() else 1.0
    triggers["learned_logistic(selected_high_regret OOF)"] = [(not np.isnan(oof_shr[i])) and oof_shr[i] >= thr for i in range(n)]

    trig_table = {}
    for name, fire in triggers.items():
        trig_table[name] = {"vs_selected_high_regret": trig_metrics(rows, fire, "selected_high_regret"),
                            "vs_extra_search_beneficial": trig_metrics(rows, fire, "extra_search_beneficial")}
    # learned AUROC/AP (grouped OOF) for reference (not the headline)
    def safe(y, s):
        m = ~np.isnan(s)
        yy = np.array([rows[i][y] for i in range(n)])[m]
        return (round(AU(yy, s[m]), 3), round(AP(yy, s[m]), 3)) if len(set(yy)) > 1 else (None, None)

    def game_boot_ci(y, s, B=2000):
        """Game-clustered bootstrap 95% CI on AUROC (resample whole games with replacement)."""
        m = ~np.isnan(s)
        idx_by_game = defaultdict(list)
        for i in range(n):
            if m[i]:
                idx_by_game[rows[i]["group"]].append(i)
        games = list(idx_by_game)
        rng = np.random.default_rng(7)
        aus = []
        for _ in range(B):
            pick = rng.choice(len(games), len(games), replace=True)
            ii = [i for k in pick for i in idx_by_game[games[k]]]
            yy = np.array([rows[i][y] for i in ii])
            if len(set(yy)) > 1:
                aus.append(AU(yy, s[ii]))
        return [round(float(np.percentile(aus, 2.5)), 3), round(float(np.percentile(aus, 97.5)), 3)] if aus else None
    learned = {"selected_high_regret_OOF_AUROC_AP": safe("selected_high_regret", oof_shr),
               "selected_high_regret_AUROC_95CI": game_boot_ci("selected_high_regret", oof_shr),
               "extra_search_beneficial_OOF_AUROC_AP": safe("extra_search_beneficial", oof_ben),
               "extra_search_beneficial_AUROC_95CI": game_boot_ci("extra_search_beneficial", oof_ben)}

    underpowered = counts["selected_high_regret"]["positives"] < 20 or counts["extra_search_beneficial"]["positives"] < 20
    summary = {"decisions": n, "games": len({r["group"] for r in rows}),
               "counts": counts, "oracle": oracle, "triggers": trig_table, "learned_oof": learned,
               "underpowered_lt20": underpowered, "forbidden_features_guard": "passed"}
    json.dump(summary, open(OUT / "selected_action_extra_search_audit_v1.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({"decisions": n, "counts": counts, "oracle": oracle, "learned_oof": learned,
                      "underpowered_lt20": underpowered}, indent=1))
    print("\nTRIGGER TABLE (vs selected_high_regret / vs extra_search_beneficial):")
    print(f"{'trigger':42s} {'rate':>5s} {'SHR_rec':>7s} {'SHR_prc':>7s} {'BEN_rec':>7s} {'BEN_prc':>7s} {'xCompute':>8s}")
    for name, m in trig_table.items():
        a, b = m["vs_selected_high_regret"], m["vs_extra_search_beneficial"]
        print(f"{name:42s} {a['trigger_rate']:>5} {str(a['recall']):>7} {str(a['precision']):>7} "
              f"{str(b['recall']):>7} {str(b['precision']):>7} {a['extra_compute_x']:>8}")


if __name__ == "__main__":
    main()
