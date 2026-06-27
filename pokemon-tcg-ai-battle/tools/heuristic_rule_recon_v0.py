"""Natural Heuristic Rule Reconstruction -- PHASE 2: feature extraction + reconstruction (Model B diagnostic).

Consumes natural_trace_raw.jsonl (pi_R = Archaludon rule agent's natural decisions) and tests whether the public
FEATURE LAYER can recover the state/action conditions behind pi_R's rules. NOT a claim that pi_R is good; a
representation diagnostic. Per decision-family we fit interpretable models (depth-2/3 trees, sparse logistic) for
y_select / y_dev / y_suppress, compare to nulls (option-zero, family/card majority, option-index), classify
identifiability, and match recovered conditions to the rule manifest.

  PYTHONIOENCODING=utf-8 python tools/heuristic_rule_recon_v0.py
"""
from __future__ import annotations
import collections
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
OUT = ROOT / "data" / "generated" / "heuristic_rule_reconstruction_v0"
# Archaludon card ids (from submissions/sub_archaludon/main.py)
DURALUDON, ARCHALUDON_EX, CINDERACE, METAL_ENERGY = 169, 190, 666, 8
BOSS, ULTRA_BALL, NIGHT_STRETCHER, ICE_CREAM, HERO_CAPE, FULL_METAL_LAB = 1182, 1121, 1097, 1147, 1159, 1244
TYPE_FAMILY = {1: "YES", 2: "NO", 3: "SELECT_CARD", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY",
               11: "DISCARD", 12: "RETREAT", 13: "ATTACK", 14: "END"}

with contextlib.redirect_stderr(io.StringIO()):
    import deck_policy_v3 as DP

from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402


def _cid_of(card):
    if isinstance(card, int):
        return card
    if isinstance(card, dict):
        for k in ("cardId", "card_id", "id"):
            if card.get(k) is not None:
                return card.get(k)
    return None


def _energy_units(ent):
    if not isinstance(ent, dict):
        return None
    return len(ent.get("energies") or []) or len(ent.get("energyCards") or [])


def feat(obs, opt, i, k):
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    me = cur.get("yourIndex", 0)
    P = players[me] if len(players) > me else {}
    typ = opt.get("type")
    with contextlib.suppress(Exception):
        src = DP.option_card_id(opt, obs)
    src = src if "src" in dir() else None
    try:
        tent = DP.option_target_entity(opt, obs)
        tcid = DP._cid(tent) if tent else None
    except Exception:
        tent, tcid = None, None
    discard = P.get("discard") or []
    metal_disc = sum(1 for c in discard if _cid_of(c) == METAL_ENERGY)
    return {
        "raw_index": i, "is_option_zero": int(i == 0), "type": typ or -1,
        "family": TYPE_FAMILY.get(typ, "OTHER"),
        "source_card_id": src if src is not None else -1, "target_card_id": tcid if tcid is not None else -1,
        "src_is_metal": int(src == METAL_ENERGY), "src_is_boss": int(src == BOSS), "src_is_ultraball": int(src == ULTRA_BALL),
        "src_is_nightstretcher": int(src == NIGHT_STRETCHER), "src_is_icecream": int(src == ICE_CREAM),
        "src_is_herocape": int(src == HERO_CAPE), "src_is_fullmetal": int(src == FULL_METAL_LAB),
        "tgt_is_archaludon": int(tcid == ARCHALUDON_EX), "tgt_is_duraludon": int(tcid == DURALUDON),
        "tgt_is_cinderace": int(tcid == CINDERACE),
        "metal_in_discard": metal_disc, "deck_count": P.get("deckCount", 0) or 0, "hand_count": P.get("handCount", 0) or 0,
        "prize_count": len(P.get("prize") or []), "turnActionCount": cur.get("turnActionCount", -1),
        "supporterPlayed": int(bool(cur.get("supporterPlayed"))), "energyAttached": int(bool(cur.get("energyAttached"))),
        "retreated": int(bool(cur.get("retreated"))),
        "energy_on_target": _energy_units(tent) if tent else -1,
    }


NUM_FEATS = ["raw_index", "is_option_zero", "type", "source_card_id", "target_card_id", "src_is_metal", "src_is_boss",
             "src_is_ultraball", "src_is_nightstretcher", "src_is_icecream", "src_is_herocape", "src_is_fullmetal",
             "tgt_is_archaludon", "tgt_is_duraludon", "tgt_is_cinderace", "metal_in_discard", "deck_count",
             "hand_count", "prize_count", "turnActionCount", "supporterPlayed", "energyAttached", "retreated",
             "energy_on_target"]


def build_rows():
    rows = []
    for line in open(OUT / "natural_trace_raw.jsonl", encoding="utf-8"):
        d = json.loads(line)
        obs = d["obs"]
        opts = (obs.get("select") or {}).get("option") or []
        k = d["maxCount"]
        pi = set(d["pi_rule_action"])
        pz = set(d["pi_zero_action"])
        for i, opt in enumerate(opts):
            if not isinstance(opt, dict):
                continue
            f = feat(obs, opt, i, k)
            f["game_id"] = d["game_id"]
            f["decision_id"] = f"{d['game_id']}:{d['step']}:{d['seat']}"
            f["y_select"] = int(i in pi)
            f["y_suppress_zero"] = int(i in pz and i not in pi)
            f["deviates"] = int(pi != pz)
            f["k"] = k
            rows.append(f)
    return rows


def _fit(rows, label, feats, depth=3):
    games = sorted({r["game_id"] for r in rows})
    test_g = set(games[::4])    # ~25% of games held out
    tr = [r for r in rows if r["game_id"] not in test_g]
    te = [r for r in rows if r["game_id"] in test_g]
    if not tr or not te or len({r[label] for r in tr}) < 2:
        return None
    Xtr = [[r[f] for f in feats] for r in tr]
    ytr = [r[label] for r in tr]
    Xte = [[r[f] for f in feats] for r in te]
    yte = [r[label] for r in te]
    if len(set(yte)) < 2:
        return None
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(5, len(tr) // 50), random_state=0).fit(Xtr, ytr)
    pred = t.predict(Xte)
    mi = mutual_info_classif(Xtr, ytr, discrete_features=True, random_state=0)
    top = sorted(zip(feats, mi), key=lambda x: -x[1])[:6]
    # nulls
    base = max(sum(yte) / len(yte), 1 - sum(yte) / len(yte))    # majority class
    zero_pred = [r["is_option_zero"] for r in te]               # option-zero null (predict the first option)
    return {
        "n_train": len(tr), "n_test": len(te), "pos_rate": round(sum(ytr) / len(tr), 3),
        "tree_f1": round(f1_score(yte, pred, zero_division=0), 3),
        "tree_precision": round(precision_score(yte, pred, zero_division=0), 3),
        "tree_recall": round(recall_score(yte, pred, zero_division=0), 3),
        "majority_null_acc": round(base, 3),
        "option_zero_null_f1": round(f1_score(yte, zero_pred, zero_division=0), 3) if label == "y_select" else None,
        "top_features_mutual_info": [(f, round(float(m), 3)) for f, m in top],
        "tree_rules": export_text(t, feature_names=list(feats), max_depth=depth).strip()[:1500],
    }


def main():
    rows = build_rows()
    fams = collections.Counter(r["family"] for r in rows)
    dec_ids = {r["decision_id"] for r in rows}
    dev_decisions = {r["decision_id"] for r in rows if r["deviates"]}
    report = {
        "n_option_rows": len(rows), "n_decisions": len(dec_ids),
        "family_distribution": dict(fams),
        "deviation_rate_from_option_zero": round(len(dev_decisions) / max(1, len(dec_ids)), 3),
        "selector_global": _fit(rows, "y_select", NUM_FEATS),
        "deviation_detector": _fit(rows, "deviates", [f for f in NUM_FEATS if f not in ("raw_index", "is_option_zero")]),
        "suppression_detector": _fit(rows, "y_suppress_zero", NUM_FEATS),
        "per_family_selector": {},
    }
    for fam in ("EVOLVE", "ATTACH", "PLAY", "SELECT_CARD", "ATTACK", "RETREAT"):
        fr = [r for r in rows if r["family"] == fam]
        if len({r["y_select"] for r in fr}) == 2 and len(fr) >= 30:
            report["per_family_selector"][fam] = {"n": len(fr), **(_fit(fr, "y_select", NUM_FEATS, depth=2) or {})}
    (OUT / "reconstruction_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"n_decisions": len(dec_ids), "n_rows": len(rows), "families": dict(fams),
                      "deviation_rate": report["deviation_rate_from_option_zero"],
                      "selector_tree_f1": report["selector_global"]["tree_f1"] if report["selector_global"] else None,
                      "selector_top": report["selector_global"]["top_features_mutual_info"] if report["selector_global"] else None,
                      "per_family": {k: {"n": v["n"], "f1": v.get("tree_f1")} for k, v in report["per_family_selector"].items()}},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
