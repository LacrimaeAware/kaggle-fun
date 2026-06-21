"""Continuous Terrain V1 -- A6 assemble + summarize. Merge repeated-label shards into the self-contained
dataset, compute the terrain summary, and report. No engine.

    python tools/finalize_terrain_v1.py
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "data" / "manifests"


def sel_opt(r):
    s = r.get("search_selected_option")
    return next((o for o in r["options"] if o["index"] == s), None)


def derive_terrain(r):
    """AUTHORITATIVE terrain flags from the REPEATED labels (more reliable than the single-pass mine tag)."""
    so = sel_opt(r) or {}
    opts = r["options"]
    hr = [o["high_regret_prob"] for o in opts]
    mixed = any(0.0 < p < 1.0 for p in hr) or (1.0 in [round(p) for p in hr] and 0.0 in [round(p) for p in hr])
    spread = r.get("value_spread", 0) or 0
    sel_hr = so.get("high_regret_prob", 0)
    modal = r.get("modal_action_stability", 1.0)
    dangerous_sibling = any(o["high_regret_prob"] >= 0.5 or o["unacceptable_prob"] >= 0.5
                            for o in opts if o["index"] != r.get("search_selected_option"))
    return {
        "repro_c1": int(sel_hr >= 0.5),
        "unstable": int(modal < 0.75),
        "boundary": int(mixed or (0.25 < sel_hr < 0.75) or (modal < 0.75 and dangerous_sibling)),
        "safe_selected": int(sel_hr == 0.0 and not dangerous_sibling),
        "has_dangerous_sibling": int(dangerous_sibling),
        "sel_high_regret_prob": round(sel_hr, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="_terrain_lab*.json")
    ap.add_argument("--out", default="continuous_terrain_v1")
    ap.add_argument("--from-jsonl", action="store_true",
                    help="recompute summary from the existing jsonl (after re-featurize); do not rebuild it")
    args = ap.parse_args()

    out = MAN / f"{args.out}.jsonl"
    failed, cost = [], 0.0
    if args.from_jsonl:
        recs = [json.loads(l) for l in open(out, encoding="utf-8")]
        failed = json.load(open(MAN / f"{args.out}_summary.json", encoding="utf-8")).get("failed", [])
    else:
        recs = []
        for fn in sorted(glob.glob(str(ROOT / "tools" / args.glob))):
            d = json.load(open(fn, encoding="utf-8"))
            recs += d.get("records", [])
            failed += d.get("failed", [])
            cost += d.get("cost_s", 0)
        uniq = {}
        for r in recs:
            uniq.setdefault(r["decision_id"], r)
        recs = list(uniq.values())
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    # attach AUTHORITATIVE derived terrain flags (from repeated labels) + rewrite jsonl
    for r in recs:
        r["terrain_authoritative"] = derive_terrain(r)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    auth = Counter()
    for r in recs:
        ta = r["terrain_authoritative"]
        for k in ("repro_c1", "unstable", "boundary", "safe_selected", "has_dangerous_sibling"):
            auth[k] += ta[k]

    n_opt = sum(len(r["options"]) for r in recs)
    games = set(r["group_id"] for r in recs)
    cls = Counter(r.get("terrain_class") for r in recs)
    rings = Counter(r.get("ring") for r in recs if r.get("ring") is not None)
    # reproduced c1 = c1/c1_seed decisions whose selected option reproduces high_regret (prob>=0.5)
    c1_recs = [r for r in recs if (r.get("terrain_class") in ("c1", "c1_seed"))]
    repro = [r for r in c1_recs if (sel_opt(r) or {}).get("high_regret_prob", 0) >= 0.5]
    repro_games = Counter(r["group_id"] for r in repro)
    top_game_share = (max(repro_games.values()) / len(repro)) if repro else 0.0
    # stability
    modal = [r["modal_action_stability"] for r in recs if "modal_action_stability" in r]
    ent = [r["live_action_entropy"] for r in recs if "live_action_entropy" in r]
    hr_opts = sum(1 for r in recs for o in r["options"] if o["high_regret_prob"] >= 0.5)
    unacc_opts = sum(1 for r in recs for o in r["options"] if o["unacceptable_prob"] >= 0.5)
    full = sum(r.get("coverage", {}).get("all_siblings_completed", 0) for r in recs)
    # semantic coverage
    sc = Counter(o["semantic_vector"]["semantic_coverage"] for r in recs for o in r["options"])

    summary = {
        "decisions": len(recs), "options": n_opt, "games": len(games),
        "class_counts": dict(cls), "ring_counts": dict(rings),
        "authoritative_terrain": dict(auth),
        "c1_candidates": len(c1_recs), "reproduced_c1": len(repro),
        "reproduced_c1_games": len(repro_games), "reproduced_c1_top_game_share": round(top_game_share, 3),
        "c2": cls.get("c2", 0) + cls.get("ring1", 0), "c3_boundary": cls.get("c3_boundary", 0) + cls.get("ring2", 0),
        "high_regret_options(prob>=.5)": hr_opts, "unacceptable_options(prob>=.5)": unacc_opts,
        "mean_modal_action_stability": round(statistics.fmean(modal), 3) if modal else None,
        "mean_live_action_entropy": round(statistics.fmean(ent), 3) if ent else None,
        "all_siblings_completed": f"{full}/{len(recs)}",
        "semantic_coverage": dict(sc),
        "semantic_trust_rate": round(sum(sc.get(t, 0) for t in ("decoded", "override", "energy", "pokemon_meta", "tool"))
                                     / max(1, sum(sc.values())), 3),
        "eval_only": sum(1 for r in recs if r.get("eval_only")),
        "failed": failed, "label_cost_s": round(cost, 0),
    }
    json.dump(summary, open(MAN / f"{args.out}_summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"-> data/manifests/{args.out}.jsonl")


if __name__ == "__main__":
    main()
