"""Branch A / Continuous Terrain V1 -- A1 candidate miner (shardable).

Scan high-criticality (and a spread band of) root decisions from replay games NOT in the round-2 13-game
dataset, run a CHEAP hand-only label (residual_risk_label k_outcome=0), and classify each into the terrain
continuum (safe .. c2 .. c3 .. unstable .. moderate-regret .. high-regret-option .. c1). Record the matching
features needed to build matched terrain rings (A2) WITHOUT re-screening. Repeated labels (A4) come later, on
the selected subset only.

Sharding is deterministic: sample one big pool, sort/stratify, each shard screens a disjoint slice.

    python tools/mine_terrain_v1.py --pool 14000 --band crit --start 0 --count 900 --out tools/_terrain_shard0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import teacher_api_v2 as T2               # noqa: E402
import audit_teacher_stability as A2      # noqa: E402
import label_requested_states as LRS      # noqa: E402
import features as FT                      # noqa: E402
import state_action_schema_v2 as SCH       # noqa: E402

MAN = ROOT / "data" / "manifests"
HRT = 5000.0
BOUND_DELTA = 2000.0
SE_HI = 4000.0
MOD_LO = 1500.0
# the 13 round-2 games to exclude (disjoint shards requirement)
R2_GAMES = {"80251230.json", "80251834.json", "80252178.json", "80252701.json", "80253044.json",
            "80253882.json", "80253888.json", "80270516.json", "80271583.json", "80275931.json",
            "80277480.json", "80279946.json", "80280539.json"}


def _deck_sig(deck):
    try:
        return abs(hash(tuple(sorted(int(x) for x in deck)))) % (10 ** 10)
    except Exception:
        return 0


def classify(lab):
    opts = {o["index"]: o for o in lab["options"]}
    sel = lab.get("search_selected_option")
    so = opts.get(sel)
    tags = []
    hr = [o["high_regret_flag"] for o in lab["options"]]
    unacc = [o["unacceptable_flag"] for o in lab["options"]]
    dmax = max((abs(o["delta_to_search"]) for o in lab["options"]), default=0)
    semax = max((o["value_se"] for o in lab["options"]), default=0)
    sel_reg = abs(so["regret"]) if so else 0
    if so and so["high_regret_flag"] == 1:
        tags.append("c1")                                   # search-selected catastrophe
    if so and so["high_regret_flag"] == 0 and any(o["unacceptable_flag"] == 1 for o in lab["options"] if o["index"] != sel):
        tags.append("c2")                                   # safe pick, dangerous siblings
    if (1 in hr and 0 in hr) or dmax > BOUND_DELTA:
        tags.append("c3_boundary")
    if 1 in hr:
        tags.append("high_regret_option")
    if so and MOD_LO < sel_reg <= HRT:
        tags.append("moderate_regret")
    if semax > SE_HI:
        tags.append("unstable_proxy")
    if not tags or (sel_reg <= MOD_LO and 1 not in hr and 1 not in unacc):
        tags.append("safe_ordinary")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="replays_20260618.json")
    ap.add_argument("--split", default="replays_20260618_split.json")
    ap.add_argument("--pool", type=int, default=14000)
    ap.add_argument("--band", choices=["crit", "spread"], default="crit")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(MAN / args.snapshot, encoding="utf-8"))
    split = json.load(open(ROOT / "data" / "splits" / args.split, encoding="utf-8"))
    pool = A2.sample_decisions(manifest, split, args.pool, verify=False)
    pool = [d for d in pool if d.get("file") not in R2_GAMES]
    for d in pool:
        d["_crit"] = T2.criticality_score(d["obs"])["score"]
    if args.band == "crit":
        pool.sort(key=lambda d: -d["_crit"])
    else:
        # spread: deterministic stride across the criticality-sorted list -> mixes high/low criticality
        pool.sort(key=lambda d: -d["_crit"])
        pool = pool[::3]
    sl = pool[args.start:args.start + args.count]

    rows, screened, t0 = [], 0, time.time()
    for d in sl:
        obs, deck = d["obs"], d["deck"]
        lab = T2.residual_risk_label(obs, deck, n_strong=32, k_outcome=0, high_regret_thresh=HRT, seed=1234)
        if not lab.get("applicable"):
            continue
        screened += 1
        f = FT.encode_state(obs)
        crit = lab["criticality"]
        cur = obs.get("current") or {}
        me = cur.get("yourIndex", 0)
        opts = obs["select"]["option"]
        sel = lab.get("search_selected_option")
        try:
            sem = list(SCH.semantic_action_key(opts[sel], cur, me)) if sel is not None and sel < len(opts) else []
        except Exception:
            sem = []
        prizes_taken = 12 - int(f.get("my_prizes_left", 6) or 6) - int(f.get("opp_prizes_left", 6) or 6)
        tags = classify(lab)
        rows.append({
            "decision_id": f"{d['file']}:{d['step']}", "group_id": d["file"], "step": d["step"],
            "player": d.get("player"), "obs_hash": LRS._hash(obs),
            "terrain_tags": tags, "criticality": round(crit["score"], 3),
            "match": {
                "action_family": int(opts[sel].get("type", -1)) if sel is not None and sel < len(opts) else -1,
                "sem_family": sem[0] if sem else None,
                "turn_proxy_prizes_taken": prizes_taken,
                "option_count": int(crit.get("n_eq_classes", len(opts))),
                "crit_band": round(crit["score"], 1),
                "prize_lead": int(f.get("prize_lead", 0) or 0),
                "can_ko": int(f.get("can_ko_opp_now", 0) > 0), "ko_back": int(crit.get("ko_back", 0)),
                "endgame": int(crit.get("endgame", 0)), "deck_sig": _deck_sig(deck),
                "board_dev": int(f.get("my_bench", 0) or 0) + int(f.get("opp_bench", 0) or 0),
                "attacker_ready": int(f.get("active_can_attack_now", 0) or 0),
            },
            "sel": sel, "stronger_argmax": lab.get("stronger_argmax_option"),
            "opt_summary": [{"i": o["index"], "regret": round(o["regret"], 1), "hr": o["high_regret_flag"],
                             "unacc": o["unacceptable_flag"], "delta": round(o["delta_to_search"], 1),
                             "se": round(o["value_se"], 1)} for o in lab["options"]],
        })
    json.dump({"band": args.band, "start": args.start, "count": args.count, "screened": screened,
               "rows": rows, "cost_s": round(time.time() - t0, 0)}, open(args.out, "w", encoding="utf-8"))
    nc1 = sum(1 for r in rows if "c1" in r["terrain_tags"])
    print(f"shard {args.band} start={args.start} screened={screened} c1={nc1} kept={len(rows)} "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
