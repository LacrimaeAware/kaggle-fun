"""Continuous Terrain V1 -- A2/A3 dataset selection + matched terrain rings (cheap, no engine).

Consumes the mine_terrain shards, selects a game-diverse dataset spanning the safe->catastrophic continuum,
and for every c1 / high-regret anchor attaches matched rings (Ring 1-4) drawn from DIFFERENT games. Writes a
selection manifest the repeated-label stage consumes. Game-balanced; c1 game-capped.

    python tools/build_terrain_v1.py --target 600 --max-per-game 24 --c1-per-game 5
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "data" / "manifests"

# canonical seeds (from round-2; held entirely out as eval_only)
SEEDS = [
    {"decision_id": "80251230.json:12", "group_id": "80251230.json", "source": {"file": "80251230.json", "step": 12, "player": 0},
     "obs_hash": "66876f66e3db", "terrain_class": "c1_seed", "ring": 0, "anchor_id": "seed_catastrophic", "eval_only": True},
    {"decision_id": "80252701.json:56", "group_id": "80252701.json", "source": {"file": "80252701.json", "step": 56, "player": 1},
     "obs_hash": "0c730f2f8916", "terrain_class": "c2_seed", "ring": 0, "anchor_id": "seed_false_positive", "eval_only": True},
]

# numeric match features for ring distance
NUMF = ["turn_proxy_prizes_taken", "option_count", "prize_lead", "board_dev", "attacker_ready"]


def _dist(a, b):
    s = 0.0
    for k in NUMF:
        s += (float(a.get(k, 0)) - float(b.get(k, 0))) ** 2
    return s ** 0.5 + 3.0 * abs(float(a.get("crit_band", 0)) - float(b.get("crit_band", 0)))


def _primary_class(tags):
    for c in ("c1", "c2", "c3_boundary", "high_regret_option", "moderate_regret", "unstable_proxy", "safe_ordinary"):
        if c in tags:
            return c
    return tags[0] if tags else "safe_ordinary"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--max-per-game", type=int, default=24)
    ap.add_argument("--c1-per-game", type=int, default=5)
    ap.add_argument("--out", default="_terrain_selection.json")
    args = ap.parse_args()

    rows = []
    shard_files = (glob.glob(str(ROOT / "tools" / "_terrain_crit*.json")) +
                   glob.glob(str(ROOT / "tools" / "_terrain_spread*.json")) +
                   glob.glob(str(ROOT / "tools" / "_terrain_sup*.json")))
    for fn in sorted(shard_files):
        rows += json.load(open(fn, encoding="utf-8")).get("rows", [])
    uniq = {}
    for r in rows:
        uniq.setdefault(r["decision_id"], r)
    rows = list(uniq.values())
    for r in rows:
        r["_cls"] = _primary_class(r["terrain_tags"])
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["_cls"]].append(r)
    total_screened = sum(json.load(open(fn, encoding="utf-8")).get("screened", 0)
                         for fn in glob.glob(str(ROOT / "tools" / "_terrain_*.json")) if "shard" not in fn and "selection" not in fn)

    selected = {}
    game_count = Counter()

    def take(r, cls, ring=None, anchor=None):
        did = r["decision_id"]
        if did in selected:
            return False
        if game_count[r["group_id"]] >= args.max_per_game:
            return False
        selected[did] = {"decision_id": did, "group_id": r["group_id"], "step": r["step"],
                         "player": r.get("player"), "obs_hash": r["obs_hash"],
                         "source": {"file": r["group_id"], "step": r["step"], "player": r.get("player")},
                         "terrain_class": cls, "ring": ring, "anchor_id": anchor, "match": r["match"]}
        game_count[r["group_id"]] += 1
        return True

    # 1. c1 anchors: round-robin across games, capped per game
    c1 = sorted(by_cls.get("c1", []), key=lambda r: -abs(r["opt_summary"][r["sel"]]["regret"]) if r.get("sel") is not None and r["sel"] < len(r["opt_summary"]) else 0)
    c1_per_game = Counter()
    anchors = []
    # multiple passes so no single game dominates
    for _pass in range(args.c1_per_game):
        for r in c1:
            if c1_per_game[r["group_id"]] != _pass:
                continue
            if c1_per_game[r["group_id"]] >= args.c1_per_game:
                continue
            if take(r, "c1", ring=0, anchor=r["decision_id"]):
                anchors.append(r)
                c1_per_game[r["group_id"]] += 1

    # 2. rings per anchor, from DIFFERENT games
    ring_pools = {1: by_cls.get("safe_ordinary", []) + by_cls.get("c2", []),
                  2: by_cls.get("c3_boundary", []),
                  3: by_cls.get("safe_ordinary", []) + by_cls.get("moderate_regret", []),
                  4: by_cls.get("safe_ordinary", [])}
    for a in anchors:
        for ring, pool in ring_pools.items():
            cand = [r for r in pool if r["group_id"] != a["group_id"]
                    and r["match"]["action_family"] == a["match"]["action_family"]
                    and r["decision_id"] not in selected]
            if not cand:
                cand = [r for r in pool if r["group_id"] != a["group_id"] and r["decision_id"] not in selected]
            if not cand:
                continue
            best = min(cand, key=lambda r: _dist(r["match"], a["match"]))
            take(best, f"ring{ring}", ring=ring, anchor=a["decision_id"])

    # 3. quota fill: c2 >=75, c3 >=75, plus moderate/unstable/high_regret_option/safe, game-balanced
    def fill(cls, need):
        pool = sorted(by_cls.get(cls, []), key=lambda r: game_count[r["group_id"]])
        for r in pool:
            if sum(1 for s in selected.values() if s["terrain_class"] == cls) >= need:
                break
            take(r, cls)

    fill("c2", 90)
    fill("c3_boundary", 90)
    fill("high_regret_option", 40)
    fill("moderate_regret", 60)
    fill("unstable_proxy", 40)
    fill("safe_ordinary", max(60, args.target // 6))

    # 4. top up to target with the most game-balanced remaining
    if len(selected) < args.target:
        rest = sorted((r for r in rows if r["decision_id"] not in selected), key=lambda r: game_count[r["group_id"]])
        for r in rest:
            if len(selected) >= args.target:
                break
            take(r, r["_cls"])

    sel_rows = list(selected.values()) + SEEDS
    out = {"rows": sel_rows, "n": len(sel_rows), "screened_total": total_screened,
           "class_counts": dict(Counter(r["terrain_class"] for r in sel_rows)),
           "ring_counts": dict(Counter(r.get("ring") for r in sel_rows if r.get("ring") is not None)),
           "games": len(set(r["group_id"] for r in sel_rows)),
           "c1_games": len(set(r["group_id"] for r in sel_rows if r["terrain_class"] == "c1")),
           "anchors": len(anchors)}
    json.dump(out, open(MAN / args.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print(f"-> data/manifests/{args.out}  ({len(sel_rows)} rows, {out['games']} games)")


if __name__ == "__main__":
    main()
