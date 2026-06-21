"""Model B / B1: build the expert-policy dataset from strong-pilot replays.

For every real decision (a select with >=2 options and a recorded action) we store a grouped-sibling
row: the root public-state features, every legal option's (type, acting-card-id, target-id), the
expert's selected option(s), the prompt context, and metadata (player, game, deck tier, outcome).
Tiered by deck similarity to OUR deck so the deck-specialist head can train on Tier 1-2 and the
generic-opponent head on all tiers.

  python tools/build_expert_policy_dataset_v1.py
Output: data/expert_policy/dataset_v1.jsonl  +  docs/workstreams/policy_guided_search_v1_b_results.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import features as FT   # encode_state / vectorize / FEATURE_KEYS
import main as M        # reference DECK

CF = json.load(open(ROOT / "agent" / "card_features.json", encoding="utf-8"))
CDB = json.load(open(ROOT / "agent" / "card_stats.json", encoding="utf-8"))
REPLAYS = ROOT / "data" / "external" / "replays"
OUT_DIR = ROOT / "data" / "expert_policy"
OUT = OUT_DIR / "dataset_v1.jsonl"
SUMMARY = ROOT / "docs" / "workstreams" / "policy_guided_search_v1_b_results.json"

ALAKA_LINE = {741, 742, 743}
DUNS_LINE = {65, 305, 66}
REF = Counter(M.DECK)                      # our production deck (DENPA92 Dudunsparce/Alakazam)
PLAY, ATTACH, EVOLVE, ABILITY, ATTACK, CARD = 7, 8, 9, 10, 13, 3


def cid(c):
    if isinstance(c, dict):
        v = c.get("id")
        return int(v) if v is not None else None
    return int(c) if isinstance(c, (int, float)) else None


def deck_of(d, seat):
    for step in d["steps"][:8]:
        if isinstance(step, list) and len(step) > seat and isinstance(step[seat], dict):
            a = step[seat].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def tier_of(deck):
    if not deck:
        return None
    c = Counter(deck)
    overlap = sum((c & REF).values())
    if overlap >= 56:
        return 1
    if overlap >= 45:
        return 2
    board = set(deck)
    if (board & ALAKA_LINE) and (board & DUNS_LINE):
        return 3
    return 4


def opt_card(opt, sel, player):
    """Best-effort acting/selected card id for an option."""
    t = opt.get("type")
    idx = opt.get("index")
    area = opt.get("area")
    if t in (PLAY, ATTACH, EVOLVE) and area in (2, None) and isinstance(idx, int):
        hand = player.get("hand") or []
        if 0 <= idx < len(hand):
            return cid(hand[idx])
    if t == CARD:
        deck = sel.get("deck") or []
        if isinstance(idx, int) and 0 <= idx < len(deck):
            return cid(deck[idx])
    if t == ATTACK:
        a = player.get("active") or []
        if a and a[0]:
            return cid(a[0])
    return None


def opt_target(opt, player, opp):
    """Target Pokemon id for in-play-targeted options (attach/ability/boss/etc.)."""
    area, idx = opt.get("inPlayArea"), opt.get("inPlayIndex")
    if area is None or idx is None:
        return None
    pidx = opt.get("playerIndex")
    p = opp if (pidx is not None and pidx != opt.get("_me", pidx)) else player
    slots = ([(p.get("active") or [None])[0]] + list(p.get("bench") or []))
    if isinstance(idx, int) and 0 <= idx < len(slots) and slots[idx]:
        return cid(slots[idx])
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(REPLAYS / "*.json")))
    rows = 0
    by_tier = Counter()
    by_ctx = Counter()
    by_seltype = Counter()
    players = set()
    n_games = 0
    with open(OUT, "w", encoding="utf-8") as fout:
        for fn in files:
            try:
                d = json.load(open(fn, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(d, dict) or not d.get("steps"):
                continue
            ep = d.get("info", {}).get("EpisodeId") or Path(fn).stem
            names = (d.get("info") or {}).get("TeamNames") or d.get("names") or [None, None]
            rew = d.get("rewards") or [None, None]
            decks = {s: deck_of(d, s) for s in (0, 1)}
            tiers = {s: tier_of(decks[s]) for s in (0, 1)}
            if tiers[0] is None and tiers[1] is None:
                continue
            n_games += 1
            for step in d["steps"]:
                if not isinstance(step, list):
                    continue
                for seat in (0, 1):
                    if seat >= len(step) or not isinstance(step[seat], dict):
                        continue
                    ag = step[seat]
                    obs = ag.get("observation") or {}
                    sel = obs.get("select") or {}
                    opts = sel.get("option") or []
                    act = ag.get("action")
                    if len(opts) < 2 or not isinstance(act, list) or not act:
                        continue
                    cur = obs.get("current") or {}
                    players_st = cur.get("players") or []
                    yi = cur.get("yourIndex", seat)
                    me = players_st[yi] if yi < len(players_st) else {}
                    opp = players_st[1 - yi] if len(players_st) > 1 else {}
                    try:
                        sf = FT.vectorize(FT.encode_state(obs))
                    except Exception:
                        continue
                    odesc = []
                    for o in opts:
                        odesc.append([o.get("type"), opt_card(o, sel, me), opt_target(o, me, opp)])
                    row = {
                        "ep": str(ep), "player": names[seat], "seat": seat, "tier": tiers[seat],
                        "turn": cur.get("turn"), "tac": cur.get("turnActionCount"),
                        "ctx": sel.get("context"), "k": sel.get("maxCount"), "mn": sel.get("minCount"),
                        "opts": odesc, "sel": act,
                        "won": (1 if (rew[seat] is not None and rew[1 - seat] is not None and rew[seat] > rew[1 - seat]) else 0),
                        "sf": [round(x, 3) for x in sf],
                    }
                    fout.write(json.dumps(row) + "\n")
                    rows += 1
                    by_tier[tiers[seat]] += 1
                    by_ctx[sel.get("context")] += 1
                    sel_type = opts[act[0]].get("type") if (sel.get("maxCount") == 1 and 0 <= act[0] < len(opts)) else "multi"
                    by_seltype[sel_type] += 1
                    if names[seat]:
                        players.add(names[seat])

    summary = {
        "rows": rows, "games": n_games, "distinct_players": len(players),
        "feature_keys": FT.FEATURE_KEYS,
        "by_tier": {str(k): v for k, v in sorted(by_tier.items(), key=lambda kv: str(kv[0]))},
        "rows_our_deck_tier1_2": by_tier[1] + by_tier[2],
        "by_selected_type": {str(k): v for k, v in by_seltype.most_common()},
        "top_contexts": {str(k): v for k, v in by_ctx.most_common(15)},
        "output": str(OUT),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("rows", "games", "distinct_players", "by_tier", "rows_our_deck_tier1_2", "by_selected_type")}, indent=1))
    print("wrote", OUT, "and", SUMMARY)


if __name__ == "__main__":
    main()
