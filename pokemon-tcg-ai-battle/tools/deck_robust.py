"""Deck-robustness sweep: run a policy pairing across a POOL of real decks, report per-deck win-rate.

Answers two things at once:
  - does a policy (e.g. search_v) pilot a given deck better or worse than another (e.g. search)?
  - how does the gap between two policies change as the DECK changes (policy adaptation across decks)?

Both sides pilot the SAME deck each block (mirror, seats swapped) so it is a pure policy comparison;
the deck is varied across blocks. Decks come from the replay DB (data/replay_db/decks.json) plus the
old embedded deck. Small n per deck for a directional read.

    python tools/deck_robust.py search_v:search --games 30
    python tools/deck_robust.py search:heuristic --games 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import cabt_arena as A  # noqa: E402
import main as M  # noqa: E402

CF = json.load(open(ROOT / "agent" / "card_features.json", encoding="utf-8"))
OLD_DECK = ([721] * 2 + [722] * 4 + [723] * 4 + [1092] + [1121] * 2 + [1145] * 2
            + [1163] * 2 + [1219] * 4 + [1227] * 4 + [1262] * 2 + [3] * 33)


def basics(deck):
    return sum(1 for c in deck if CF.get(str(c), {}).get("ct") == 0 and CF.get(str(c), {}).get("stage") == "basic")


def deck_pool():
    """Labelled decks: top archetypes from the replay DB by owner + the old embedded deck."""
    decks = json.load(open(ROOT / "data" / "replay_db" / "decks.json", encoding="utf-8"))
    want = [("onechan1", 9), ("DENPA92", 8), ("Heisei", 8)]
    pool = {}
    for name, b in want:
        cand = [d for d in decks if d["basics"] == b and any(n == name for n, _ in d["top_names"])]
        if cand:
            pool[f"{name}({b}b)"] = cand[0]["deck"]
    pool["old-Abomasnow(6b)"] = OLD_DECK
    return pool


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matchup", nargs="?", default="search_v:search")
    ap.add_argument("--games", type=int, default=30)
    args = ap.parse_args()
    a, b = args.matchup.split(":")
    pool = deck_pool()
    print(f"deck-robustness: {a} (A) vs {b} (B), {args.games} games/deck, mirror deck, seat-swapped\n", flush=True)
    print(f"  {'deck':<22} {'basics':>6} {'A win':>7} {'wilson95':>16}  (n)", flush=True)
    rows = []
    orig = M.DECK
    try:
        for label, deck in pool.items():
            M.DECK = list(deck)                      # both policies pilot this deck (read at select)
            r = A.run(args.games, A.AGENTS[a], A.AGENTS[b], label=label, progress=10)
            dec = r["wins_a"] + r["wins_b"]
            lo, hi = wilson(r["wins_a"], dec)
            print(f"  {label:<22} {basics(deck):>6} {r['a_win_rate_decided']:>7.3f} [{lo:.2f},{hi:.2f}]"
                  f"      ({r['wins_a']}-{r['wins_b']}, {r['errors']}e)", flush=True)
            rows.append((label, r["a_win_rate_decided"]))
    finally:
        M.DECK = orig
    print("\nRead: how A's edge over B shifts across decks. A flat ~0.5 everywhere = no policy difference;"
          " a deck where A >> B = A adapts to that deck better.", flush=True)


if __name__ == "__main__":
    main()
