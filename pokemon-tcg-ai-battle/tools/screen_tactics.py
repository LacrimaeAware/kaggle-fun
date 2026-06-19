"""Cheap directional A/B screen for Branch A tactical-floor candidates vs production agent_search.

Same deck both sides (DENPA92), seat-swapped, Wilson 95% CI. Exploration screen, not a promotion test.

    python tools/screen_tactics.py draw --games 30
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import cabt_arena as A          # noqa: E402
import main as M                # noqa: E402
import search_live_v2 as V2     # noqa: E402

CANDS = {"draw": V2.agent_search_draw, "gust": V2.agent_search_gust,
         "evolve": V2.agent_search_evolve, "tactical": V2.agent_search_tactical}


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - m) / d, 3), round((c + m) / d, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand", choices=list(CANDS))
    ap.add_argument("--games", type=int, default=30)
    args = ap.parse_args()
    a, b = CANDS[args.cand], M.agent_search
    print(f"SCREEN: agent_search_{args.cand} (A) vs production agent_search (B) | {args.games} games | "
          f"DENPA92 both sides | seat-swapped\n", flush=True)
    r = A.run(args.games, a, b, label=f"{args.cand} vs search", progress=5)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = wilson(r["wins_a"], dec)
    verdict = "A BETTER" if lo > 0.5 else ("A WORSE" if hi < 0.5 else "tie / inconclusive (CI spans 0.5)")
    print(f"\n=> agent_search_{args.cand} {r['a_win_rate_decided']:.3f}  Wilson95 [{lo},{hi}]  {verdict}"
          f"  ({r['wins_a']}-{r['wins_b']}, {r['draws']}d {r['errors']}e, {r['s_per_game']}s/g)")


if __name__ == "__main__":
    main()
