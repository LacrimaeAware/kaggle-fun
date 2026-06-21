"""Depth A/B: does deeper search buy win rate? Runs 2-ply (agent_search2, branches the opponent's
top-k replies) vs 1-ply (agent_search), same pilot deck, head-to-head, seat-swapped. Both are in the
SAME baseline module, so no isolation needed. The harness is fair (base-vs-base ~0.5, measured).

  python tools/ab_depth_v1.py --games 16
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_candidate_v1 as AB
sys.path.insert(0, str(ROOT / "agent"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--progress", type=int, default=4)
    args = ap.parse_args()

    base = importlib.import_module("main")
    if not hasattr(base, "agent_search2"):
        print("ERROR: baseline main has no agent_search2 (2-ply). Available:",
              [a for a in dir(base) if a.startswith("agent")])
        return
    PILOT = AB.pilot_deck()
    base.DECK = PILOT

    print(f"2-ply (agent_search2) [A] vs 1-ply (agent_search) [B], pilot deck both, "
          f"{args.games} games seat-swapped")
    r = AB.run(args.games, base.agent_search2, base.agent_search, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = AB.wilson(r["wins_a"], dec)
    wr = r["wins_a"] / dec if dec else 0.0
    print(f"\n=> 2-ply vs 1-ply: {wr:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  "
          f"({r['wins_a']}-{r['wins_b']}, {r['draws']}d, {r['errors']}e, {r['seconds']}s)")
    print("   >0.5 = depth helps even at the same 0.6s budget.")


if __name__ == "__main__":
    main()
