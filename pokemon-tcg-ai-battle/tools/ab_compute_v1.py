"""Does spending the real time budget convert to wins? Runs an EXPENSIVE search (configurable
opp_k / N_DETERM / per-decision budget) against the DEPLOYED 1-ply (opp_k=0, N=8, 0.6s), same
pilot deck, head-to-head. The expensive agent runs in an isolated baseline copy (_candv1) so its
N_DETERM can be raised independently. >0.5 means the 600s/game headroom is a real lever.

  python tools/ab_compute_v1.py --oppk 2 --ndeterm 16 --budget 2.5 --games 12
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_candidate_v1 as AB
AB.CAND_SRC = ROOT / "agent"            # build _candv1 from the BASELINE
AB.MODS = ["main", "search", "eval", "features"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oppk", type=int, default=2)
    ap.add_argument("--ndeterm", type=int, default=16)
    ap.add_argument("--budget", type=float, default=2.5)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--progress", type=int, default=3)
    args = ap.parse_args()

    AB.build_candidate_pkg()            # _candv1 = isolated baseline copy
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "agent"))
    base = importlib.import_module("main")
    exp = importlib.import_module("_candv1.main")
    es = importlib.import_module("_candv1.search")
    PILOT = AB.pilot_deck()
    base.DECK = PILOT
    exp.DECK = PILOT
    es.N_DETERM = args.ndeterm          # read at call time inside the determinization loop

    def expensive(obs):
        try:
            if obs.get("select") is None:
                return list(exp.DECK)
            mv = exp._forced_move(obs)
            if mv is not None:
                return mv
            mv = es.best_option(obs, exp.DECK, leaf_mode="hand", opp_k=args.oppk, time_budget=args.budget)
            if mv is not None:
                return mv
        except Exception:
            pass
        return exp.agent(obs)

    print(f"EXPENSIVE (opp_k={args.oppk}, N_DETERM={args.ndeterm}, budget={args.budget}s) "
          f"vs DEPLOYED 1-ply (opp_k=0, N=8, 0.6s), pilot deck, {args.games} games seat-swapped")
    r = AB.run(args.games, expensive, base.agent_search, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = AB.wilson(r["wins_a"], dec)
    wr = r["wins_a"] / dec if dec else 0.0
    print(f"\n=> expensive vs deployed: {wr:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  "
          f"({r['wins_a']}-{r['wins_b']}, {r['draws']}d, {r['errors']}e, {r['seconds']}s)")
    print("   >0.5 = spending the time budget (depth + sampling) converts to wins.")


if __name__ == "__main__":
    main()
