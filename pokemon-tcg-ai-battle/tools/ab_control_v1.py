"""CONTROL: run the baseline against an ISOLATED COPY OF ITSELF through the same head-to-head
harness used for the candidate A/B. Two identical agents must score ~0.5 if the rig is fair.
If this is skewed, the head-to-head harness (two forward-model searchers sharing one native cg
engine in one process) is biased, and the candidate's 5-95 is an artifact, not a real result.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_candidate_v1 as AB

# Point the isolated-package builder at the BASELINE agent dir instead of the candidate.
AB.CAND_SRC = ROOT / "agent"
AB.MODS = ["main", "search", "eval", "features"]   # baseline has no deck_policy


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--progress", type=int, default=4)
    args = ap.parse_args()

    AB.build_candidate_pkg()   # _candv1 is now an isolated copy of the BASELINE
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "agent"))
    base_main = importlib.import_module("main")
    ctrl_main = importlib.import_module("_candv1.main")
    PILOT = AB.pilot_deck()
    base_main.DECK = PILOT
    ctrl_main.DECK = PILOT

    print(f"CONTROL: baseline-copy (A) vs baseline (B), pilot deck both, {args.games} games seat-swapped")
    r = AB.run(args.games, ctrl_main.agent_search, base_main.agent_search, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = AB.wilson(r["wins_a"], dec)
    wr = r["wins_a"] / dec if dec else 0.0
    print(f"\n=> base-copy vs base: {wr:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  "
          f"({r['wins_a']}-{r['wins_b']}, {r['draws']}d, {r['errors']}e)")
    print("   Expected ~0.5 if the head-to-head rig is fair. Far from 0.5 = the rig is the bug.")


if __name__ == "__main__":
    main()
