"""Ablation: candidate vs baseline with selected deck_policy features DISABLED, to localize the
head-to-head regression. The rig is fair (base-copy vs base ~0.5), so whichever feature, when
disabled, restores ~0.5 is the culprit.

  python tools/ab_ablate_v1.py --off subdecision priors rollout --games 12

Features:
  subdecision  -> DP.choose_subdecision returns None (live + rollout takeover off)
  priors       -> DP.root_option_priors returns None (search option ordering/tie-break off)
  rollout      -> DP.rollout_choice returns None (rollout falls back to baseline aggro pick)
With all three off, the candidate ~= baseline + (final-prize forced KO + dynamic Powerful Hand).
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_candidate_v1 as AB   # CAND_SRC defaults to the inbox candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", nargs="*", default=[], choices=["subdecision", "priors", "rollout"])
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--progress", type=int, default=4)
    args = ap.parse_args()

    AB.build_candidate_pkg()   # rebuild _candv1 FROM THE CANDIDATE (overwrites any scratch)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "agent"))
    base_main = importlib.import_module("main")
    cand_main = importlib.import_module("_candv1.main")
    cDP = importlib.import_module("_candv1.deck_policy")
    PILOT = AB.pilot_deck()
    base_main.DECK = PILOT
    cand_main.DECK = PILOT

    off = set(args.off)
    if "subdecision" in off:
        cDP.choose_subdecision = lambda *a, **k: None
    if "priors" in off:
        cDP.root_option_priors = lambda *a, **k: None
    if "rollout" in off:
        cDP.rollout_choice = lambda *a, **k: None

    label = "candidate[" + ("+".join(sorted(off)) + " OFF" if off else "full") + "]"
    print(f"{label} (A) vs baseline (B), pilot deck both, {args.games} games seat-swapped")
    r = AB.run(args.games, cand_main.agent_search, base_main.agent_search, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = AB.wilson(r["wins_a"], dec)
    wr = r["wins_a"] / dec if dec else 0.0
    print(f"\n=> {label} vs baseline: {wr:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  "
          f"({r['wins_a']}-{r['wins_b']}, {r['draws']}d, {r['errors']}e)")


if __name__ == "__main__":
    main()
