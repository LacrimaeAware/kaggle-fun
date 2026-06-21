"""Clean cross-check: candidate vs first_agent AND baseline vs first_agent, same pilot deck.

Each game has exactly ONE searching agent (first_agent does not search). This matches every
historical measurement and removes the two-searchers-in-one-process confound of the head-to-head.
If candidate-vs-first is much worse than baseline-vs-first, the candidate is genuinely weaker;
if they are close, the head-to-head blowout was a mirror-match artifact.

    python tools/ab_vs_first_v1.py --games 30
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_candidate_v1 as AB   # reuse build_candidate_pkg, pilot_deck, wilson, run, make, cabt


def first_samedeck_factory(deck):
    def ag(obs):
        return list(deck) if obs.get("select") is None else AB.cabt.first_agent(obs)
    return ag


def report(name, r):
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = AB.wilson(r["wins_a"], dec)
    wr = r["wins_a"] / dec if dec else 0.0
    print(f"=> {name} vs first: {wr:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  "
          f"({r['wins_a']}-{r['wins_b']}, {r['draws']}d, {r['errors']}e)", flush=True)
    return {"agent": name, "win_rate": round(wr, 3), "wilson95": [round(lo, 3), round(hi, 3)],
            "wins": r["wins_a"], "losses": r["wins_b"], "draws": r["draws"], "errors": r["errors"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--progress", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "tools" / "_ab_vs_first_v1.json"))
    args = ap.parse_args()

    AB.build_candidate_pkg()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "agent"))
    base_main = importlib.import_module("main")
    cand_main = importlib.import_module("_candv1.main")
    PILOT = AB.pilot_deck()
    base_main.DECK = PILOT
    cand_main.DECK = PILOT
    first = first_samedeck_factory(PILOT)

    import json
    print(f"baseline vs first ({args.games} games)", flush=True)
    rb = AB.run(args.games, base_main.agent_search, first, progress=args.progress)
    print(f"candidate vs first ({args.games} games)", flush=True)
    rc = AB.run(args.games, cand_main.agent_search, first, progress=args.progress)
    out = {"baseline": report("baseline", rb), "candidate": report("candidate", rc)}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
