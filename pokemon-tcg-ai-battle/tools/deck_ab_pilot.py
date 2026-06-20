"""Deck A/B: our SAME agent_search piloting the best pilot's near-identical list (episode 80723114, player 0)
vs our DENPA92 deck, head-to-head, seat-swapped, Wilson CI. Deck value is policy-coupled, so this our-search-
on-both comparison is the relevant one (the only thing that has moved our LB).

    python tools/deck_ab_pilot.py --games 60
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import main as M               # noqa: E402
import search_sprint as SS      # noqa: E402

EPISODE = ROOT / "data" / "external" / "replays" / "80723114.json"


def pilot_deck():
    d = json.load(open(EPISODE, encoding="utf-8"))
    for step in d["steps"][:6]:
        if isinstance(step, list) and step and isinstance(step[0], dict):
            act = step[0].get("action")
            if isinstance(act, list) and len(act) == 60:
                return list(act)
    raise SystemExit("pilot deck not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--progress", type=int, default=10)
    args = ap.parse_args()
    pilot = pilot_deck()
    ours = list(M.DECK)
    pc, oc = Counter(pilot), Counter(ours)
    overlap = sum((pc & oc).values())
    print(f"Pilot deck: {len(pilot)} cards, {len(pc)} distinct | overlap with DENPA92: {overlap}/60\n"
          f"DECK A/B: agent_search on PILOT list vs agent_search on DENPA92, {args.games} games, seat-swapped.\n"
          f"Read: if Pilot's Wilson lower bound > 0.5, our search pilots their list better -> candidate swap.\n",
          flush=True)
    SS.ab("Pilot-deck vs DENPA92 (A=Pilot)", SS.mk_deck(pilot), SS.mk_deck(ours), args.games, args.progress)


if __name__ == "__main__":
    main()
