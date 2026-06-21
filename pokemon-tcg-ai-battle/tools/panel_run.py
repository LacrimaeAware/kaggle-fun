"""Robustness panel: NEW agent (top-pilot deck + Powerful-Hand hoard term, W=15) and OLD agent (DENPA92 +
plain agent_search) each vs a field of opponents, seat-swapped, Wilson CI. One process per (side, opponent)
so the panel runs in parallel. Answers "is the new agent robustly better than the previous one across a
variety of decks", not just in the mirror.

    python tools/panel_run.py --side new --opp lucario --games 40 --out tools/_panel_new_lucario.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import main as M               # noqa: E402
import eval as EV               # noqa: E402
import search_sprint as SS      # noqa: E402
import cabt_arena as A          # noqa: E402


def pilot_deck():
    d = json.load(open(ROOT / "data" / "external" / "replays" / "80723114.json", encoding="utf-8"))
    for step in d["steps"][:6]:
        if isinstance(step, list) and step and isinstance(step[0], dict):
            act = step[0].get("action")
            if isinstance(act, list) and len(act) == 60:
                return list(act)
    raise SystemExit("pilot deck not found")


def with_w(fn, w):
    """Set the gated hoard weight before each call (sequential arena calls -> no cross-arm leakage)."""
    def ag(obs):
        EV.W_POWERFUL_HAND = w
        return fn(obs)
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["new", "old"], required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--progress", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    slate = json.load(open(ROOT / "tools" / "_deckoff_slate.json", encoding="utf-8"))
    PILOT, DENPA = pilot_deck(), list(M.DECK)
    new = with_w(SS.mk_deck(PILOT), 15.0)
    old = with_w(SS.mk_deck(DENPA), 0.0)
    opponents = {
        "first": with_w(A.AGENTS["first"], 0.0),
        "heuristic": with_w(A.AGENTS["heuristic"], 0.0),
        "lucario": with_w(SS.mk_deck(slate[0]["deck"]), 0.0),
        "iono": with_w(SS.mk_deck(slate[1]["deck"]), 0.0),
        "dragapult": with_w(SS.mk_deck(slate[4]["deck"]), 0.0),
        "old": old,
    }
    side = new if args.side == "new" else old
    opp = opponents[args.opp]
    label = f"{args.side} vs {args.opp}"
    print(f"{label}, {args.games} games seat-swapped\n", flush=True)
    r = A.run(args.games, side, opp, label=label, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = SS.wilson(r["wins_a"], dec)
    res = {"side": args.side, "opp": args.opp, "win_rate": round(r["a_win_rate_decided"], 3),
           "wilson95": [round(lo, 3), round(hi, 3)], "wins": r["wins_a"], "losses": r["wins_b"],
           "errors": r.get("errors_a", 0) + r.get("errors_b", 0), "s_per_game": r["s_per_game"]}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)
    EV.W_POWERFUL_HAND = 0.0
    print(f"=> {label}: {res['win_rate']} Wilson {res['wilson95']} ({res['wins']}-{res['losses']}, err {res['errors']})", flush=True)


if __name__ == "__main__":
    main()
