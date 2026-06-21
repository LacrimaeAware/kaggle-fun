"""Heuristic A/B: agent_search WITH the gated 'hoard for Powerful Hand' leaf term (eval.W_POWERFUL_HAND>0)
vs WITHOUT it (=0), same deck, seat-swapped. Both arms keep the forced-move floor (lethal/KO still taken),
so the term only affects non-lethal decisions. The win-rate A/B is the only judge.

    python tools/hoard_ab.py --deck denpa92 --w 15 --games 60 --out tools/_hoard_denpa92.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", choices=["denpa92", "pilot"], default="denpa92")
    ap.add_argument("--w", type=float, default=15.0)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--progress", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    deck = list(M.DECK) if args.deck == "denpa92" else pilot_deck()
    base = SS.mk_deck(deck)

    def hoard(obs):
        EV.W_POWERFUL_HAND = args.w
        return base(obs)

    def plain(obs):
        EV.W_POWERFUL_HAND = 0.0
        return base(obs)

    label = f"hoard(W={args.w}) vs plain on {args.deck}"
    print(f"{label}, {args.games} games seat-swapped. Read: Wilson LB > 0.5 = hoard term helps.\n", flush=True)
    r = A.run(args.games, hoard, plain, label=label, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = SS.wilson(r["wins_a"], dec)
    verdict = "hoard helps" if lo > 0.5 else ("hoard hurts" if hi < 0.5 else "tie (CI spans 0.5)")
    res = {"deck": args.deck, "w": args.w, "win_rate_hoard": round(r["a_win_rate_decided"], 3),
           "wilson95": [round(lo, 3), round(hi, 3)], "wins": r["wins_a"], "losses": r["wins_b"],
           "errors": r.get("errors_a", 0) + r.get("errors_b", 0), "s_per_game": r["s_per_game"], "verdict": verdict}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)
    EV.W_POWERFUL_HAND = 0.0
    print(f"=> {label}: {res['win_rate_hoard']} Wilson {res['wilson95']} {verdict} ({res['wins']}-{res['losses']}, err {res['errors']})", flush=True)


if __name__ == "__main__":
    main()
