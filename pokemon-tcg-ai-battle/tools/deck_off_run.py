"""Deck-off: run our agent_search piloting one slate deck vs agent_search on DENPA92, seat-swapped, save the
result. One process per deck so the whole slate runs in parallel. The relevant question is "which deck does
OUR search pilot best" (deck value is policy-coupled), not the deck's expert-hands win rate.

    python tools/deck_off_run.py --index 0 --games 40 --out tools/_deckoff_res_0.json
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
import search_sprint as SS      # noqa: E402
import cabt_arena as A          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--progress", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    slate = json.load(open(ROOT / "tools" / "_deckoff_slate.json", encoding="utf-8"))
    s = slate[args.index]
    label = s["label"]
    print(f"[{args.index}] {label}: our-search on this deck vs DENPA92, {args.games} games seat-swapped\n", flush=True)
    r = A.run(args.games, SS.mk_deck(s["deck"]), SS.mk_deck(list(M.DECK)), label=label, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = SS.wilson(r["wins_a"], dec)
    verdict = "deck better than DENPA92" if lo > 0.5 else ("worse" if hi < 0.5 else "tie (CI spans 0.5)")
    res = {"index": args.index, "label": label, "win_rate_vs_denpa92": round(r["a_win_rate_decided"], 3),
           "wilson95": [round(lo, 3), round(hi, 3)], "wins": r["wins_a"], "losses": r["wins_b"],
           "errors": r.get("errors_a", 0) + r.get("errors_b", 0), "s_per_game": r["s_per_game"],
           "verdict": verdict}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"=> [{args.index}] {label}: {res['win_rate_vs_denpa92']} Wilson {res['wilson95']} {verdict} "
          f"({res['wins']}-{res['losses']}, err {res['errors']})", flush=True)


if __name__ == "__main__":
    main()
