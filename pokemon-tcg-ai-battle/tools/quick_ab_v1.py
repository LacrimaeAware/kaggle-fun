"""Fast, MONITORABLE, small-n exploratory A/B. Single process, prints a running tally every --report games,
and silences the OpenSpiel C-level stderr spam so the output is readable. For "is this above or below 50%?"
exploratory reads, NOT strong statistics. Stops early once the CI clears 50% if --early is set.

  python tools/quick_ab_v1.py --a phaware_search_deckout --b phaware_search --games 80 --report 10

Variants are reused from run_heuristic_ab_v1 (first, choose, eff, search, phaware, phaware_search,
phaware_search_ca, phaware_search_deckout, phaware_search_ph, phaware_search_v3).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

# Silence OpenSpiel / engine C-level stderr (written at fd 2, so Python redirect_stderr cannot catch it).
# Our progress goes to stdout (fd 1), which stays clean.
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import run_heuristic_ab_v1 as H  # noqa: E402  (lazy heavy imports happen inside H._load)


def _hw(p: float, n: int) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--report", type=int, default=10)
    ap.add_argument("--early", action="store_true", help="stop once the 95% CI clears 50% (either side)")
    ap.add_argument("--out", default=None, help="durable result file; defaults to data/ab_runs/<a>_vs_<b>.json")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (ROOT / "data" / "ab_runs" / f"{args.a}_vs_{args.b}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(done, dec, p, hw, side, finished):
        # written EVERY report batch so the result survives any interruption (no end-of-run-only writes)
        out.write_text(json.dumps({
            "a": args.a, "b": args.b, "games_target": args.games, "games_done": done,
            "wins_a": wa, "wins_b": wb, "draws": dr, "winrate_a": round(p, 4),
            "ci95_halfwidth": round(hw, 4), "verdict": side, "finished": finished,
        }, indent=2), encoding="utf-8")

    D = H._load()
    make = D["make"]
    A, B = H._make_agent(args.a, D), H._make_agent(args.b, D)

    print(f"{args.a} (A) vs {args.b} (B), up to {args.games} games, seat-alternated -> {out}", flush=True)
    wa = wb = dr = 0
    for i in range(args.games):
        a_seat = i % 2
        pair = [A, B] if a_seat == 0 else [B, A]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt")
                env.run(pair)
            last = env.steps[-1]
            r0, r1 = last[0].get("reward"), last[1].get("reward")
        except Exception:
            dr += 1
            continue
        if r0 is None or r1 is None or r0 == r1:
            dr += 1
        else:
            w = 0 if r0 > r1 else 1
            (wa, wb) = (wa + 1, wb) if w == a_seat else (wa, wb + 1)

        done = i + 1
        if done % args.report == 0 or done == args.games:
            dec = wa + wb
            p = wa / dec if dec else 0.0
            hw = _hw(p, dec)
            side = ">50%" if p - hw > 0.5 else "<50%" if p + hw < 0.5 else "~50% (CI spans)"
            print(f"[{done}/{args.games}] A={p:.3f} +/-{hw:.3f}  ({wa}-{wb}, {dr}d)  {side}", flush=True)
            stop = args.early and dec >= 20 and (p - hw > 0.5 or p + hw < 0.5)
            save(done, dec, p, hw, side, finished=(done == args.games or stop))
            if stop:
                print(f"early stop: CI cleared 50% at n={dec}", flush=True)
                break


if __name__ == "__main__":
    main()
