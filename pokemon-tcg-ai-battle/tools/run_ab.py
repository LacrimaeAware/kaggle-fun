"""Run win-rate A/B matchups on the real cabt engine with LIVE progress + Wilson CIs.

Never a black box: every matchup streams `done/total (pct) | A win-rate | elapsed, ETA | errors`
lines (flushed), so a background run is watchable and silent errors surface immediately.

    python tools/run_ab.py --games 200 combine:heuristic combine:search search:heuristic
    python tools/run_ab.py --games 120 combine:heuristic          # one matchup

Agents: random, first, heuristic, search, search_v, combine (all pilot agent.DECK -> isolates policy).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
import cabt_arena as A  # noqa: E402


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matchups", nargs="*", default=["combine:heuristic", "combine:search", "search:heuristic"],
                    help="a:b pairs (A's win-rate reported)")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--progress", type=int, default=20, help="print a live line every N games")
    args = ap.parse_args()

    print(f"win-rate A/B | {args.games} games each | DECK=agent.DECK | seat-swapped\n", flush=True)
    results = []
    for i, mu in enumerate(args.matchups):
        a, b = mu.split(":")
        if a not in A.AGENTS or b not in A.AGENTS:
            print(f"  skip {mu}: unknown agent (have {list(A.AGENTS)})", flush=True)
            continue
        print(f"[{i+1}/{len(args.matchups)}] {a} vs {b}:", flush=True)
        r = A.run(args.games, A.AGENTS[a], A.AGENTS[b], label=f"{a} vs {b}", progress=args.progress)
        dec = r["wins_a"] + r["wins_b"]
        lo, hi = wilson(r["wins_a"], dec)
        verdict = "A wins" if lo > 0.5 else ("B wins" if hi < 0.5 else "tie (CI spans 0.5)")
        line = (f"  => {a} {r['a_win_rate_decided']:.3f}  Wilson95 [{lo:.3f},{hi:.3f}]  {verdict}"
                f"  ({r['wins_a']}-{r['wins_b']}, {r['draws']}d {r['errors']}e, {r['s_per_game']}s/g)")
        print(line + "\n", flush=True)
        results.append((mu, line))
    print("=== SUMMARY ===", flush=True)
    for mu, line in results:
        print(line, flush=True)


if __name__ == "__main__":
    main()
