"""Single-elimination tournament among agents. Seeded bracket (seed i vs seed N-1-i each round), n games
per match (seat-alternated), winner = more game-wins (tie -> higher seed advances). Matches within a round
run in parallel across cores; progress streams; result written to data/ab_runs/tournament.json.

  python tools/tournament_v1.py --agents heuristic_first,phaware_search_planner,phaware_search,phaware --games 20

Agent names are run_heuristic_ab_v1 variants. Provide them in seed order (strongest first is conventional).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if not os.environ.get("TOURNEY_DEBUG"):   # silence OpenSpiel fd-2 spam (skip with TOURNEY_DEBUG=1 to see errors)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import run_heuristic_ab_v1 as H  # noqa: E402


def run_chunk(task):
    a, b, n, seat0 = task
    D = H._load()
    make = D["make"]
    A, B = H._make_agent(a, D), H._make_agent(b, D)
    wa = wb = dr = 0
    for i in range(n):
        a_seat = (seat0 + i) % 2
        pair = [A, B] if a_seat == 0 else [B, A]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt")
                env.run(pair)
            w = H._winner(env)
        except Exception:
            dr += 1
            continue
        if w is None:
            dr += 1
        elif w == a_seat:
            wa += 1
        else:
            wb += 1
    return (a, b, wa, wb, dr)


def play_match(ex, a, b, games, chunk=4):
    tasks, rem, seat = [], games, 0
    while rem > 0:
        k = min(chunk, rem)
        tasks.append((a, b, k, seat))
        seat = (seat + k) % 2
        rem -= k
    wa = wb = dr = 0
    for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
        _, _, cwa, cwb, cdr = f.result()
        wa += cwa; wb += cwb; dr += cdr
    return wa, wb, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", required=True, help="comma list, in seed order")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    field = [a.strip() for a in args.agents.split(",") if a.strip()]
    for a in field:
        assert a in H.VARIANTS, f"unknown agent {a}"
    print(f"Tournament: {len(field)} agents, n={args.games}/match, {args.workers} workers", flush=True)
    print("Seeds: " + ", ".join(f"{i+1}.{a}" for i, a in enumerate(field)), flush=True)

    bracket = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        alive = list(field)
        rnd = 1
        while len(alive) > 1:
            print(f"\n=== Round {rnd} ({len(alive)} in) ===", flush=True)
            pairs = []
            n = len(alive)
            for i in range(n // 2):
                pairs.append((alive[i], alive[n - 1 - i]))
            bye = [alive[n // 2]] if n % 2 == 1 else []
            winners_top, winners_bot = [], []
            round_rows = []
            for i, (a, b) in enumerate(pairs):
                wa, wb, dr = play_match(ex, a, b, args.games)
                win = a if wa >= wb else b
                tie = " (tie -> higher seed)" if wa == wb else ""
                print(f"  {a}  {wa}-{wb}  {b}   ({dr}d) -> {win}{tie}", flush=True)
                round_rows.append({"a": a, "b": b, "wins_a": wa, "wins_b": wb, "draws": dr, "winner": win})
                winners_top.append(win)
            for b in bye:
                print(f"  {b}: bye -> advances", flush=True)
            # reassemble winners in seed order so the next round re-seeds correctly
            alive = winners_top + bye
            bracket.append({"round": rnd, "matches": round_rows, "bye": bye})
            rnd += 1

    champ = alive[0]
    print(f"\nCHAMPION: {champ}", flush=True)
    out = ROOT / "data" / "ab_runs" / "tournament.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"field": field, "games_per_match": args.games, "bracket": bracket,
                               "champion": champ}, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
