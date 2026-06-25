"""PARALLEL, durable, STREAMING A/B. Runs many matchups at once across all cores, silences the OpenSpiel
fd-2 spam, and after every finished chunk it (a) prints a running tally and (b) overwrites a per-matchup
JSON in data/ab_runs/. So you get fast results that stream in and survive any interruption, instead of
waiting blind for a single-process run to finish.

  python tools/par_ab_v1.py --matchups phaware_vs_first,phaware_search_vs_phaware --games 120 --chunk 8

Matchup names use the run_heuristic_ab_v1 variants: first, choose, eff, search, phaware, phaware_search,
phaware_search_ca, phaware_search_deckout, phaware_search_ph, phaware_search_v3, phaware_search_dev,
phaware_search_planner. Self-mirror (x_vs_x) is the fairness control (should sit near 0.500).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Silence OpenSpiel / engine C-level stderr (fd 2) in EVERY process (main + workers re-import this module).
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import run_heuristic_ab_v1 as H  # noqa: E402  (heavy imports are lazy inside H._load)


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


def _hw(p: float, n: int) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchups", required=True, help="comma list of a_vs_b")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    pairs = []
    for m in args.matchups.split(","):
        a, b = m.split("_vs_", 1)
        assert a in H.VARIANTS and b in H.VARIANTS, f"unknown variant in {m}"
        pairs.append((a, b))

    outdir = ROOT / "data" / "ab_runs"
    outdir.mkdir(parents=True, exist_ok=True)
    agg = {}      # (a,b) -> [wa, wb, dr, done]
    tasks = []
    for a, b in pairs:
        agg[(a, b)] = [0, 0, 0, 0]
        rem, seat = args.games, 0
        while rem > 0:
            k = min(args.chunk, rem)
            tasks.append((a, b, k, seat))
            seat = (seat + k) % 2
            rem -= k

    print(f"{len(tasks)} chunks / {len(pairs)} matchups / {args.workers} workers / {args.games} games each",
          flush=True)

    def save(a, b):
        wa, wb, dr, done = agg[(a, b)]
        dec = wa + wb
        p = wa / dec if dec else 0.0
        h = _hw(p, dec)
        side = ">50%" if p - h > 0.5 else "<50%" if p + h < 0.5 else "~50% (CI spans)"
        (outdir / f"{a}_vs_{b}.json").write_text(json.dumps({
            "a": a, "b": b, "games_target": args.games, "games_done": done,
            "wins_a": wa, "wins_b": wb, "draws": dr, "winrate_a": round(p, 4),
            "ci95_halfwidth": round(h, 4), "verdict": side,
        }, indent=2), encoding="utf-8")
        return p, h, side

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_chunk, t) for t in tasks]
        for f in as_completed(futs):
            a, b, wa, wb, dr = f.result()
            g = agg[(a, b)]
            g[0] += wa; g[1] += wb; g[2] += dr; g[3] += wa + wb + dr
            p, h, side = save(a, b)
            print(f"[{a}_vs_{b}] {g[3]}/{args.games}  A={p:.3f} +/-{h:.3f}  ({g[0]}-{g[1]}, {g[2]}d)  {side}",
                  flush=True)

    print("DONE", flush=True)
    for (a, b), g in agg.items():
        dec = g[0] + g[1]
        p = g[0] / dec if dec else 0.0
        print(f"  {a}_vs_{b}: A={p:.3f} ({g[0]}-{g[1]}, {g[2]}d, n={dec})", flush=True)


if __name__ == "__main__":
    main()
