"""Fair multi-deck arena. Every deck is piloted by the SAME generic policy (static/PH KO floor + 1-ply
search_v3, hand leaf) -- NO deck-specific heuristics, so no deck gets an edge the others lack. Round-robin
among our candidate decks PLUS the top real meta decks, so a deck's score reflects performance against the
FIELD, not a single matchup against our own deck. Ranks decks by average win rate across all opponents.

  python tools/deck_arena_v1.py --games 16 --meta-top 3

This answers "is deck X actually good vs the field" rather than "does X counter our deck". It does NOT
include any Alakazam-specific heuristics; to compare PILOTS (heuristics) use the other harnesses.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@contextlib.contextmanager
def _quiet_import():
    """Suppress ONLY the OpenSpiel fd-2 import spam, then restore real stderr so errors stay visible."""
    old = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(dn)
        os.close(old)
WINRATE = ROOT / "data" / "deck_winrate_v1.json"   # best-performing decks (by corpus win rate), not most-common


def _slug(label, i):
    s = (label or "").split(",")[0].strip().lower().replace("'", "").replace("’", "").replace(" ", "_")
    return (s[:16] or f"deck{i}")

ALAKAZAM = ([5]*3 + [13] + [19]*4 + [66]*3 + [305]*4 + [741]*4 + [742]*4 + [743]*4 + [1079]*4 + [1081]*4
            + [1086]*4 + [1097]*3 + [1129] + [1152]*4 + [1182]*3 + [1184] + [1225]*4 + [1231]*4 + [1264])
STARMIE = ([3]*9 + [17]*4 + [666]*4 + [1030]*3 + [1031]*3 + [1086]*4 + [1097]*2 + [1120]*4 + [1121]
           + [1122]*4 + [1145]*4 + [1159] + [1182] + [1189]*4 + [1223]*2 + [1225]*2 + [1227]*4 + [1229]*4)
DENPA92 = ([5]*3 + [19]*4 + [65]*4 + [66]*4 + [741]*4 + [742]*4 + [743]*3 + [1079]*3 + [1081]*3 + [1086]*4
           + [1097] + [1129] + [1146] + [1152]*4 + [1159] + [1182]*3 + [1184] + [1225]*4 + [1231]*4 + [1264]*4)

_DECKS: dict = {}


def _load_decks(meta_top: int) -> dict:
    # our candidates + the top-`meta_top` BEST-PERFORMING corpus decks (by win rate), named by archetype.
    decks = {"alakazam": ALAKAZAM, "starmie": STARMIE, "denpa92": DENPA92}
    try:
        wr = json.loads(WINRATE.read_text(encoding="utf-8"))
        ours = {tuple(sorted(d)) for d in decks.values()}
        added = 0
        for i, entry in enumerate(wr.get("decks", [])):   # already sorted by win rate, descending
            dk = entry.get("deck")
            if not dk or len(dk) != 60 or tuple(sorted(dk)) in ours:
                continue
            nm = _slug(entry.get("label"), i)
            while nm in decks:
                nm += "2"
            decks[nm] = list(dk)
            ours.add(tuple(sorted(dk)))
            added += 1
            if added >= meta_top:
                break
    except Exception:
        pass
    return decks


def _pilot(deck, DP3, S):
    def agent(obs):
        if obs.get("select") is None:
            return list(deck)
        try:
            ko = DP3.best_ko_attack(obs)
            if ko is not None:
                return [ko[0]]
            mv = S.best_option(obs, deck, leaf_mode="hand")
            if mv:
                return mv
        except Exception:
            pass
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        k = sel.get("minCount") or 1
        return list(range(min(max(k, 1), len(opts)))) if opts else []
    return agent


def _winner(env):
    last = env.steps[-1]
    r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def run_chunk(task):
    a, b, n, seat0, meta_top = task
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet_import(), contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make
        import deck_policy_v3 as DP3
        import search_v3 as S
    S.USE_DYNAMIC_ATTACKS = True
    decks = _load_decks(meta_top)
    A, B = _pilot(decks[a], DP3, S), _pilot(decks[b], DP3, S)
    wa = wb = dr = 0
    for i in range(n):
        a_seat = (seat0 + i) % 2
        pair = [A, B] if a_seat == 0 else [B, A]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt")
                env.run(pair)
            w = _winner(env)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--meta-top", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()

    decks = _load_decks(args.meta_top)
    names = list(decks)
    print(f"Fair arena: {len(names)} decks (same generic pilot for all), round-robin n={args.games}/pair", flush=True)
    print("decks:", ", ".join(names), flush=True)
    wins = Counter(); losses = Counter()
    head = {}
    tasks = []
    for a, b in combinations(names, 2):
        rem, seat = args.games, 0
        while rem > 0:
            k = min(args.chunk, rem)
            tasks.append((a, b, k, seat, args.meta_top))
            seat = (seat + k) % 2
            rem -= k
    agg = Counter()
    total, done = len(tasks), 0
    print(f"running {total} chunks across {args.workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
            done += 1
            try:
                a, b, wa, wb, dr = f.result()
            except Exception as exc:
                print(f"  [chunk {done}/{total}] ERROR: {exc!r}", flush=True)
                continue
            agg[(a, b, "wa")] += wa; agg[(a, b, "wb")] += wb
            print(f"  [chunk {done}/{total}] {a} vs {b}: +{wa}-{wb}  "
                  f"(pair so far {agg[(a, b, 'wa')]}-{agg[(a, b, 'wb')]})", flush=True)
    for a, b in combinations(names, 2):
        wa, wb = agg[(a, b, "wa")], agg[(a, b, "wb")]
        wins[a] += wa; losses[a] += wb; wins[b] += wb; losses[b] += wa
        head[f"{a}_vs_{b}"] = f"{wa}-{wb}"
        print(f"  {a} {wa}-{wb} {b}", flush=True)
    print("\n=== deck ranking (avg winrate vs the field, fair generic pilot) ===", flush=True)
    rows = []
    for nm in names:
        dec = wins[nm] + losses[nm]
        wr = wins[nm] / dec if dec else 0.0
        rows.append((wr, nm, wins[nm], losses[nm]))
    for wr, nm, w, l in sorted(rows, reverse=True):
        print(f"  {nm:10s} {wr:.3f}  ({w}-{l})", flush=True)
    out = ROOT / "data" / "ab_runs" / "deck_arena.json"
    out.write_text(json.dumps({"games": args.games, "decks": names,
                               "ranking": [{"deck": nm, "winrate": round(wr, 4), "wins": w, "losses": l}
                                           for wr, nm, w, l in sorted(rows, reverse=True)],
                               "head_to_head": head}, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
