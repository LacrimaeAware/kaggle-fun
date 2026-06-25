"""Cross-deck pilot A/B: does OUR generic forward-model pilot do better with the Starmie deck or the
Alakazam deck? Each agent plays its OWN deck via the deck-agnostic path (PH/static KO floor + search_v3,
leaf=hand), so the comparison isolates the DECK, not Alakazam-specific heuristics. Parallel, durable.

  python tools/deck_pilot_ab_v1.py --games 40

Caveat: this is our-pilot self-play, not the ladder. It tells us which deck our agent pilots better head to
head; the human ~70% Starmie winrate is THEIR pilot. Measure ours before switching.
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

if not os.environ.get("PILOT_DEBUG"):
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

ROOT = Path(__file__).resolve().parent.parent

ALAKAZAM = (
    [5] * 3 + [13] * 1 + [19] * 4 + [66] * 3 + [305] * 4 + [741] * 4 + [742] * 4 + [743] * 4
    + [1079] * 4 + [1081] * 4 + [1086] * 4 + [1097] * 3 + [1129] * 1 + [1152] * 4
    + [1182] * 3 + [1184] * 1 + [1225] * 4 + [1231] * 4 + [1264] * 1
)
STARMIE = [3]*9 + [17]*4 + [666]*4 + [1030]*3 + [1031]*3 + [1086]*4 + [1097]*2 + [1120]*4 + [1121]*1 \
    + [1122]*4 + [1145]*4 + [1159]*1 + [1182]*1 + [1189]*4 + [1223]*2 + [1225]*2 + [1227]*4 + [1229]*4
DECKS = {"alakazam": ALAKAZAM, "starmie": STARMIE}


def _load():
    sys.path.insert(0, str(ROOT / "agent"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
        import deck_policy_v3 as DP3
        import search_v3 as S
    S.USE_DYNAMIC_ATTACKS = True
    return make, DP3, S


def _pilot(deck, DP3, S):
    """Deck-agnostic pilot: take a listed KO, else 1-ply search develop, else legal-first."""
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


def _build(name, DP3, S):
    """A side is either a deck name (generic pilot on that deck) or 'heuristic_first' (the full Alakazam
    heuristics-first agent), so we can test Starmie-generic vs our BEST Alakazam agent, not just generic."""
    if name == "heuristic_first":
        sub = str(ROOT / "submissions" / "sub_heuristic")
        if sub not in sys.path:
            sys.path.insert(0, sub)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import agent_impl as AI
        return AI.agent
    return _pilot(DECKS[name], DP3, S)


def run_chunk(task):
    a_deck, b_deck, n, seat0 = task
    make, DP3, S = _load()
    A, B = _build(a_deck, DP3, S), _build(b_deck, DP3, S)
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
    return (wa, wb, dr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="starmie")
    ap.add_argument("--b", default="alakazam")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()

    out = ROOT / "data" / "ab_runs" / f"pilot_{args.a}_vs_{args.b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"PILOT A/B: {args.a} deck vs {args.b} deck (both our generic pilot), n={args.games}", flush=True)
    tasks, rem, seat = [], args.games, 0
    while rem > 0:
        k = min(args.chunk, rem)
        tasks.append((args.a, args.b, k, seat))
        seat = (seat + k) % 2
        rem -= k
    wa = wb = dr = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
            cwa, cwb, cdr = f.result()
            wa += cwa; wb += cwb; dr += cdr
            done = wa + wb + dr
            dec = wa + wb
            p = wa / dec if dec else 0.0
            hw = 1.96 * math.sqrt(p * (1 - p) / dec) if dec else 0.0
            out.write_text(json.dumps({"a": args.a, "b": args.b, "games_done": done, "wins_a": wa,
                                       "wins_b": wb, "draws": dr, "winrate_a": round(p, 4)}, indent=2),
                           encoding="utf-8")
            print(f"[{done}/{args.games}] {args.a} A={p:.3f} +/-{hw:.3f} ({wa}-{wb}, {dr}d)", flush=True)
    print(f"\nRESULT: {args.a} {wa}-{wb} {args.b} ({dr}d) -> {args.a if wa>wb else args.b} pilots better", flush=True)


if __name__ == "__main__":
    main()
