"""Audit the cabt game structure to ground the landscape: option-type histogram, a full
mid-game state dump, win-condition read, and distributions over many games.

Run: python audit_game.py  (writes a readable report to stdout; engine logs suppressed)
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
from collections import Counter

logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt
    from kaggle_environments import make

DECK = list(cabt.deck)

type_counts: Counter = Counter()
type_example: dict[int, dict] = {}
maxcount_counts: Counter = Counter()
mid_state = {}
decisions = [0]


def probe(obs):
    sel = obs.get("select")
    if sel is None:
        return DECK
    opts = sel.get("option") or []
    mc = sel.get("maxCount")
    maxcount_counts[mc] += 1
    for o in opts:
        t = o.get("type")
        type_counts[t] += 1
        type_example.setdefault(t, o)
    decisions[0] += 1
    # capture a full mid-game state once
    if decisions[0] == 20 and not mid_state:
        cur = obs.get("current") or {}
        mid_state["dump"] = json.dumps(cur, default=str)[:2400]
        mid_state["n_options"] = len(opts)
    return list(range(mc)) if mc else []


print("==== one annotated game (probe vs random) ====")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    env = make("cabt")
    env.run([probe, cabt.random_agent])
last = env.steps[-1]
print("final rewards:", [last[0].get("reward"), last[1].get("reward")])
print("final result field:", (last[0].get("observation") or {}).get("current"))
print("decisions made by probe:", decisions[0], "| engine steps:", len(env.steps))
print("\nmaxCount distribution:", dict(maxcount_counts))
print("\noption TYPE histogram (type: count) with an example option dict:")
for t in sorted(type_counts):
    print(f"  type {t:>2}: {type_counts[t]:>4}   eg {json.dumps(type_example[t])}")
print("\n---- full mid-game current state (truncated) ----")
print(mid_state.get("dump", "(not captured)"))

# distributions over many games
print("\n==== distributions over 150 random-vs-random games ====")
lengths = []
seat0_wins = draws = 0
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    for g in range(150):
        e = make("cabt")
        e.run([cabt.random_agent, cabt.random_agent])
        lengths.append(len(e.steps))
        r0 = e.steps[-1][0].get("reward")
        r1 = e.steps[-1][1].get("reward")
        if r0 == r1:
            draws += 1
        elif r0 > r1:
            seat0_wins += 1
lengths.sort()
n = len(lengths)
print(f"games: {n}")
print(f"engine-steps per game: min {lengths[0]}, median {lengths[n//2]}, "
      f"p90 {lengths[int(n*0.9)]}, max {lengths[-1]}")
print(f"seat-0 (first player) wins: {seat0_wins}/{n} = {seat0_wins/n:.3f}  | draws: {draws}")
