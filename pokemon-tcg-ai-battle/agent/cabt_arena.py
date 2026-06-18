"""Run real cabt matches between two agent callables and report a win rate.

This uses the actual organizer engine (kaggle_environments cabt), so the numbers here have
provenance `local-sim` against the REAL ruleset, unlike the mock. Seats are swapped each
game to cancel the first-player advantage.

    python cabt_arena.py --games 60 --a heuristic --b random

Agents available by name: random, first, heuristic (from main.py). Quiet by default;
the engine's own logging is suppressed.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import time

logging.disable(logging.CRITICAL)           # silence kaggle_environments / open_spiel logs

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt
    from kaggle_environments import make

import main as agent_mod                      # our agent (heuristic)


# Wrap the baselines so every agent pilots the SAME deck (agent_mod.DECK). This isolates
# policy skill from deck choice: any win-rate gap is the policy, not the cards.
def _random_samedeck(obs):
    return list(agent_mod.DECK) if obs.get("select") is None else cabt.random_agent(obs)


def _first_samedeck(obs):
    return list(agent_mod.DECK) if obs.get("select") is None else cabt.first_agent(obs)


AGENTS = {
    "random": _random_samedeck,
    "first": _first_samedeck,
    "heuristic": agent_mod.agent,
    "search": agent_mod.agent_search,
    "search2": agent_mod.agent_search2,
    "search_v": agent_mod.agent_search_v,
    "combine": agent_mod.agent_combine,
}


def winner_of(env) -> int | None:
    """0, 1, or None (draw), read from the final per-agent reward."""
    last = env.steps[-1]
    r0 = last[0].get("reward")
    r1 = last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def run(games: int, a, b, label: str = "", progress: int = 20) -> dict:
    """progress: print a live line every `progress` games (0 = silent). Each line shows games done,
    the running decided win-rate for A, elapsed, ETA, and the error count -- so a background run is
    never a black box (and silent errors surface immediately)."""
    wins_a = wins_b = draws = errors = 0
    t0 = time.time()
    tag = f"[{label}] " if label else ""
    for g in range(games):
        # swap seats each game; track which seat our 'a' sits in
        a_seat = g % 2
        agents = [a, b] if a_seat == 0 else [b, a]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env = make("cabt")
                env.run(agents)
            w = winner_of(env)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  {tag}ERROR game {g+1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        if w is None:
            draws += 1
        elif w == a_seat:
            wins_a += 1
        else:
            wins_b += 1
        done = g + 1
        if progress and (done % progress == 0 or done == games):
            el = time.time() - t0
            dec = wins_a + wins_b
            wr = wins_a / dec if dec else 0.0
            eta = el / done * (games - done)
            print(f"  {tag}{done}/{games} ({100*done//games}%) | A {wr:.3f} ({wins_a}-{wins_b}, {draws}d {errors}e)"
                  f" | {el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)
        # Note: do NOT call cabt.battle_finish() here. env.run drives the engine through
        # the interpreter, which manages the single global battle_ptr itself; finishing it
        # externally double-frees the native battle and aborts the process.
    dt = time.time() - t0
    decided = wins_a + wins_b
    return {
        "games": games, "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "errors": errors, "a_win_rate": wins_a / games if games else 0.0,
        "a_win_rate_decided": wins_a / decided if decided else 0.0,
        "seconds": round(dt, 1), "s_per_game": round(dt / games, 2) if games else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--a", default="heuristic", choices=list(AGENTS))
    ap.add_argument("--b", default="random", choices=list(AGENTS))
    args = ap.parse_args()
    r = run(args.games, AGENTS[args.a], AGENTS[args.b], label=f"{args.a} vs {args.b}")
    print(
        f"{args.a} (A) vs {args.b} (B) over {r['games']} real cabt games:\n"
        f"  A wins {r['wins_a']}, B wins {r['wins_b']}, draws {r['draws']}, errors {r['errors']}\n"
        f"  A win rate {r['a_win_rate']:.3f} (decided {r['a_win_rate_decided']:.3f})\n"
        f"  {r['s_per_game']} s/game, {r['seconds']}s total"
    )


if __name__ == "__main__":
    main()
