"""Run games between two policies in the mock environment and report a win rate.

Usage:
    python selfplay.py --games 400 --seed 0           # heuristic vs random
    python selfplay.py --games 400 --p0 heuristic --p1 heuristic

The win rate is for p0. Players alternate who moves first across games to remove the
first-move advantage from the measurement. Every game is bounded (max sub-actions per
turn and max turns), so the harness cannot hang. A timed-out game (hit the turn cap) is
scored as a draw and counted separately; a healthy policy pair should produce ~zero.
"""
from __future__ import annotations

import argparse
import time

from ptcg_mock import MockPTCG, GameState
from policies import RandomLegalPolicy, HeuristicPolicy

MAX_SUBACTIONS_PER_TURN = 16
MAX_TURNS = 200


def play_game(env: MockPTCG, p0, p1, seed: int, first: int) -> int | None:
    """Return the winning player index (0 or 1), or None for a capped draw."""
    state = env.new_game(seed)
    state.to_move = first
    policies = (p0, p1)
    while not env.is_terminal(state) and state.turn < MAX_TURNS:
        mover = state.to_move
        subactions = 0
        # a single turn: keep acting until the turn ends (attack/pass) or the cap hits
        turn_at_start = state.turn
        while state.to_move == mover and not env.is_terminal(state):
            action = policies[mover].act(env, state)
            state = env.step(state, action)
            subactions += 1
            if subactions >= MAX_SUBACTIONS_PER_TURN and state.to_move == mover:
                # force the turn to end to guarantee progress
                state = env.step(state, ("pass",))
            if state.turn != turn_at_start:
                break
    return env.winner(state)


def run(games: int, p0_kind: str, p1_kind: str, seed: int) -> dict:
    env = MockPTCG()

    def make(kind: str, s: int):
        return HeuristicPolicy(s) if kind == "heuristic" else RandomLegalPolicy(s)

    wins0 = wins1 = draws = 0
    t0 = time.time()
    for g in range(games):
        p0 = make(p0_kind, seed + g)
        p1 = make(p1_kind, seed + 10_000 + g)
        first = g % 2                     # alternate the first move
        w = play_game(env, p0, p1, seed + g, first)
        if w is None:
            draws += 1
        elif w == 0:
            wins0 += 1
        else:
            wins1 += 1
    dt = time.time() - t0
    decided = wins0 + wins1
    return {
        "games": games, "p0": p0_kind, "p1": p1_kind,
        "wins0": wins0, "wins1": wins1, "draws": draws,
        "p0_win_rate": wins0 / games,
        "p0_win_rate_decided": (wins0 / decided) if decided else 0.0,
        "seconds": round(dt, 2), "ms_per_game": round(1000 * dt / games, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--p0", default="heuristic", choices=["heuristic", "random"])
    ap.add_argument("--p1", default="random", choices=["heuristic", "random"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = run(args.games, args.p0, args.p1, args.seed)
    print(
        f"{r['p0']} (p0) vs {r['p1']} (p1) over {r['games']} games:\n"
        f"  p0 wins {r['wins0']}, p1 wins {r['wins1']}, draws {r['draws']}\n"
        f"  p0 win rate {r['p0_win_rate']:.3f} (excluding draws {r['p0_win_rate_decided']:.3f})\n"
        f"  {r['ms_per_game']} ms/game, {r['seconds']}s total"
    )


if __name__ == "__main__":
    main()
