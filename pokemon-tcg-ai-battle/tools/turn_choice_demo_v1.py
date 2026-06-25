"""Demo: 'look at candidate whole turns, simulate each, score the board, pick the best.'

Makes the turn-level idea VISIBLE and tests the metric. On real captured states (my turn, with BOTH a
developmental option and an attack available -- a real "attack now vs develop" choice), it plays out the
rest of the turn under several SEQUENCING strategies via the cg forward model, scores the resulting board
AFTER the opponent's reply, and prints each strategy's score + the spread.

The question it answers: does the leaf metric meaningfully separate whole-turn strategies (so "pick the
best turn" has signal), or do they all score ~the same (metric too weak to plan with)?

  python tools/turn_choice_demo_v1.py --capture-games 24 --worlds 12 --max-states 24
"""
from __future__ import annotations

import argparse
import contextlib
import io
import logging
import statistics
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import main as M                       # old production agent (realistic state generator)
import search_v3 as S                  # rollout + cg helpers
import deck_policy_v3 as DP3            # PH-aware attack value
import ab_candidate_v1 as AB           # pilot_deck + engine import
from ko_sequencing_state_test_v1 import _build_pool, _simulate_line   # proven paired-world sim

logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt   # noqa: F401
    from kaggle_environments import make

S.USE_DYNAMIC_ATTACKS = True

ATTACK, PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, END = 13, 7, 8, 9, 10, 12, 14
DEV = (PLAY, ATTACH, EVOLVE, ABILITY)


def _idx(opts, types):
    return [i for i, o in enumerate(opts) if isinstance(o, dict) and o.get("type") in types]


def _best_attack(obsd, me, opts):
    atk = _idx(opts, (ATTACK,))
    if not atk:
        return None
    try:
        return max(atk, key=lambda i: DP3.attack_profile(opts[i], obsd, me).get("value", 0.0))
    except Exception:
        return atk[0]


# --- candidate whole-turn strategies: each returns the choice for one of MY decisions in the rollout ---
def strat_attack_asap(obsd, me):
    opts = obsd.get("select", {}).get("option") or []
    a = _best_attack(obsd, me, opts)
    return [a] if a is not None else None


def strat_develop_first(obsd, me):
    opts = obsd.get("select", {}).get("option") or []
    dev = _idx(opts, DEV)
    if dev:
        return [dev[0]]
    a = _best_attack(obsd, me, opts)
    return [a] if a is not None else None


def strat_draw_first(obsd, me):
    opts = obsd.get("select", {}).get("option") or []
    ab = _idx(opts, (ABILITY,))            # draw abilities (Psychic Draw / Run Away Draw) first
    if ab:
        return [ab[0]]
    dev = _idx(opts, (PLAY, ATTACH, EVOLVE))
    if dev:
        return [dev[0]]
    a = _best_attack(obsd, me, opts)
    return [a] if a is not None else None


STRATS = {"attack_asap": strat_attack_asap, "develop_first": strat_develop_first, "draw_first": strat_draw_first}


def capture_strategic_states(games: int, max_states: int, deck: list) -> list:
    """Self-play the production agent; capture MY single-pick decisions that have BOTH a developmental
    option AND an attack option (a real 'attack now vs develop' choice)."""
    states = []
    M.DECK = deck

    def logger(obs):
        mv = M.agent_search(obs)
        try:
            sel = obs.get("select")
            if sel is not None and len(states) < max_states:
                opts = sel.get("option") or []
                if (sel.get("maxCount") == 1 and len(opts) >= 3
                        and any(isinstance(o, dict) and o.get("type") in (PLAY, EVOLVE, ATTACH) for o in opts)
                        and any(isinstance(o, dict) and o.get("type") == ATTACK for o in opts)):
                    states.append(deepcopy(obs))
        except Exception:
            pass
        return mv

    for g in range(games):
        if len(states) >= max_states:
            break
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = make("cabt")
            env.run([logger, M.agent_search] if g % 2 == 0 else [M.agent_search, logger])
    return states[:max_states]


def score_strategies(obs: dict, deck: list, worlds: int) -> dict | None:
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    A = S._api()
    if A is None:
        return None
    sums = {k: 0.0 for k in STRATS}
    n = 0
    for _ in range(worlds):
        pool = _build_pool(obs, deck, me)                  # one shared world (paired across strategies)
        vals = {}
        ok = True
        for name, fn in STRATS.items():
            v = _simulate_line(A, obs, pool, me, (lambda o, f=fn: f(o, me)))
            if v is None:
                ok = False
                break
            vals[name] = v
        if ok:
            for k, v in vals.items():
                sums[k] += v
            n += 1
    if n < max(3, worlds // 2):
        return None
    return {k: sums[k] / n for k in STRATS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-games", type=int, default=24)
    ap.add_argument("--worlds", type=int, default=12)
    ap.add_argument("--max-states", type=int, default=24)
    args = ap.parse_args()

    deck = AB.pilot_deck()
    print(f"capturing strategic states (my turn, dev + attack both available) from up to {args.capture_games} games...", flush=True)
    states = capture_strategic_states(args.capture_games, args.max_states, deck)
    print(f"captured {len(states)} states\n", flush=True)

    rows = []
    for i, obs in enumerate(states):
        sc = score_strategies(obs, deck, args.worlds)
        if sc is None:
            continue
        rows.append(sc)
        best = max(sc, key=sc.get)
        spread = max(sc.values()) - min(sc.values())
        line = "  ".join(f"{k}={sc[k]:+8.1f}" for k in STRATS)
        print(f"state {len(rows):2d}: {line}   -> best={best:<13s} spread={spread:7.1f}", flush=True)

    if not rows:
        print("no scorable states")
        return
    spreads = [max(r.values()) - min(r.values()) for r in rows]
    wins = {k: sum(1 for r in rows if max(r, key=r.get) == k) for k in STRATS}
    meaningful = sum(1 for s in spreads if s > 30.0)    # >30 leaf pts ~ one body of board difference
    print(f"\n=== summary over {len(rows)} states ===")
    print(f"  mean score spread between strategies: {statistics.fmean(spreads):.1f}  (median {statistics.median(spreads):.1f})")
    print(f"  states where the spread is meaningful (>30 leaf pts): {meaningful}/{len(rows)}")
    print(f"  best-strategy tally: " + ", ".join(f"{k} {wins[k]}" for k in STRATS))
    print("  => if the spread is ~0, the metric can't tell turns apart (planning has no signal).")
    print("  => if it's large and the winner varies by state, 'enumerate turns -> pick best' is viable.")


if __name__ == "__main__":
    main()
