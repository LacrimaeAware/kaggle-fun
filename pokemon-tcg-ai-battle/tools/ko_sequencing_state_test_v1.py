"""Applicable-state test for the KO-sequencing principle (bank the KO; never endanger it).

Whole-game A/B can't resolve effects below ~0.10 at feasible n (selfbase = 0.567 at n=120: identical
code, pure noise). This sidesteps that. It finds the SPECIFIC states where a lethal, non-game-winning
KO exists, and for each, has the ENGINE score two lines on the SAME paired hidden worlds:

  A. attack now   -- take the KO immediately (current behavior; attacking ends the turn)
  B. bank the KO  -- develop every non-endangering action first, take the KO last

Paired per world, so determinization noise cancels. Reports the mean paired leaf difference (B - A),
its SE, and how often B beats A. Measures the principle directly instead of through whole-game noise.

  python tools/ko_sequencing_state_test_v1.py --capture-games 30 --worlds 12 --max-states 40
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import statistics
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "agent"))
import ab_candidate_v1 as AB          # pilot_deck + engine import
import main as M                      # production bot (state generator)
import search_v3 as S                 # rollout helpers (_hidden_pool/_rollout_pick/_api/_obs_dict)
import deck_policy_v3 as DP3           # best_ko_attack + safe_pre_attack_indices
import eval as EV                      # leaf eval

logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt   # noqa: F401
    from kaggle_environments import make

S.USE_DYNAMIC_ATTACKS = True          # rollouts see Powerful Hand = 20 x hand


def capture_states(games: int, max_states: int, deck: list) -> list:
    """Self-play the production bot; capture obs where a lethal NON-game-winning KO is available
    for the mover (the exact states where 'attack now vs develop-first' is the decision)."""
    states = []
    M.DECK = deck

    def logger(obs):
        mv = M.agent_search(obs)
        try:
            if obs.get("select") is not None and len(states) < max_states:
                ko = DP3.best_ko_attack(obs)
                if ko is not None and not ko[1].get("game_win"):
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


def _build_pool(obsd: dict, deck: list, me: int) -> dict:
    """One determinization of the hidden zones, shared by BOTH lines so the world is paired."""
    cur = obsd["current"]
    players = cur["players"]
    P, O = players[me], players[1 - me]
    n_my_deck = P.get("deckCount", 0) or 0
    n_op_deck = O.get("deckCount", 0) or 0
    n_my_prize = len(P.get("prize") or [])
    n_op_prize = len(O.get("prize") or [])
    n_op_hand = O.get("handCount", 0) or 0
    mp = S._hidden_pool(deck, P, exclude_hand=False); mp += [3] * max(0, (n_my_deck + n_my_prize) - len(mp))
    op = S._hidden_pool(deck, O, exclude_hand=True);  op += [3] * max(0, (n_op_deck + n_op_prize + n_op_hand) - len(op))
    return dict(your_deck=mp[:n_my_deck], your_prize=mp[n_my_deck:n_my_deck + n_my_prize],
                opponent_deck=op[n_op_hand + n_op_prize:n_op_hand + n_op_prize + n_op_deck],
                opponent_prize=op[n_op_hand:n_op_hand + n_op_prize], opponent_hand=op[:n_op_hand],
                opponent_active=[])


def _simulate_line(A, obsd: dict, pool: dict, me: int, my_pick):
    """search_begin on the shared world, play MY turn via my_pick, opp aggro reply, eval at my next
    turn. my_pick(obsd_at_decision) -> choice list, or None to use the default development pick."""
    obsc = A.to_observation_class(obsd)
    try:
        root = A.search_begin(obsc, **pool)
    except Exception:
        return None
    st = root
    saw_opp = False
    val = None
    try:
        for _ in range(S.DEPTH_CAP * 2):
            ob = st.observation
            c = ob.current
            if c is not None and c.result != -1:
                break
            sub = ob.select
            if sub is None:
                break
            my_move = c is not None and c.yourIndex == me
            if saw_opp and my_move:                  # control back to me after the opponent -> eval
                break
            if not my_move:
                saw_opp = True
                choice = S._rollout_pick(ob, is_me=False)
            else:
                choice = my_pick(S._obs_dict(ob))
                if choice is None:
                    choice = S._rollout_pick(ob, is_me=True)
            st = A.search_step(st.searchId, choice)
        val = EV.evaluate_obs(S._obs_dict(st.observation), me)
    except Exception:
        val = None
    finally:
        try:
            A.search_end()
        except Exception:
            pass
    return val


def _pick_ko_now(obsd: dict, me: int):
    ko = DP3.best_ko_attack(obsd, me)
    return [ko[0]] if ko is not None else None


def _pick_bank_ko(obsd: dict, me: int):
    ko = DP3.best_ko_attack(obsd, me)
    if ko is None:
        return None                                  # no KO here -> default development
    ko_index, profile = ko
    if profile.get("game_win"):
        return [ko_index]
    safe = DP3.safe_pre_attack_indices(obsd, ko_index, me)
    if safe:
        return [safe[0]]                             # develop ONE non-endangering action; loop re-checks
    return [ko_index]                                # no safe development left -> take the KO


def eval_state(obs: dict, deck: list, worlds: int):
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    A = S._api()
    if A is None:
        return None
    diffs = []
    for _ in range(worlds):
        pool = _build_pool(obs, deck, me)
        vA = _simulate_line(A, obs, pool, me, lambda o: _pick_ko_now(o, me))
        vB = _simulate_line(A, obs, pool, me, lambda o: _pick_bank_ko(o, me))
        if vA is not None and vB is not None:
            diffs.append(vB - vA)
    if len(diffs) < max(3, worlds // 2):
        return None
    mean = statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("inf")
    return {"mean": mean, "se": se, "n": len(diffs), "b_better": sum(1 for d in diffs if d > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-games", type=int, default=30)
    ap.add_argument("--worlds", type=int, default=12)
    ap.add_argument("--max-states", type=int, default=40)
    args = ap.parse_args()

    deck = AB.pilot_deck()
    print(f"capturing KO-available states from up to {args.capture_games} self-play games...", flush=True)
    states = capture_states(args.capture_games, args.max_states, deck)
    print(f"captured {len(states)} KO-available states", flush=True)

    results = []
    for i, obs in enumerate(states):
        r = eval_state(obs, deck, args.worlds)
        if r is not None:
            results.append(r)
        if (i + 1) % 5 == 0:
            print(f"  evaluated {i + 1}/{len(states)} states", flush=True)
    if not results:
        print("no evaluable states")
        return

    state_means = [r["mean"] for r in results]       # per-state paired mean (B - A)
    overall = statistics.fmean(state_means)
    se = statistics.stdev(state_means) / math.sqrt(len(state_means)) if len(state_means) > 1 else float("inf")
    b_better = sum(1 for m in state_means if m > 1.0)        # B clearly better (eval prize term is 1000)
    a_better = sum(1 for m in state_means if m < -1.0)
    print(f"\n=== KO-SEQUENCING applicable-state test (states={len(results)}, worlds/state={args.worlds}) ===")
    print(f"  mean leaf diff (bank-KO minus attack-now): {overall:+.1f}  (SE {se:.1f})")
    print(f"  states where banking clearly better: {b_better}/{len(results)}   "
          f"attack-now clearly better: {a_better}/{len(results)}")
    print("  >0 => developing non-endangering first, KO last, beats attacking immediately")
    out = {"states": len(results), "worlds": args.worlds, "mean_diff": round(overall, 1),
           "se": round(se, 1), "bank_better": b_better, "attack_better": a_better,
           "state_means": [round(m, 1) for m in state_means]}
    (ROOT / "docs" / "workstreams" / "ko_sequencing_state_test.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote -> docs/workstreams/ko_sequencing_state_test.json")


if __name__ == "__main__":
    main()
