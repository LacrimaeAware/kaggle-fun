"""L3 v0: 1-ply forward-model search over the current decision.

At one of my MAIN single-pick decisions with several legal options, simulate EACH option with
the competition's own forward model (cg.api search_begin/search_step), greedily finish the rest
of MY turn with the engine's default policy, and score the resulting state with the L2 eval
(eval.py). Return the option whose line leads to the best resulting state.

Why this is the lever: a hand-scored heuristic cannot see the consequence of a play (it ties
first_agent, measured). Search picks the move by its RESULT, so taking/setting up a knockout
beats a default move it cannot tell apart.

Crash-safe by contract: best_option returns None on ANY problem (no cg, bad determinization,
search error, time budget) so the caller falls back to the heuristic. It never raises.

Determinization (v0, honest scope): hidden zones are filled to the correct COUNT, and for the
opponent's deck we assume the same deck list (true for the local same-deck self-play used to
measure this). Evaluating MY own turn barely depends on the opponent's hidden cards (they do
not act and rarely draw during my turn). A sampled / opponent-modelled determinization is a
later refinement, NOT required to test whether search helps.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
import sys
import time
from collections import Counter

import eval as EV

DEPTH_CAP = 80            # max sub-decisions to roll out my turn + the opponent's reply
DEFAULT_BUDGET = 0.6      # seconds/decision hard cap (measured need ~0.14s max; bounds match-time forfeit risk)
N_DETERM = 4              # determinization samples averaged per decision (cuts single-world noise)
ATTACK_OPT = 13           # OptionType.ATTACK

_API = None


def _load_atk() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("attack_stats.json", os.path.join(here, "attack_stats.json"),
              "/kaggle_simulations/agent/attack_stats.json"):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


ATK = _load_atk()           # attackId(str) -> {d(dmg), ...}


def _rollout_pick(sel) -> list:
    """Rollout policy for the playout (both players): take the highest-damage attack if any is
    legal, else the engine default order. A default-order opponent never punishes (it just plays
    option 0), so leaves look uniformly safe; an attacking opponent makes the punish real."""
    opts = sel.option
    n = len(opts)
    k = sel.maxCount or 0
    mn = sel.minCount or 0
    if k == 1 and n > 0:
        best_i, best_d = None, -1
        for i, o in enumerate(opts):
            if getattr(o, "type", None) == ATTACK_OPT:
                d = ATK.get(str(getattr(o, "attackId", None)), {}).get("d", 0) or 0
                if d > best_d:
                    best_d, best_i = d, i
        if best_i is not None:
            return [best_i]
    return list(range(max(min(k, n), min(mn, n))))


def _api():
    """Import cg.api once (bundled with the agent at submission, or the sample dir locally)."""
    global _API
    if _API is not None:
        return _API or None
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here,
              os.path.join(here, ".."),
              os.path.join(here, "..", "data", "external", "official", "sample_submission"),
              "/kaggle_simulations/agent"):
        ap = os.path.abspath(p)
        if os.path.isfile(os.path.join(ap, "cg", "api.py")) and ap not in sys.path:
            sys.path.insert(0, ap)
    try:
        import cg.api as A
        _API = A
    except Exception:
        _API = False
    return _API or None


def _obs_dict(observation) -> dict:
    if observation is None:
        return {}
    try:
        return dataclasses.asdict(observation)
    except Exception:
        return {}


def _hidden_pool(deck: list, player: dict, exclude_hand: bool) -> list:
    """Cards in `deck` that are NOT visible on `player`'s board/discard -- i.e. their hidden
    cards. With exclude_hand=False (used for myself, since I see my own hand) the pool is
    deck+prize; with exclude_hand=True (the opponent) it is deck+prize+hand. Shuffled so the
    later slice into hand/prize/deck is a representative draw from the real deck composition,
    not all-energy. This is what makes the simulated opponent reply actually play threats."""
    cnt = Counter(deck)

    def rm(cid):
        if cid and cnt.get(cid, 0) > 0:
            cnt[cid] -= 1

    def strip(p):
        if not p:
            return
        rm(p.get("id"))
        for e in (p.get("energyCards") or []):
            rm(e.get("id"))
        for t in (p.get("tools") or []):
            rm(t.get("id"))
        for pe in (p.get("preEvolution") or []):
            rm(pe.get("id"))

    for p in (player.get("active") or []):
        strip(p)
    for p in (player.get("bench") or []):
        strip(p)
    for c in (player.get("discard") or []):
        rm(c.get("id"))
    if not exclude_hand:
        for c in (player.get("hand") or []):
            rm(c.get("id"))
    pool = list(cnt.elements())
    random.shuffle(pool)
    return pool


def _search(obs: dict, deck: list, time_budget: float = DEFAULT_BUDGET, leaf_mode: str = "hand"):
    """Core search. Returns (best_option_index, best_backed_up_value), or (None, None) if search
    is not applicable. The backed-up value is the max over options of the determinization-averaged
    leaf value -- the A0GB-style search-bootstrapped value of this state (used as a learning target
    by datagen --bootstrap)."""
    A = _api()
    if A is None:
        return None, None
    sel = obs.get("select")
    cur = obs.get("current")
    if not sel or not cur:
        return None, None
    if (sel.get("maxCount") or 0) != 1:        # only single-pick decisions
        return None, None
    opts = sel.get("option") or []
    if len(opts) < 2:
        return None, None
    players = cur.get("players") or []
    if len(players) < 2:
        return None, None
    me = cur.get("yourIndex", 0)
    P, O = players[me], players[1 - me]

    # don't search when we'd have to predict a face-down opponent active (mostly setup) -> heuristic
    oa = O.get("active") or []
    if oa and oa[0] is None:
        return None, None

    n_my_deck = P.get("deckCount", 0) or 0
    n_op_deck = O.get("deckCount", 0) or 0
    n_my_prize = len(P.get("prize") or [])
    n_op_prize = len(O.get("prize") or [])
    n_op_hand = O.get("handCount", 0) or 0

    # Average each option's value over N_DETERM sampled hidden worlds. A single determinization
    # makes the leaf value high-variance (one lucky/unlucky opponent draw decides the move); the
    # average is a determinized-ISMCTS-style estimate over the real deck composition.
    obsd = A.to_observation_class(obs)
    sums = counts = None
    t0 = time.time()
    for _ in range(N_DETERM):
        if time.time() - t0 > time_budget:
            break
        my_pool = _hidden_pool(deck, P, exclude_hand=False)
        my_pool += [3] * max(0, (n_my_deck + n_my_prize) - len(my_pool))
        your_deck = my_pool[:n_my_deck]
        your_prize = my_pool[n_my_deck:n_my_deck + n_my_prize]
        op_pool = _hidden_pool(deck, O, exclude_hand=True)
        op_pool += [3] * max(0, (n_op_deck + n_op_prize + n_op_hand) - len(op_pool))
        opp_hand = op_pool[:n_op_hand]
        opp_prize = op_pool[n_op_hand:n_op_hand + n_op_prize]
        opp_deck = op_pool[n_op_hand + n_op_prize:n_op_hand + n_op_prize + n_op_deck]
        try:
            root = A.search_begin(
                obsd,
                # empty hidden zones stay empty: the engine requires list length == zone count,
                # and a 0-count zone must be [], not a phantom energy (was `zone or [3]`).
                your_deck=your_deck, your_prize=your_prize,
                opponent_deck=opp_deck, opponent_prize=opp_prize,
                opponent_hand=opp_hand, opponent_active=[],
            )
        except Exception:
            continue
        nn = len(root.observation.select.option)
        if sums is None:
            sums, counts = [0.0] * nn, [0] * nn
        try:
            for i in range(min(len(sums), nn)):
                if time.time() - t0 > time_budget:
                    break
                try:
                    v = _simulate(A, root.searchId, i, me, leaf_mode)
                except Exception:
                    continue
                sums[i] += v
                counts[i] += 1
        finally:
            try:
                A.search_end()
            except Exception:
                pass

    if not sums or not any(counts):
        return None, None
    best_i, best_avg = None, None
    for i in range(len(sums)):
        if counts[i] > 0:
            avg = sums[i] / counts[i]
            if best_avg is None or avg > best_avg:
                best_avg, best_i = avg, i
    return best_i, best_avg


def best_option(obs: dict, deck: list, time_budget: float = DEFAULT_BUDGET, leaf_mode: str = "hand"):
    """The chosen option as a 1-element list, or None if search does not apply (caller falls back)."""
    i, _v = _search(obs, deck, time_budget, leaf_mode)
    return [i] if i is not None else None


def best_option_value(obs: dict, deck: list, time_budget: float = DEFAULT_BUDGET, leaf_mode: str = "hand"):
    """(move, backed_up_value). move is None if search does not apply. The value is the search's
    A0GB-style bootstrapped estimate of this state -- the training target for the value model."""
    i, v = _search(obs, deck, time_budget, leaf_mode)
    return ([i] if i is not None else None), v


def _simulate(A, root_id, first_choice: int, me: int, leaf_mode: str = "hand") -> float:
    """Take `first_choice`, then play out my turn AND the opponent's reply with the engine
    default policy, and evaluate the state at the start of my next turn (so the score reflects
    the opponent's punish, not just how my board looks before they answer).

    leaf_mode: "hand" -> seat-absolute hand eval (its own scale, terminal +/-1e6).
               "learned"/"blend" -> a [0,1] scale (terminals 1/0/0.5) so the determinization
               average is a real mean; the learned/blended value is only queried on a CLEAN
               non-terminal start-of-my-turn leaf (its training distribution), else neutral 0.5."""
    st = A.search_step(root_id, [first_choice])
    saw_opp = False
    for _ in range(DEPTH_CAP):
        ob = st.observation
        cur = ob.current
        if cur is not None and cur.result != -1:        # game decided in the line
            break
        sel = ob.select
        if sel is None:
            break
        my_move = cur is not None and cur.yourIndex == me
        if saw_opp and my_move:                         # control is back to me -> evaluate here
            break
        if not my_move:
            saw_opp = True
        st = A.search_step(st.searchId, _rollout_pick(sel))     # aggressive playout (both sides)
    obs = _obs_dict(st.observation)
    cur = obs.get("current") or {}
    res = cur.get("result", -1)
    if leaf_mode == "hand":
        return EV.evaluate_obs(obs, me)
    # learned / blend: one [0,1] scale
    if res == me:
        return 1.0
    if res == (1 - me):
        return 0.0
    if res == 2:
        return 0.5
    clean = (st.observation.select is not None) and cur.get("yourIndex") == me
    if not clean:
        return 0.5                                      # neutral on off-distribution leaves
    return EV.evaluate_blend(obs, me) if leaf_mode == "blend" else EV.evaluate_learned(obs, me)
