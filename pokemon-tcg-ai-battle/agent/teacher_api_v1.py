"""SPLIT_BASE_V2 / P4 -- Teacher API V1 (frozen shared interface).

Wraps the EXISTING forward-model search (agent/search.py) WITHOUT modifying it. For one single-pick
decision it returns the rich per-action signal both branches consume:

    per option:   semantic_action_key, mean_value, value_variance,
                  completed_determinizations, normalized_advantage, eq_class
    per decision: top_two_margin, soft_policy_target, acceptable_action_set,
                  forced_action_flag, chosen_option (forced floor else search argmax),
                  teacher seed + config hash

It reproduces the DEPLOYED teacher exactly -- agent_search = main._forced_move floor (lethal/KO,
go-first) THEN search.best_option (1-ply, opp_prior=None same-deck, opp_k=0, leaf_mode='hand'). The
ONLY additions over search are per-determinization bookkeeping and derived summaries; the planner is
NOT redesigned. Decision-level metrics are computed over SEMANTIC EQUIVALENCE CLASSES, never raw
option index (so two identical moves are one class and two distinct PLAYs are two classes).

Reproducibility (IMPORTANT, measured -- do not overclaim): pass seed=<int> to fix the determinization
DRAW (which hidden cards are sampled) via Python's RNG. The seed does NOT control the native engine's
internal rollout RNG -- coin flips (manual_coin=False, matching the deployed agent) and shuffle effects
resolve inside cg.dll, and sim.py exposes no Seed hook. Measured on real strategic decisions, ~93% have
engine rollout RNG, so same-seed queries are NOT bit-identical there: each per-option value is a Monte
Carlo estimate and value_variance reports the COMBINED (determinization + engine) noise. For low-noise
labels, raise n_determ offline (the noise averages down). Branch A's A2 stability audit must treat
per-seed values as noisy and average enough worlds; cross-seed disagreement conflates the two sources.

It mirrors search._search (the deployed agent: _forced_move floor then 1-ply hand-eval search), so its
argmax equals the agent's move IN EXPECTATION; it is deliberately not bound to option_evals' want_features
world-skip (immaterial -- both estimate the same value).

This module is shared and frozen after the split (auditor-gated changes only). It performs no
training and chooses no research direction.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time

import search as S
import main as M
import state_action_schema_v2 as SCH

# defaults match the DEPLOYED agent so teacher labels equal the agent that plays
DEFAULT_N_DETERM = 8
DEFAULT_BUDGET = S.DEFAULT_BUDGET          # 0.6 s/decision (the match cap)
DEFAULT_LEAF = "hand"
_SOFT_EPS = 1e-9


def teacher_config(n_determ: int = DEFAULT_N_DETERM, time_budget: float = DEFAULT_BUDGET,
                   leaf_mode: str = DEFAULT_LEAF, deck: list | None = None, seed=None) -> dict:
    """The reproducible identity of a teacher query: tunables + the frozen search-module constants
    the value depends on. Two queries with the same config_hash are the same teacher."""
    deck_sig = SCH.deck_signature(deck) if deck else {"hash": None, "n_cards": 0, "n_distinct": 0}
    payload = {
        "schema_version": SCH.SCHEMA_VERSION,
        "teacher": "v1",
        "n_determ": n_determ,
        "time_budget": time_budget,
        "leaf_mode": leaf_mode,
        "opp_prior": None,          # same-deck determinization (the deployed teacher)
        "opp_k": 0,                 # 1-ply (the deployed teacher)
        "deck_hash": deck_sig["hash"],
        "seed": seed,
        "search_DEFAULT_BUDGET": S.DEFAULT_BUDGET,
        "search_DEPTH_CAP": S.DEPTH_CAP,
        "search_MY_CONT": S.MY_CONT,
        "search_N_DETERM_default": S.N_DETERM,
    }
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return {"hash": h, "config": payload}


def _per_world_values(obs: dict, deck: list, n_determ: int, time_budget: float, leaf_mode: str):
    """Mirror search._search's determinization loop EXACTLY (the deployed teacher: opp_prior=None,
    opp_k=0), but RETAIN each option's per-world leaf values instead of collapsing to a running mean.
    Reuses search's own helpers (_api, _hidden_pool, _simulate) so the sampled worlds and leaf values
    are identical to the agent's under a shared RNG seed. Returns list[option] -> list[float], or
    None if search does not apply. Never raises."""
    A = S._api()
    if A is None:
        return None
    sel, cur = obs.get("select"), obs.get("current")
    if not sel or not cur or (sel.get("maxCount") or 0) != 1 or len(sel.get("option") or []) < 2:
        return None
    players = cur.get("players") or []
    if len(players) < 2:
        return None
    me = cur.get("yourIndex", 0)
    P, O = players[me], players[1 - me]
    oa = O.get("active") or []
    if oa and oa[0] is None:                 # face-down opponent active -> search does not apply
        return None
    n_my_deck, n_op_deck = P.get("deckCount", 0) or 0, O.get("deckCount", 0) or 0
    n_my_prize, n_op_prize = len(P.get("prize") or []), len(O.get("prize") or [])
    n_op_hand = O.get("handCount", 0) or 0
    obsd = A.to_observation_class(obs)
    per_world = None
    t0 = time.time()
    for _ in range(n_determ):
        if time.time() - t0 > time_budget:
            break
        # SAME draw order as search._search (P pool, then O pool) -> identical determinization
        # DRAW under a seed (the engine's rollout coin/shuffle RNG still varies; see module docstring)
        mp = S._hidden_pool(deck, P, exclude_hand=False)
        mp += [3] * max(0, (n_my_deck + n_my_prize) - len(mp))
        op = S._hidden_pool(deck, O, exclude_hand=True)
        op += [3] * max(0, (n_op_deck + n_op_prize + n_op_hand) - len(op))
        try:
            root = A.search_begin(
                obsd,
                your_deck=mp[:n_my_deck], your_prize=mp[n_my_deck:n_my_deck + n_my_prize],
                opponent_deck=op[n_op_hand + n_op_prize:n_op_hand + n_op_prize + n_op_deck],
                opponent_prize=op[n_op_hand:n_op_hand + n_op_prize], opponent_hand=op[:n_op_hand],
                opponent_active=[],
            )
        except Exception:
            continue
        nn = len(root.observation.select.option)
        if per_world is None:
            per_world = [[] for _ in range(nn)]
        try:
            for i in range(min(len(per_world), nn)):
                if time.time() - t0 > time_budget:
                    break
                try:
                    v = S._simulate(A, root.searchId, i, me, leaf_mode)   # opp_k=0, want_features=False
                except Exception:
                    continue
                per_world[i].append(v)
        finally:
            try:
                A.search_end()
            except Exception:
                pass
    return per_world


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _variance(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)   # sample variance


def query(obs: dict, deck: list, *, n_determ: int = DEFAULT_N_DETERM,
          time_budget: float = DEFAULT_BUDGET, leaf_mode: str = DEFAULT_LEAF,
          seed=None, accept_z: float = 1.0) -> dict:
    """Query Teacher V1 on one decision. Returns a structured result (see module docstring), or
    {'applicable': False, ...} when search does not apply (the agent would fall back to the
    heuristic). Pure w.r.t. `obs` (does not mutate it). Never raises.

    accept_z: an eq-class joins the acceptable-action set if it is within accept_z combined standard
    errors of the best class (a "statistically indistinguishable from best" set; near-ties are not
    forced into one hard winner)."""
    cfg = teacher_config(n_determ, time_budget, leaf_mode, deck, seed)
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    me = cur.get("yourIndex", 0)
    base = {"applicable": False, "config_hash": cfg["hash"], "config": cfg["config"],
            "seed": seed, "n_options": len(opts), "me": me}

    # forced-move floor (the deployed agent applies this BEFORE search) -- always available
    forced_opt = None
    try:
        fm = M._forced_move(obs)
        if fm is not None and 0 <= fm[0] < len(opts):
            forced_opt = fm[0]
    except Exception:
        forced_opt = None

    if not SCH.is_single_pick_decision(obs):
        base["forced_action_flag"] = forced_opt is not None
        base["forced_option"] = forced_opt
        base["chosen_option"] = forced_opt
        return base

    if seed is not None:
        random.seed(seed)
    t0 = time.time()
    per_world = _per_world_values(obs, deck, n_determ, time_budget, leaf_mode)
    elapsed = time.time() - t0
    if per_world is None:
        base["forced_action_flag"] = forced_opt is not None
        base["forced_option"] = forced_opt
        base["chosen_option"] = forced_opt
        base["elapsed_s"] = elapsed
        return base

    nn = len(per_world)
    keys = [SCH.semantic_action_key(opts[i], cur, me) if i < len(opts) and isinstance(opts[i], dict)
            else ("__nondict__", i) for i in range(nn)]
    eq_index = {}
    for k in keys:
        if k not in eq_index:
            eq_index[k] = len(eq_index)
    opt_eq = [eq_index[k] for k in keys]

    means = [_mean(per_world[i]) for i in range(nn)]
    varis = [_variance(per_world[i]) for i in range(nn)]
    counts = [len(per_world[i]) for i in range(nn)]
    present = [m for m in means if m is not None]
    if not present:
        base["forced_action_flag"] = forced_opt is not None
        base["forced_option"] = forced_opt
        base["chosen_option"] = forced_opt
        base["elapsed_s"] = elapsed
        return base
    grand_mean = sum(present) / len(present)             # centering baseline (matches dataset 'adv')

    options = []
    for i in range(nn):
        options.append({
            "index": i,
            "semantic_action_key": keys[i],
            "eq_class": opt_eq[i],
            "mean_value": means[i],
            "value_variance": varis[i],
            "completed_determinizations": counts[i],
            "normalized_advantage": (means[i] - grand_mean) if means[i] is not None else None,
        })

    # ---- aggregate to semantic equivalence classes (the canonical decision unit) ----
    classes = []
    for k, ci in sorted(eq_index.items(), key=lambda kv: kv[1]):
        members = [i for i in range(nn) if opt_eq[i] == ci and means[i] is not None]
        if not members:
            classes.append({"eq_class": ci, "key": k, "members": [], "mean_value": None,
                            "value_variance": None, "completed_determinizations": 0})
            continue
        cm = sum(means[i] for i in members) / len(members)
        cv = sum(varis[i] for i in members) / len(members)
        cc = sum(counts[i] for i in members)
        classes.append({"eq_class": ci, "key": k, "members": members, "mean_value": cm,
                        "value_variance": cv, "completed_determinizations": cc})

    valued = [c for c in classes if c["mean_value"] is not None]
    valued.sort(key=lambda c: -c["mean_value"])
    best = valued[0]
    top_two_margin = (best["mean_value"] - valued[1]["mean_value"]) if len(valued) >= 2 else None

    # soft policy over classes: scale-adaptive softmax of class means centered on the best
    cmeans = [c["mean_value"] for c in valued]
    spread = (max(cmeans) - min(cmeans)) if len(cmeans) >= 2 else 0.0
    temp = max(spread, _SOFT_EPS)
    exps = [math.exp((m - best["mean_value"]) / temp) for m in cmeans]
    z = sum(exps) or 1.0
    soft = {valued[j]["eq_class"]: exps[j] / z for j in range(len(valued))}

    # acceptable set: classes within accept_z combined std-errors of the best (indistinguishable)
    def se(c):
        n = max(1, c["completed_determinizations"])
        return math.sqrt(max(0.0, c["value_variance"] or 0.0) / n)
    acceptable = []
    se_best = se(best)
    for c in valued:
        if best["mean_value"] - c["mean_value"] <= accept_z * (se_best + se(c)) + _SOFT_EPS:
            acceptable.append(c["eq_class"])

    for c in classes:
        c["soft_policy_target"] = soft.get(c["eq_class"], 0.0)
        c["acceptable"] = c["eq_class"] in acceptable

    argmax_option = max((i for i in range(nn) if means[i] is not None), key=lambda i: means[i])
    chosen_option = forced_opt if forced_opt is not None else argmax_option

    return {
        "applicable": True,
        "config_hash": cfg["hash"],
        "config": cfg["config"],
        "seed": seed,
        "me": me,
        "n_options": nn,
        "options": options,
        "eq_classes": classes,
        "argmax_option": argmax_option,
        "argmax_eq_class": opt_eq[argmax_option],
        "top_two_margin": top_two_margin,
        "soft_policy_target": soft,
        "acceptable_action_set": acceptable,
        "forced_action_flag": forced_opt is not None,
        "forced_option": forced_opt,
        "forced_eq_key": keys[forced_opt] if forced_opt is not None and forced_opt < nn else None,
        "chosen_option": chosen_option,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    print("Teacher API V1 -- shared, frozen after split.")
    print("  defaults:", teacher_config()["config"])
