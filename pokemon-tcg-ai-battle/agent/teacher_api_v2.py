"""Branch A / A4 -- Teacher V2: selective high-criticality labels with a signal stronger than hand-eval.

Gates expensive evaluation behind a cheap criticality score (extra compute only where it matters), then for
a triggered decision combines:
  * a LOW-NOISE hand-eval advantage  -- high-N determinizations (Teacher V1 machinery at N>=32), the
    primary counterfactual-advantage target, and
  * a TERMINAL-OUTCOME auxiliary     -- K full playouts per option to a decided result -> per-option
    win-rate, a signal beyond the 1-ply hand leaf (the plan's auxiliary critic).

Output per decision is the rich label Model B requested (siblings, soft policy, advantage, criticality,
stability/uncertainty, determinizations, action spread, outcome, metadata). Wraps existing search; no
planner redesign. Reproducible determinization via seed; engine rollout RNG remains MC noise (see A2).
"""
from __future__ import annotations

import random
import statistics
import time

import search as S
import main as M
import state_action_schema_v2 as SCH
import teacher_api_v1 as T1
import features as FT

DEPTH_CAP = getattr(S, "DEPTH_CAP", 80)


def criticality_score(obs: dict) -> dict:
    """Cheap criticality from the obs. Returns components + a scalar in ~[0,1]. No engine."""
    f = FT.encode_state(obs)
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    me = cur.get("yourIndex", 0)
    n_eq = len(set(SCH.equivalence_classes(opts, cur, me))) if cur.get("players") else 1
    can_ko = 1.0 if f.get("can_ko_opp_now", 0) > 0 else 0.0
    # KO-back exposure: opponent's active can plausibly KO my active (rough proxy via HP vs a big hit)
    my_hp = f.get("my_active_hp", 0) or 0
    ko_back = 1.0 if (0 < my_hp <= 120) else 0.0
    endgame = 1.0 if min(f.get("my_prizes_left", 6), f.get("opp_prizes_left", 6)) <= 2 else 0.0
    branching = min(1.0, n_eq / 6.0)
    score = 0.45 * can_ko + 0.2 * ko_back + 0.2 * endgame + 0.15 * branching
    return {"score": round(score, 3), "can_ko": can_ko, "ko_back": ko_back,
            "endgame": endgame, "n_eq_classes": n_eq}


def _playout_result(A, root_id: int, first_choice: int, me: int) -> float:
    """Take first_choice, then play BOTH sides to a decided result with the rollout policy.
    Returns 1.0 win / 0.0 loss / 0.5 draw-or-undecided, from my seat. Never raises out."""
    st = A.search_step(root_id, [first_choice])
    for _ in range(DEPTH_CAP * 4):
        cur = st.observation.current
        if cur is not None and cur.result != -1:
            break
        sel = st.observation.select
        if sel is None:
            break
        is_me = cur is not None and cur.yourIndex == me
        st = A.search_step(st.searchId, S._rollout_pick(sel, is_me=is_me))
    cur = st.observation.current
    res = cur.result if cur is not None else -1
    return 1.0 if res == me else (0.0 if res == (1 - me) else 0.5)


def _outcome_winrate(obs: dict, deck: list, k: int, time_budget: float):
    """Per-option terminal win-rate over k full playouts (fresh determinization each). list[option] of
    (winrate, n) or None where not run. The 'stronger than hand-eval' signal. Never raises."""
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
    if oa and oa[0] is None:
        return None
    n_my_deck, n_op_deck = P.get("deckCount", 0) or 0, O.get("deckCount", 0) or 0
    n_my_prize, n_op_prize = len(P.get("prize") or []), len(O.get("prize") or [])
    n_op_hand = O.get("handCount", 0) or 0
    obsd = A.to_observation_class(obs)
    wins = counts = None
    t0 = time.time()
    for _ in range(k):
        if time.time() - t0 > time_budget:
            break
        mp = S._hidden_pool(deck, P, exclude_hand=False); mp += [3] * max(0, (n_my_deck + n_my_prize) - len(mp))
        op = S._hidden_pool(deck, O, exclude_hand=True); op += [3] * max(0, (n_op_deck + n_op_prize + n_op_hand) - len(op))
        try:
            root = A.search_begin(obsd, your_deck=mp[:n_my_deck], your_prize=mp[n_my_deck:n_my_deck + n_my_prize],
                                  opponent_deck=op[n_op_hand + n_op_prize:n_op_hand + n_op_prize + n_op_deck],
                                  opponent_prize=op[n_op_hand:n_op_hand + n_op_prize], opponent_hand=op[:n_op_hand],
                                  opponent_active=[])
        except Exception:
            continue
        nn = len(root.observation.select.option)
        if wins is None:
            wins, counts = [0.0] * nn, [0] * nn
        try:
            for i in range(min(len(wins), nn)):
                if time.time() - t0 > time_budget:
                    break
                try:
                    wins[i] += _playout_result(A, root.searchId, i, me)
                    counts[i] += 1
                except Exception:
                    continue
        finally:
            try:
                A.search_end()
            except Exception:
                pass
    if not wins or not any(counts):
        return None
    return [(wins[i] / counts[i], counts[i]) if counts[i] else None for i in range(len(wins))]


def outcome_playouts(obs: dict, deck: list, k: int, time_budget: float):
    """Per-option list of terminal results over k PAIRED playouts: within each of the k iterations all
    siblings are played out from the SAME determinized world (shared-world paired comparison). Returns
    list[option] -> [r1, r2, ...] aligned by world index, or None. The richer form of _outcome_winrate
    (keeps the per-world sequence so convergence and variance can be read off). Never raises."""
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
    if oa and oa[0] is None:
        return None
    n_my_deck, n_op_deck = P.get("deckCount", 0) or 0, O.get("deckCount", 0) or 0
    n_my_prize, n_op_prize = len(P.get("prize") or []), len(O.get("prize") or [])
    n_op_hand = O.get("handCount", 0) or 0
    obsd = A.to_observation_class(obs)
    results = None
    t0 = time.time()
    for _ in range(k):
        if time.time() - t0 > time_budget:
            break
        mp = S._hidden_pool(deck, P, exclude_hand=False); mp += [3] * max(0, (n_my_deck + n_my_prize) - len(mp))
        op = S._hidden_pool(deck, O, exclude_hand=True); op += [3] * max(0, (n_op_deck + n_op_prize + n_op_hand) - len(op))
        try:
            root = A.search_begin(obsd, your_deck=mp[:n_my_deck], your_prize=mp[n_my_deck:n_my_deck + n_my_prize],
                                  opponent_deck=op[n_op_hand + n_op_prize:n_op_hand + n_op_prize + n_op_deck],
                                  opponent_prize=op[n_op_hand:n_op_hand + n_op_prize], opponent_hand=op[:n_op_hand],
                                  opponent_active=[])
        except Exception:
            continue
        nn = len(root.observation.select.option)
        if results is None:
            results = [[] for _ in range(nn)]
        try:
            for i in range(min(len(results), nn)):
                if time.time() - t0 > time_budget:
                    break
                try:
                    results[i].append(_playout_result(A, root.searchId, i, me))
                except Exception:
                    continue
        finally:
            try:
                A.search_end()
            except Exception:
                pass
    return results


def query_v2(obs: dict, deck: list, *, n_determ: int = 32, hand_budget: float = 8.0,
             k_outcome: int = 6, outcome_budget: float = 12.0, seed=None, crit_threshold: float = 0.3) -> dict:
    """Teacher V2 label for one decision. Always returns criticality; runs the expensive stronger
    evaluation (high-N hand advantage + outcome win-rate) only when criticality >= threshold."""
    crit = criticality_score(obs)
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    out = {"criticality": crit, "evaluated": False, "config": {"n_determ": n_determ, "k_outcome": k_outcome,
            "hand_budget": hand_budget, "outcome_budget": outcome_budget, "seed": seed},
           "applicable": SCH.is_single_pick_decision(obs)}
    if not out["applicable"] or crit["score"] < crit_threshold:
        return out
    if seed is not None:
        random.seed(seed)
    v1 = T1.query(obs, deck, n_determ=n_determ, time_budget=hand_budget, leaf_mode="hand", seed=seed)
    if not v1.get("applicable"):
        return out
    outcome = _outcome_winrate(obs, deck, k_outcome, outcome_budget)
    opts = obs["select"]["option"]
    options = []
    for o in v1["options"]:
        i = o["index"]
        ow = outcome[i] if (outcome and i < len(outcome) and outcome[i]) else None
        options.append({
            "index": i, "semantic_action_key": o["semantic_action_key"], "eq_class": o["eq_class"],
            "hand_mean_value": o["mean_value"], "hand_value_variance": o["value_variance"],
            "hand_norm_advantage": o["normalized_advantage"],
            "completed_determinizations": o["completed_determinizations"],
            "outcome_winrate": round(ow[0], 3) if ow else None, "outcome_playouts": ow[1] if ow else 0,
        })
    # action spread (hand advantage range) + whether outcome and hand agree on the top action
    hand_best = v1.get("argmax_eq_class")
    ow_vals = [(o["index"], o["outcome_winrate"]) for o in options if o["outcome_winrate"] is not None]
    outcome_best = max(ow_vals, key=lambda x: x[1])[0] if ow_vals else None
    _hv = [o["mean_value"] for o in v1["options"] if o["mean_value"] is not None]
    spread = (max(_hv) - min(_hv)) if len(_hv) >= 2 else 0.0
    out.update({
        "evaluated": True, "me": me,
        "options": options,
        "soft_policy_target": v1["soft_policy_target"],
        "acceptable_action_set": v1["acceptable_action_set"],
        "top_two_margin": v1["top_two_margin"],
        "hand_argmax_eq_class": hand_best,
        "outcome_argmax_option": outcome_best,
        "hand_outcome_agree": (outcome_best is not None and outcome_best in
                               [o["index"] for o in options if o["eq_class"] == hand_best]),
        "action_spread": round(spread, 1),
        "forced_action_flag": v1["forced_action_flag"],
        "config_hash": v1["config_hash"],
    })
    return out


if __name__ == "__main__":
    print("Teacher V2: selective high-criticality labels (high-N hand advantage + terminal-outcome auxiliary).")
