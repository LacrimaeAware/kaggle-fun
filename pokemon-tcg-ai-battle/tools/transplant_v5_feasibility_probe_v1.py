"""Transplant V5 runtime-feature feasibility probe (Model B support pack, OFFLINE/READ-ONLY).

Empirically tests -- on recorded golden-state observations -- whether the context / action / delta axes a
state-conditioned T(s,a,delta) transplant would need are computable from the LIVE runtime without a full turn
rollout, without simulating the opponent, and without hidden cards or the game result.

Three mechanisms, all already in the repo (we ADD nothing to the agent; this is an audit):
  * CONTEXT  -> turn_context_v0.extract_turn_context(obs) + learned_selector_bridge.tactical_state_features(obs)
  * ACTION   -> learned_proposer_adapter.option_index_to_key(obs) + learned_selector_bridge.option_features(opt)
  * DELTA    -> (a) realized one-ply public diff via search_v3.option_deltas (cg.api search_begin+search_step),
                (b) capability/threshold deltas by recomputing the STATIC tactical levels on the post-apply obs
                    (pre vs post), for NON-terminal options (post obs is still my own next sub-decision).

Writes immediate_delta_probe.json + example_payloads.jsonl. Run:
  PYTHONIOENCODING=utf-8 python tools/transplant_v5_feasibility_probe_v1.py
"""
from __future__ import annotations
import collections
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
OUT = ROOT / "data" / "generated" / "transplant_v5_runtime_support"
FIXTURES = ROOT / "tests" / "golden_state_action_fixtures" / "fixtures.json"
TERMINAL_FAMILIES = {"ATTACK", "END", "RETREAT"}
TYPE_TO_FAMILY = {3: "CARD", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 12: "RETREAT", 13: "ATTACK", 14: "END"}

with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    import search_v3 as S
    import learned_selector_bridge as BR
    import learned_proposer_adapter as AD
    import turn_context_v0 as TC
    import deck_policy_v3 as DP
    import starmie_heuristics as SH

DECK = list(SH.STARMIE_DECK)
# the static tactical "levels" we diff pre/post to get capability deltas. names are bridge.tactical_state_features keys.
CAP_LEVELS = ["board.my_ready_main_attackers", "board.my_backup_ready", "board.my_main_one_short",
              "commitment.guaranteed_ko_available", "commitment.game_winning_attack_available",
              "commitment.nonterminal_attack_available", "value.ready_attacker_diff"]
# V5 delta axis -> how we source it
DELTA_SOURCE = {
    "hand_count_delta": ("apply_realized", "option_deltas.cards_drawn"),
    "deck_count_delta": ("apply_realized", "option_deltas.deck_used"),
    "board_hp_delta": ("apply_realized_partial", "option_deltas.dmg_dealt (opp active only)"),
    "energy_attached_delta": ("apply_realized", "option_deltas.energy_attached"),
    "ko_realized": ("apply_realized", "option_deltas.opp_ko / prizes_taken"),
    "board_dev_delta": ("apply_realized", "option_deltas.board_dev"),
    "ready_attacker_delta": ("apply_recompute", "tactical board.my_ready_main_attackers pre/post"),
    "ko_available_delta": ("apply_recompute", "tactical commitment.guaranteed_ko_available pre/post"),
    "attack_affordability_delta": ("apply_recompute", "tactical board.my_main_one_short pre/post"),
    "energy_shortfall_delta": ("static_or_recompute", "SH energy-unit shortfall (ATTACH descriptor) or pre/post"),
    "line_completion_delta": ("static_projection", "deck_policy_v3 _projected_hand_after / safe_pre_attack"),
    "threshold_crossing_flags": ("apply_recompute", "my_main_one_short True->False == crossed attack threshold"),
}


def _flat_tac(obs):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return BR.tactical_state_features(obs)
    except Exception:
        return {}


def _context(obs):
    ctx = {}
    try:
        tc = TC.extract_turn_context(obs)
        for k in ("global_turn_number", "turn_action_count", "supporter_used_this_turn",
                  "energy_attached_this_turn", "retreated_this_turn", "safe_development_legal_count",
                  "terminal_legal_option_count", "nonterminal_legal_option_count", "first_player_resolved"):
            ctx[k] = tc.get(k)
    except Exception:
        pass
    tac = _flat_tac(obs)
    for k in ("board.prize_diff", "board.my_ready_main_attackers", "board.my_backup_ready", "board.my_units",
              "board.opp_units", "board.my_deck_count", "value.deckout_pressure",
              "commitment.safe_development_available", "commitment.guaranteed_ko_available"):
        ctx[k] = tac.get(k)
    return ctx


def _action(opt, obs, key):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            of = BR.option_features(opt, obs)
    except Exception:
        of = {}
    fam = of.get("action_family") or TYPE_TO_FAMILY.get(opt.get("type"), "?")
    return {"family": fam, "compact_semantic_key": key, "source_card_id": of.get("source_card_id"),
            "target_card_id": of.get("target_card_id"), "target_owner": of.get("target_owner"),
            "target_zone": of.get("target_zone"), "ends_turn": of.get("ends_turn"),
            "terminal": fam in TERMINAL_FAMILIES}


def _begin(obs):
    """Replicate search_v3.option_deltas' begin block; return (api, root, base_snapshot, me) or (None,...)."""
    A = S._api()
    if A is None:
        return None, None, None, None
    sel, cur = obs.get("select"), obs.get("current")
    if not sel or not cur or (sel.get("maxCount") or 0) != 1 or len(sel.get("option") or []) < 2:
        return None, None, None, None
    players = cur.get("players") or []
    if len(players) < 2:
        return None, None, None, None
    me = cur.get("yourIndex", 0)
    P, O = players[me], players[1 - me]
    oa = O.get("active") or []
    if oa and oa[0] is None:
        return None, None, None, None
    base = S._snapshot(cur, me)
    if not base:
        return None, None, None, None
    n_my_deck, n_op_deck = P.get("deckCount", 0) or 0, O.get("deckCount", 0) or 0
    n_my_prize, n_op_prize = len(P.get("prize") or []), len(O.get("prize") or [])
    n_op_hand = O.get("handCount", 0) or 0
    obsd = A.to_observation_class(obs)
    mp = S._hidden_pool(DECK, P, exclude_hand=False); mp += [3] * max(0, (n_my_deck + n_my_prize) - len(mp))
    op = S._hidden_pool(DECK, O, exclude_hand=True); op += [3] * max(0, (n_op_deck + n_op_prize + n_op_hand) - len(op))
    try:
        root = A.search_begin(obsd, your_deck=mp[:n_my_deck], your_prize=mp[n_my_deck:n_my_deck + n_my_prize],
                              opponent_deck=op[n_op_hand + n_op_prize:n_op_hand + n_op_prize + n_op_deck],
                              opponent_prize=op[n_op_hand:n_op_hand + n_op_prize], opponent_hand=op[:n_op_hand],
                              opponent_active=[])
    except Exception:
        return None, None, None, None
    return A, root, base, me


def analyze_fixture(obs):
    """Per option: realized delta + capability delta (recompute tactical levels on post obs). One begin/fixture."""
    A, root, base, me = _begin(obs)
    if A is None:
        return None
    pre_tac = _flat_tac(obs)
    nn = len(root.observation.select.option)
    rows = [None] * nn
    try:
        for i in range(nn):
            try:
                st = A.search_step(root.searchId, [i])
                post = S._obs_dict(st.observation)
                post_cur = post.get("current") or {}
                snap = S._snapshot(post_cur, me)
                realized = S._delta(base, snap, me) if snap else None
                # capability deltas: recompute static tactical levels on the post obs (valid for non-terminal,
                # where post is still MY decision: yourIndex == me and select is not None)
                cap = {}
                non_terminal_post = (post_cur.get("yourIndex") == me) and (post.get("select") is not None)
                if non_terminal_post:
                    post_tac = _flat_tac(post)
                    for lvl in CAP_LEVELS:
                        a, b = pre_tac.get(lvl), post_tac.get(lvl)
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                            cap[lvl + "::delta"] = b - a
                rows[i] = {"realized": realized, "capability": cap, "non_terminal_post": non_terminal_post}
            except Exception:
                continue
    finally:
        with contextlib.suppress(Exception):
            A.search_end()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = json.load(open(FIXTURES, encoding="utf-8"))["fixtures"]
    payloads = []
    fam_axis = collections.defaultdict(lambda: collections.Counter())   # family -> Counter(axis computed)
    fam_count = collections.Counter()
    realized_keys_seen = collections.Counter()
    cap_examples = []
    n_searchable = 0
    t0 = time.time()
    det_check = None

    for fi, fx in enumerate(fixtures):
        obs = fx.get("observation")
        if not obs or not obs.get("select"):
            continue
        sel = obs["select"]
        if (sel.get("maxCount") or 0) != 1 or len(sel.get("option") or []) < 2:
            continue
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                keys = AD.option_index_to_key(obs)
        except Exception:
            keys = {}
        ctx = _context(obs)
        rows = analyze_fixture(obs)
        if rows is None:
            continue
        n_searchable += 1
        # determinism: re-run the first searchable fixture and compare realized deltas
        if det_check is None:
            rows2 = analyze_fixture(obs)
            det_check = {"fixture": fi, "match": _same_realized(rows, rows2)}
        options = sel.get("option") or []
        for i, opt in enumerate(options):
            if not isinstance(opt, dict) or rows[i] is None:
                continue
            act = _action(opt, obs, keys.get(i))
            fam = act["family"]
            fam_count[fam] += 1
            realized = rows[i].get("realized") or {}
            cap = rows[i].get("capability") or {}
            for k, v in realized.items():
                if v not in (0, 0.0, None):
                    realized_keys_seen[k] += 1
            # which V5 axes are populated for this option
            v5 = _map_v5_deltas(realized, cap, fam)
            for axis, val in v5.items():
                if val is not None:
                    fam_axis[fam][axis] += 1
            if cap and len(cap_examples) < 40 and any(abs(x) > 0 for x in cap.values() if isinstance(x, (int, float))):
                cap_examples.append({"fixture": fi, "option": i, "family": fam,
                                     "compact_key": act["compact_semantic_key"], "capability_delta": cap})
            if len(payloads) < 200:
                missing = [k for k, val in {**ctx}.items() if val is None]
                payloads.append({
                    "decision_id": f"fixture:{fi}", "legal_option_index": i,
                    "context": ctx, "action": act, "delta": v5,
                    "missing_fields": missing,
                    "runtime_safe": True,   # all inputs from public obs + the engine search sandbox the agent already uses
                })

    elapsed = time.time() - t0
    report = {
        "source": "tests/golden_state_action_fixtures/fixtures.json",
        "searchable_fixtures_used": n_searchable,
        "mechanisms": {
            "context": "turn_context_v0.extract_turn_context + learned_selector_bridge.tactical_state_features (STATIC obs)",
            "action": "learned_proposer_adapter.option_index_to_key + learned_selector_bridge.option_features (STATIC obs)",
            "delta_realized": "search_v3.option_deltas: cg.api search_begin + one search_step/option, public-board diff",
            "delta_capability": "recompute static tactical levels on the post-apply obs (pre/post), NON-terminal options only",
        },
        "determinism_check": det_check,
        "realized_delta_keys_nonzero_counts": dict(realized_keys_seen),
        "v5_delta_axis_computed_by_family": {f: dict(c) for f, c in fam_axis.items()},
        "family_option_counts": dict(fam_count),
        "delta_axis_source_map": {k: {"mechanism": v[0], "from": v[1]} for k, v in DELTA_SOURCE.items()},
        "capability_delta_worked_examples": cap_examples[:20],
        "cost": {"searchable_fixtures": n_searchable, "wall_clock_s": round(elapsed, 1),
                 "per_fixture_s": round(elapsed / max(1, n_searchable), 3),
                 "note": "one search_begin + one search_step per option per fixture; non-terminal options also recompute tactical levels."},
    }
    (OUT / "immediate_delta_probe.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with open(OUT / "example_payloads.jsonl", "w", encoding="utf-8") as fh:
        for p in payloads:
            fh.write(json.dumps(p, default=str) + "\n")
    print(json.dumps({"searchable_fixtures": n_searchable, "payloads": len(payloads),
                      "determinism": det_check, "realized_keys": dict(realized_keys_seen),
                      "v5_axis_by_family": {f: dict(c) for f, c in fam_axis.items()},
                      "cost_per_fixture_s": report["cost"]["per_fixture_s"]}, indent=2, default=str))
    return 0


def _map_v5_deltas(realized, cap, fam):
    """Map what we actually computed onto the V5 delta axis names (None where not populated for this option)."""
    def cz(key):  # capability-zero-aware get
        return cap.get(key + "::delta")
    out = {
        "hand_count_delta": realized.get("cards_drawn"),
        "deck_count_delta": realized.get("deck_used"),
        "board_hp_delta": realized.get("dmg_dealt"),
        "energy_attached_delta": realized.get("energy_attached"),
        "ko_realized": realized.get("opp_ko"),
        "board_dev_delta": realized.get("board_dev"),
        "ends_turn": realized.get("ends_turn"),
        "ready_attacker_delta": cz("board.my_ready_main_attackers"),
        "ko_available_delta": cz("commitment.guaranteed_ko_available"),
        "attack_affordability_delta": cz("board.my_main_one_short"),
    }
    return {k: v for k, v in out.items()}


def _same_realized(r1, r2):
    if r1 is None or r2 is None or len(r1) != len(r2):
        return False
    for a, b in zip(r1, r2):
        if (a is None) != (b is None):
            return False
        if a and b and a.get("realized") != b.get("realized"):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
