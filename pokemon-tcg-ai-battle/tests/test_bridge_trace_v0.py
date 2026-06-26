"""Tests for the proposer-bridge trace logger (Section 10): runtime/eval separation, trace-mode-no-behaviour-
change, decision_id alignment, option-index->key mapping, forbidden-field exclusion. Runs on the produced
traces (skips with a notice if not built). Run: python tests/test_bridge_trace_v0.py
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
ATLAS = os.path.join(HERE, "..", "data", "generated", "starmie_bridge_trace_v0")
FORBIDDEN_IN_RUNTIME = ("pilot_name", "outcome_won", "future_same_turn_sequence", "replay_link")


def _sample(name, k=200):
    p = os.path.join(ATLAS, name)
    if not os.path.exists(p):
        return None
    rows = []
    for line in open(p, encoding="utf-8"):
        rows.append(json.loads(line))
        if len(rows) >= k:
            break
    return rows


def test_runtime_eval_separation_and_forbidden_fields():
    rows = _sample("yushin_trace.jsonl")
    if rows is None:
        print("SKIP separation: traces not built"); return
    for r in rows:
        assert set(r.keys()) >= {"runtime", "eval_meta"}, r.keys()
        rt = json.dumps(r["runtime"])
        for f in FORBIDDEN_IN_RUNTIME:
            assert f not in rt, f"forbidden field {f} leaked into runtime"
    print(f"PASS separation: {len(rows)} rows, no forbidden field in runtime")


def test_option_index_to_key_mapping():
    rows = _sample("yushin_trace.jsonl")
    if rows is None:
        print("SKIP mapping: traces not built"); return
    for r in rows:
        idx2key = r["runtime"]["option_index_to_semantic_key"]
        assert len(idx2key) == r["runtime"]["n_legal"], (len(idx2key), r["runtime"]["n_legal"])
        keys = set(idx2key.values())
        ak = r["runtime"]["current_agent_action_key"]
        pk = r["eval_meta"]["pilot_action_key"]
        assert ak is None or ak in keys, ak
        assert pk is None or pk in keys, pk
    print(f"PASS mapping: every option has a key; agent/pilot keys resolve ({len(rows)} rows)")


def test_decision_id_alignment():
    rows = _sample("yushin_trace.jsonl")
    if rows is None:
        print("SKIP alignment: traces not built"); return
    for r in rows:
        em = r["eval_meta"]
        assert r["decision_id"] == f"{em['episode_id']}_{em['step']}_{em['seat']}", r["decision_id"]
    print(f"PASS alignment: decision_id == episode_step_seat ({len(rows)} rows)")


def test_trace_mode_does_not_mutate_obs():
    """The tracing helpers (tactical extractor + semantic keys) are READ-ONLY -- they must not mutate the
    observation, so logging cannot change the agent's chosen action."""
    import starmie_tactical_state as TS
    import deck_policy_v3 as DP  # noqa
    obs = {"current": {"yourIndex": 0, "players": [
        {"active": [{"id": 1031, "hp": 330, "energies": [{"id": 17}], "energyCards": [{"id": 17}]}],
         "bench": [{"id": 666, "hp": 160, "energies": [], "energyCards": []}],
         "prize": [1] * 5, "hand": [1, 2], "handCount": 2, "deckCount": 30},
        {"active": [{"id": 743, "hp": 120, "energies": []}], "bench": [], "prize": [1] * 6, "handCount": 5, "deckCount": 40}]},
        "select": {"maxCount": 1, "minCount": 1, "option": [
            {"type": 13, "attackId": 1488}, {"type": 13, "attackId": 1487}, {"type": 14}]}}
    before = copy.deepcopy(obs)
    TS.extract(obs)
    TS.extract(obs)
    assert obs == before, "tactical extraction mutated the observation"
    print("PASS trace-mode: tactical extraction is read-only (obs unchanged)")


def test_bridge_input_join_fields():
    p = os.path.join(ATLAS, "model_a_bridge_input.jsonl")
    if not os.path.exists(p):
        print("SKIP bridge-input: not built"); return
    r = json.loads(open(p, encoding="utf-8").readline())
    for f in ("decision_id", "option_index_to_semantic_key", "current_agent_action_key", "eval_meta"):
        assert f in r, f
    assert "pilot_action_key" in r["eval_meta"]
    # forbidden fields stay under eval_meta, never at top-level runtime fields
    assert "pilot_name" not in {k for k in r if k != "eval_meta"}
    print("PASS bridge-input: join fields present, eval_meta separated")


def main() -> int:
    rc = 0
    for t in (test_runtime_eval_separation_and_forbidden_fields, test_option_index_to_key_mapping,
              test_decision_id_alignment, test_trace_mode_does_not_mutate_obs, test_bridge_input_join_fields):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}"); rc = 1
    print(f"\n{'ALL BRIDGE-TRACE TESTS PASS' if rc == 0 else 'SOME FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
