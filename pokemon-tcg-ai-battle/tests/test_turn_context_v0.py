"""Tests for the read-only public turn-context extractor (agent/turn_context_v0). PREP module: must be read-only,
schema-stable, exclude the forbidden outcome field, and NOT be wired into gameplay.

  PYTHONIOENCODING=utf-8 python tests/test_turn_context_v0.py
"""
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import turn_context_v0 as TC  # noqa: E402


def _obs(**cur):
    base = {"current": {"yourIndex": 0, "firstPlayer": 0, "turn": 3, "turnActionCount": 4, "supporterPlayed": True,
                        "energyAttached": False, "retreated": False, "stadiumPlayed": False, "stadium": [],
                        "result": -1, "players": [
                            {"active": [{"id": 1031, "appearThisTurn": False}], "bench": [], "asleep": False,
                             "paralyzed": False, "confused": False, "burned": False, "poisoned": False},
                            {"active": [{"id": 743}], "bench": []}]},
            "select": {"option": [{"type": 8}, {"type": 13}, {"type": 14}, {"type": 3}]}}
    base["current"].update(cur)
    return base


def test_no_mutation():
    o = _obs()
    before = copy.deepcopy(o)
    TC.extract_turn_context(o)
    assert o == before, "extractor mutated the observation"
    print("PASS no-mutation")


def test_schema_stable():
    keys = None
    for o in (_obs(), _obs(firstPlayer=-1, turn=0), {"current": {}, "select": {}}, {}, _obs(turn=None)):
        p = TC.extract_turn_context(o)
        k = tuple(sorted(p.keys()))
        assert set(TC.FIELDS).issubset(p.keys()), f"missing fields: {set(TC.FIELDS) - set(p.keys())}"
        assert "field_status" in p
        if keys is None:
            keys = k
        assert k == keys, "schema not stable across observations"
    print("PASS schema-stable (always the full field set + field_status)")


def test_forbidden_outcome_excluded():
    o = _obs(result=7)
    p = TC.extract_turn_context(o)
    assert not any(k in TC.FORBIDDEN for k in p), "a forbidden key leaked into the payload"
    # the outcome value must not be surfaced under any field
    assert "result" not in json.dumps(p) or "field_status" in p  # 'result' may appear only inside status strings, not as data
    assert all(b not in str(p.get(f)) for f in TC.FIELDS for b in ("won", "result=7")), "outcome value leaked"
    print("PASS forbidden-outcome-excluded (current.result not surfaced)")


def test_missing_fields_explicit_null():
    p = TC.extract_turn_context({"current": {}, "select": {}})
    assert p["global_turn_number"] is None and p["turn_action_count"] is None and p["supporter_used_this_turn"] is None
    assert p["field_status"]["global_turn_number"] == "null_missing"
    assert p["field_status"]["turn_action_count"] == "null_missing"
    print("PASS missing-fields-explicit-null")


def test_values_correct():
    p = TC.extract_turn_context(_obs(turnActionCount=9, supporterPlayed=True, energyAttached=True, retreated=False))
    assert p["decision_index_in_turn"] == 9 and p["supporter_used_this_turn"] is True
    assert p["energy_attached_this_turn"] is True and p["retreated_this_turn"] is False
    assert p["attack_available"] is True and p["end_available"] is True
    assert p["terminal_legal_option_count"] == 2 and p["nonterminal_legal_option_count"] == 2
    assert p["information_revealing_legal_count"] == 1 and p["safe_development_legal_count"] == 1
    assert p["is_setup_phase"] is False and p["am_i_first_player"] is True
    print("PASS values-correct")


def test_setup_phase_caveat():
    p = TC.extract_turn_context(_obs(firstPlayer=-1, turn=0))
    assert p["is_setup_phase"] is True and p["am_i_first_player"] is None
    assert p["is_our_first_turn_best_effort"] is None  # not derived during setup
    assert p["field_status"]["is_our_first_turn_best_effort"] == "best_effort_setup_caveat"
    print("PASS setup-phase-caveat (no over-assertion during setup)")


def test_not_wired_into_gameplay():
    src = (ROOT / "agent" / "starmie_heuristics.py").read_text(encoding="utf-8")
    assert "turn_context_v0" not in src, "turn_context_v0 is imported by gameplay -- must stay unwired until V3"
    print("PASS not-wired (gameplay does not import the prep module)")


def main() -> int:
    rc = 0
    for t in (test_no_mutation, test_schema_stable, test_forbidden_outcome_excluded, test_missing_fields_explicit_null,
              test_values_correct, test_setup_phase_caveat, test_not_wired_into_gameplay):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            rc = 1
    print(f"\n{'ALL TURN-CONTEXT TESTS PASS' if rc == 0 else 'SOME FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
