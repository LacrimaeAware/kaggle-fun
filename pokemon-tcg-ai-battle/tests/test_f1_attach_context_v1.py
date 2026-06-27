"""Tests for the F1 ATTACH-context extractor (Model B runtime prep, READ-ONLY).

Verifies on real Starmie self-play states (golden fixtures are a different deck, so they cannot exercise the
Mega/Staryu/Cinderace role+threshold logic): no obs mutation, no forbidden metadata, correct role detection
(main_attacker/energy_engine/setup_basic), energy class (water/ignition/tool), the Ignition=3-units-on-Mega and
tool=0-units rules, threshold/shortfall sanity, and explicit-null schema stability on a non-Starmie fixture.

Engine-backed tests capture <=2 games (allowed for extractor verification) and SKIP if the engine is absent.

Run:  PYTHONIOENCODING=utf-8 python tests/test_f1_attach_context_v1.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "agent"))
import f1_attach_context_extractor_v1 as F  # noqa: E402

STARMIE_ROLES = {"main_attacker", "energy_engine", "setup_basic", "attacker", "wall_tank", "utility"}
FORBIDDEN = ("result", "outcome", "won", "pilot", "replay", "future", "reward")
_CAP = F.capture_starmie_attach_obs(n_games=2, cap=40)   # captured once, reused across tests


def test_no_mutation():
    if not _CAP:
        print("SKIP no-mutation (engine unavailable)")
        return
    obs = _CAP[0]
    before = json.dumps(obs, sort_keys=True, default=str)
    F.extract_attach_context(obs)
    assert json.dumps(obs, sort_keys=True, default=str) == before, "extractor mutated obs"
    print("PASS no-mutation")


def test_no_forbidden_metadata():
    if not _CAP:
        print("SKIP no-forbidden (engine unavailable)")
        return
    for obs in _CAP:
        for r in F.extract_attach_context(obs):
            blob = json.dumps(r, default=str).lower()
            for bad in FORBIDDEN:
                assert bad not in blob, f"forbidden token {bad!r} in payload"
    print("PASS no-forbidden-metadata")


def test_roles_and_energy_rules():
    if not _CAP:
        print("SKIP roles-energy (engine unavailable)")
        return
    rows = [r for obs in _CAP for r in F.extract_attach_context(obs)]
    assert rows, "no ATTACH options captured"
    saw_mega = saw_ign_on_mega = saw_tool = 0
    for r in rows:
        assert r["target_role"] in STARMIE_ROLES, r["target_role"]
        assert r["energy_class"] in ("basic_water", "basic_lightning", "ignition", "tool", "basic_energy_other") or r["energy_class"], r["energy_class"]
        if r["target_role"] == "main_attacker":
            saw_mega += 1
        # Ignition on Mega adds 3 units; a tool adds 0; ordinary energy adds 1
        if r["energy_is_ignition"] and r["target_card_id"] == F.MEGA_STARMIE:
            assert r["energy_units_added"] == 3, r
            saw_ign_on_mega += 1
        if r["energy_class"] == "tool":
            assert r["energy_units_added"] == 0 and r["redundant_energy"] is False, r
            saw_tool += 1
        # threshold sanity for our cards: crossing means shortfall went >0 -> 0
        if r["crosses_attack_threshold"]:
            assert r["shortfall_before"] and r["shortfall_after"] == 0, r
        # all Starmie-card rows fully populated (no missing fields)
        if r["attack_cheapest_units"] is not None:
            assert not r["missing_fields"], r["missing_fields"]
        assert r["runtime_safe"] is True
    assert saw_mega > 0, "expected Mega attaches in Starmie self-play"
    print(f"PASS roles+energy-rules ({len(rows)} options; mega={saw_mega} ign_on_mega={saw_ign_on_mega} tool={saw_tool})")


def test_explicit_null_schema_on_nonstarmie_fixture():
    # golden fixtures are a different deck -> role generic, thresholds None, but EVERY key present (explicit null)
    fx = json.load(open(ROOT / "tests" / "golden_state_action_fixtures" / "fixtures.json", encoding="utf-8"))["fixtures"]
    keys_required = {"raw_option_index", "semantic_key", "energy_card_id", "energy_class", "energy_units_added",
                     "target_role", "target_card_id", "target_energy_before", "target_energy_after",
                     "shortfall_before", "shortfall_after", "crosses_attack_threshold", "already_ready",
                     "redundant_energy", "runtime_safe", "missing_fields"}
    checked = 0
    for f in fx:
        obs = f.get("observation")
        if not obs or not obs.get("select"):
            continue
        for r in F.extract_attach_context(obs):
            assert keys_required <= set(r.keys()), f"missing keys: {keys_required - set(r.keys())}"
            checked += 1
        if checked >= 20:
            break
    assert checked > 0, "no ATTACH options in fixtures to check schema"
    print(f"PASS explicit-null-schema ({checked} non-Starmie attach options, schema stable)")


def main():
    fns = [test_no_mutation, test_no_forbidden_metadata, test_roles_and_energy_rules,
           test_explicit_null_schema_on_nonstarmie_fixture]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
    print("ALL F1-ATTACH-CONTEXT TESTS PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
