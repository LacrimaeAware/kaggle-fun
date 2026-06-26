"""Tests for the learned-proposer adapter + safety spec (bridge adapter v0). The adapter must be DISABLED by
default and unable to change the agent action. Run: python tests/test_proposer_adapter_v0.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
import learned_proposer_adapter as AD  # noqa: E402

M, C, S, IGN, W = 1031, 666, 1030, 17, 3


def _obs():
    return {"current": {"yourIndex": 0, "players": [
        {"active": [{"id": M, "hp": 330, "energies": [{"id": IGN}], "energyCards": [{"id": IGN}]}],
         "bench": [{"id": C, "hp": 160, "energies": [], "energyCards": []}],
         "hand": [{"id": W}], "handCount": 1, "deckCount": 30, "prize": [1] * 4},
        {"active": [{"id": 743, "hp": 120, "energies": []}], "bench": [], "prize": [1] * 4, "deckCount": 30}]},
        "select": {"maxCount": 1, "minCount": 1, "option": [
            {"type": 13, "attackId": 1488}, {"type": 13, "attackId": 1487},
            {"type": 8, "cardId": W, "area": 4, "index": 0}, {"type": 8, "cardId": W, "area": 5, "index": 0},
            {"type": 7, "cardId": 1229}, {"type": 14}]}}


def test_disabled_by_default():
    assert AD.PROPOSER_ENABLED is False
    h = AD.load_proposer(None)
    assert h.status == "MISSING"
    r = AD.rank_actions(h, _obs())
    assert r["ranked_actions"] == [] and r["status"] in ("MISSING", "DISABLED")
    print("PASS disabled-by-default: no ranked actions -> cannot change agent action")


def test_missing_artifact_cannot_change_action():
    for p in (None, "C:/does/not/exist.json"):
        h = AD.load_proposer(p)
        assert h.status == "MISSING"
        assert AD.rank_actions(h, _obs())["ranked_actions"] == []
    print("PASS missing-artifact: returns MISSING, empty ranking")


def test_semantic_key_mapping():
    obs = _obs()
    keys = AD.option_index_to_key(obs)
    assert keys[0] == "ATTACK:Nebula" and keys[1] == "ATTACK:Jetting"
    assert keys[2] == "ATTACH:Water:Mega" and keys[3] == "ATTACH:Water:Cinderace"
    assert keys[4] == "PLAY:Wally's Compassion" and keys[5] == "END"
    # determinism
    assert AD.option_index_to_key(obs) == keys
    print("PASS semantic-keys: canonical + deterministic", )


def test_raw_index_roundtrip():
    obs = _obs()
    k2i = AD.key_to_indices(obs)
    i2k = AD.option_index_to_key(obs)
    for k, idxs in k2i.items():
        for i in idxs:
            assert i2k[i] == k
    # every option index appears exactly once across the reverse map
    flat = [i for idxs in k2i.values() for i in idxs]
    assert sorted(flat) == list(range(len(i2k)))
    print("PASS roundtrip: key<->index mapping consistent")


def test_safety_filter_examples():
    obs = _obs()  # Nebula (idx0) is a KO; Jetting (idx1) KO too
    # proposing END (idx5) loses the KO -> v2 soft flag
    assert AD.safety_check(obs, 5)["filters"]["v2_loses_guaranteed_ko"]["veto"] is True
    # proposing Wally (idx4) -> v4
    assert AD.safety_check(obs, 4)["filters"]["v4_wally_strips_ko_energy"]["veto"] is True
    # proposing the KO attack (idx0) -> no hard veto
    assert AD.safety_check(obs, 0)["hard_veto"] is False
    # illegal index -> hard veto
    assert AD.safety_check(obs, 99)["hard_veto"] is True
    print("PASS safety: KO-loss/Wally flagged, KO ok, illegal hard-vetoed")


def test_adapter_does_not_mutate_obs():
    obs = _obs()
    before = copy.deepcopy(obs)
    AD.option_index_to_key(obs)
    AD.rank_actions(AD.load_proposer(None), obs)
    AD.safety_check(obs, 0)
    assert obs == before, "adapter mutated the observation"
    print("PASS no-mutation: adapter calls are read-only")


def test_no_forbidden_metadata_consumed():
    """The adapter takes only the runtime observation + an option index; adding eval-only fields to the obs must
    not change its output (it never reads pilot/outcome/future)."""
    obs = _obs()
    k1 = AD.option_index_to_key(obs)
    obs2 = _obs()
    obs2["eval_meta_leak"] = {"pilot_name": "Yushin Ito", "outcome_won": True, "future": ["ATTACK"]}
    obs2["current"]["players"][0]["__pilot"] = "Yushin Ito"
    assert AD.option_index_to_key(obs2) == k1
    assert AD.safety_check(obs2, 0) == AD.safety_check(obs, 0)
    print("PASS no-forbidden-metadata: output independent of any eval fields")


def main() -> int:
    rc = 0
    for t in (test_disabled_by_default, test_missing_artifact_cannot_change_action, test_semantic_key_mapping,
              test_raw_index_roundtrip, test_safety_filter_examples, test_adapter_does_not_mutate_obs,
              test_no_forbidden_metadata_consumed):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}"); rc = 1
    print(f"\n{'ALL PROPOSER-ADAPTER TESTS PASS' if rc == 0 else 'SOME FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
