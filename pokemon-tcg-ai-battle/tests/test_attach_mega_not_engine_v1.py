"""Fixed-state tests for ATTACH_MEGA_NOT_ENGINE_V1 (default-off attach-targeting probe).
Run: python tests/test_attach_mega_not_engine_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
import starmie_heuristics as SH  # noqa: E402

M, C, S, IGN, W = SH.MEGA_STARMIE, SH.CINDERACE, SH.STARYU, SH.IGNITION, SH.BASIC_WATER
ATTACH = 8


def _ent(cid, energy=()):
    return {"id": cid, "hp": (330 if cid == M else 160 if cid == C else 70),
            "energyCards": [{"id": e} for e in energy], "energies": [{"id": e} for e in energy]}


def _obs(active, bench, options, opp_hp=120):
    me = {"active": [active] if active else [], "bench": list(bench),
          "hand": [{"id": W}, {"id": W}], "prize": [1] * 4, "deckCount": 30, "handCount": 2}
    opp = {"active": [{"id": 743, "hp": opp_hp}], "bench": [], "prize": [1] * 4, "deckCount": 30}
    return {"current": {"yourIndex": 0, "players": [me, opp]},
            "select": {"maxCount": 1, "minCount": 1, "option": options}}


def _att(energy, area, index=0):
    return {"type": ATTACH, "cardId": energy, "area": area, "index": index}


def _with(flag, fn):
    old = SH.ATTACH_MEGA
    SH.ATTACH_MEGA = flag
    try:
        return fn()
    finally:
        SH.ATTACH_MEGA = old


def test_1_redirect_cinderace_to_mega():
    """A Basic-Water attach aimed at Cinderace (0e) is redirected to the active Mega (1e, building toward Nebula)."""
    obs = _obs(_ent(M, [W]), [_ent(C)], [_att(W, SH.DP.A_ACTIVE), _att(W, SH.DP.A_BENCH, 0)])
    # baseline _attach_score: Cinderace-0 = 70 > Mega-building = 62 (the bug)
    assert SH._attach_score(obs, obs["select"]["option"][1], obs["current"]["players"][0]) == 70.0
    assert SH._attach_score(obs, obs["select"]["option"][0], obs["current"]["players"][0]) == 62.0
    off = _with(False, lambda: SH._attach_mega_pref(obs, [1]))   # toggle off -> Cinderace stands
    on = _with(True, lambda: SH._attach_mega_pref(obs, [1]))     # toggle on -> redirect to Mega (index 0)
    assert off == [1], off
    assert on == [0], on
    print("PASS 1: Cinderace attach redirected to the Mega line")


def test_2_mega_attach_left_alone():
    """A pick already targeting the Mega is unchanged."""
    obs = _obs(_ent(M, [W]), [_ent(C)], [_att(W, SH.DP.A_ACTIVE), _att(W, SH.DP.A_BENCH, 0)])
    assert _with(True, lambda: SH._attach_mega_pref(obs, [0])) == [0]
    print("PASS 2: an existing Mega attach is left alone")


def test_3_ignition_not_redirected():
    """Ignition keeps its Nebula-only gate -- never redirected by this rule."""
    obs = _obs(_ent(M, [W]), [_ent(C)], [{"type": ATTACH, "cardId": IGN, "area": SH.DP.A_BENCH, "index": 0},
                                         _att(W, SH.DP.A_ACTIVE)])
    assert _with(True, lambda: SH._attach_mega_pref(obs, [0])) == [0]   # Ignition->Cinderace pick stays
    print("PASS 3: Ignition attach not redirected")


def test_4_no_useful_line_attach_keeps_cinderace():
    """If the Mega is already capped (>=3 units) and there is no Staryu, no useful line attach exists -> the
    Cinderace attach stands (engine not starved)."""
    obs = _obs(_ent(M, [W, W, W]), [_ent(C)], [_att(W, SH.DP.A_ACTIVE), _att(W, SH.DP.A_BENCH, 0)])
    # Mega at 3 units -> _attach_score 0 (redundant); only the Cinderace attach is useful
    assert SH._attach_score(obs, obs["select"]["option"][0], obs["current"]["players"][0]) == 0.0
    assert _with(True, lambda: SH._attach_mega_pref(obs, [1])) == [1]
    print("PASS 4: no useful line attach -> Cinderace attach stands")


def test_5_redirect_to_staryu_when_no_mega_use():
    """With a benched Staryu and the Mega capped, redirect the Cinderace attach to the Staryu line (builds the
    attacker), matching the #1 pilot's line-first attaching."""
    obs = _obs(_ent(M, [W, W, W]), [_ent(C), _ent(S)],
               [_att(W, SH.DP.A_ACTIVE), _att(W, SH.DP.A_BENCH, 0), _att(W, SH.DP.A_BENCH, 1)])
    assert _with(True, lambda: SH._attach_mega_pref(obs, [1])) == [2]   # -> Staryu (bench index 1 = option 2)
    print("PASS 5: redirect to Staryu line when Mega is capped")


def test_6_toggle_off_is_baseline():
    """Default off: no redirect ever happens."""
    obs = _obs(_ent(M, [W]), [_ent(C)], [_att(W, SH.DP.A_ACTIVE), _att(W, SH.DP.A_BENCH, 0)])
    assert SH.ATTACH_MEGA == (os.environ.get("STARMIE_ATTACH_MEGA_NOT_ENGINE_V1", "") == "1")
    assert _with(False, lambda: SH._attach_mega_pref(obs, [1])) == [1]
    print("PASS 6: toggle off reproduces baseline")


def main() -> int:
    rc = 0
    for t in (test_1_redirect_cinderace_to_mega, test_2_mega_attach_left_alone, test_3_ignition_not_redirected,
              test_4_no_useful_line_attach_keeps_cinderace, test_5_redirect_to_staryu_when_no_mega_use,
              test_6_toggle_off_is_baseline):
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}"); rc = 1
    print(f"\n{'ALL ATTACH-MEGA TESTS PASS' if rc == 0 else 'SOME FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
