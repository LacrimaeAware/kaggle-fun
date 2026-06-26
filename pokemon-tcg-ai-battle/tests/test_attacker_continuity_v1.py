"""Fixed-state tests for ATTACKER_CONTINUITY_V1 (tactical-leaf task, Section 5). No engine games -- instant
correctness checks on frozen states. Run: python tests/test_attacker_continuity_v1.py

The term is DISABLED by default; these tests exercise it via the explicit `continuity=True/False` parameter so
they never depend on the env toggle.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
import eval as EV  # noqa: E402

M, C, IGN, W = 1031, 666, 17, 3
DK = dict(deckout_weight=EV.DECKOUT_TEST)


def _ent(cid, hp=None, energy=()):
    return {"id": cid, "hp": (hp if hp is not None else (330 if cid == M else 160)),
            "energies": [{"id": e} for e in energy]}


def _state(active=None, bench=(), opp_active=None, my_prizes=4, opp_prizes=4, result=-1, opp_hand=None):
    me = {"active": ([active] if active else []), "bench": list(bench),
          "prize": [1] * my_prizes, "deckCount": 30, "handCount": 5}
    opp = {"active": ([opp_active] if opp_active else [_ent(743, 120)]), "bench": [],
           "prize": [1] * opp_prizes, "deckCount": 30, "handCount": (len(opp_hand) if opp_hand else 5)}
    if opp_hand is not None:
        opp["hand"] = list(opp_hand)
    return {"players": [me, opp], "result": result}


def _leaf(s, continuity):
    return EV.evaluate_deck_v3(s, 0, deckout_weight=EV.DECKOUT_TEST, continuity=continuity)


def _score(s):
    return EV.attacker_continuity_score(s["players"][0], s["players"][1])


def _vec(s):
    return EV.attacker_continuity_vector(s["players"][0], s["players"][1])


def test_1_ready_mega_over_cinderace_engine():
    """Prefer the state that retains a ready Mega attacker over the Cinderace engine (same energy)."""
    cind = _state(active=_ent(C, energy=[W, W, W]), bench=[_ent(M, energy=[])])
    mega = _state(active=_ent(M, energy=[W, W, W]), bench=[_ent(C, energy=[])])
    assert _leaf(cind, False) == _leaf(mega, False), "baseline must be deck-blind (identical)"
    assert _leaf(mega, True) > _leaf(cind, True), (_leaf(mega, True), _leaf(cind, True))
    print("PASS 1: ready Mega preferred over Cinderace engine")


def test_2_backup_continuity():
    """With the active Mega already ready, powering a viable backup Mega scores higher than a bare backup."""
    backup_ready = _state(active=_ent(M, energy=[W]), bench=[_ent(M, energy=[W])])
    backup_bare = _state(active=_ent(M, energy=[W]), bench=[_ent(M, energy=[])])
    assert _vec(backup_ready)["viable_backups"] == 1 and _vec(backup_bare)["viable_backups"] == 0
    assert _score(backup_ready) > _score(backup_bare)
    print("PASS 2: viable backup continuity rewarded")


def test_3_one_short_threshold():
    """An action that makes a Mega attack-ready (0 -> 1 unit) increases the continuity score."""
    zero = _state(active=_ent(M, energy=[]))
    one = _state(active=_ent(M, energy=[W]))
    assert _vec(zero)["one_short_main"] == 1 and _vec(zero)["ready_main_active"] == 0
    assert _vec(one)["ready_main_active"] == 1
    assert _score(one) > _score(zero), (_score(one), _score(zero))
    print("PASS 3: crossing the readiness threshold is rewarded")


def test_4_redundant_overattachment():
    """A 2nd unit on a Jetting-ready Mega crosses no new threshold (still only Jetting) -> NOT rewarded; the term
    penalizes the redundant unit. (Nebula needs 3.)"""
    one = _state(active=_ent(M, energy=[W]))
    two = _state(active=_ent(M, energy=[W, W]))
    assert _vec(two)["redundant_energy"] == 1
    assert _score(two) <= _score(one), (_score(two), _score(one))
    print("PASS 4: redundant over-attachment not rewarded")


def test_5_ignition_three_units():
    """One Ignition on a Mega = 3 functional units (Nebula-ready), NOT one card; treated as ready, not one-short,
    not redundant. (The baseline leaf's card-count would undervalue it.)"""
    mega_ign = _state(active=_ent(M, energy=[IGN]))
    v = _vec(mega_ign)
    assert EV._cont_units(_ent(M, energy=[IGN])) == 3, EV._cont_units(_ent(M, energy=[IGN]))
    assert v["ready_main_active"] == 1 and v["one_short_main"] == 0 and v["redundant_energy"] == 0, v
    # Ignition on a non-Mega (Cinderace) is 1 unit, not 3
    assert EV._cont_units(_ent(C, energy=[IGN])) == 1
    print("PASS 5: Ignition = 3 units on a Mega (unit-aware)")


def test_6_terminal_and_ko_preservation():
    """The term cannot override a terminal win, nor a prize/KO advantage (W_PRIZE >> continuity)."""
    # (a) terminal win returns +WIN regardless of continuity
    won = _state(active=_ent(C, energy=[W]), result=0)  # result==me, awful continuity (engine active)
    assert _leaf(won, True) == EV.WIN
    # (b) a state that TOOK a prize (MY prizes 3 left -> I am ahead) but has bad continuity still beats a level
    # -prize state with great continuity (W_PRIZE=1000 >> continuity ~tens). Prizes are REMAINING; KO takes from
    # MY pile, so taking a prize DECREASES my prizes-left.
    took_prize = _state(active=_ent(C, energy=[W]), bench=[], my_prizes=3, opp_prizes=4)   # +1 prize, no main online
    great_cont = _state(active=_ent(M, energy=[W]), bench=[_ent(M, energy=[W])], my_prizes=4, opp_prizes=4)  # ready main + backup
    assert _leaf(took_prize, True) > _leaf(great_cont, True), (_leaf(took_prize, True), _leaf(great_cont, True))
    print("PASS 6: terminal/prize/KO terms remain dominant")


def test_7_cinderace_engine_value_retained():
    """Do NOT blindly penalize all Cinderace development: Turbo-level energy (1 unit) on the engine is NOT
    penalized (engine_overinvest counts only units BEYOND Turbo's 1)."""
    cind_1 = _state(active=_ent(C, energy=[W]))
    cind_2 = _state(active=_ent(C, energy=[W, W]))
    assert _vec(cind_1)["engine_overinvest"] == 0, _vec(cind_1)
    assert _vec(cind_2)["engine_overinvest"] == 1, _vec(cind_2)
    print("PASS 7: first (Turbo) energy on Cinderace not penalized")


def test_8_three_prize_exposure():
    """Record exposure for a DAMAGED powered Mega (<=half HP, 3 prizes + invested energy at risk); a full-HP
    powered Mega is NOT treated as exposed."""
    full = _state(active=_ent(M, hp=330, energy=[W, W, W]))
    damaged = _state(active=_ent(M, hp=150, energy=[W, W, W]))
    assert _vec(full)["exposed_concentration"] == 0, _vec(full)
    assert _vec(damaged)["exposed_concentration"] == 1, _vec(damaged)
    print("PASS 8: three-prize exposure recorded only when damaged")


def test_9_no_hidden_information():
    """The term reads only public board slots; opponent hidden hand/prize IDENTITIES never change the score."""
    base = _state(active=_ent(M, energy=[W]), opp_hand=[{"id": 9999}, {"id": 8888}])
    other = _state(active=_ent(M, energy=[W]), opp_hand=[{"id": 1}, {"id": 2}])  # same handCount, different ids
    assert _score(base) == _score(other)
    # mutating opponent prize identities (still 4 prizes) does not change the score
    base["players"][1]["prize"] = [{"id": 1234}] * 4
    assert _score(base) == _score(other)
    print("PASS 9: no hidden-information dependence")


def test_10_toggle_off_reproduces_baseline():
    """continuity=False must omit EXACTLY the term: leaf(False) == leaf(True) - continuity_score, on states with
    a nonzero term, and identical when the term is zero."""
    states = [
        _state(active=_ent(M, energy=[W, W, W]), bench=[_ent(C, energy=[])]),
        _state(active=_ent(C, energy=[W, W]), bench=[_ent(M, energy=[])]),
        _state(active=_ent(M, energy=[IGN]), bench=[_ent(M, energy=[W])]),
    ]
    for s in states:
        off = _leaf(s, False)
        on = _leaf(s, True)
        assert abs((on - off) - _score(s)) < 1e-9, (on, off, _score(s))
    # default-off env behaviour: ATTACKER_CONTINUITY_ON is False unless STARMIE_LEAF_ATTACKER_CONTINUITY=1
    assert EV.ATTACKER_CONTINUITY_ON == (os.environ.get("STARMIE_LEAF_ATTACKER_CONTINUITY", "") == "1")
    print("PASS 10: toggle-off reproduces baseline exactly")


def main() -> int:
    tests = [test_1_ready_mega_over_cinderace_engine, test_2_backup_continuity, test_3_one_short_threshold,
             test_4_redundant_overattachment, test_5_ignition_three_units, test_6_terminal_and_ko_preservation,
             test_7_cinderace_engine_value_retained, test_8_three_prize_exposure, test_9_no_hidden_information,
             test_10_toggle_off_reproduces_baseline]
    rc = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            rc = 1
    print(f"\n{'ALL CONTINUITY TESTS PASS' if rc == 0 else 'SOME CONTINUITY TESTS FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
