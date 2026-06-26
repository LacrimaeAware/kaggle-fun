"""Frozen-state tests for the Model B audit fixes (no engine games; instant correctness checks).
Run: python tests/test_starmie_audit_fixes_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
import starmie_heuristics as SH  # noqa: E402


def test_F_ignition_energy_units():
    """Audit F: one Ignition on Mega Starmie = 3 energy units (not 1 card), so Nebula (210) is affordable."""
    mega_1ign = {"id": SH.MEGA_STARMIE, "hp": 330, "energyCards": [{"id": SH.IGNITION}]}
    assert SH._energy_units(mega_1ign) == 3, SH._energy_units(mega_1ign)
    mega_1water = {"id": SH.MEGA_STARMIE, "hp": 330, "energyCards": [{"id": SH.BASIC_WATER}]}
    assert SH._energy_units(mega_1water) == 1
    player = {"active": [mega_1ign], "bench": []}
    assert SH._our_max_hit(player) == 210.0, "1 Ignition should fund Nebula (210), not be read as 120"
    print("PASS F: ignition energy-units")


def test_G_sel_card_respects_zone():
    """Audit G: a CARD option with area=DECK(1) must resolve from sel.deck, not from another zone at the same
    index. With deck AND discard both populated, the old fixed-order code could return the wrong card."""
    sel = {"deck": [{"id": SH.MEGA_STARMIE}, {"id": SH.STARYU}],
           "discard": [{"id": SH.CRUSHING_HAMMER}, {"id": SH.BOSS}]}
    o_deck = {"type": 3, "area": 1, "index": 0}
    assert SH._sel_card(sel, o_deck) == SH.MEGA_STARMIE
    # unknown area + ambiguous (both zones populated) -> defer (None), never guess wrong
    o_unknown = {"type": 3, "index": 0}
    assert SH._sel_card(sel, o_unknown) is None
    # unknown area + only one zone populated -> resolve it
    sel1 = {"discard": [{"id": SH.STARYU}]}
    assert SH._sel_card(sel1, {"type": 3, "index": 0}) == SH.STARYU
    print("PASS G: _sel_card respects zone")


def test_B_boss_not_played_when_active_ko_available():
    """Audit B: Boss must NOT be played before a KO when the active is already KO-able (gusting re-targets the
    KO to a lower-value bench mon). _high_value_play returns None so the KO floor takes the active KO."""
    obs = {"current": {"yourIndex": 0, "players": [
        {"active": [{"id": SH.MEGA_STARMIE, "hp": 330, "energyCards": [{"id": SH.BASIC_WATER}]}],
         "bench": [], "hand": [{"id": SH.BOSS}], "prize": [1, 2, 3, 4]},
        {"active": [{"id": SH.STARYU, "hp": 100}], "bench": [{"id": SH.STARYU, "hp": 60}], "prize": [1, 2, 3, 4]},
    ]}, "select": {"maxCount": 1, "option": [
        {"type": 13, "attackId": SH.JETTING_BLOW},   # Jetting Blow 120 KOs the 100-hp active
        {"type": 7, "area": 2, "index": 0},          # PLAY Boss (hand[0])
    ]}}
    player, opp, _ = SH._me_opp(obs)
    opts = obs["select"]["option"]
    assert SH._best_ko_index(obs, opts, opp) == 0, "Jetting should KO the 100-hp active"
    assert SH._high_value_play(obs, opts, player, opp) is None, "Boss must be skipped when an active KO exists"
    print("PASS B: boss skipped when active KO available")


def test_J_no_suicide_only_self_removing_ability():
    """Audit J: with one Pokemon in play and a SAFE ability (no self-shuffle/discard effect), _no_suicide must
    NOT force END. The Starmie deck has no Run-Away-Draw-style ability, so the rule should stay silent."""
    obs = {"current": {"yourIndex": 0, "players": [
        {"active": [{"id": SH.CINDERACE, "hp": 160}], "bench": [], "hand": []},
        {"active": [{"id": SH.STARYU, "hp": 70}], "bench": []},
    ]}, "select": {"maxCount": 1, "option": [
        {"type": 10, "area": 4, "index": 0},   # an ABILITY (Cinderace Explosiveness = safe, not self-removing)
        {"type": 14},                          # END
    ]}}
    assert SH._no_suicide(obs) is None, "must not force END for a safe ability"
    print("PASS J: no_suicide only suppresses self-removing abilities")


if __name__ == "__main__":
    test_F_ignition_energy_units()
    test_G_sel_card_respects_zone()
    test_B_boss_not_played_when_active_ko_available()
    test_J_no_suicide_only_self_removing_ability()
    print("\nALL AUDIT-FIX TESTS PASS")
