"""STARMIE TACTICAL-STATE EXTRACTOR (V1) -- Section 1 of the tactical-leaf task.

Read-only, PUBLIC-INFORMATION-ONLY diagnostics for the Cinderace/Mega-Starmie deck. Turns a public observation
(+ legal options) into structured entity features, board features, and tactical coordinates (RACE/SWEEP/WALL/
VALUE/COMMITMENT). It is a DIAGNOSTIC: coordinates are computed quantities, not asserted ground-truth classes.

PUBLIC INPUTS ONLY. Uses: my own board+hand (visible to me at runtime), opponent's VISIBLE board, and COUNTS
(handCount, deckCount, prize-pile length). NEVER uses: opponent hidden hand/prize identities, future actions,
the final outcome, pilot identity, replay id, or leaderboard score. (Those are added as evaluation-only metadata
by the EXPORT step, never inside the runtime feature payload.)

Affordability/damage policy (task 1A): the engine's offered ATTACK menu is ground truth for MY active when it is
an attack decision; for everything else we use the VERIFIED Starmie attack mechanics (below) for our own cards
and a card_stats estimate for opponents, with an explicit `uncertainty` flag when damage cannot be resolved
safely. attack_stats.json is treated as advisory, not complete ground truth (conditional/flat attacks).

Verified Starmie mechanics used here:
- Mega Starmie ex (1031): Jetting Blow 120 at >=1 energy unit (+50 bench snipe); Nebula Beam 210 at >=3 units,
  FLAT (ignores weakness/resistance).
- Cinderace (666): Turbo Flare 50 at >=1 unit (the energy ENGINE, not the primary attacker).
- Staryu (1030): Water Gun 20 at >=1 unit (the basic that evolves into Mega Starmie).
- Ignition Energy (17): provides 3 energy UNITS on an Evolution (Mega Starmie), 1 otherwise; discarded EOT.
"""
from __future__ import annotations

import os
from typing import Any

import deck_policy_v3 as DP

# ---- card ids / roles (verified from card_stats.json + starmie_heuristics constants) ----
MEGA_STARMIE, CINDERACE, STARYU, IGNITION, BASIC_WATER = 1031, 666, 1030, 17, 3
JETTING_BLOW, NEBULA_BEAM, TURBO_FLARE = 1487, 1488, 965
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END, CARD = 7, 8, 9, 10, 11, 12, 13, 14, 3

# semantic roles
R_MAIN_ATTACKER = "main_attacker"   # Mega Starmie ex -- our prize-taking attacker
R_ENERGY_ENGINE = "energy_engine"   # Cinderace -- accelerates energy, NOT the primary attacker
R_SETUP_BASIC = "setup_basic"       # Staryu -- evolves into the Mega
R_WALL_TANK = "wall_tank"           # high-HP body used to stall
R_ATTACKER = "attacker"             # generic opposing attacker (ex/mega or real damage)
R_UTILITY = "utility"               # everything else in play

CDB = DP.CDB           # card_stats.json (id -> {n,hp,ex,mega,wk,rs,ty,atk:[{cost,cE,dmg,name}]})
CF = getattr(DP, "CF", {})   # card_features.json (id -> {stage,retreat,best_dmg,...}) if loaded

# Verified attacks for OUR deck: (attackId, name, energy units required, damage, flat?, bench_snipe)
_VERIFIED = {
    MEGA_STARMIE: [
        {"attackId": JETTING_BLOW, "name": "Jetting Blow", "units": 1, "dmg": 120.0, "flat": False, "snipe": 50.0},
        {"attackId": NEBULA_BEAM, "name": "Nebula Beam", "units": 3, "dmg": 210.0, "flat": True, "snipe": 0.0},
    ],
    CINDERACE: [{"attackId": TURBO_FLARE, "name": "Turbo Flare", "units": 1, "dmg": 50.0, "flat": False, "snipe": 0.0}],
    STARYU: [{"attackId": None, "name": "Water Gun", "units": 1, "dmg": 20.0, "flat": False, "snipe": 0.0}],
}


def _meta(cid) -> dict:
    out = {}
    try:
        out.update(CF.get(str(cid), {}) or {})
    except Exception:
        pass
    out.update(CDB.get(str(cid), {}) or {})
    return out


def _is_evolution(cid) -> bool:
    """True if the card is an evolution (Stage1+), so Ignition Energy provides 3 units. Mega Starmie qualifies."""
    if cid == MEGA_STARMIE:
        return True
    stage = str(_meta(cid).get("stage", "") or "").lower()
    return stage not in ("", "basic")


def energy_units(entity) -> int:
    """Energy UNITS provided by attached energy. EXACTLY mirrors starmie_heuristics._energy_units so affordability
    matches the live agent (the leaf interacts with it): Ignition Energy gives 3 units on Mega Starmie (the only
    evolution we ever energize), 1 otherwise; every other energy gives 1. (Per the card Ignition is 3 on ANY
    evolution, but the agent only ever puts it on the Mega, so we keep the agent's exact rule.)"""
    cards = DP._get(entity, "energyCards", None)
    if cards is None:
        cards = DP._get(entity, "energies", None)
    cards = DP._items(cards)
    if not cards:
        return 0
    is_mega = DP._cid(entity) == MEGA_STARMIE
    units = 0
    for c in cards:
        units += 3 if (DP._cid(c) == IGNITION and is_mega) else 1
    return units


def semantic_role(cid, owner_is_me: bool) -> str:
    if cid == MEGA_STARMIE:
        return R_MAIN_ATTACKER
    if cid == CINDERACE:
        return R_ENERGY_ENGINE
    if cid == STARYU:
        return R_SETUP_BASIC
    m = _meta(cid)
    hp = float(m.get("hp", 0) or 0)
    best = float(m.get("best_dmg", 0) or 0)
    if not best:
        best = max([float(a.get("dmg", 0) or 0) for a in (m.get("atk") or [])] or [0.0])
    if m.get("mega") or m.get("ex") or best >= 120:
        return R_ATTACKER
    if hp >= 150:
        return R_WALL_TANK
    if best > 0:
        return R_ATTACKER
    return R_UTILITY


def _weakness_mult(attacker_cid, defender) -> float:
    """x2 if defender's weakness matches attacker type; ignored for flat attacks by the caller."""
    if not defender:
        return 1.0
    atype = str(_meta(attacker_cid).get("ty", "") or "")
    dwk = str(_meta(DP._cid(defender)).get("wk", "") or "")
    return 2.0 if (atype and dwk and atype == dwk) else 1.0


def affordable_attacks(entity, defender, menu_attack_ids=None):
    """Return (list of affordable attack dicts, uncertainty:bool). Each dict: name, units, base_dmg, eff_dmg
    (vs `defender`, weakness applied unless flat), flat, snipe. `menu_attack_ids` (set) = the engine-offered
    attackIds for MY active on an attack decision -> ground-truth affordability. Otherwise unit-based estimate."""
    cid = DP._cid(entity)
    units = energy_units(entity)
    out = []
    uncertainty = False
    if cid in _VERIFIED:
        for a in _VERIFIED[cid]:
            offered = (menu_attack_ids is None) or (a["attackId"] in menu_attack_ids) or (a["attackId"] is None and not menu_attack_ids)
            affordable = (units >= a["units"]) if menu_attack_ids is None else (a["attackId"] in (menu_attack_ids or set()))
            if not affordable:
                continue
            base = a["dmg"]
            eff = base if a["flat"] else base * _weakness_mult(cid, defender)
            out.append({"name": a["name"], "units": a["units"], "base_dmg": base, "eff_dmg": eff,
                        "flat": a["flat"], "snipe": a["snipe"]})
        if cid == STARYU and menu_attack_ids is not None:
            # Staryu's attackId is unknown to us; if a menu is present we cannot match it -> mild uncertainty.
            uncertainty = uncertainty or False
    else:
        # opponent / unknown card: estimate from card_stats.atk (cost field is unreliable -> uncertainty).
        uncertainty = True
        for a in (_meta(cid).get("atk") or []):
            dmg = float(a.get("dmg", 0) or 0)
            cost = int(a.get("cost", 0) or 0)
            if dmg <= 0:
                continue  # ability / conditional -> cannot resolve safely
            if units >= cost:
                eff = dmg * _weakness_mult(cid, defender)
                out.append({"name": a.get("name", "?"), "units": cost, "base_dmg": dmg, "eff_dmg": eff,
                            "flat": False, "snipe": 0.0})
    return out, uncertainty


def _maxhp(cid) -> float:
    return float(_meta(cid).get("hp", 0) or 0)


def _remaining_hp(entity) -> float:
    hp = DP._get(entity, "hp", None)
    if hp is not None:
        return float(hp or 0)
    cid = DP._cid(entity)
    dmg = float(DP._get(entity, "damage", 0) or 0)
    return max(0.0, _maxhp(cid) - dmg)


def _retreat_cost(cid) -> int:
    r = _meta(cid).get("retreat", None)
    try:
        return int(r) if r is not None else 0
    except Exception:
        return 0


def entity_features(entity, owner_is_me: bool, slot: str, opposing_active, my_prizes_left, opp_prizes_left,
                    menu_attack_ids=None) -> dict:
    """Public per-entity tactical features (task 1A)."""
    cid = DP._cid(entity)
    role = semantic_role(cid, owner_is_me)
    units = energy_units(entity)
    rem_hp = _remaining_hp(entity)
    maxhp = _maxhp(cid)
    defender = opposing_active
    atks, uncertain = affordable_attacks(entity, defender, menu_attack_ids if owner_is_me else None)
    max_dmg = max([a["eff_dmg"] for a in atks] or [0.0])
    # KO-now on the opposing active
    def_hp = _remaining_hp(defender) if defender else 0.0
    can_ko_active = bool(defender and def_hp > 0 and max_dmg >= def_hp)
    # one ordinary attachment away from being attack-ready (its cheapest attack)
    cheapest = min([a["units"] for a in _VERIFIED.get(cid, [])] or [99])
    ready = (cid in _VERIFIED) and (units >= cheapest)
    one_short = (cid in _VERIFIED) and (not ready) and (units + 1 >= cheapest)
    # prize liability + retreat
    rc = _retreat_cost(cid)
    return {
        "card_id": cid, "name": _meta(cid).get("n", f"#{cid}"), "role": role, "owner": "me" if owner_is_me else "opp",
        "slot": slot,  # "active" | "bench"
        "hp_remaining": rem_hp, "hp_max": maxhp, "damage": max(0.0, maxhp - rem_hp),
        "evolution_stage": str(_meta(cid).get("stage", "basic") or "basic"),
        "prize_liability": DP._prize_value(cid),
        "attached_cards": DP._attached_count(entity), "attached_units": units,
        "retreat_cost": rc, "retreat_affordable": units >= rc,
        "affordable_attacks": [a["name"] for a in atks],
        "max_affordable_damage": max_dmg,
        "min_unit_shortfall_to_attack": (0 if ready else (cheapest - units if cid in _VERIFIED else None)),
        "can_ko_opposing_active": can_ko_active,
        "attack_ready": bool(ready),
        "one_attachment_from_ready": bool(one_short),
        "is_main_attacker": role == R_MAIN_ATTACKER,
        "is_energy_engine": role == R_ENERGY_ENGINE,
        "status": DP._get(entity, "status", None) or DP._get(entity, "specialConditions", None),
        "damage_uncertainty": bool(uncertain),
    }


def _menu_attack_ids(opts) -> set:
    return {o.get("attackId") for o in opts if o.get("type") == ATTACK and o.get("attackId") is not None}


def _all_entities(player):
    a = DP._active(player)
    ents = [("active", a)] if a else []
    ents += [("bench", b) for b in DP._bench(player) if b]
    return ents


def board_features(me, opp, obs, opts) -> dict:
    """Public board-level aggregates (task 1B)."""
    menu_atk = _menu_attack_ids(opts)
    opp_active = DP._active(opp)
    my_active = DP._active(me)
    my_prizes = len(DP._items(DP._get(me, "prize", [])))
    opp_prizes = len(DP._items(DP._get(opp, "prize", [])))

    my_ents = [entity_features(e, True, slot, opp_active, my_prizes, opp_prizes, menu_atk) for slot, e in _all_entities(me)]
    opp_ents = [entity_features(e, False, slot, my_active, opp_prizes, my_prizes, None) for slot, e in _all_entities(opp)]

    def count(ents, pred):
        return sum(1 for f in ents if pred(f))

    my_ready_main = count(my_ents, lambda f: f["is_main_attacker"] and f["attack_ready"])
    opp_ready_resp = count(opp_ents, lambda f: f["attack_ready"] or f["role"] in (R_ATTACKER,) and f["attached_units"] >= 1)
    my_total_units = sum(f["attached_units"] for f in my_ents)
    engine_overinv = sum(max(0, f["attached_units"] - 1) for f in my_ents if f["is_energy_engine"])  # units on engine beyond Turbo's 1
    max_conc = max([f["attached_units"] for f in my_ents] or [0])
    exposed3 = count(my_ents, lambda f: f["prize_liability"] >= 3 and f["attached_units"] >= 1)

    return {
        "prize_diff": opp_prizes - my_prizes,  # +ve = I am ahead (opp has more prizes left to take)
        "my_prizes_left": my_prizes, "opp_prizes_left": opp_prizes,
        "my_pokemon": len(my_ents), "opp_pokemon": len(opp_ents),
        "my_board_hp": sum(f["hp_remaining"] for f in my_ents), "opp_board_hp": sum(f["hp_remaining"] for f in opp_ents),
        "my_units": my_total_units, "opp_units": sum(f["attached_units"] for f in opp_ents),
        "my_ready_attackers": count(my_ents, lambda f: f["attack_ready"]),
        "opp_ready_attackers": count(opp_ents, lambda f: f["attack_ready"] or (f["role"] == R_ATTACKER and f["attached_units"] >= 1)),
        "my_one_short": count(my_ents, lambda f: f["one_attachment_from_ready"]),
        "my_ready_main_attackers": my_ready_main,
        "my_main_one_short": count(my_ents, lambda f: f["is_main_attacker"] and f["one_attachment_from_ready"]),
        "my_backup_ready": max(0, my_ready_main - (1 if (my_active and DP._cid(my_active) == MEGA_STARMIE and energy_units(my_active) >= 1) else 0)),
        "my_engine_count": count(my_ents, lambda f: f["is_energy_engine"]),
        "my_immediate_ko": count(my_ents, lambda f: f["slot"] == "active" and f["can_ko_opposing_active"]),
        "opp_immediate_ko": count(opp_ents, lambda f: f["slot"] == "active" and f["can_ko_opposing_active"]),
        "my_hand_count": DP._hand_count(me), "opp_hand_count": DP._hand_count(opp),
        "my_deck_count": int(DP._get(me, "deckCount", 0) or 0), "opp_deck_count": int(DP._get(opp, "deckCount", 0) or 0),
        "my_deckout_risk": int(DP._get(me, "deckCount", 99) or 99) <= 5,
        "my_bench_capacity": max(0, 5 - len(DP._bench(me))),
        "max_energy_concentration": (max_conc / my_total_units) if my_total_units else 0.0,
        "engine_overinvestment_units": engine_overinv,
        "energy_on_main_attackers": sum(f["attached_units"] for f in my_ents if f["is_main_attacker"]),
        "exposed_three_prize_liability": exposed3,
        "_my_entities": my_ents, "_opp_entities": opp_ents,
    }


def tactical_coordinates(b: dict) -> dict:
    """Diagnostic coordinates (task 1C). NOT collapsed into one scalar."""
    return {
        "RACE_STATE": {
            "my_immediate_ko": b["my_immediate_ko"], "opp_immediate_ko": b["opp_immediate_ko"],
            "both_mains_online": b["my_ready_main_attackers"] >= 1 and b["opp_ready_attackers"] >= 1,
            "prize_diff": b["prize_diff"], "backup_continuity_diff": b["my_backup_ready"] - 0,
        },
        "SWEEP_PRESSURE": {
            "my_ready_main": b["my_ready_main_attackers"], "my_ready_backup": b["my_backup_ready"],
            "opp_ready_response": b["opp_ready_attackers"], "opp_one_short_response": 0,
            "expected_consecutive_kos": b["my_ready_main_attackers"] + b["my_backup_ready"],
        },
        "WALL_PRESSURE": {
            "opp_active_hp": (b["_opp_entities"][0]["hp_remaining"] if b["_opp_entities"] and b["_opp_entities"][0]["slot"] == "active" else None),
            "my_max_damage": max([f["max_affordable_damage"] for f in b["_my_entities"] if f["slot"] == "active"] or [0.0]),
            "wall_survival_margin": None, "opp_bench_behind_active": max(0, b["opp_pokemon"] - 1),
        },
        "VALUE_STATE": {
            "prize_diff": b["prize_diff"], "ready_attacker_diff": b["my_ready_attackers"] - b["opp_ready_attackers"],
            "backup_continuity_diff": b["my_backup_ready"], "energy_dev_diff": b["my_units"] - b["opp_units"],
            "hand_count_diff": b["my_hand_count"] - b["opp_hand_count"], "board_hp_diff": b["my_board_hp"] - b["opp_board_hp"],
            "deckout_pressure": b["my_deckout_risk"],
        },
        "COMMITMENT_STATE": b.get("_commitment", {}),
    }


def _commitment_state(obs, opts, me, opp) -> dict:
    """Which families of action are available right now (task 1C COMMITMENT_STATE)."""
    has = {t: any(o.get("type") == t for o in opts) for t in (PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END, CARD)}
    gw = ko = nonterm_atk = False
    for o in opts:
        if o.get("type") != ATTACK:
            continue
        try:
            prof = DP.attack_profile(o, obs)
        except Exception:
            prof = {}
        if o.get("attackId") == NEBULA_BEAM:  # flat 210, ignores wk/rs -> recompute KO ourselves
            d = DP._active(opp)
            dhp = _remaining_hp(d) if d else 0.0
            prof = {"ko": bool(d and dhp > 0 and 210.0 >= dhp), "game_win": bool(d and dhp > 0 and 210.0 >= dhp and DP._prize_value(DP._cid(d)) >= len(DP._items(DP._get(me, "prize", []))))}
        gw = gw or bool(prof.get("game_win"))
        ko = ko or bool(prof.get("ko"))
        nonterm_atk = nonterm_atk or not bool(prof.get("ko"))
    # supporter / info-revealing among PLAY options
    supporter_unused = info_action = False
    for o in opts:
        if o.get("type") != PLAY:
            continue
        cid = DP.option_card_id(o, obs)
        eff = (DP.CEFF.get(str(cid), {}) if hasattr(DP, "CEFF") else {}) or {}
        if eff.get("draw") or eff.get("search") or eff.get("search_to_bench"):
            info_action = True
    return {
        "game_winning_attack_available": gw, "guaranteed_ko_available": ko, "nonterminal_attack_available": nonterm_atk,
        "safe_development_available": has[PLAY] or has[ATTACH] or has[EVOLVE] or has[ABILITY],
        "attachment_unused": has[ATTACH], "supporter_or_play_unused": has[PLAY],
        "information_action_available": info_action, "retreat_available": has[RETREAT], "end_available": has[END],
    }


def extract(obs: dict, deck=None) -> dict:
    """Top-level: public tactical-state record for one decision. Returns {} on a non-decision or any failure."""
    try:
        sel = DP._selection(obs)
        if sel is None:
            return {}
        cur = DP._current(obs)
        me_idx = DP._perspective(cur)
        me, opp = DP._player(cur, me_idx), DP._player(cur, 1 - me_idx)
        opts = DP._items(DP._get(sel, "option", []))
        b = board_features(me, opp, obs, opts)
        b["_commitment"] = _commitment_state(obs, opts, me, opp)
        coords = tactical_coordinates(b)
        entities = b.pop("_my_entities") + b.pop("_opp_entities")
        b.pop("_commitment", None)
        return {
            "entity_features": entities,
            "board_features": b,
            "tactical_coordinates": coords,
            "n_options": len(opts),
            "option_types": sorted({int(o.get("type")) for o in opts if o.get("type") is not None}),
            "schema_version": "starmie_tactical_state_v1",
        }
    except Exception as e:
        return {"error": repr(e), "schema_version": "starmie_tactical_state_v1"}


if __name__ == "__main__":  # tiny smoke test on a synthetic obs
    demo = {
        "current": {"yourIndex": 0, "players": [
            {"active": [{"id": MEGA_STARMIE, "hp": 330, "energyCards": [{"id": IGNITION}]}],
             "bench": [{"id": CINDERACE, "hp": 160, "energyCards": [{"id": BASIC_WATER}]}],
             "prize": [1, 1, 1, 1, 1], "hand": [1, 2, 3], "handCount": 3, "deckCount": 30},
            {"active": [{"id": 743, "hp": 60}], "bench": [], "prize": [1, 1, 1, 1, 1, 1], "handCount": 5, "deckCount": 40},
        ]},
        "select": {"maxCount": 1, "minCount": 1, "option": [
            {"type": ATTACK, "attackId": NEBULA_BEAM}, {"type": ATTACK, "attackId": JETTING_BLOW}, {"type": END}]},
    }
    import json
    print(json.dumps(extract(demo), indent=2, default=str))
