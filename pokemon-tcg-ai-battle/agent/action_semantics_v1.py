"""Continuous Terrain V1 -- A5 fine-grained action/effect semantics.

Produces a flat, self-contained semantic vector per legal option by assembling already-decoded sources
(card_features + card_effects + attack_stats + forward-model option_deltas + encode_state context) plus a
runtime own-switch/opp-gust disambiguation and a deterministic OVERRIDE table for frequent cards the base
decoder conflates or under-decodes. Fields the engine layer cannot reliably decode (coin-flip EV/variance,
energy-type restriction) are emitted as 0 with a `coverage` flag rather than fabricated.

    from action_semantics_v1 import semantic_vector, CARD_NAME
    v = semantic_vector(obs, option_index, deltas[i], feats, cur, me)
"""
from __future__ import annotations

import json
from pathlib import Path

import state_action_schema_v2 as SCH

_AG = Path(__file__).resolve().parent
CARD_FEATURES = json.load(open(_AG / "card_features.json", encoding="utf-8"))
CARD_EFFECTS = json.load(open(_AG / "card_effects.json", encoding="utf-8"))
ATTACK_STATS = json.load(open(_AG / "attack_stats.json", encoding="utf-8"))
CARD_NAME = {int(k): v.get("n", "") for k, v in CARD_FEATURES.items()}


def _norm_name(s):
    return (s or "").replace("’", "'").replace("‘", "'").strip().lower()


_NAME_TO_ID = {}
for _k, _v in CARD_FEATURES.items():
    _NAME_TO_ID.setdefault(_norm_name(_v.get("n", "")), int(_k))

EFFECT_FIELDS = ["draw", "search", "search_to_bench", "energy_accel", "heal", "recover_discard",
                 "status", "disrupt", "discard_cost", "shuffle_hand", "has_ability"]
# coarse card_type code from card_features.type (stable small int)
_TYPE_CODES = {"basic_energy": 0, "special_energy": 1, "pokemon": 2, "basic_pokemon": 2, "stage1": 3,
               "stage2": 4, "item": 5, "supporter": 6, "stadium": 7, "tool": 8, "mega": 9}

# Deterministic OVERRIDES for frequent meta cards the regex decoder conflates/under-decodes.
# Keyed by card NAME (resolved to id at import). Values OVERWRITE the decoded effect fields.
_OVERRIDE_BY_NAME = {
    "Buddy-Buddy Poffin":      {"search_to_bench": 2, "search": 0},
    "Ultra Ball":              {"search": 1, "discard_cost": 2},
    "Nest Ball":               {"search_to_bench": 1, "search": 0},
    "Dusk Ball":               {"search": 1},
    "Earthen Vessel":          {"search": 2, "discard_cost": 1},
    "Professor's Research":    {"draw": 7, "discard_hand": 1},
    "Iono":                    {"shuffle_hand": 1, "draw": 0, "disrupt": 1},
    "Boss's Orders":           {"opp_gust": 1, "switch_gust": 0},
    "Switch":                  {"own_switch": 1, "switch_gust": 0},
    "Counter Catcher":         {"opp_gust": 1, "switch_gust": 0},
}
# card_features.type -> coverage category (when no decoded text effect / override exists)
_ENERGY_TYPES = {"basic_energy", "special_energy"}
_POKEMON_TYPES = {"pokemon", "basic_pokemon", "stage1", "stage2", "mega"}
OVERRIDES = {}
for _nm, _ov in _OVERRIDE_BY_NAME.items():
    _cid = _NAME_TO_ID.get(_norm_name(_nm))
    if _cid is not None:
        OVERRIDES[_cid] = _ov


def _meta(card_id):
    cf = CARD_FEATURES.get(str(card_id)) if card_id is not None else None
    if not cf:
        return {"card_type": -1, "card_stage": -1, "card_hp": 0, "card_prize": 0, "is_ex": 0,
                "is_mega": 0, "is_tera": 0, "is_ace_spec": 0, "card_retreat": 0}
    stage_map = {None: 0, "basic": 1, "stage1": 2, "stage2": 3, "mega": 4}
    return {"card_type": _TYPE_CODES.get(cf.get("type"), -1),
            "card_stage": stage_map.get(cf.get("stage"), 0),
            "card_hp": int(cf.get("hp", 0) or 0), "card_prize": int(cf.get("prize", 0) or 0),
            "is_ex": int(bool(cf.get("ex"))), "is_mega": int(bool(cf.get("mega"))),
            "is_tera": int(bool(cf.get("tera"))), "is_ace_spec": int(bool(cf.get("ace_spec"))),
            "card_retreat": int(cf.get("retreat", 0) or 0)}


def _effects(card_id, target_side):
    """Decoded effect magnitudes + own_switch/opp_gust split + override. Returns (dict, coverage)."""
    out = {f"eff_{k}": 0 for k in EFFECT_FIELDS}
    out["own_switch"] = 0
    out["opp_gust"] = 0
    out["discard_hand"] = 0
    cov = "unknown"
    raw = CARD_EFFECTS.get(str(card_id)) if card_id is not None else None
    if isinstance(raw, dict):
        cov = "decoded"
        for k, v in raw.items():
            if k == "switch_gust":
                # disambiguate by who the option targets: opp -> gust, me/own -> switch; unknown -> split half
                if target_side == 1:
                    out["opp_gust"] = float(v)
                elif target_side == 0:
                    out["own_switch"] = float(v)
                else:
                    out["opp_gust"] = out["own_switch"] = float(v) * 0.5
            elif f"eff_{k}" in out:
                out[f"eff_{k}"] = float(v)
    ov = OVERRIDES.get(card_id)
    if ov:
        cov = "override"
        for k, v in ov.items():
            key = f"eff_{k}" if k in EFFECT_FIELDS else k
            out[key] = float(v)
    return out, cov


def _delta_vec(d):
    keys = ["prizes_taken", "opp_prizes_taken", "opp_ko", "dmg_dealt", "cards_drawn", "energy_attached",
            "board_dev", "deck_used", "discard_gain", "ends_turn", "wins_now", "loses_now"]
    if not d:
        return {f"d_{k}": 0.0 for k in keys}
    return {f"d_{k}": float(d.get(k, 0) or 0) for k in keys}


def _context(feats):
    f = feats or {}
    return {
        "ctx_can_ko": int(f.get("can_ko_opp_now", 0) or 0 > 0),
        "ctx_lethal": int((f.get("can_ko_opp_now", 0) or 0) > 0 and int(f.get("opp_prizes_left", 6) or 6) <= int(f.get("ko_prize_value", 1) or 1)),
        "ctx_ko_back": int(0 < (f.get("my_active_hp", 0) or 0) <= 120),
        "ctx_prize_lead": int(f.get("prize_lead", 0) or 0),
        "ctx_backup_attacker": int(f.get("my_bench", 0) or 0),
        "ctx_bench_slots": 5 - int(f.get("my_bench", 0) or 0),
        "ctx_hand_size": int(f.get("hand_size", 0) or 0),
        "ctx_deckout_risk": int(f.get("deckout_risk", 0) or 0),
        "ctx_energy_short": int(f.get("active_energy_short", 0) or 0),
        "ctx_supporter_avail": int(f.get("supporter_available", 0) or 0),
        "ctx_attach_done": int(f.get("energy_attach_done", 0) or 0),
        "ctx_can_ability": int(f.get("can_use_ability", 0) or 0),
        "ctx_board_dev": int(f.get("my_bench", 0) or 0) + int(f.get("opp_bench", 0) or 0),
        "ctx_attacker_ready": int(f.get("active_can_attack_now", 0) or 0),
    }


def semantic_vector(obs, i, deltas_i, feats, cur=None, me=None):
    """Flat semantic vector for legal option i. Self-contained for downstream featurization."""
    cur = cur or obs.get("current") or {}
    me = me if me is not None else cur.get("yourIndex", 0)
    players = cur.get("players") or []
    me_player = players[me] if me < len(players) else {}
    opt = (obs.get("select") or {}).get("option", [])[i]
    cid = SCH.card_identity(opt, me_player)
    tgt = SCH.target_entity(opt)
    tpi = tgt.get("player_index")
    target_side = (0 if tpi == me else 1) if tpi is not None else -1
    attack_id = opt.get("attackId")
    atk = ATTACK_STATS.get(str(attack_id)) if attack_id is not None else None
    eff, cov = _effects(cid, target_side)
    if cov not in ("override", "decoded"):
        cf = CARD_FEATURES.get(str(cid)) or {}
        ctype = cf.get("type")
        if ctype in _ENERGY_TYPES:
            cov = "energy"                       # plain/typed energy: the action IS "attach energy of type X"
        elif ctype in _POKEMON_TYPES or (cf.get("hp", 0) or 0) > 0:
            cov = "pokemon_meta"                 # play/evolve: card meta + deltas capture establishment
        elif ctype == "tool":
            cov = "tool"                         # attach tool: meta captures it (specific tool effect partial)
        # else genuinely-undecoded item/supporter/stadium -> stays "unknown"
    v = {
        # action identity
        "opt_type": int(opt.get("type", -1)),
        "acting_card_id": int(cid) if cid is not None else -1,
        "attack_id": int(attack_id) if attack_id is not None else -1,
        "ability_flag": int(opt.get("type") == SCH.OptType.ABILITY),
        "target_side": target_side,
        "target_zone": int(tgt.get("in_play_area") if tgt.get("in_play_area") is not None else -1),
        "target_index": int(tgt.get("in_play_index") if tgt.get("in_play_index") is not None else -1),
        # attack effect
        "atk_damage": float(atk.get("d", 0)) if atk else 0.0,
        "atk_cost": float(atk.get("c", 0)) if atk else 0.0,
        "semantic_coverage": cov,
    }
    v.update(_meta(cid))
    v.update(eff)
    v.update(_delta_vec(deltas_i))
    v.update(_context(feats))
    return v


if __name__ == "__main__":
    print(f"action_semantics_v1: {len(CARD_FEATURES)} cards, {len(CARD_EFFECTS)} decoded, "
          f"{len(OVERRIDES)} overrides resolved.")
