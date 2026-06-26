"""CABT observation -> public Feature-V2 payload for Model A's official packer + portable selector runtime.

This is the Model-B side adapter. It produces ONLY the public fields the runtime model consumes
(state counts, board-entity buckets, option exact-ID features) and hands them to the OFFICIAL packer
(agent/vendor/portable_selector_v1/starmie_feature_v2_packer.py). It never copies hidden/future/pilot/
outcome data and never re-implements the model or the packer.

Field resolution reuses deck_policy_v3's proven CABT resolvers (option_card_id / option_target_entity).
Validated field-by-field against Model A's exported raw_observation payloads before any live use.

The runtime model (verified from feature_vocab.json) reads exactly:
  state: bias, our/opp_hand_size, our/opp_deck_count, our/opp_prize_count, our_bench_count,
         our/opp_attack_ready_count, option_count
  entity (our/opp active+bench, and the option's target): hp_bucket(30), damage_bucket(30),
         attached_energy_count(1)
  option features: type_id, attack_id, ability_id, source_card_id, target_card_id, context_card_id,
         target_owner/zone/slot ; printed action_family
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE / "vendor" / "portable_selector_v1"
for _p in (str(_HERE), str(_VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import deck_policy_v3 as DP  # noqa: E402  proven CABT resolvers
import starmie_tactical_state as TS  # noqa: E402  public tactical-state extractor

A_ACTIVE, A_BENCH, A_HAND, A_DECK, A_DISCARD = 4, 5, 2, 1, 6
_ZONE = {A_DECK: "deck", A_HAND: "hand", A_ACTIVE: "active", A_BENCH: "bench", A_DISCARD: "discard"}
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END, CARD = 7, 8, 9, 10, 11, 12, 13, 14, 3
YES, NO = 1, 2

# Model A's deterministic type_id -> action_family convention (learned from the exported parity ground truth).
FAMILY = {YES: "YES_PROMPT", NO: "NO_PROMPT", CARD: "SELECT_CARD", PLAY: "PLAY", ATTACH: "ATTACH",
          EVOLVE: "EVOLVE", ABILITY: "ABILITY", RETREAT: "RETREAT", ATTACK: "ATTACK", END: "END"}


def _zone_str(area):
    if area is None:
        return None
    try:
        area = int(area)
    except Exception:
        return None
    return _ZONE.get(area, f"area_{area}")


# ---------------------------------------------------------------- entity scalars
def _maxhp(ent: Any) -> int:
    v = DP._get(ent, "maxHp", None)
    if v is None:
        v = DP._get(ent, "hp", 0)
    try:
        return int(v or 0)
    except Exception:
        return 0


def _remaining_hp(ent: Any) -> int:
    try:
        return int(DP._get(ent, "hp", 0) or 0)
    except Exception:
        return 0


def _damage(ent: Any) -> int:
    d = DP._get(ent, "damage", None)
    try:
        return int(d) if d is not None else 0
    except Exception:
        return 0


def _energy_count(ent: Any) -> int:
    # Model A counts energy UNITS (an Ignition card supplies 3 units), which CABT pre-expands in the
    # entity's `energies` list. Verified against exported board_entities (Mega+Ignition = 4 units, not 2 cards).
    en = DP._get(ent, "energies", None)
    if isinstance(en, list):
        return len(en)
    return DP._attached_count(ent)


def _attack_ready(ent: Any) -> bool:
    """An entity is attack-ready if it can afford at least one of its attacks with attached energy."""
    if not ent:
        return False
    n = DP._attached_count(ent)
    meta = DP._meta(DP._cid(ent)) or {}
    atks = meta.get("atks", []) or meta.get("atk", []) or []
    for attack in atks:
        cost = attack.get("cE", attack.get("cost", [])) or []
        if n >= len(cost):
            return True
    return False


# ---------------------------------------------------------------- board + state
def board_entities(obs: Any) -> list[dict]:
    cur = DP._current(obs)
    me = DP._perspective(cur)
    rows: list[dict] = []
    for role, pidx in (("our", me), ("opp", 1 - me)):
        player = DP._player(cur, pidx)
        # Model A always emits an active-slot row (placeholder when the slot is empty).
        rows.append(_entity_row(DP._active(player), role, "active", 0))
        for slot, b in enumerate(DP._bench(player)):
            if b:
                rows.append(_entity_row(b, role, "bench", slot))
    return rows


def _entity_row(ent: Any, role: str, zone: str, slot: int) -> dict:
    # Model A reports hp = CABT remaining hp (already net of damage), damage = the explicit CABT
    # damage-counter field (0 when absent). Empty slots are emitted with card_id None / hp 0.
    if not ent:
        return {"player_role": role, "zone": zone, "slot_index": slot,
                "card_id": None, "hp": 0, "damage": 0, "attached_energy_count": 0}
    return {
        "player_role": role,
        "zone": zone,
        "slot_index": slot,
        "card_id": DP._cid(ent),
        "hp": _remaining_hp(ent),
        "damage": _damage(ent),
        "attached_energy_count": _energy_count(ent),
    }


# The selector model is dominated by tactical features crossed with action family (149 of 185 weights).
# These are decision-level (shared across options). Keys must match the runtime's expected token names.
_TAC_BOARD = ("prize_diff", "my_ready_main_attackers", "my_backup_ready", "my_main_one_short", "my_units",
              "opp_units", "engine_overinvestment_units", "energy_on_main_attackers",
              "exposed_three_prize_liability", "my_deck_count")
_TAC_COMMIT = ("game_winning_attack_available", "guaranteed_ko_available", "nonterminal_attack_available",
               "safe_development_available", "attachment_unused", "retreat_available", "end_available")
_TAC_VALUE = ("ready_attacker_diff", "energy_dev_diff", "deckout_pressure")


def tactical_state_features(obs: Any) -> dict:
    """Flat board./commitment./value. tactical features consumed by the selector model.

    Built from the public starmie_tactical_state extractor (the same tactical schema Model A trained on).
    """
    cur = DP._current(obs)
    me_i = DP._perspective(cur)
    me = DP._player(cur, me_i)
    opp = DP._player(cur, 1 - me_i)
    opts = DP._items(DP._get(DP._selection(obs) or {}, "option", []))
    try:
        board = TS.board_features(me, opp, obs, opts)
        commit = TS._commitment_state(obs, opts, me, opp)
        value = TS.tactical_coordinates({**board, "_commitment": commit}).get("VALUE_STATE", {})
    except Exception:
        return {}
    tsf: dict[str, Any] = {}
    for k in _TAC_BOARD:
        if k in board:
            tsf[f"board.{k}"] = board[k]
    for k in _TAC_COMMIT:
        if k in commit:
            tsf[f"commitment.{k}"] = bool(commit[k])
    for k in _TAC_VALUE:
        if k in value:
            tsf[f"value.{k}"] = value[k]
    return tsf


def state_features(obs: Any, options: list) -> dict:
    cur = DP._current(obs)
    me = DP._perspective(cur)
    out: dict[str, Any] = {"option_count": len(options)}
    for role, pidx in (("our", me), ("opp", 1 - me)):
        player = DP._player(cur, pidx)
        out[f"{role}_hand_size"] = int(DP._get(player, "handCount", 0) or 0)
        out[f"{role}_deck_count"] = int(DP._get(player, "deckCount", 0) or 0)
        out[f"{role}_prize_count"] = len(DP._items(DP._get(player, "prize", [])))
        out[f"{role}_bench_count"] = len(DP._bench(player))
        ready = 0
        if _attack_ready(DP._active(player)):
            ready += 1
        for b in DP._bench(player):
            if _attack_ready(b):
                ready += 1
        out[f"{role}_attack_ready_count"] = ready
    return out


# ---------------------------------------------------------------- option features
def _owner(pidx: Any, me: int) -> str:
    try:
        return "our" if int(pidx) == me else "opp"
    except Exception:
        return "our"


_ZONE_KEY = {A_HAND: "hand", A_ACTIVE: "active", A_BENCH: "bench", A_DISCARD: "discard"}


def _card_at(cur: Any, opt: Any, me: int):
    """Resolve a selected card's id from a visible zone (hand/active/bench/discard) by area+index."""
    area = DP._get(opt, "area", None)
    index = DP._get(opt, "index", None)
    key = _ZONE_KEY.get(area if not isinstance(area, str) else None)
    if key is None or index is None:
        return None
    pidx = DP._get(opt, "playerIndex", me)
    try:
        player = DP._player(cur, int(pidx))
        zone = DP._items(DP._get(player, key, []))
        ent = zone[int(index)] if 0 <= int(index) < len(zone) else None
        return DP._cid(ent)
    except Exception:
        return None


def _target_loc(opt: Any, me: int) -> tuple[Any, Any, Any]:
    """Recipient (board target) location: inPlayArea/inPlayIndex, falling back to area/index."""
    area = DP._get(opt, "inPlayArea", DP._get(opt, "area", None))
    index = DP._get(opt, "inPlayIndex", DP._get(opt, "targetIndex", None))
    if index is None:
        index = DP._get(opt, "index", None)
    owner = _owner(DP._get(opt, "targetPlayerIndex", DP._get(opt, "playerIndex", me)), me)
    try:
        slot = int(index) if index is not None else None
    except Exception:
        slot = None
    return owner, _zone_str(area), slot


def option_features(opt: Any, obs: Any) -> dict:
    """Resolve the model-consumed exact-ID features from a raw CABT option (no hidden/future data).

    Matches Model A's exported conventions: deterministic family per type_id, select_context_id from the
    selection context, ATTACK source = our active attacker, source/target location from area/index aliases.
    """
    cur = DP._current(obs)
    me = DP._perspective(cur)
    t = DP._get(opt, "type", None)
    feats: dict[str, Any] = {
        "type_id": t,
        "action_family": FAMILY.get(t),
        "attack_id": DP._get(opt, "attackId", None),
        "ability_id": DP._get(opt, "abilityId", None),
    }
    if t in (ATTACK, END):
        feats["ends_turn"] = True

    src = None
    try:
        src = DP.option_card_id(opt, obs)
    except Exception:
        src = None
    src = src if isinstance(src, int) else None

    if t in (PLAY, ATTACH, EVOLVE):
        feats["source_card_id"] = src
        feats["source_owner"] = _owner(DP._get(opt, "playerIndex", me), me)
        # PLAY/ATTACH/EVOLVE are always played from hand; CABT omits the area on bare PLAY options.
        feats["source_zone"] = _zone_str(DP._get(opt, "area", None)) or "hand"
        si = DP._get(opt, "index", None)
        try:
            feats["source_slot"] = int(si) if si is not None else None
        except Exception:
            pass
    elif t == ATTACK:
        feats["source_card_id"] = DP._cid(DP._active(DP._player(cur, me)))
        feats["source_owner"] = "our"
        feats["source_zone"] = "active"
        feats["source_slot"] = 0
    elif t == CARD:  # SELECT_CARD: resolved card is the selected card (a target)
        feats["target_card_id"] = src

    # board recipient (attach/evolve target, or selected card location)
    tent = None
    try:
        tent = DP.option_target_entity(opt, obs)
    except Exception:
        tent = None
    if t in (ATTACH, EVOLVE, CARD):
        owner, zone, slot = _target_loc(opt, me)
        if feats.get("target_card_id") is None and tent is not None:
            feats["target_card_id"] = DP._cid(tent)
        if t == CARD and feats.get("target_card_id") is None:
            feats["target_card_id"] = _card_at(cur, opt, me)
        if zone is not None:
            feats["target_owner"] = owner
            feats["target_zone"] = zone
            feats["target_slot"] = slot

    # selection-level context (shared by every option in this prompt)
    sel = DP._selection(obs) or {}
    sctx = DP._get(sel, "context", None)
    if sctx is not None:
        try:
            feats["select_context_id"] = int(sctx)
        except Exception:
            pass
    ctx_card = DP._cid(DP._get(sel, "contextCard", None))
    if isinstance(ctx_card, int):
        feats["context_card_id"] = ctx_card
    if t == YES:
        feats["yes_no_value"] = 1
        feats["is_yes"] = True
    elif t == NO:
        feats["yes_no_value"] = -1
        feats["is_no"] = True

    return {k: v for k, v in feats.items() if v is not None}


def shim_option(opt: Any, obs: Any, raw_index: int, tactical: dict | None = None) -> dict:
    feats = option_features(opt, obs)
    out = {
        "raw_option_index": raw_index,
        "type_id": DP._get(opt, "type", None),
        "action_family": feats.get("action_family"),
        "features": feats,
    }
    if tactical:
        out["tactical_state_features"] = tactical
    return out


# ---------------------------------------------------------------- payload
def cabt_to_payload(obs: Any, baseline_action: Any = None, search_action: Any = None) -> dict:
    """Build the public Feature-V2 adapter payload consumed by the official packer."""
    sel = DP._selection(obs) or {}
    options = DP._items(DP._get(sel, "option", []))
    tactical = tactical_state_features(obs)
    payload: dict[str, Any] = {
        "state_features": state_features(obs, options),
        "board_entities": board_entities(obs),
        "tactical_state_features": tactical,
        "raw_legal_options": [shim_option(o, obs, i, tactical) for i, o in enumerate(options)],
    }
    if baseline_action is not None:
        payload["baseline_action"] = baseline_action
    if search_action is not None:
        payload["search_action"] = search_action
    return payload


if __name__ == "__main__":  # pragma: no cover - smoke
    print("learned_selector_bridge: CABT -> Feature-V2 payload adapter (read-only)")
