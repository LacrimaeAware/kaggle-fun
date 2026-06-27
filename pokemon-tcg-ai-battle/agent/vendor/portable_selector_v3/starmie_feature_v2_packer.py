"""Standalone public Feature-V2 packer for the Starmie selector bundle.

The exported selector runtime consumes a packed public observation plus grouped
legal options. This module is the official normalizer for the public payload
that a CABT adapter should produce before calling the selector. It is dependency
free so it can be copied next to ``starmie_selector_runtime.py``.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "starmie_feature_v2_packer_v1"
ACTION_TYPE_FAMILIES = {
    "0": "no_op",
    "1": "yes_prompt",
    "2": "no_prompt",
    "3": "select_card",
    "7": "play",
    "8": "attach",
    "9": "evolve",
    "10": "ability",
    "12": "retreat",
    "13": "attack",
    "14": "end_turn",
}
FORBIDDEN_INPUT_KEYS = {
    "pilot",
    "team",
    "outcome",
    "winner",
    "won",
    "replay_id",
    "decision_id",
    "split",
    "selected",
    "label",
    "future",
}


def pack_cabt_observation(observation: Mapping[str, Any], legal_options: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Pack a public CABT-derived observation into selector-ready Feature-V2.

    The input may already be close to the training-pack shape, or it may use
    CABT-style option aliases such as ``card_id``, ``card``, ``inPlayArea`` and
    ``inPlayIndex``. Hidden cards, future actions, pilot/team IDs and outcomes
    are never copied into model-facing fields.
    """

    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    raw_options = legal_options
    if raw_options is None:
        for key in ("legal_options", "options", "raw_legal_options"):
            candidate = observation.get(key)
            if _is_sequence(candidate):
                raw_options = candidate  # type: ignore[assignment]
                break
    if raw_options is None:
        raw_options = ()
    state_features = _copy_public_mapping(observation.get("state_features"))
    board_entities = _copy_public_rows(observation.get("board_entities"))
    zone_inventory = _copy_public_rows(observation.get("zone_inventory"))
    tactical_state_features = _copy_public_mapping(observation.get("tactical_state_features"))
    runtime_tactical = _copy_public_mapping(observation.get("runtime_tactical"))
    packed_options = [
        _pack_option(option, index, board_entities)
        for index, option in enumerate(raw_options)
        if isinstance(option, Mapping)
    ]
    baseline_action = _pack_action_reference(observation.get("baseline_action"), packed_options)
    search_action = _pack_action_reference(observation.get("search_action"), packed_options)
    forbidden = sorted(key for key in observation if _is_forbidden_key(str(key)))
    out = {
        "schema_version": SCHEMA_VERSION,
        "state_features": state_features,
        "board_entities": board_entities,
        "zone_inventory": zone_inventory,
        "tactical_state_features": tactical_state_features,
        "runtime_tactical": runtime_tactical,
        "packed_options": packed_options,
        "legal_options": packed_options,
        "baseline_action": baseline_action,
        "search_action": search_action,
        "forbidden_metadata_ignored": forbidden,
        "support_status": "PACKED",
        "packer_notes": {
            "input_contract": "public CABT adapter payload; no hidden/future/pilot/outcome fields are model-facing",
            "select_card_rule": "type_id=3 card_id/card resolves to target_card_id; duplicate raw options remain in raw_option_indexes",
            "play_rule": "type_id=7 source_card_id is played card; context_card_id uses explicit context/effect card when present",
            "attach_rule": "type_id=8 source_card_id is attached card; target owner/zone/slot/card describe recipient",
        },
    }
    if observation.get("top_k") is not None:
        out["top_k"] = observation.get("top_k")
    return out


def _pack_option(option: Mapping[str, Any], fallback_index: int, board_entities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    features = _merged_features(option)
    type_id = _int_value(_first_present(features, option, ("type_id", "type", "action_type", "actionType")))
    family = _normalize_family(str(option.get("action_family") or _family_from_features(features, type_id)))
    _resolve_card_identity(features, option, type_id, family, board_entities)
    packed_index = _int_value(option.get("packed_option_index"))
    if packed_index is None:
        packed_index = fallback_index
    packed_indexes = _int_list(option.get("packed_option_indexes"))
    if not packed_indexes:
        packed_indexes = [packed_index]
    raw_indexes = _int_list(option.get("raw_option_indexes"))
    if not raw_indexes:
        raw_indexes = _int_list(option.get("source_option_indexes"))
    raw_index = _int_value(option.get("raw_option_index"))
    if raw_index is None:
        raw_index = _int_value(option.get("source_option_index"))
    if raw_index is None:
        raw_index = fallback_index
    if not raw_indexes:
        raw_indexes = [raw_index]
    features["type_id"] = type_id
    if "action_group_index" not in features and option.get("source_action_group_index") is not None:
        features["action_group_index"] = option.get("source_action_group_index")
    if "action_group_size" not in features and len(raw_indexes) > 1:
        features["action_group_size"] = len(raw_indexes)
    out = {
        "packed_option_index": packed_index,
        "packed_option_indexes": packed_indexes,
        "raw_option_index": raw_index,
        "raw_option_indexes": raw_indexes,
        "source_option_index": _int_value(option.get("source_option_index")),
        "source_option_indexes": _int_list(option.get("source_option_indexes")) or raw_indexes,
        "semantic_action_key": str(option.get("semantic_action_key") or _semantic_action_key(features, family)),
        "action_family": family,
        "type_id": type_id,
        "attack_id": _int_value(features.get("attack_id")),
        "ability_id": _int_value(features.get("ability_id")),
        "source_card_id": _int_value(features.get("source_card_id")),
        "target_card_id": _int_value(features.get("target_card_id")),
        "target_owner": features.get("target_owner"),
        "target_zone": features.get("target_zone"),
        "target_slot": _int_value(features.get("target_slot")),
        "context_card_id": _int_value(features.get("context_card_id")),
        "features": _jsonable(features),
        "hard_safety_flags": _copy_public_mapping(option.get("hard_safety_flags")),
        "tactical_state_features": _copy_public_mapping(option.get("tactical_state_features")),
        "is_source_option0": bool(option.get("is_source_option0")) or raw_index == 0 or 0 in set(raw_indexes),
    }
    return {key: value for key, value in out.items() if value is not None}


def _merged_features(option: Mapping[str, Any]) -> dict[str, Any]:
    features = copy.deepcopy(option.get("features")) if isinstance(option.get("features"), Mapping) else {}
    aliases = (
        "type_id",
        "type",
        "action_type",
        "actionType",
        "area_id",
        "select_context_id",
        "context_card_id",
        "effect_card_id",
        "source_card_id",
        "target_card_id",
        "card_id",
        "attack_id",
        "ability_id",
        "source_owner",
        "source_zone",
        "source_slot",
        "target_owner",
        "target_zone",
        "target_slot",
        "target_index",
        "target_area",
        "target_player_index",
        "inPlayArea",
        "inPlayIndex",
        "yes_no_value",
        "number_value",
        "ends_turn",
        "energy_attachment_delta",
        "evolves_via_search",
        "is_yes",
        "is_no",
    )
    for key in aliases:
        if key in option and key not in features:
            features[key] = option.get(key)
    for key, alias in (
        ("sourceCardId", "source_card_id"),
        ("targetCardId", "target_card_id"),
        ("contextCardId", "context_card_id"),
        ("effectCardId", "effect_card_id"),
        ("attackId", "attack_id"),
        ("abilityId", "ability_id"),
    ):
        if key in option and alias not in features:
            features[alias] = option.get(key)
    return features


def _resolve_card_identity(
    features: dict[str, Any],
    option: Mapping[str, Any],
    type_id: int | None,
    family: str,
    board_entities: Sequence[Mapping[str, Any]],
) -> None:
    explicit_card = _card_id_from_any(
        option.get("card"),
        option.get("card_id"),
        features.get("card_id"),
        option.get("selected_card"),
        option.get("selectedCard"),
    )
    context_card = _card_id_from_any(
        option.get("context_card"),
        option.get("effect_card"),
        option.get("effectCard"),
        features.get("context_card_id"),
        features.get("effect_card_id"),
    )
    if context_card is not None and features.get("context_card_id") is None:
        features["context_card_id"] = context_card
    if family == "SELECT_CARD" or type_id == 3:
        if features.get("target_card_id") is None and explicit_card is not None:
            features["target_card_id"] = explicit_card
        return
    if family in {"PLAY", "ATTACH", "EVOLVE"} or type_id in {7, 8, 9}:
        if features.get("source_card_id") is None and explicit_card is not None:
            features["source_card_id"] = explicit_card
    target_entity = _target_entity(board_entities, features, option)
    if target_entity:
        if features.get("target_card_id") is None and target_entity.get("card_id") is not None:
            features["target_card_id"] = target_entity.get("card_id")
        if features.get("target_owner") is None:
            features["target_owner"] = target_entity.get("player_role")
        if features.get("target_zone") is None:
            features["target_zone"] = target_entity.get("zone")
        if features.get("target_slot") is None:
            features["target_slot"] = target_entity.get("slot_index")


def _target_entity(
    board_entities: Sequence[Mapping[str, Any]],
    features: Mapping[str, Any],
    option: Mapping[str, Any],
) -> Mapping[str, Any]:
    owner = _first_not_none(features.get("target_owner"), option.get("target_owner"), option.get("targetPlayer"), option.get("target_player_index"))
    zone = _first_not_none(features.get("target_zone"), option.get("target_zone"), option.get("inPlayArea"), option.get("target_area"))
    slot = _int_value(_first_not_none(features.get("target_slot"), option.get("target_slot"), option.get("inPlayIndex"), option.get("target_index")))
    if owner is None and zone is None and slot is None:
        return {}
    for entity in board_entities:
        if owner is not None and str(entity.get("player_role")) != str(owner):
            continue
        if zone is not None and str(entity.get("zone")) != str(zone):
            continue
        if slot is not None and _int_value(entity.get("slot_index")) != slot:
            continue
        return entity
    return {}


def _pack_action_reference(action: Any, options: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, int) and not isinstance(action, bool):
        return _reference_from_indexes([action], options)
    if not isinstance(action, Mapping):
        semantic = str(action)
        indexes = [index for index, option in enumerate(options) if semantic == option.get("semantic_action_key")]
        return _reference_from_indexes(indexes, options, semantic_action_key=semantic)
    indexes = _int_list(action.get("packed_option_indexes"))
    if not indexes:
        packed = _int_value(action.get("packed_option_index"))
        if packed is not None:
            indexes = [packed]
    raw_indexes = _int_list(action.get("raw_option_indexes"))
    semantic = action.get("semantic_action_key")
    ref = _reference_from_indexes(indexes, options, raw_indexes=raw_indexes, semantic_action_key=semantic)
    for key in ("raw_option_index", "source"):
        if key in action:
            ref[key] = copy.deepcopy(action.get(key))
    return ref


def _reference_from_indexes(
    indexes: Sequence[int],
    options: Sequence[Mapping[str, Any]],
    *,
    raw_indexes: Sequence[int] | None = None,
    semantic_action_key: Any = None,
) -> dict[str, Any]:
    packed = sorted({int(index) for index in indexes if isinstance(index, int) and not isinstance(index, bool)})
    valid = [index for index in packed if 0 <= index < len(options)]
    raw = list(raw_indexes or [])
    if not raw:
        for index in valid:
            raw.extend(_int_list(options[index].get("raw_option_indexes")) or [_int_value(options[index].get("raw_option_index")) or index])
    first = valid[0] if valid else (packed[0] if packed else None)
    out: dict[str, Any] = {
        "packed_option_indexes": packed,
        "raw_option_indexes": sorted({int(index) for index in raw if isinstance(index, int) and not isinstance(index, bool)}),
    }
    if first is not None:
        out["packed_option_index"] = first
    if out["raw_option_indexes"]:
        out["raw_option_index"] = out["raw_option_indexes"][0]
    if semantic_action_key is not None:
        out["semantic_action_key"] = str(semantic_action_key)
    elif valid:
        out["semantic_action_key"] = options[valid[0]].get("semantic_action_key")
    return out


def _semantic_action_key(features: Mapping[str, Any], family: str) -> str:
    fields = {
        "family": family,
        "type_id": features.get("type_id"),
        "source_card_id": features.get("source_card_id"),
        "source_owner": features.get("source_owner"),
        "source_zone": features.get("source_zone"),
        "source_slot": features.get("source_slot"),
        "target_card_id": features.get("target_card_id"),
        "target_owner": features.get("target_owner"),
        "target_zone": features.get("target_zone"),
        "target_slot": features.get("target_slot"),
        "attack_id": features.get("attack_id"),
        "ability_id": features.get("ability_id"),
        "select_context_id": features.get("select_context_id"),
        "yes_no_value": features.get("yes_no_value"),
        "number_value": features.get("number_value"),
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _family_from_features(features: Mapping[str, Any], type_id: int | None) -> str:
    if features.get("ends_turn") or type_id == 14:
        return "END"
    if features.get("energy_attachment_delta") or type_id == 8:
        return "ATTACH"
    if features.get("attack_id") is not None or type_id == 13:
        return "ATTACK"
    if features.get("ability_id") is not None or features.get("has_ability_hint") or type_id == 10:
        return "ABILITY"
    if features.get("evolves_via_search") or type_id == 9:
        return "EVOLVE"
    if features.get("select_context_id") not in (None, 0) or type_id == 3:
        return "SELECT_CARD"
    if features.get("is_yes") or type_id == 1:
        return "YES_PROMPT"
    if features.get("is_no") or type_id == 2:
        return "NO_PROMPT"
    if features.get("source_zone") == "hand" or type_id == 7:
        return "PLAY"
    mapped = ACTION_TYPE_FAMILIES.get(str(type_id))
    if mapped:
        return mapped
    if isinstance(type_id, int) and not isinstance(type_id, bool):
        return f"TYPE_{type_id}"
    return "UNKNOWN"


def _normalize_family(family: str) -> str:
    family = family.upper()
    aliases = {
        "END_TURN": "END",
        "NO_OP": "END",
        "CARD": "SELECT_CARD",
        "TUTOR": "SELECT_CARD",
        "ATTACH_ENERGY": "ATTACH",
        "YES": "YES_PROMPT",
        "NO": "NO_PROMPT",
        "YES_PROMPT": "YES_PROMPT",
        "NO_PROMPT": "NO_PROMPT",
    }
    return aliases.get(family, family)


def _copy_public_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _jsonable(item)
        for key, item in value.items()
        if not _is_forbidden_key(str(key))
    }


def _copy_public_rows(value: Any) -> list[dict[str, Any]]:
    if not _is_sequence(value):
        return []
    return [_copy_public_mapping(item) for item in value if isinstance(item, Mapping)]


def _card_id_from_any(*values: Any) -> int | None:
    for value in values:
        parsed = _int_value(value)
        if parsed is not None:
            return parsed
        if isinstance(value, Mapping):
            for key in ("card_id", "id", "cardId", "definition_id"):
                parsed = _int_value(value.get(key))
                if parsed is not None:
                    return parsed
    return None


def _first_present(features: Mapping[str, Any], option: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in features:
            return features.get(key)
        if key in option:
            return option.get(key)
    return None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed
    return None


def _int_list(value: Any) -> list[int]:
    if not _is_sequence(value):
        return []
    out = []
    for item in value:
        parsed = _int_value(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_forbidden_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in FORBIDDEN_INPUT_KEYS)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items() if not _is_forbidden_key(str(key))}
    if _is_sequence(value):
        return [_jsonable(item) for item in value]
    return value
