"""Standalone Starmie proposer selector runtime.

This file is intentionally dependency-light so it can be copied into the
Starmie heuristic repository. It supports the packed public Feature-V2
observation shape plus grouped legal options. Live CABT-to-packed conversion
belongs to the importing adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EPSILON = 1e-12
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


class StarmiePortableRuntimeError(RuntimeError):
    """Raised for malformed portable artifacts."""


class StarmieSelectorRuntime:
    def __init__(
        self,
        *,
        checkpoint: Mapping[str, Any],
        selector_config: Mapping[str, Any],
        api_contract: Mapping[str, Any] | None = None,
        model_hash: str | None = None,
        selector_hash: str | None = None,
    ) -> None:
        weights = checkpoint.get("weights")
        if not isinstance(weights, Mapping):
            raise StarmiePortableRuntimeError("checkpoint is missing sparse proposer weights")
        selected = _mapping(selector_config.get("selected_selector"))
        selector_weights = selected.get("weights")
        if not isinstance(selector_weights, Mapping):
            raise StarmiePortableRuntimeError("selector config is missing selected S5 weights")
        self.proposer_weights = {str(key): float(value) for key, value in weights.items()}
        self.selector_weights = {str(key): float(value) for key, value in selector_weights.items()}
        self.selector_id = str(selected.get("selector_id") or "S5_LIGHTWEIGHT_SELECTOR")
        self.candidate_set_id = str(selected.get("candidate_set_id") or "C4")
        self.model_hash = model_hash
        self.selector_hash = selector_hash
        self.api_contract = dict(api_contract or {})

    @classmethod
    def from_dir(cls, artifact_dir: str | Path) -> "StarmieSelectorRuntime":
        root = Path(artifact_dir)
        checkpoint_path = root / "model_checkpoint.json"
        selector_path = root / "selector_config.json"
        api_path = root / "api_contract.json"
        return cls(
            checkpoint=_read_json(checkpoint_path),
            selector_config=_read_json(selector_path),
            api_contract=_read_json(api_path) if api_path.exists() else {},
            model_hash=_sha256_file(checkpoint_path),
            selector_hash=_sha256_file(selector_path),
        )

    def rank_and_select(
        self,
        observation: Mapping[str, Any],
        legal_options: Sequence[Mapping[str, Any]],
        baseline_action: Any = None,
        search_action: Any = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        try:
            return self._rank_and_select(observation, legal_options, baseline_action, search_action, top_k)
        except Exception as exc:  # pragma: no cover - fail-closed runtime guard.
            return {
                "status": "ERROR",
                "selected_semantic_key": None,
                "selected_raw_option_index": None,
                "source": "fallback",
                "ranked_actions": [],
                "selector_features": {},
                "selector_score": None,
                "entropy": 0.0,
                "top1_margin": 0.0,
                "support_status": f"ERROR:{exc}",
                "model_hash": self.model_hash,
                "selector_hash": self.selector_hash,
            }

    def _rank_and_select(
        self,
        observation: Mapping[str, Any],
        legal_options: Sequence[Mapping[str, Any]],
        baseline_action: Any,
        search_action: Any,
        top_k: int,
    ) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            return _abstain("UNSUPPORTED", "observation is not a mapping", self.model_hash, self.selector_hash)
        options = tuple(option for option in legal_options if isinstance(option, Mapping))
        if not options:
            return _abstain("ABSTAIN", "no legal options supplied", self.model_hash, self.selector_hash)
        forbidden_present = sorted(key for key in observation if _is_forbidden_key(str(key)))
        record = _runtime_record(observation, options)
        rows = _feature_rows_v0(record)
        if len(rows) != len(options):
            return _abstain("UNSUPPORTED", "feature row count mismatch", self.model_hash, self.selector_hash)

        logits = [sum(self.proposer_weights.get(token, 0.0) * value for token, value in row.items()) for row in rows]
        probabilities = _softmax(logits)
        ranking = _ranking(logits)
        entropy = _entropy(probabilities)
        margin = _top_margin(probabilities, ranking)
        semantic_keys = [_semantic_action_key(option) for option in options]
        baseline_indexes = _resolve_action_indexes(baseline_action, options, semantic_keys)
        search_indexes = _resolve_action_indexes(search_action, options, semantic_keys)
        option_zero_indexes = {index for index, option in enumerate(options) if _is_option_zero(option)}
        candidate_indexes = _candidate_indexes(
            self.candidate_set_id,
            ranking,
            baseline_indexes,
            search_indexes,
        )
        if not candidate_indexes:
            return _abstain("ABSTAIN", "candidate set is empty", self.model_hash, self.selector_hash)

        selector_features_by_index: dict[int, dict[str, float]] = {}
        selector_scores: dict[int, float] = {}
        for index in candidate_indexes:
            features = _selector_features(
                index,
                options,
                logits,
                probabilities,
                ranking,
                baseline_indexes,
                search_indexes,
                option_zero_indexes,
                observation,
            )
            selector_features_by_index[index] = features
            selector_scores[index] = sum(self.selector_weights.get(token, 0.0) * value for token, value in features.items())
        ordered = sorted(candidate_indexes, key=lambda idx: (-selector_scores.get(idx, 0.0), idx))
        raw_selected = ordered[0]
        vetoes = _safety_flags(options[raw_selected])
        selected = raw_selected
        source = "selector"
        support = "SUPPORTED"
        if any(vetoes.values()):
            fallback = sorted(baseline_indexes)
            if fallback:
                selected = fallback[0]
                source = "fallback"
                support = "SAFETY_FALLBACK"
            else:
                source = "fallback"
                support = "SAFETY_VETO_NO_BASELINE"
        selected_option = options[selected]
        ranked_actions = []
        for rank, index in enumerate(ranking[: max(0, int(top_k))], start=1):
            ranked_actions.append(
                {
                    "semantic_action_key": semantic_keys[index],
                    "raw_option_index": _raw_option_index(options[index], index),
                    "packed_option_index": index,
                    "probability": probabilities[index],
                    "logit": logits[index],
                    "rank": rank,
                    "action_family": _selector_family(options[index]),
                }
            )
        return {
            "status": "READY",
            "selected_semantic_key": semantic_keys[selected],
            "selected_raw_option_index": _raw_option_index(selected_option, selected),
            "selected_packed_option_index": selected,
            "raw_selector_packed_option_index": raw_selected,
            "source": source,
            "ranked_actions": ranked_actions,
            "selector_features": selector_features_by_index.get(raw_selected, {}),
            "selector_score": selector_scores.get(raw_selected),
            "selector_scores_by_packed_index": {str(k): v for k, v in sorted(selector_scores.items())},
            "entropy": entropy,
            "top1_margin": margin,
            "support_status": support,
            "forbidden_metadata_ignored": forbidden_present,
            "model_hash": self.model_hash,
            "selector_hash": self.selector_hash,
        }


def rank_and_select(
    observation: Mapping[str, Any],
    legal_options: Sequence[Mapping[str, Any]],
    baseline_action: Any = None,
    search_action: Any = None,
    top_k: int = 5,
    *,
    artifact_dir: str | Path | None = None,
    runtime: StarmieSelectorRuntime | None = None,
) -> dict[str, Any]:
    if runtime is None:
        if artifact_dir is None:
            return _abstain("ERROR", "artifact_dir or runtime is required", None, None)
        runtime = StarmieSelectorRuntime.from_dir(artifact_dir)
    return runtime.rank_and_select(observation, legal_options, baseline_action, search_action, top_k)


def _runtime_record(observation: Mapping[str, Any], options: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "state_features": _mapping(observation.get("state_features")),
        "board_entities": list(_rows(observation.get("board_entities"))),
        "zone_inventory": list(_rows(observation.get("zone_inventory"))),
        "option_count": len(options),
        "options": [dict(option) for option in options],
    }


def _feature_rows_v0(record: Mapping[str, Any]) -> list[dict[str, float]]:
    context = _context_features(record)
    rows = []
    for option in _options(record):
        features = dict(context)
        features.update(_action_features(record, option))
        _add_interactions(features)
        rows.append(features)
    return rows


def _context_features(record: Mapping[str, Any]) -> dict[str, float]:
    features = {"state:bias": 1.0}
    state = _mapping(record.get("state_features"))
    for key in (
        "our_hand_size",
        "our_deck_count",
        "our_prize_count",
        "our_bench_count",
        "our_attack_ready_count",
        "opp_hand_size",
        "opp_deck_count",
        "opp_prize_count",
        "opp_attack_ready_count",
        "option_count",
    ):
        _add_numeric(features, f"state:{key}", state.get(key))
    for entity in _rows(record.get("board_entities")):
        role = str(entity.get("player_role") or "unknown")
        zone = str(entity.get("zone") or "unknown")
        prefix = f"entity:{role}:{zone}"
        _add_bucketed(features, f"{prefix}:hp_bucket", entity.get("hp"), bucket=30)
        _add_bucketed(features, f"{prefix}:damage_bucket", entity.get("damage"), bucket=30)
        _add_bucketed(features, f"{prefix}:attached_energy_count", entity.get("attached_energy_count"), bucket=1)
    return features


def _action_features(record: Mapping[str, Any], option: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    option_features = _mapping(option.get("features"))
    features[f"printed:action_family={_action_family(option)}"] = 1.0
    for role, key in (
        ("source", "source_card_id"),
        ("target", "target_card_id"),
        ("context", "context_card_id"),
    ):
        value = _int_value(option_features.get(key))
        if value is not None:
            features[f"exact:{role}:card_id={value}"] = 1.0
    attack_id = _int_value(option_features.get("attack_id"))
    ability_id = _int_value(option_features.get("ability_id"))
    type_id = _int_value(option_features.get("type_id"))
    if attack_id is not None:
        features[f"exact:attack_id={attack_id}"] = 1.0
    if ability_id is not None:
        features[f"exact:ability_id={ability_id}"] = 1.0
    if type_id is not None:
        features[f"exact:type_id={type_id}"] = 1.0
    target = _target_entity(record, option_features)
    if target:
        _add_bucketed(features, "entity:target:hp_bucket", target.get("hp"), bucket=30)
        _add_bucketed(features, "entity:target:damage_bucket", target.get("damage"), bucket=30)
        _add_bucketed(features, "entity:target:attached_energy_count", target.get("attached_energy_count"), bucket=1)
    return features


def _add_interactions(features: dict[str, float]) -> None:
    state_tokens = [token for token in features if token.startswith(("state:", "entity:", "tactical:"))]
    action_tokens = [token for token in features if token.startswith(("exact:", "printed:", "effect:", "attack:", "dynamic:"))]
    for state_token in state_tokens[:120]:
        state_value = features[state_token]
        if not state_value:
            continue
        for action_token in action_tokens[:160]:
            action_value = features[action_token]
            if action_value:
                features[f"cross:{state_token}&&{action_token}"] = state_value * action_value


def _selector_features(
    index: int,
    options: Sequence[Mapping[str, Any]],
    logits: Sequence[float],
    probabilities: Sequence[float],
    ranking: Sequence[int],
    baseline_indexes: set[int],
    search_indexes: set[int],
    option_zero_indexes: set[int],
    observation: Mapping[str, Any],
) -> dict[str, float]:
    rank = ranking.index(index) + 1 if index in ranking else len(ranking) + 1
    family = _selector_family(options[index])
    safety = _safety_flags(options[index])
    tactical = _tactical_features(observation, options[index])
    features: dict[str, float] = {
        "bias": 1.0,
        "source:heuristic": 1.0 if index in baseline_indexes else 0.0,
        "source:search": 1.0 if index in search_indexes else 0.0,
        "source:proposer_top1": 1.0 if rank == 1 else 0.0,
        "source:proposer_top3": 1.0 if rank <= 3 else 0.0,
        "source:proposer_top5": 1.0 if rank <= 5 else 0.0,
        "source:heuristic_proposer_agree": 1.0 if index in baseline_indexes and rank == 1 else 0.0,
        "candidate:option_zero": 1.0 if index in option_zero_indexes else 0.0,
        "proposer:probability": probabilities[index] if 0 <= index < len(probabilities) else 0.0,
        "proposer:logit_scaled": _scaled(logits[index]) if 0 <= index < len(logits) else 0.0,
        "proposer:rank_reciprocal": 1.0 / rank if rank else 0.0,
        "proposer:entropy_scaled": _scaled(_entropy(probabilities)),
        "proposer:margin": _top_margin(probabilities, ranking),
        f"family:{family}": 1.0,
    }
    for key, active in safety.items():
        if active:
            features[f"safety:{key}"] = 1.0
            features[f"safety:{key}&&family:{family}"] = 1.0
    for key, value in tactical.items():
        numeric = _number(value)
        if numeric is not None:
            features[f"tactical:{key}"] = _scaled(numeric)
            features[f"tactical:{key}&&family:{family}"] = _scaled(numeric)
        elif isinstance(value, bool) and value:
            features[f"tactical:{key}=True"] = 1.0
            features[f"tactical:{key}=True&&family:{family}"] = 1.0
    return {key: value for key, value in features.items() if value}


def _candidate_indexes(
    candidate_set_id: str,
    ranking: Sequence[int],
    baseline_indexes: set[int],
    search_indexes: set[int],
) -> set[int]:
    top1 = set(ranking[:1])
    top3 = set(ranking[:3])
    top5 = set(ranking[:5])
    if candidate_set_id == "C0":
        return set(baseline_indexes)
    if candidate_set_id == "C1":
        return top1
    if candidate_set_id == "C2":
        return set(baseline_indexes) | top1
    if candidate_set_id == "C3":
        return set(baseline_indexes) | top3
    if candidate_set_id == "C4":
        return set(baseline_indexes) | top5
    if candidate_set_id == "C5":
        return set(baseline_indexes) | set(search_indexes) | top3
    return set(baseline_indexes) | top5


def _resolve_action_indexes(action: Any, options: Sequence[Mapping[str, Any]], semantic_keys: Sequence[str]) -> set[int]:
    if action is None:
        return set()
    if isinstance(action, int) and not isinstance(action, bool):
        return {
            index
            for index, option in enumerate(options)
            if index == action or _raw_option_index(option, index) == action or _packed_option_index(option, index) == action
        }
    if isinstance(action, Mapping):
        indexes: set[int] = set()
        packed_many = action.get("packed_option_indexes")
        if isinstance(packed_many, Sequence) and not isinstance(packed_many, (str, bytes)):
            packed_values = {item for item in packed_many if isinstance(item, int) and not isinstance(item, bool)}
            indexes.update(index for index, option in enumerate(options) if _packed_option_index(option, index) in packed_values)
        raw_many = action.get("raw_option_indexes")
        if isinstance(raw_many, Sequence) and not isinstance(raw_many, (str, bytes)):
            raw_values = {item for item in raw_many if isinstance(item, int) and not isinstance(item, bool)}
            indexes.update(index for index, option in enumerate(options) if _raw_option_index(option, index) in raw_values)
        packed = action.get("packed_option_index")
        if isinstance(packed, int) and not isinstance(packed, bool):
            indexes.update(index for index, option in enumerate(options) if _packed_option_index(option, index) == packed)
        raw = action.get("raw_option_index")
        if isinstance(raw, int) and not isinstance(raw, bool):
            indexes.update(index for index, option in enumerate(options) if _raw_option_index(option, index) == raw)
        semantic_many = action.get("semantic_action_keys")
        if isinstance(semantic_many, Sequence) and not isinstance(semantic_many, (str, bytes)):
            semantic_values = {str(item) for item in semantic_many if str(item)}
            indexes.update(
                index
                for index, semantic in enumerate(semantic_keys)
                if str(semantic) in semantic_values or bool(semantic_values & set(str(semantic).split("|")))
            )
        for key in ("semantic_action_key", "action_key"):
            if key in action:
                action_key = str(action.get(key))
                indexes.update(
                    index
                    for index, semantic in enumerate(semantic_keys)
                    if action_key == semantic or action_key in str(semantic).split("|")
                )
        return indexes
    action_key = str(action)
    return {
        index
        for index, semantic in enumerate(semantic_keys)
        if action_key == semantic or action_key in str(semantic).split("|")
    }


def _tactical_features(observation: Mapping[str, Any], option: Mapping[str, Any]) -> Mapping[str, Any]:
    option_tactical = option.get("tactical_state_features")
    if isinstance(option_tactical, Mapping):
        return option_tactical
    obs_tactical = observation.get("tactical_state_features")
    if isinstance(obs_tactical, Mapping):
        return obs_tactical
    runtime_tactical = _mapping(observation.get("runtime_tactical"))
    board = _mapping(runtime_tactical.get("board"))
    commit = _mapping(runtime_tactical.get("COMMITMENT_STATE"))
    value = _mapping(runtime_tactical.get("VALUE_STATE"))
    out: dict[str, Any] = {}
    for key in (
        "prize_diff",
        "my_ready_main_attackers",
        "my_backup_ready",
        "my_main_one_short",
        "my_units",
        "opp_units",
        "engine_overinvestment_units",
        "energy_on_main_attackers",
        "exposed_three_prize_liability",
        "my_deck_count",
    ):
        if key in board:
            out[f"board.{key}"] = board.get(key)
    for key in (
        "game_winning_attack_available",
        "guaranteed_ko_available",
        "nonterminal_attack_available",
        "safe_development_available",
        "attachment_unused",
        "retreat_available",
        "end_available",
    ):
        if key in commit:
            out[f"commitment.{key}"] = bool(commit.get(key))
    for key in ("ready_attacker_diff", "energy_dev_diff", "deckout_pressure"):
        if key in value:
            out[f"value.{key}"] = value.get(key)
    return out


def _safety_flags(option: Mapping[str, Any]) -> dict[str, bool]:
    flags = option.get("hard_safety_flags")
    if isinstance(flags, Mapping):
        return {str(key): bool(value) for key, value in flags.items()}
    return {}


def _is_option_zero(option: Mapping[str, Any]) -> bool:
    if bool(option.get("is_source_option0")):
        return True
    source_indexes = option.get("source_option_indexes")
    if isinstance(source_indexes, Sequence) and not isinstance(source_indexes, (str, bytes)):
        return 0 in {item for item in source_indexes if isinstance(item, int) and not isinstance(item, bool)}
    return option.get("source_option_index") == 0


def _semantic_action_key(option: Mapping[str, Any]) -> str:
    existing = option.get("semantic_action_key")
    if isinstance(existing, str) and existing:
        return existing
    features = _mapping(option.get("features")) if "features" in option else _mapping(option)
    fields = {
        "family": _selector_family(option),
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


def _selector_family(option: Mapping[str, Any]) -> str:
    existing = option.get("action_family")
    if isinstance(existing, str) and existing:
        return _normalize_family(existing)
    features = _mapping(option.get("features"))
    type_id = features.get("type_id")
    if features.get("ends_turn") or type_id == 14:
        return "END"
    if features.get("energy_attachment_delta"):
        return "ATTACH"
    if features.get("attack_id") is not None:
        return "ATTACK"
    if features.get("ability_id") is not None or features.get("has_ability_hint"):
        return "ABILITY"
    if features.get("evolves_via_search"):
        return "EVOLVE"
    if features.get("select_context_id") not in (None, 0):
        return "SELECT_CARD"
    if features.get("is_yes") or features.get("is_no"):
        return "YES_NO"
    if features.get("source_zone") == "hand" or type_id == 7:
        return "PLAY"
    mapped = ACTION_TYPE_FAMILIES.get(str(type_id))
    if mapped:
        return _normalize_family(mapped)
    if isinstance(type_id, int) and not isinstance(type_id, bool):
        return f"TYPE_{type_id}"
    return "UNKNOWN"


def _action_family(option: Mapping[str, Any]) -> str:
    features = _mapping(option.get("features"))
    type_id = features.get("type_id")
    if isinstance(type_id, int) and not isinstance(type_id, bool):
        return ACTION_TYPE_FAMILIES.get(str(type_id), f"type_{type_id}")
    return "unknown"


def _normalize_family(family: str) -> str:
    family = family.upper()
    aliases = {
        "END_TURN": "END",
        "NO_OP": "END",
        "CARD": "SELECT_CARD",
        "TUTOR": "SELECT_CARD",
        "ATTACH_ENERGY": "ATTACH",
        "YES_PROMPT": "YES_PROMPT",
        "NO_PROMPT": "NO_PROMPT",
    }
    return aliases.get(family, family)


def _raw_option_index(option: Mapping[str, Any], fallback: int) -> int:
    value = option.get("raw_option_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    features = _mapping(option.get("features"))
    value = features.get("raw_option_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    value = option.get("source_option_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def _target_entity(record: Mapping[str, Any], option_features: Mapping[str, Any]) -> Mapping[str, Any]:
    target_owner = option_features.get("target_owner")
    target_zone = option_features.get("target_zone")
    target_slot = _int_value(option_features.get("target_slot"))
    for entity in _rows(record.get("board_entities")):
        if target_owner is not None and str(entity.get("player_role")) != str(target_owner):
            continue
        if target_zone is not None and str(entity.get("zone")) != str(target_zone):
            continue
        if target_slot is not None and _int_value(entity.get("slot_index")) != target_slot:
            continue
        return entity
    return {}


def _packed_option_index(option: Mapping[str, Any], fallback: int) -> int:
    value = option.get("packed_option_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def _options(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    options = record.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return ()
    return tuple(option for option in options if isinstance(option, Mapping))


def _add_numeric(features: dict[str, float], token: str, value: Any) -> None:
    parsed = _number(value)
    if parsed is not None:
        features[token] = _scaled(parsed)


def _add_bucketed(features: dict[str, float], token: str, value: Any, *, bucket: int) -> None:
    parsed = _number(value)
    if parsed is None:
        return
    if bucket <= 1:
        features[token] = _scaled(parsed)
    else:
        features[f"{token}={int(parsed // bucket) * bucket}"] = 1.0


def _scaled(value: float) -> float:
    if -1.0 <= value <= 1.0:
        return value
    return math.copysign(math.log1p(abs(value)), value)


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        return [1.0 / len(scores) for _ in scores]
    max_score = max(finite)
    values = [math.exp(max(-60.0, min(60.0, score - max_score))) if math.isfinite(score) else 0.0 for score in scores]
    total = sum(values)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in values]


def _ranking(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(prob * math.log(max(EPSILON, prob)) for prob in probabilities if prob > 0.0)


def _top_margin(probabilities: Sequence[float], ranking: Sequence[int]) -> float:
    if len(ranking) < 2:
        return 0.0
    return max(0.0, probabilities[ranking[0]] - probabilities[ranking[1]])


def _rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(row for row in value if isinstance(row, Mapping))


def _int_value(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_forbidden_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in FORBIDDEN_INPUT_KEYS)


def _abstain(status: str, reason: str, model_hash: str | None, selector_hash: str | None) -> dict[str, Any]:
    return {
        "status": status,
        "selected_semantic_key": None,
        "selected_raw_option_index": None,
        "source": "fallback",
        "ranked_actions": [],
        "selector_features": {},
        "selector_score": None,
        "entropy": 0.0,
        "top1_margin": 0.0,
        "support_status": reason,
        "model_hash": model_hash,
        "selector_hash": selector_hash,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()
