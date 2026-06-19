"""Align Teacher V2 decision labels to Contextual Action Ranker rows.

This is a preparation tool for Branch B. It does not train a model and it does
not treat missing/placeholder Teacher V2 labels as evidence. A usable Teacher
V2 artifact must align each root decision and each legal sibling action to the
contextual dataset by decision id / obs_hash plus option index and semantic key.

Accepted Teacher V2 shapes:

  * JSON with a top-level `decisions`, `records`, or `labels` list;
  * JSONL with one decision record per line.

Expected per-decision fields, with tolerant aliases:

  decision_id/id/root_decision_id, obs_hash, legal_siblings/options/actions,
  hand_soft_policy/soft_policy, acceptable_set/acceptable_action_set,
  criticality_score, coverage/timing/determinization metadata.

Expected per-option fields:

  option_index/index, semantic_action_key/key, eq_class,
  hand_mean_value, hand_value_variance, hand_norm_advantage,
  outcome_winrate, outcome_playouts, outcome_variance/confidence.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "docs" / "workstreams" / "contextual_action_ranker_v1_dataset.json"
DEFAULT_OUTPUT = ROOT / "docs" / "workstreams" / "teacher_v2_alignment_check.json"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def canonical_json(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def load_contextual_dataset(path: Path) -> list[dict]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = blob.get("decisions")
    if not isinstance(rows, list):
        raise SystemExit(f"contextual dataset has no decisions list: {path}")
    return rows


def load_teacher(path: Path | None) -> list[dict]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    blob = json.loads(text)
    if isinstance(blob, list):
        return blob
    for key in ("decisions", "records", "labels"):
        if isinstance(blob.get(key), list):
            return blob[key]
    raise SystemExit(f"Teacher V2 artifact has no decisions/records/labels list: {path}")


def get_any(d: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in d:
            return d[name]
    return default


def norm_key(key) -> tuple:
    if key is None:
        return ()
    if isinstance(key, str):
        try:
            key = json.loads(key)
        except Exception:
            return (key,)
    if isinstance(key, (list, tuple)):
        return tuple(key)
    return (key,)


def key_string(key) -> str:
    return canonical_json(list(norm_key(key)))


def contextual_decision_ids(d: dict) -> list[str]:
    ids = []
    if d.get("decision_id"):
        ids.append(str(d["decision_id"]))
    if d.get("id"):
        ids.append(str(d["id"]))
    if d.get("obs_hash"):
        ids.append("obs:" + str(d["obs_hash"]))
    if d.get("game_file") is not None:
        step = d.get("step", d.get("call", ""))
        ids.append(f"replay:{d.get('game_file')}:{step}")
    if d.get("game") is not None and d.get("call") is not None:
        ids.append(f"recovery:{d.get('source')}:{d.get('game')}:{d.get('call')}")
    return ids


def teacher_decision_ids(d: dict) -> list[str]:
    ids = []
    for name in ("decision_id", "id", "root_decision_id"):
        if d.get(name) is not None:
            ids.append(str(d[name]))
    if d.get("obs_hash") is not None:
        ids.append("obs:" + str(d["obs_hash"]))
    if d.get("game_file") is not None:
        step = d.get("step", d.get("call", ""))
        ids.append(f"replay:{d.get('game_file')}:{step}")
    if d.get("game") is not None and d.get("call") is not None:
        src = d.get("source", d.get("student", "unknown"))
        ids.append(f"recovery:{src}:{d.get('game')}:{d.get('call')}")
    return ids


def contextual_index(rows: list[dict]) -> tuple[dict[str, int], list[dict]]:
    idx = {}
    collisions = []
    for i, d in enumerate(rows):
        for did in contextual_decision_ids(d):
            if did in idx and idx[did] != i:
                collisions.append({"decision_id": did, "first": idx[did], "second": i})
            else:
                idx[did] = i
    return idx, collisions


def teacher_options(d: dict) -> list[dict]:
    opts = get_any(d, ("legal_siblings", "options", "actions", "siblings"), [])
    return opts if isinstance(opts, list) else []


def soft_policy(d: dict) -> dict:
    raw = get_any(d, ("hand_soft_policy", "soft_policy", "soft_policy_target"), {})
    return raw if isinstance(raw, dict) else {}


def acceptable_set(d: dict):
    raw = get_any(d, ("acceptable_set", "acceptable_action_set", "hand_acceptable_set"), [])
    if isinstance(raw, dict):
        return set(raw.keys())
    if isinstance(raw, list):
        return set(raw)
    return set()


def option_field_presence(opt: dict) -> dict[str, bool]:
    return {
        "hand_mean_value": get_any(opt, ("hand_mean_value", "mean_value")) is not None,
        "hand_value_variance": get_any(opt, ("hand_value_variance", "value_variance")) is not None,
        "hand_norm_advantage": get_any(opt, ("hand_norm_advantage", "hand_normalized_advantage", "normalized_advantage")) is not None,
        "outcome_winrate": get_any(opt, ("outcome_winrate", "terminal_outcome_winrate")) is not None,
        "outcome_playouts": get_any(opt, ("outcome_playouts", "terminal_outcome_playouts")) is not None,
        "outcome_variance": get_any(opt, ("outcome_variance", "outcome_confidence", "terminal_outcome_variance")) is not None,
    }


def compare_decision(trec: dict, crec: dict) -> dict:
    opts = teacher_options(trec)
    ckeys = [norm_key(k) for k in (crec.get("keys") or [])]
    ceq = [int(x) for x in (crec.get("eq") or [])]
    option_rows = []
    counts = Counter()
    eq_by_key = {}
    for j, key in enumerate(ckeys):
        eq_by_key.setdefault(key_string(key), ceq[j] if j < len(ceq) else None)

    for opt in opts:
        raw_i = get_any(opt, ("option_index", "index"))
        try:
            oi = int(raw_i)
        except Exception:
            oi = None
        tkey = norm_key(get_any(opt, ("semantic_action_key", "key")))
        teq_raw = get_any(opt, ("eq_class", "eq"))
        teq = int(teq_raw) if isinstance(teq_raw, int) or (isinstance(teq_raw, str) and teq_raw.isdigit()) else None
        row = {
            "option_index": oi,
            "teacher_key": list(tkey),
            "teacher_eq_class": teq,
            "index_in_range": oi is not None and 0 <= oi < len(ckeys),
            "semantic_key_match": False,
            "eq_match_or_remappable": False,
            "fields": option_field_presence(opt),
        }
        if row["index_in_range"]:
            ckey = ckeys[oi]
            row["contextual_key"] = list(ckey)
            row["contextual_eq_class"] = ceq[oi] if oi < len(ceq) else None
            row["semantic_key_match"] = tkey == ckey
            row["eq_match_or_remappable"] = (
                teq is None
                or teq == row["contextual_eq_class"]
                or eq_by_key.get(key_string(tkey)) == row["contextual_eq_class"]
            )
        for k, v in row["fields"].items():
            if v:
                counts[f"field_{k}"] += 1
        counts["options"] += 1
        counts["index_in_range"] += int(row["index_in_range"])
        counts["semantic_key_match"] += int(row["semantic_key_match"])
        counts["eq_match_or_remappable"] += int(row["eq_match_or_remappable"])
        option_rows.append(row)

    return {
        "n_teacher_options": len(opts),
        "n_contextual_options": len(ckeys),
        "option_counts": dict(counts),
        "all_option_indices_match": counts["index_in_range"] == len(opts) and len(opts) == len(ckeys),
        "all_semantic_keys_match": counts["semantic_key_match"] == len(opts) and len(opts) == len(ckeys),
        "all_eq_match_or_remappable": counts["eq_match_or_remappable"] == len(opts) and len(opts) == len(ckeys),
        "has_soft_policy": bool(soft_policy(trec)),
        "has_acceptable_set": bool(acceptable_set(trec)),
        "has_criticality": get_any(trec, ("criticality_score", "criticality")) is not None,
        "has_coverage_metadata": any(k in trec for k in ("coverage", "candidate_coverage", "all_options_completed", "actual_search_time")),
        "has_seed_metadata": any(k in trec for k in ("seed", "seeds", "paired_seed", "seed_info")),
        "sample_option_mismatches": [
            row for row in option_rows
            if not (row["index_in_range"] and row["semantic_key_match"] and row["eq_match_or_remappable"])
        ][:5],
    }


def summarize_alignment(teacher_rows: list[dict], contextual_rows: list[dict]) -> dict:
    cidx, collisions = contextual_index(contextual_rows)
    matched = []
    unmatched_teacher = []
    decision_reports = []
    for i, trec in enumerate(teacher_rows):
        dids = teacher_decision_ids(trec)
        match_i = next((cidx[did] for did in dids if did in cidx), None)
        if match_i is None:
            unmatched_teacher.append({"teacher_index": i, "ids": dids[:5]})
            continue
        comp = compare_decision(trec, contextual_rows[match_i])
        comp["teacher_index"] = i
        comp["contextual_index"] = match_i
        comp["matched_ids"] = [did for did in dids if did in cidx]
        matched.append(comp)
        if not (comp["all_option_indices_match"] and comp["all_semantic_keys_match"] and comp["all_eq_match_or_remappable"]):
            decision_reports.append(comp)

    totals = Counter()
    usable_primary = 0
    usable_outcome = 0
    for comp in matched:
        totals["teacher_options"] += comp["n_teacher_options"]
        for k, v in comp["option_counts"].items():
            totals[k] += v
        if comp["all_semantic_keys_match"] and comp["option_counts"].get("field_hand_norm_advantage", 0) == comp["n_teacher_options"]:
            usable_primary += 1
        if comp["all_semantic_keys_match"] and comp["option_counts"].get("field_outcome_winrate", 0) == comp["n_teacher_options"]:
            usable_outcome += 1

    return {
        "status": "implemented_loader_no_teacher_rows" if not teacher_rows else "alignment_checked",
        "teacher_decisions": len(teacher_rows),
        "contextual_decisions": len(contextual_rows),
        "matched_decisions": len(matched),
        "unmatched_teacher_decisions": len(unmatched_teacher),
        "contextual_id_collisions": collisions[:20],
        "option_totals": dict(totals),
        "usable_primary_hand_advantage_decisions": usable_primary,
        "usable_outcome_aux_decisions": usable_outcome,
        "alignment_ready_for_training": bool(teacher_rows) and len(matched) == len(teacher_rows) and not decision_reports,
        "sample_unmatched_teacher": unmatched_teacher[:20],
        "sample_alignment_mismatches": decision_reports[:20],
    }


def required_schema() -> dict:
    return {
        "decision_required": [
            "decision_id or obs_hash",
            "legal_siblings/options/actions",
            "hand_soft_policy or soft_policy",
            "acceptable_set",
            "criticality_score",
            "coverage/timing/determinization metadata",
        ],
        "option_required": [
            "option_index",
            "semantic_action_key",
            "eq_class",
            "hand_mean_value",
            "hand_value_variance",
            "hand_norm_advantage",
        ],
        "option_auxiliary": [
            "outcome_winrate",
            "outcome_playouts",
            "outcome_variance or outcome_confidence",
        ],
        "primary_target_rule": "hand_norm_advantage weighted by criticality and inverse variance/stability",
        "outcome_rule": "outcome_winrate is auxiliary only unless Model A validates higher-k reliability",
    }


def self_test_rows() -> tuple[list[dict], list[dict]]:
    contextual = [{
        "obs_hash": "abc",
        "keys": [[7, 100, None, None, None], [13, 200, 300, None, None]],
        "eq": [0, 1],
    }]
    teacher = [{
        "decision_id": "obs:abc",
        "obs_hash": "abc",
        "criticality_score": 2.5,
        "hand_soft_policy": {"0": 0.25, "1": 0.75},
        "acceptable_set": [1],
        "candidate_coverage": 1.0,
        "seeds": [1, 2],
        "options": [
            {
                "option_index": 0,
                "semantic_action_key": [7, 100, None, None, None],
                "eq_class": 0,
                "hand_mean_value": 1.0,
                "hand_value_variance": 0.1,
                "hand_norm_advantage": -0.5,
                "outcome_winrate": 0.4,
                "outcome_playouts": 32,
                "outcome_variance": 0.2,
            },
            {
                "option_index": 1,
                "semantic_action_key": [13, 200, 300, None, None],
                "eq_class": 1,
                "hand_mean_value": 2.0,
                "hand_value_variance": 0.2,
                "hand_norm_advantage": 0.5,
                "outcome_winrate": 0.6,
                "outcome_playouts": 32,
                "outcome_variance": 0.2,
            },
        ],
    }]
    return teacher, contextual


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contextual-dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--teacher-v2", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--self-test", action="store_true", help="run an internal positive alignment fixture")
    args = ap.parse_args()

    if args.self_test:
        teacher_rows, contextual_rows = self_test_rows()
        dataset_path = None
        teacher_path = None
    else:
        contextual_rows = load_contextual_dataset(resolve(args.contextual_dataset))
        teacher_rows = load_teacher(resolve(args.teacher_v2)) if args.teacher_v2 else []
        dataset_path = str(resolve(args.contextual_dataset))
        teacher_path = str(resolve(args.teacher_v2)) if args.teacher_v2 else None

    report = {
        "artifact_version": "teacher_v2_alignment_check.1",
        "branch": "exp/robust-learner-v2",
        "contextual_dataset": dataset_path,
        "teacher_v2_artifact": teacher_path,
        "self_test": bool(args.self_test),
        "required_schema": required_schema(),
        "alignment": summarize_alignment(teacher_rows, contextual_rows),
        "live_agent_consumed": "none; this tool is offline preparation only",
        "status": "implemented" if args.self_test or not teacher_rows else "alignment_checked",
    }
    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "teacher_decisions": report["alignment"]["teacher_decisions"],
        "matched_decisions": report["alignment"]["matched_decisions"],
        "alignment_ready_for_training": report["alignment"]["alignment_ready_for_training"],
        "output": str(out.relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
