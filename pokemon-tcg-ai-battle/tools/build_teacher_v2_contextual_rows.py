"""Build contextual-ranker rows directly from Teacher V2 labels.

This is Branch B's Path B bridge: Teacher V2 labels do not need to pre-exist
inside the old contextual dataset if their replay root state can be recovered.
For each Teacher V2 record, this tool reconstructs the root observation from
source file/step (or uses an embedded observation if present), runs the same
contextual feature path used at inference, and aligns labels by option index,
semantic action key, and eq class/remap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import contextual_ranker as CR  # noqa: E402
import state_action_schema_v2 as SCH  # noqa: E402
import train_contextual_action_ranker as TRAIN  # noqa: E402

DEFAULT_TEACHER_V2 = ROOT / "data" / "manifests" / "teacher_v2_labels_scaled.jsonl"
DEFAULT_DIRECT_OUT = ROOT / "docs" / "workstreams" / "teacher_v2_contextual_scaled_dataset.json"
DEFAULT_MIXED_OUT = ROOT / "docs" / "workstreams" / "contextual_action_ranker_v1_teacher_v2_mixed_dataset.json"
DEFAULT_REPORT = ROOT / "docs" / "workstreams" / "teacher_v2_contextual_scaled_featurization.json"
DEFAULT_BASE_DATASET = ROOT / "docs" / "workstreams" / "contextual_action_ranker_v1_dataset.json"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def canonical_json(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def teacher_options(label: dict) -> list[dict]:
    opts = label.get("options") or label.get("legal_siblings") or label.get("actions") or []
    return opts if isinstance(opts, list) else []


def option_key(opt: dict):
    return tuple(opt.get("semantic_action_key") or opt.get("key") or ())


def option_index(opt: dict) -> int | None:
    raw = opt.get("index", opt.get("option_index"))
    try:
        return int(raw)
    except Exception:
        return None


def option_eq(opt: dict) -> int | None:
    raw = opt.get("eq_class", opt.get("eq"))
    try:
        return int(raw)
    except Exception:
        return None


def get_label_observation(label: dict) -> dict | None:
    for key in ("obs", "observation", "root_obs", "root_observation"):
        obs = label.get(key)
        if isinstance(obs, dict):
            return obs
    return None


def source_identity(label: dict) -> tuple[str | None, int | None]:
    src = label.get("source") if isinstance(label.get("source"), dict) else {}
    file_name = src.get("file") or label.get("game_file")
    step = src.get("step", label.get("step", label.get("call")))
    try:
        step = int(step)
    except Exception:
        step = None
    return file_name, step


def load_replay(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def align_label_to_feat(label: dict, feat: dict) -> dict:
    opts = teacher_options(label)
    fkeys = [tuple(k) for k in (feat.get("keys") or [])]
    feqs = [int(x) for x in (feat.get("eq") or [])]
    counts = Counter()
    mismatches = []
    t_to_f: dict[int, int] = {}
    f_to_t: dict[int, set[int]] = defaultdict(set)
    if len(opts) != len(fkeys):
        mismatches.append({"kind": "option_count", "teacher": len(opts), "feature": len(fkeys)})
    for opt in opts:
        counts["teacher_options"] += 1
        idx = option_index(opt)
        tkey = option_key(opt)
        teq = option_eq(opt)
        in_range = idx is not None and 0 <= idx < len(fkeys)
        counts["index_in_range"] += int(in_range)
        if not in_range:
            mismatches.append({"kind": "index", "index": idx, "teacher_key": list(tkey)})
            continue
        fkey = fkeys[idx]
        feq = feqs[idx]
        semantic_match = tkey == fkey
        counts["semantic_key_match"] += int(semantic_match)
        if not semantic_match:
            mismatches.append({
                "kind": "semantic_key",
                "index": idx,
                "teacher_key": list(tkey),
                "feature_key": list(fkey),
            })
        if teq is not None:
            old = t_to_f.get(teq)
            if old is not None and old != feq:
                mismatches.append({"kind": "eq_remap_conflict", "teacher_eq": teq, "first": old, "second": feq})
            t_to_f[teq] = feq
            f_to_t[feq].add(teq)
        counts["eq_exact_match"] += int(teq == feq)
        counts["eq_remappable"] += int(teq is None or t_to_f.get(teq) == feq)
    for feq, teqs in f_to_t.items():
        if len(teqs) > 1:
            mismatches.append({"kind": "feature_eq_multiple_teacher_eq", "feature_eq": feq, "teacher_eqs": sorted(teqs)})
    return {
        "ok": not mismatches and len(opts) == len(fkeys),
        "counts": dict(counts),
        "teacher_to_feature_eq": t_to_f,
        "mismatches": mismatches[:10],
    }


def average(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def remap_soft(label: dict, t_to_f: dict[int, int]) -> dict[int, float]:
    raw = label.get("soft_policy_target") or label.get("soft_policy") or {}
    out = defaultdict(float)
    for key, value in raw.items():
        try:
            teq = int(key)
            val = float(value)
        except Exception:
            continue
        if teq in t_to_f:
            out[int(t_to_f[teq])] += val
    total = sum(out.values())
    if total <= 0:
        return {}
    return {int(k): float(v / total) for k, v in out.items()}


def remap_acceptable(label: dict, t_to_f: dict[int, int]) -> dict[int, float]:
    raw = label.get("acceptable_action_set") or label.get("acceptable_set") or []
    vals = raw.keys() if isinstance(raw, dict) else raw
    out = {}
    for value in vals:
        try:
            teq = int(value)
        except Exception:
            continue
        if teq in t_to_f:
            out[int(t_to_f[teq])] = 1.0
    return out


def aggregate_option_maps(label: dict, feat: dict) -> dict:
    acc = {
        "adv": defaultdict(list),
        "variance": defaultdict(list),
        "completed": defaultdict(list),
        "outcome": defaultdict(list),
        "outcome_se": defaultdict(list),
    }
    for opt in teacher_options(label):
        idx = option_index(opt)
        if idx is None or idx < 0 or idx >= len(feat["eq"]):
            continue
        feq = int(feat["eq"][idx])
        if opt.get("hand_norm_advantage") is not None:
            acc["adv"][feq].append(float(opt["hand_norm_advantage"]))
        if opt.get("hand_value_variance") is not None:
            acc["variance"][feq].append(float(opt["hand_value_variance"]))
        if opt.get("completed_determinizations") is not None:
            acc["completed"][feq].append(float(opt["completed_determinizations"]))
        if opt.get("outcome_winrate") is not None:
            acc["outcome"][feq].append(float(opt["outcome_winrate"]))
        if opt.get("outcome_se") is not None:
            acc["outcome_se"][feq].append(float(opt["outcome_se"]))
    return {
        "adv": {int(k): average(v) for k, v in acc["adv"].items()},
        "variance": {int(k): average(v) for k, v in acc["variance"].items()},
        "completed": {int(k): average(v) for k, v in acc["completed"].items()},
        "outcome": {int(k): average(v) for k, v in acc["outcome"].items()},
        "outcome_se": {int(k): average(v) for k, v in acc["outcome_se"].items()},
    }


def coverage_weight(label: dict) -> float:
    cov = label.get("coverage") if isinstance(label.get("coverage"), dict) else {}
    n = float(cov.get("n_options") or len(teacher_options(label)) or 0)
    if n <= 0:
        return 0.0
    hand = float(cov.get("determ_full_options", 0) or 0) / n
    out = float(cov.get("outcome_full_options", 0) or 0) / n
    return max(0.0, min(1.0, 0.75 * hand + 0.25 * out))


def criticality_score(label: dict) -> float:
    crit = label.get("criticality") if isinstance(label.get("criticality"), dict) else {}
    return float(crit.get("score", label.get("criticality_score", 0.0)) or 0.0)


def outcome_confidence_by_eq(outcome_se: dict[int, float]) -> dict[int, float]:
    out = {}
    for eq, se in outcome_se.items():
        # 0.05 SE stays useful but weak; high-SE values fade quickly.
        out[int(eq)] = 1.0 / (1.0 + max(0.0, float(se)) / 0.05)
    return out


def row_signature(row: dict) -> tuple:
    return (row.get("game_file"), canonical_json(row.get("keys") or []))


def partition_for(index: int) -> str:
    mod = index % 10
    if mod == 8:
        return "val"
    if mod == 9:
        return "test"
    return "train"


def make_row(label: dict, feat: dict, obs: dict, deck: list[int], game: dict | None,
             source_file: str | None, step: int | None, player: int | None,
             label_index: int, align: dict, args) -> dict | None:
    maps = aggregate_option_maps(label, feat)
    soft = remap_soft(label, align["teacher_to_feature_eq"])
    acceptable = remap_acceptable(label, align["teacher_to_feature_eq"])
    adv = maps["adv"]
    if not soft or not adv or not acceptable:
        return None
    old_eq = TRAIN.old_ranker_eq(obs, deck, feat) if args.old_ranker_baseline else None
    best_adv = max(adv.values()) if adv else 0.0
    old_regret = None
    high_regret = False
    if old_eq is not None and old_eq in adv:
        old_regret = best_adv - float(adv[old_eq])
        high_regret = old_regret >= args.high_regret_threshold
    cov_w = coverage_weight(label)
    crit = criticality_score(label)
    variance_values = list(maps["variance"].values())
    variance_mean = average(variance_values) if variance_values else None
    outcome_conf = outcome_confidence_by_eq(maps["outcome_se"])
    player_name = TRAIN.player_name(game, player) if game is not None and player is not None else None
    rec = {
        "source": args.teacher_source_name,
        "partition": partition_for(label_index),
        "obs_hash": TRAIN.obs_hash(obs),
        "teacher_v2_obs_hash": label.get("obs_hash"),
        "deck_hash": TRAIN.deck_hash(deck),
        "player": player_name,
        "game_file": source_file,
        "game": None,
        "call": step,
        "step": step,
        "turn": (obs.get("current") or {}).get("turn"),
        "turn_action_count": (obs.get("current") or {}).get("turnActionCount"),
        "cids": [int(x) for x in feat["cids"]],
        "dense": feat["dense"],
        "eq": [int(x) for x in feat["eq"]],
        "keys": feat["keys"],
        "soft": {str(k): float(v) for k, v in soft.items()},
        "adv": {str(k): float(v) for k, v in adv.items()},
        "acceptable": {str(k): float(v) for k, v in acceptable.items()},
        "outcome_winrate": {str(k): float(v) for k, v in maps["outcome"].items()},
        "outcome_confidence": {str(k): float(v) for k, v in outcome_conf.items()},
        "outcome_se": {str(k): float(v) for k, v in maps["outcome_se"].items()},
        "teacher_label_agreement": 1.0,
        "teacher_stability": "teacher_v2_covered" if cov_w >= 0.999 else "teacher_v2_partial",
        "teacher_confidence": 0.0,  # filled after variance calibration
        "teacher_applicable_repeats": 1,
        "teacher_not_applicable_repeats": 0,
        "top_two_margin": label.get("top_two_margin"),
        "criticality_score": crit,
        "coverage_weight": cov_w,
        "value_variance_mean": variance_mean,
        "completed_determinizations_mean": average(list(maps["completed"].values())) if maps["completed"] else None,
        "chosen_eq": None,
        "student_eq": old_eq,
        "student_regret": old_regret,
        "high_regret": high_regret,
        "old_ranker_eq": old_eq,
        "option0_eq": int(feat["eq"][0]),
        "weight": 0.0,  # filled after variance calibration
        "teacher_v2_label_index": label_index,
        "teacher_v2_seed": label.get("seed"),
        "teacher_v2_config": label.get("config"),
        "teacher_v2_paired_world": label.get("paired_world"),
        "teacher_v2_hand_argmax_eq_class": label.get("hand_argmax_eq_class"),
        "teacher_v2_outcome_argmax_option": label.get("outcome_argmax_option"),
        "teacher_v2_hand_outcome_agree": label.get("hand_outcome_agree"),
    }
    return rec


def calibrate_weights(rows: list[dict], args) -> None:
    variances = sorted(float(r["value_variance_mean"]) for r in rows if r.get("value_variance_mean") is not None)
    median_var = variances[len(variances) // 2] if variances else 1.0
    median_var = max(1.0, median_var)
    for row in rows:
        var = max(1.0, float(row.get("value_variance_mean") or median_var))
        variance_weight = math.sqrt(median_var / var)
        variance_weight = min(args.max_variance_weight, max(args.min_variance_weight, variance_weight))
        criticality_weight = 0.75 + float(row.get("criticality_score") or 0.0)
        confidence = float(row.get("coverage_weight") or 0.0) * variance_weight
        row["teacher_confidence"] = float(confidence)
        weight = args.teacher_v2_weight * criticality_weight * confidence
        if row.get("high_regret"):
            weight *= args.high_regret_weight_scale
        row["weight"] = float(max(args.min_row_weight, weight))
        if row.get("outcome_confidence"):
            avg_outcome_conf = average(list(row["outcome_confidence"].values()))
            row["outcome_weight"] = float(args.outcome_aux_weight * avg_outcome_conf * float(row.get("coverage_weight") or 0.0))


def find_reconstructed(label: dict, replay_dir: Path) -> tuple[dict | None, list[dict]]:
    file_name, step = source_identity(label)
    embedded = get_label_observation(label)
    attempts = []
    if embedded is not None and (file_name is None or step is None):
        cur = embedded.get("current") or {}
        me = cur.get("yourIndex", 0)
        deck = None
        feat = CR.decision_features(embedded, deck or [])
        attempts.append({"source": "embedded_observation", "feature_built": bool(feat)})
        if feat:
            align = align_label_to_feat(label, feat)
            if align["ok"]:
                return {"obs": embedded, "deck": deck or [], "feat": feat, "align": align, "game": None,
                        "file": None, "step": None, "player": me}, attempts

    if file_name is None or step is None:
        attempts.append({"source": "source_identity", "error": "missing_file_or_step"})
        return None, attempts
    replay_path = replay_dir / file_name
    game = load_replay(replay_path)
    if game is None:
        attempts.append({"source": "replay", "file": file_name, "error": "missing_or_bad_json"})
        return None, attempts
    steps = game.get("steps") or []
    if step < 0 or step >= len(steps) or not isinstance(steps[step], list):
        attempts.append({"source": "replay", "file": file_name, "step": step, "error": "bad_step"})
        return None, attempts
    matches = []
    for player, rec in enumerate(steps[step]):
        if not isinstance(rec, dict):
            continue
        obs = rec.get("observation") or {}
        cur = obs.get("current") or {}
        if not SCH.is_single_pick_decision(obs):
            attempts.append({"player": player, "single_pick": False})
            continue
        me = cur.get("yourIndex", player)
        deck = TRAIN.player_deck(game, me) or TRAIN.player_deck(game, player)
        if not deck:
            attempts.append({"player": player, "single_pick": True, "feature_built": False, "error": "missing_deck"})
            continue
        feat = CR.decision_features(obs, deck)
        if feat is None:
            attempts.append({"player": player, "single_pick": True, "feature_built": False})
            continue
        align = align_label_to_feat(label, feat)
        attempts.append({
            "player": player,
            "single_pick": True,
            "feature_built": True,
            "alignment_ok": align["ok"],
            "counts": align["counts"],
            "mismatches": align["mismatches"],
        })
        if align["ok"]:
            matches.append({"obs": obs, "deck": deck, "feat": feat, "align": align, "game": game,
                            "file": file_name, "step": step, "player": player})
    if len(matches) == 1:
        return matches[0], attempts
    if len(matches) > 1:
        attempts.append({"source": "replay", "error": "ambiguous_reconstruction", "matches": len(matches)})
    return None, attempts


def dataset_summary(rows: list[dict]) -> dict:
    return {
        "total_decisions": len(rows),
        "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
        "by_partition": dict(sorted(Counter(r["partition"] for r in rows).items())),
        "by_teacher_stability": dict(sorted(Counter(r.get("teacher_stability", "unknown") for r in rows).items())),
        "high_regret": sum(1 for r in rows if r.get("high_regret")),
        "players": len({r.get("player") for r in rows if r.get("player")}),
        "decks": len({r.get("deck_hash") for r in rows if r.get("deck_hash")}),
    }


def write_dataset(path: Path, rows: list[dict], config: dict, collection: dict, version: str) -> None:
    payload = {
        "artifact_version": version,
        "config": config,
        "collection": collection,
        "summary": dataset_summary(rows),
        "decisions": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-v2", type=Path, default=DEFAULT_TEACHER_V2)
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    ap.add_argument("--direct-output", type=Path, default=DEFAULT_DIRECT_OUT)
    ap.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE_DATASET)
    ap.add_argument("--mixed-output", type=Path, default=DEFAULT_MIXED_OUT)
    ap.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--teacher-source-name", default="teacher_v2_scaled")
    ap.add_argument("--teacher-v2-weight", type=float, default=1.25)
    ap.add_argument("--outcome-aux-weight", type=float, default=0.35)
    ap.add_argument("--high-regret-threshold", type=float, default=1000.0)
    ap.add_argument("--high-regret-weight-scale", type=float, default=1.5)
    ap.add_argument("--min-variance-weight", type=float, default=0.2)
    ap.add_argument("--max-variance-weight", type=float, default=1.4)
    ap.add_argument("--min-row-weight", type=float, default=0.05)
    ap.add_argument("--old-ranker-baseline", action="store_true", default=True)
    args = ap.parse_args()

    args.teacher_v2 = resolve(args.teacher_v2)
    args.replay_dir = resolve(args.replay_dir)
    args.direct_output = resolve(args.direct_output)
    args.base_dataset = resolve(args.base_dataset)
    args.mixed_output = resolve(args.mixed_output)
    args.report_output = resolve(args.report_output)

    labels = load_jsonl(args.teacher_v2)
    rows = []
    stats = Counter()
    option_counts = Counter()
    failures = []
    for i, label in enumerate(labels):
        stats["teacher_decisions"] += 1
        missing = []
        if not teacher_options(label):
            missing.append("options")
        if not (label.get("soft_policy_target") or label.get("soft_policy")):
            missing.append("soft_policy_target")
        if not (label.get("acceptable_action_set") or label.get("acceptable_set")):
            missing.append("acceptable_action_set")
        for opt in teacher_options(label):
            for field in ("index", "semantic_action_key", "eq_class", "hand_norm_advantage", "hand_value_variance"):
                if opt.get(field) is None:
                    missing.append(f"option.{field}")
            if opt.get("outcome_winrate") is None:
                missing.append("option.outcome_winrate")
            if opt.get("outcome_se") is None:
                missing.append("option.outcome_se")
        if missing:
            stats["missing_required_fields"] += 1
            failures.append({"teacher_index": i, "reason": "missing_fields", "fields": sorted(set(missing))})
            continue
        rec, attempts = find_reconstructed(label, args.replay_dir)
        if rec is None:
            stats["reconstruction_failed"] += 1
            failures.append({"teacher_index": i, "reason": "reconstruction_failed", "attempts": attempts[:6]})
            continue
        row = make_row(
            label,
            rec["feat"],
            rec["obs"],
            rec["deck"],
            rec["game"],
            rec["file"],
            rec["step"],
            rec["player"],
            i,
            rec["align"],
            args,
        )
        if row is None:
            stats["target_translation_failed"] += 1
            failures.append({"teacher_index": i, "reason": "target_translation_failed"})
            continue
        for k, v in rec["align"]["counts"].items():
            option_counts[k] += v
        rows.append(row)
        stats["featurized"] += 1

    calibrate_weights(rows, args)

    config = {
        "teacher_v2": str(args.teacher_v2),
        "replay_dir": str(args.replay_dir),
        "primary_target": "hand_norm_advantage",
        "auxiliary_target": "outcome_winrate_confidence_weighted_by_outcome_se",
        "outcome_argmax_primary": False,
        "feature_path": "agent/contextual_ranker.py decision_features",
        "partition_rule": "index % 10: 8=val, 9=test, otherwise train",
        "teacher_source_name": args.teacher_source_name,
        "teacher_v2_weight": args.teacher_v2_weight,
        "outcome_aux_weight": args.outcome_aux_weight,
    }
    collection = {
        "teacher_v2": dict(stats),
        "option_alignment": dict(option_counts),
    }
    write_dataset(args.direct_output, rows, config, collection, "teacher_v2_contextual_rows.v1")

    mixed_rows = None
    removed_base = 0
    if args.base_dataset.exists():
        base = json.loads(args.base_dataset.read_text(encoding="utf-8"))
        base_rows = list(base.get("decisions") or [])
        new_sigs = {row_signature(r) for r in rows}
        kept_base = [r for r in base_rows if row_signature(r) not in new_sigs]
        removed_base = len(base_rows) - len(kept_base)
        mixed_rows = kept_base + rows
        mixed_config = dict(base.get("config") or {})
        mixed_config.update({
            "teacher_v2_direct_dataset": str(args.direct_output),
            "base_dataset": str(args.base_dataset),
            "base_rows_removed_by_teacher_v2_signature": removed_base,
            "note": "Teacher V2 rows replace old rows with the same replay-file + ordered semantic sibling signature.",
        })
        mixed_collection = dict(base.get("collection") or {})
        mixed_collection["teacher_v2_direct"] = collection
        write_dataset(args.mixed_output, mixed_rows, mixed_config, mixed_collection,
                      "contextual_action_ranker_v1.dataset.teacher_v2_mixed")

    total_options = option_counts.get("teacher_options", 0)
    report = {
        "artifact_version": "teacher_v2_contextual_featurization.v1",
        "branch": "exp/robust-learner-v2",
        "teacher_v2_artifact": str(args.teacher_v2),
        "direct_dataset": str(args.direct_output),
        "mixed_dataset": str(args.mixed_output) if mixed_rows is not None else None,
        "base_dataset": str(args.base_dataset) if args.base_dataset.exists() else None,
        "teacher_v2_decisions_loaded": len(labels),
        "featurized_decisions": len(rows),
        "training_ready": len(rows) == len(labels) and not failures,
        "missing_fields_failures": stats.get("missing_required_fields", 0),
        "reconstruction_failures": stats.get("reconstruction_failed", 0),
        "target_translation_failures": stats.get("target_translation_failed", 0),
        "option_level_alignment": {
            "teacher_options": total_options,
            "index_alignment_rate": option_counts.get("index_in_range", 0) / max(1, total_options),
            "semantic_key_alignment_rate": option_counts.get("semantic_key_match", 0) / max(1, total_options),
            "eq_exact_match_rate": option_counts.get("eq_exact_match", 0) / max(1, total_options),
            "eq_remappable_rate": option_counts.get("eq_remappable", 0) / max(1, total_options),
            "counts": dict(option_counts),
        },
        "direct_summary": dataset_summary(rows),
        "mixed_summary": dataset_summary(mixed_rows) if mixed_rows is not None else None,
        "base_rows_removed_by_teacher_v2_signature": removed_base,
        "failures": failures[:20],
        "live_agent_consumed": "none; offline replay reconstruction only",
        "baseline": "agent_search",
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "teacher_v2_decisions_loaded": report["teacher_v2_decisions_loaded"],
        "featurized_decisions": report["featurized_decisions"],
        "training_ready": report["training_ready"],
        "option_level_alignment": report["option_level_alignment"],
        "direct_dataset": str(args.direct_output.relative_to(ROOT)),
        "mixed_dataset": str(args.mixed_output.relative_to(ROOT)) if mixed_rows is not None else None,
        "report": str(args.report_output.relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
