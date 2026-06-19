"""Branch B / B1.1 -- representation-ceiling audit.

This is the first Robust Learner V2 diagnostic. It asks whether the old
compressed state representation can even support stable Teacher V1 action
labels before we build a larger student.

The script is deliberately manifest-aware: it reads only replay files named by
the frozen split and verifies their hashes against the immutable manifest. That
keeps the rolling replay downloader from silently changing this experiment.

Examples:

    python tools/audit_representation_ceiling.py \
        --replay-dir C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/pokemon-tcg-ai-battle/data/external/replays \
        --partition train --max-decisions 24 --repeats 3 --n-determ 4

    python tools/audit_representation_ceiling.py \
        --replay-dir C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/pokemon-tcg-ai-battle/data/external/replays \
        --partition val --max-decisions 64 --output docs/workstreams/robust_learner_v2_b1_1_val.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))

import main as M  # noqa: E402
import state_action_schema_v2 as SCH  # noqa: E402
import teacher_api_v1 as T  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "replays_20260618.json"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "replays_20260618_split.json"
MARGIN_BINS = [1.0, 10.0, 100.0, 1000.0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_snapshot(manifest_path: Path, split_path: Path, partition: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("manifest") != manifest_path.name:
        raise SystemExit(f"split {split_path} points at {split.get('manifest')}, not {manifest_path.name}")
    wanted = set(split.get(partition) or [])
    if not wanted:
        raise SystemExit(f"partition {partition!r} is empty or missing in {split_path}")
    rows = []
    by_name = {f["file"]: f for f in manifest.get("files", [])}
    for name in sorted(wanted):
        rec = by_name.get(name)
        if not rec:
            raise SystemExit(f"{name} is in split but missing from manifest")
        if rec.get("skipped_reason"):
            continue
        rows.append(rec)
    return rows


def verify_replay(path: Path, rec: dict) -> None:
    if not path.exists():
        raise SystemExit(f"missing replay from frozen snapshot: {path}")
    actual = sha256_file(path)
    expected = rec.get("sha256")
    if actual != expected:
        raise SystemExit(f"hash mismatch for {path.name}: manifest {expected}, actual {actual}")


def player_deck(game: dict, player: int) -> list[int] | None:
    for step in game.get("steps", []):
        if player < len(step) and isinstance(step[player], dict):
            action = step[player].get("action")
            if isinstance(action, list) and len(action) == 60:
                return action
    return None


def iter_decisions(game: dict, file_name: str) -> Iterable[dict]:
    for step_i, step in enumerate(game.get("steps", [])):
        if not isinstance(step, list):
            continue
        for agent_i, agent_step in enumerate(step):
            if not isinstance(agent_step, dict):
                continue
            obs = agent_step.get("observation") or {}
            if not SCH.is_single_pick_decision(obs):
                continue
            opts = ((obs.get("select") or {}).get("option") or [])
            if len(opts) < 2:
                continue
            cur = obs.get("current") or {}
            if not cur.get("players"):
                continue
            me = cur.get("yourIndex", agent_i)
            deck = player_deck(game, me)
            if not deck:
                continue
            yield {"file": file_name, "step": step_i, "agent": agent_i, "obs": obs, "deck": deck}


def _soft_lookup(soft: dict, key: int) -> float:
    return float(soft.get(key, soft.get(str(key), 0.0)) or 0.0)


def _repeat_metrics(result: dict, label: tuple) -> dict:
    opts = result.get("options") or []
    chosen = result.get("chosen_option")
    opt = opts[chosen] if chosen is not None and chosen < len(opts) else {}
    eq_class = opt.get("eq_class")
    acceptable = set(result.get("acceptable_action_set") or [])
    return {
        "label": label,
        "action_type": label[0] if label else None,
        "eq_class": eq_class,
        "top_two_margin": result.get("top_two_margin"),
        "mean_value": opt.get("mean_value"),
        "value_variance": opt.get("value_variance"),
        "completed_determinizations": opt.get("completed_determinizations"),
        "normalized_advantage": opt.get("normalized_advantage"),
        "acceptable_action_count": len(acceptable),
        "chosen_in_acceptable_set": eq_class in acceptable if eq_class is not None else False,
        "soft_policy_for_chosen": _soft_lookup(result.get("soft_policy_target") or {}, eq_class),
    }


def _num_mean(xs: list) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def teacher_label_repeats(decision: dict, repeats: int, n_determ: int, time_budget: float,
                          seed0: int, accept_z: float) -> dict:
    labels = []
    margins = []
    forced = []
    repeat_metrics = []
    elapsed = 0.0
    last = None
    for r in range(repeats):
        result = T.query(
            decision["obs"],
            decision["deck"],
            n_determ=n_determ,
            time_budget=time_budget,
            leaf_mode="hand",
            seed=seed0 + r,
            accept_z=accept_z,
        )
        elapsed += float(result.get("elapsed_s") or 0.0)
        if not result.get("applicable"):
            return {
                "status": "not_applicable",
                "elapsed_s": elapsed,
                "reason": "teacher_api_not_applicable",
                "last": result,
            }
        last = result
        opts = result.get("options") or []
        chosen = result.get("chosen_option")
        if chosen is None or chosen >= len(opts):
            return {
                "status": "not_applicable",
                "elapsed_s": elapsed,
                "reason": "no_chosen_option",
                "last": result,
            }
        label = tuple(opts[chosen]["semantic_action_key"])
        labels.append(label)
        margins.append(result.get("top_two_margin"))
        forced.append(bool(result.get("forced_action_flag")))
        repeat_metrics.append(_repeat_metrics(result, label))
    counts = Counter(labels)
    label, n_label = counts.most_common(1)[0]
    agreement = n_label / max(1, repeats)
    numeric_margins = [float(m) for m in margins if m is not None]
    majority_metrics = [m for m in repeat_metrics if m["label"] == label]
    return {
        "status": "labelled",
        "label": label,
        "agreement": agreement,
        "label_counts": counts,
        "label_entropy_bits": entropy(counts),
        "mean_margin": sum(numeric_margins) / len(numeric_margins) if numeric_margins else None,
        "forced_any": any(forced),
        "elapsed_s": elapsed,
        "repeat_metrics": repeat_metrics,
        "mean_value_variance": _num_mean([m["value_variance"] for m in majority_metrics]),
        "mean_completed_determinizations": _num_mean([m["completed_determinizations"] for m in majority_metrics]),
        "mean_acceptable_action_count": _num_mean([m["acceptable_action_count"] for m in repeat_metrics]),
        "chosen_in_acceptable_rate": _num_mean([1.0 if m["chosen_in_acceptable_set"] else 0.0
                                                for m in repeat_metrics]),
        "mean_soft_policy_for_chosen": _num_mean([m["soft_policy_for_chosen"] for m in repeat_metrics]),
        "last": last,
    }


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for n in counts.values():
        p = n / total
        out -= p * math.log2(p)
    return out


def canonical_json(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def label_to_json_key(label: tuple) -> str:
    return canonical_json(list(label))


def margin_bin(margin) -> str:
    if margin is None:
        return "none"
    x = abs(float(margin))
    for upper in MARGIN_BINS:
        if x < upper:
            return f"<{upper:g}"
    return f">={MARGIN_BINS[-1]:g}"


def state_fingerprint(obs: dict) -> str:
    payload = {"current": obs.get("current"), "select": obs.get("select")}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def feature_keys(obs: dict, result: dict) -> list[dict]:
    cur = obs.get("current") or {}
    me = result["me"]
    root = tuple(SCH.encode_vector(obs))
    rows = []
    for cls in result.get("eq_classes") or []:
        if cls.get("mean_value") is None:
            continue
        key = tuple(cls["key"])
        members = cls.get("members") or []
        opt = ((obs.get("select") or {}).get("option") or [])[members[0]] if members else {}
        desc = SCH.action_descriptor(opt, cur, me) if isinstance(opt, dict) else {}
        rows.append({
            "eq_class": cls["eq_class"],
            "semantic_key": key,
            "root_only": root,
            "root_plus_type": root + (key[0],),
            "root_plus_action": root + (canonical_json(desc),),
            "root_plus_semantic_key": root + key,
        })
    return rows


def memorizer_expected_top1(decisions: list[dict], feature_name: str) -> tuple[float, float]:
    label_rate = defaultdict(lambda: [0, 0])
    for d in decisions:
        for row in d["rows"]:
            bucket = label_rate[row[feature_name]]
            bucket[1] += 1
            if row["semantic_key"] == d["label"]:
                bucket[0] += 1

    expected = 0.0
    optimistic = 0.0
    total = 0
    for d in decisions:
        scored = []
        for row in d["rows"]:
            pos, n = label_rate[row[feature_name]]
            scored.append((pos / n if n else 0.0, row["semantic_key"]))
        if not scored:
            continue
        best = max(s for s, _ in scored)
        tied = [k for s, k in scored if s == best]
        total += 1
        if d["label"] in tied:
            optimistic += 1.0
            expected += 1.0 / len(tied)
    if total == 0:
        return 0.0, 0.0
    return expected / total, optimistic / total


def summarize_stable_decisions(decisions: list[dict]) -> dict:
    by_root_states = defaultdict(set)
    labels_by_root = defaultdict(Counter)
    for d in decisions:
        by_root_states[d["root"]].add(d["obs_hash"])
        labels_by_root[d["root"]][d["label"]] += 1

    collision_roots = {k: v for k, v in by_root_states.items() if len(v) > 1}
    entropies = [entropy(c) for c in labels_by_root.values()]
    weighted_entropy = (
        sum(entropy(c) * sum(c.values()) for c in labels_by_root.values()) / len(decisions)
        if decisions else 0.0
    )
    memorizer = {}
    for name in ["root_only", "root_plus_type", "root_plus_action", "root_plus_semantic_key"]:
        expected, optimistic = memorizer_expected_top1(decisions, name)
        memorizer[name] = {"expected_top1": expected, "optimistic_top1": optimistic}
    return {
        "representation": {
            "distinct_compressed_roots": len(by_root_states),
            "roots_containing_multiple_raw_states": len(collision_roots),
            "raw_states_inside_collided_roots": sum(len(v) for v in collision_roots.values()),
            "teacher_label_entropy_root_mean_bits": (sum(entropies) / len(entropies)) if entropies else 0.0,
            "teacher_label_entropy_root_weighted_bits": weighted_entropy,
        },
        "memorizer_ceiling": memorizer,
    }


def jsonable_decision(d: dict) -> dict:
    return {
        "id": d["id"],
        "obs_hash": d["obs_hash"],
        "label": list(d["label"]),
        "agreement": d["agreement"],
        "mean_margin": d["mean_margin"],
        "forced_any": d["forced_any"],
        "elapsed_s": d["elapsed_s"],
        "n_eq_classes": len(d["rows"]),
    }


def write_output(path: Path, payload: dict) -> None:
    path = path if path.is_absolute() else ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def stability_record(decision: dict, label: dict, accepted: bool) -> dict:
    return {
        "id": f"{decision['file']}:{decision['step']}:{decision['agent']}",
        "accepted_stable": accepted,
        "agreement": label["agreement"],
        "majority_label": list(label["label"]),
        "majority_action_type": str(label["label"][0]),
        "label_counts": {label_to_json_key(k): v for k, v in label["label_counts"].items()},
        "label_entropy_bits": label["label_entropy_bits"],
        "mean_margin": label["mean_margin"],
        "margin_bin": margin_bin(label["mean_margin"]),
        "forced_any": label["forced_any"],
        "elapsed_s": label["elapsed_s"],
        "mean_value_variance": label["mean_value_variance"],
        "mean_completed_determinizations": label["mean_completed_determinizations"],
        "mean_acceptable_action_count": label["mean_acceptable_action_count"],
        "chosen_in_acceptable_rate": label["chosen_in_acceptable_rate"],
        "mean_soft_policy_for_chosen": label["mean_soft_policy_for_chosen"],
    }


def _rate_row(counts: dict) -> dict:
    stable = counts.get("stable", 0)
    unstable = counts.get("unstable", 0)
    total = stable + unstable
    return {
        "stable": stable,
        "unstable": unstable,
        "total": total,
        "stable_rate": stable / total if total else 0.0,
    }


def summarize_stability(records: list[dict]) -> dict:
    by_action_type = defaultdict(lambda: {"stable": 0, "unstable": 0})
    by_margin_bin = defaultdict(lambda: {"stable": 0, "unstable": 0})
    by_action_type_margin = defaultdict(lambda: {"stable": 0, "unstable": 0})
    margins = defaultdict(list)
    variances = defaultdict(list)
    acceptable_counts = defaultdict(list)
    label_entropies = defaultdict(list)
    completed = defaultdict(list)
    for rec in records:
        status = "stable" if rec["accepted_stable"] else "unstable"
        action_type = rec["majority_action_type"]
        mb = rec["margin_bin"]
        by_action_type[action_type][status] += 1
        by_margin_bin[mb][status] += 1
        by_action_type_margin[f"{action_type}|{mb}"][status] += 1
        for k, v in [
            ("stable" if rec["accepted_stable"] else "unstable", rec["mean_margin"]),
            ("all", rec["mean_margin"]),
        ]:
            if v is not None:
                margins[k].append(float(v))
        if rec["mean_value_variance"] is not None:
            variances[status].append(float(rec["mean_value_variance"]))
            variances["all"].append(float(rec["mean_value_variance"]))
        if rec["mean_acceptable_action_count"] is not None:
            acceptable_counts[status].append(float(rec["mean_acceptable_action_count"]))
            acceptable_counts["all"].append(float(rec["mean_acceptable_action_count"]))
        if rec["mean_completed_determinizations"] is not None:
            completed[status].append(float(rec["mean_completed_determinizations"]))
            completed["all"].append(float(rec["mean_completed_determinizations"]))
        label_entropies[status].append(float(rec["label_entropy_bits"]))
        label_entropies["all"].append(float(rec["label_entropy_bits"]))

    def means(d):
        return {k: (sum(v) / len(v) if v else None) for k, v in sorted(d.items())}

    return {
        "by_action_type": {k: _rate_row(v) for k, v in sorted(by_action_type.items())},
        "by_margin_bin": {k: _rate_row(v) for k, v in sorted(by_margin_bin.items())},
        "by_action_type_margin_bin": {k: _rate_row(v) for k, v in sorted(by_action_type_margin.items())},
        "mean_margin": means(margins),
        "mean_value_variance": means(variances),
        "mean_acceptable_action_count": means(acceptable_counts),
        "mean_completed_determinizations": means(completed),
        "mean_label_entropy_bits": means(label_entropies),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    ap.add_argument("--partition", choices=["train", "val", "test"], default="train")
    ap.add_argument("--max-files", type=int, default=80)
    ap.add_argument("--max-decisions", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-determ", type=int, default=4)
    ap.add_argument("--time-budget", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--accept-z", type=float, default=1.0)
    ap.add_argument("--min-agreement", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=None, help="optional JSON result path, relative to project root")
    ap.add_argument("--include-decisions", action="store_true", help="include accepted decision IDs/labels in JSON output")
    args = ap.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    records = load_snapshot(args.manifest, args.split, args.partition)
    rng = random.Random(args.seed)
    records = records[:args.max_files]

    audited = []
    scanned = 0
    not_applicable = 0
    skipped_unstable = 0
    skipped_no_rows = 0
    agreement_hist = Counter()
    unstable_examples = []
    stability_records = []
    for rec in records:
        path = args.replay_dir / rec["file"]
        verify_replay(path, rec)
        game = json.loads(path.read_text(encoding="utf-8"))
        for decision in iter_decisions(game, rec["file"]):
            scanned += 1
            label = teacher_label_repeats(
                decision,
                repeats=args.repeats,
                n_determ=args.n_determ,
                time_budget=args.time_budget,
                seed0=rng.randrange(1_000_000_000),
                accept_z=args.accept_z,
            )
            if label["status"] != "labelled":
                not_applicable += 1
                continue
            agreement_hist[f"{label['agreement']:.3f}"] += 1
            accepted = label["agreement"] >= args.min_agreement
            stability_records.append(stability_record(decision, label, accepted))
            if not accepted:
                skipped_unstable += 1
                if len(unstable_examples) < 12:
                    unstable_examples.append({
                        "id": f"{decision['file']}:{decision['step']}:{decision['agent']}",
                        "agreement": label["agreement"],
                        "label_counts": {label_to_json_key(k): v for k, v in label["label_counts"].items()},
                        "label_entropy_bits": label["label_entropy_bits"],
                        "mean_margin": label["mean_margin"],
                        "margin_bin": margin_bin(label["mean_margin"]),
                        "majority_action_type": str(label["label"][0]),
                        "mean_value_variance": label["mean_value_variance"],
                        "mean_acceptable_action_count": label["mean_acceptable_action_count"],
                        "forced_any": label["forced_any"],
                    })
                continue
            rows = feature_keys(decision["obs"], label["last"])
            if not rows:
                skipped_no_rows += 1
                continue
            audited.append({
                "id": f"{decision['file']}:{decision['step']}:{decision['agent']}",
                "obs_hash": state_fingerprint(decision["obs"]),
                "root": tuple(SCH.encode_vector(decision["obs"])),
                "label": label["label"],
                "agreement": label["agreement"],
                "mean_margin": label["mean_margin"],
                "forced_any": label["forced_any"],
                "elapsed_s": label["elapsed_s"],
                "rows": rows,
            })
            if len(audited) >= args.max_decisions:
                break
        if len(audited) >= args.max_decisions:
            break

    if not audited:
        raise SystemExit("no stable teacher-labelled decisions collected")

    stable_summary = summarize_stable_decisions(audited)
    stability_summary = summarize_stability(stability_records)
    rep = stable_summary["representation"]
    memorizer = stable_summary["memorizer_ceiling"]
    labelled = len(audited) + skipped_unstable + skipped_no_rows
    payload = {
        "audit_version": "branch_b_b1_2.0",
        "config": {
            "manifest": str(args.manifest.relative_to(ROOT) if args.manifest.is_relative_to(ROOT) else args.manifest),
            "split": str(args.split.relative_to(ROOT) if args.split.is_relative_to(ROOT) else args.split),
            "replay_dir": str(args.replay_dir),
            "partition": args.partition,
            "max_files": args.max_files,
            "max_decisions": args.max_decisions,
            "repeats": args.repeats,
            "n_determ": args.n_determ,
            "time_budget": args.time_budget,
            "seed": args.seed,
            "accept_z": args.accept_z,
            "min_agreement": args.min_agreement,
        },
        "summary": {
            "scanned_decisions": scanned,
            "teacher_labelled_decisions": labelled,
            "accepted_stable_decisions": len(audited),
            "skipped_unstable_decisions": skipped_unstable,
            "not_applicable_decisions": not_applicable,
            "skipped_no_rows": skipped_no_rows,
            "forced_any_stable": sum(1 for d in audited if d["forced_any"]),
            "agreement_histogram": dict(sorted(agreement_hist.items())),
            "unstable_examples": unstable_examples,
        },
        "stability": stability_summary,
        **stable_summary,
    }
    if args.include_decisions:
        payload["accepted_decisions"] = [jsonable_decision(d) for d in audited]
        payload["labelled_decisions"] = stability_records

    print("Branch B / B1.1-B1.2 representation + teacher-stability audit")
    print(f"  manifest        : {args.manifest.relative_to(ROOT)}")
    print(f"  split partition : {args.partition}")
    print(f"  replay dir      : {args.replay_dir}")
    print(f"  scanned         : {scanned} decisions")
    print(f"  teacher labelled: {labelled} decisions")
    print(f"  audited stable  : {len(audited)} decisions")
    print(f"  skipped unstable: {skipped_unstable}")
    print(f"  not applicable  : {not_applicable}")
    print(f"  teacher query   : repeats={args.repeats}, n_determ={args.n_determ}, "
          f"time_budget={args.time_budget}, min_agreement={args.min_agreement}")
    print(f"  forced-any      : {sum(1 for d in audited if d['forced_any'])}/{len(audited)}")
    print(f"  agreement hist  : {dict(sorted(agreement_hist.items()))}")
    print(f"  stability by type: {stability_summary['by_action_type']}")
    print(f"  stability by margin: {stability_summary['by_margin_bin']}")
    print()
    print("State-representation collisions")
    print(f"  distinct compressed roots          : {rep['distinct_compressed_roots']}")
    print(f"  roots containing >1 raw state      : {rep['roots_containing_multiple_raw_states']}")
    print(f"  raw states inside collided roots   : {rep['raw_states_inside_collided_roots']}")
    print(f"  teacher-label entropy | root mean  : {rep['teacher_label_entropy_root_mean_bits']:.4f} bits")
    print(f"  teacher-label entropy | root weight: {rep['teacher_label_entropy_root_weighted_bits']:.4f} bits")
    print()
    print("Exact memorizer ceiling on the audited stable subset")
    print("  expected counts a max-score tie fractionally; optimistic counts any max-score tie as a hit.")
    for name, vals in memorizer.items():
        expected = vals["expected_top1"]
        optimistic = vals["optimistic_top1"]
        print(f"  {name:24s}: expected_top1={expected:.3f} optimistic_top1={optimistic:.3f}")
    if args.output:
        write_output(args.output, payload)
        out_path = args.output if args.output.is_absolute() else ROOT / args.output
        print()
        print(f"wrote JSON -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
