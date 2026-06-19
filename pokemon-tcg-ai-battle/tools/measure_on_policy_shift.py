"""Branch B / B1.3 -- direct on-policy shift measurement for the old student.

This runs the old learned policy in real cabt games, records the states it
actually visits, and then queries Teacher API V1 on those visited decisions.
That is the first point where Branch B may talk about on-policy shift; offline
top-1 failures and teacher label wobble alone are not enough.

Default is a small smoke run. The plan-required run is at least 100 games.

Example:

    python tools/measure_on_policy_shift.py --student rank --opponent heuristic --games 100 \
        --output docs/workstreams/robust_learner_v2_b1_3_rank_vs_heuristic_100g.json
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))

logging.disable(logging.CRITICAL)

import cabt_arena as A  # noqa: E402
import main as M  # noqa: E402
import state_action_schema_v2 as SCH  # noqa: E402
import teacher_api_v1 as T  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "replays_20260618.json"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "replays_20260618_split.json"
MARGIN_BINS = [1.0, 10.0, 100.0, 1000.0]
DISTANCE_BINS = [0.0, 5.0, 20.0, 100.0]


def canonical_json(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def obs_hash(obs: dict) -> str:
    payload = {"current": obs.get("current"), "select": obs.get("select")}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def margin_bin(margin) -> str:
    if margin is None:
        return "none"
    x = abs(float(margin))
    for upper in MARGIN_BINS:
        if x < upper:
            return f"<{upper:g}"
    return f">={MARGIN_BINS[-1]:g}"


def distance_bin(dist) -> str:
    if dist is None:
        return "none"
    x = float(dist)
    for upper in DISTANCE_BINS:
        if x <= upper:
            return f"<={upper:g}"
    return f">{DISTANCE_BINS[-1]:g}"


def mean(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for n in counts.values():
        p = n / total
        out -= p * math.log2(p)
    return out


def normalize_obs(obs: dict) -> dict:
    """The local env may call agents with either the raw observation or a Kaggle wrapper."""
    inner = obs.get("observation") if isinstance(obs, dict) else None
    return inner if isinstance(inner, dict) else obs


def label_key(label) -> str:
    return canonical_json(list(label))


def soft_lookup(soft: dict, key: int) -> float:
    return float(soft.get(key, soft.get(str(key), 0.0)) or 0.0)


def load_split_names(split_path: Path, partition: str) -> list[str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return list(split.get(partition) or [])


def build_reference_roots(replay_dir: Path, split_path: Path, partition: str,
                          max_files: int, max_decisions: int) -> list[tuple[float, ...]]:
    """Sample frozen replay roots for a rough nearest-L1 distance diagnostic."""
    roots = []
    for name in load_split_names(split_path, partition)[:max_files]:
        path = replay_dir / name
        if not path.exists():
            continue
        try:
            game = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for step in game.get("steps", []):
            if not isinstance(step, list):
                continue
            for agent_step in step:
                if not isinstance(agent_step, dict):
                    continue
                obs = agent_step.get("observation") or {}
                if not SCH.is_single_pick_decision(obs):
                    continue
                try:
                    roots.append(tuple(float(x) for x in SCH.encode_vector(obs)))
                except Exception:
                    continue
                if len(roots) >= max_decisions:
                    return roots
    return roots


def nearest_l1(root: tuple[float, ...], refs: list[tuple[float, ...]]) -> float | None:
    if not refs:
        return None
    best = None
    for r in refs:
        if len(r) != len(root):
            continue
        d = sum(abs(a - b) for a, b in zip(root, r))
        if best is None or d < best:
            best = d
    return best


def make_traced_agent(name: str, fn, records: list[dict], game_id: int, seat: int):
    call_i = 0

    def traced_agent(obs: dict):
        nonlocal call_i
        raw_obs = normalize_obs(obs)
        action = fn(raw_obs)
        call_i += 1
        try:
            if SCH.is_single_pick_decision(raw_obs) and isinstance(action, list) and len(action) == 1:
                opts = ((raw_obs.get("select") or {}).get("option") or [])
                idx = action[0]
                if isinstance(idx, int) and 0 <= idx < len(opts) and isinstance(opts[idx], dict):
                    cur = raw_obs.get("current") or {}
                    me = cur.get("yourIndex", seat)
                    records.append({
                        "game": game_id,
                        "seat": seat,
                        "call": call_i,
                        "turn": cur.get("turn"),
                        "turn_action_count": cur.get("turnActionCount"),
                        "student_action": idx,
                        "student_key": tuple(SCH.semantic_action_key(opts[idx], cur, me)),
                        "obs_hash": obs_hash(raw_obs),
                        "obs": copy.deepcopy(raw_obs),
                    })
        except Exception:
            pass
        return action

    traced_agent.__name__ = f"traced_{name}"
    return traced_agent


def run_traced_games(student: str, opponent: str, games: int, progress: int) -> tuple[list[dict], dict]:
    if student not in A.AGENTS:
        raise SystemExit(f"unknown student {student!r}; choices: {sorted(A.AGENTS)}")
    if opponent not in A.AGENTS:
        raise SystemExit(f"unknown opponent {opponent!r}; choices: {sorted(A.AGENTS)}")
    student_fn = A.AGENTS[student]
    opponent_fn = A.AGENTS[opponent]
    decisions = []
    wins_student = wins_opp = draws = errors = 0
    t0 = time.time()
    for g in range(games):
        student_seat = g % 2
        game_records = []
        traced = make_traced_agent(student, student_fn, game_records, g, student_seat)
        agents = [traced, opponent_fn] if student_seat == 0 else [opponent_fn, traced]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env = A.make("cabt")
                env.run(agents)
            winner = A.winner_of(env)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ERROR game {g + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        decisions.extend(game_records)
        if winner is None:
            draws += 1
        elif winner == student_seat:
            wins_student += 1
        else:
            wins_opp += 1
        done = g + 1
        if progress and (done % progress == 0 or done == games):
            elapsed = time.time() - t0
            print(f"  traced {done}/{games} games | decisions={len(decisions)} | "
                  f"student {wins_student}-{wins_opp}, {draws}d {errors}e | {elapsed:.0f}s", flush=True)
    return decisions, {
        "games": games,
        "wins_student": wins_student,
        "wins_opponent": wins_opp,
        "draws": draws,
        "errors": errors,
        "seconds": time.time() - t0,
    }


def _compare_one_result(decision: dict, result: dict) -> dict | None:
    opts = result.get("options") or []
    idx = decision["student_action"]
    if idx >= len(opts):
        return None

    classes = {c["eq_class"]: c for c in result.get("eq_classes") or []}
    student_eq = opts[idx].get("eq_class")
    chosen = result.get("chosen_option")
    teacher_eq = opts[chosen].get("eq_class") if chosen is not None and chosen < len(opts) else None
    teacher_key = tuple(opts[chosen]["semantic_action_key"]) if chosen is not None and chosen < len(opts) else None
    acceptable = set(result.get("acceptable_action_set") or [])
    if result.get("forced_action_flag") and teacher_eq is not None:
        acceptable.add(teacher_eq)

    valued = [c for c in classes.values() if c.get("mean_value") is not None]
    best_value = max((c["mean_value"] for c in valued), default=None)
    student_class = classes.get(student_eq) or {}
    teacher_class = classes.get(teacher_eq) or {}
    student_value = student_class.get("mean_value")
    regret = (best_value - student_value) if best_value is not None and student_value is not None else None
    return {
        "teacher_eq": teacher_eq,
        "teacher_key": teacher_key,
        "student_eq": student_eq,
        "hard_top1_agree": student_eq == teacher_eq,
        "acceptable_agree": student_eq in acceptable,
        "regret": regret,
        "top_two_margin": result.get("top_two_margin"),
        "forced_action_flag": bool(result.get("forced_action_flag")),
        "acceptable_action_count": len(acceptable),
        "teacher_elapsed_s": result.get("elapsed_s"),
        "student_value": student_value,
        "teacher_best_value": best_value,
        "student_value_variance": student_class.get("value_variance"),
        "teacher_value_variance": teacher_class.get("value_variance"),
        "student_completed_determinizations": student_class.get("completed_determinizations"),
        "teacher_completed_determinizations": teacher_class.get("completed_determinizations"),
        "student_normalized_advantage": opts[idx].get("normalized_advantage"),
        "soft_policy_for_student": soft_lookup(result.get("soft_policy_target") or {}, student_eq),
    }


def teacher_compare(decision: dict, refs: list[tuple[float, ...]], n_determ: int,
                    time_budget: float, seed: int, accept_z: float, repeats: int,
                    min_teacher_agreement: float, min_acceptable_rate: float) -> dict:
    base = {
        "game": decision["game"],
        "seat": decision["seat"],
        "call": decision["call"],
        "turn": decision["turn"],
        "turn_action_count": decision["turn_action_count"],
        "obs_hash": decision["obs_hash"],
        "student_action": decision["student_action"],
        "student_key": list(decision["student_key"]),
        "student_action_type": str(decision["student_key"][0]),
    }
    try:
        root = tuple(float(x) for x in SCH.encode_vector(decision["obs"]))
    except Exception:
        root = ()
    dist = nearest_l1(root, refs) if root else None
    base["nearest_train_l1"] = dist
    base["distance_bin"] = distance_bin(dist)

    per_repeat = []
    not_applicable = 0
    for r in range(repeats):
        result = T.query(
            decision["obs"],
            M.DECK,
            n_determ=n_determ,
            time_budget=time_budget,
            leaf_mode="hand",
            seed=seed + r,
            accept_z=accept_z,
        )
        if not result.get("applicable"):
            not_applicable += 1
            continue
        one = _compare_one_result(decision, result)
        if one is None:
            not_applicable += 1
            continue
        per_repeat.append(one)

    if not per_repeat:
        return {
            **base,
            "teacher_applicable": False,
            "teacher_stability_class": "not_applicable",
            "teacher_repeat_count": repeats,
            "teacher_repeat_applicable_count": 0,
            "teacher_repeat_not_applicable_count": not_applicable,
        }

    label_counts = Counter(x["teacher_key"] for x in per_repeat if x["teacher_key"] is not None)
    teacher_key, teacher_key_n = label_counts.most_common(1)[0]
    label_agreement = teacher_key_n / len(per_repeat)
    majority = [x for x in per_repeat if x["teacher_key"] == teacher_key]
    stability_class = "stable" if label_agreement >= min_teacher_agreement else "unstable"
    acceptable_rate = mean([1.0 if x["acceptable_agree"] else 0.0 for x in per_repeat]) or 0.0
    hard_rate = mean([1.0 if x["hard_top1_agree"] else 0.0 for x in per_repeat]) or 0.0
    regret = mean([x["regret"] for x in majority])
    top_two_margin = mean([x["top_two_margin"] for x in majority])
    student_eq = majority[0]["student_eq"]
    teacher_eq = majority[0]["teacher_eq"]
    return {
        **base,
        "teacher_applicable": True,
        "teacher_stability_class": stability_class,
        "teacher_label_agreement": label_agreement,
        "teacher_label_entropy_bits": entropy(label_counts),
        "teacher_label_counts": {label_key(k): v for k, v in label_counts.items()},
        "teacher_repeat_count": repeats,
        "teacher_repeat_applicable_count": len(per_repeat),
        "teacher_repeat_not_applicable_count": not_applicable,
        "teacher_chosen_eq": teacher_eq,
        "student_eq": student_eq,
        "hard_top1_agree": hard_rate >= 0.5,
        "hard_top1_agree_rate": hard_rate,
        "acceptable_agree": acceptable_rate >= min_acceptable_rate,
        "acceptable_agree_rate": acceptable_rate,
        "regret": regret,
        "top_two_margin": top_two_margin,
        "margin_bin": margin_bin(top_two_margin),
        "forced_action_flag": any(x["forced_action_flag"] for x in per_repeat),
        "acceptable_action_count": mean([x["acceptable_action_count"] for x in per_repeat]),
        "teacher_elapsed_s": mean([x["teacher_elapsed_s"] for x in per_repeat]),
        "student_value": mean([x["student_value"] for x in majority]),
        "teacher_best_value": mean([x["teacher_best_value"] for x in majority]),
        "student_value_variance": mean([x["student_value_variance"] for x in majority]),
        "teacher_value_variance": mean([x["teacher_value_variance"] for x in majority]),
        "student_completed_determinizations": mean([x["student_completed_determinizations"] for x in majority]),
        "teacher_completed_determinizations": mean([x["teacher_completed_determinizations"] for x in majority]),
        "student_normalized_advantage": mean([x["student_normalized_advantage"] for x in majority]),
        "soft_policy_for_student": mean([x["soft_policy_for_student"] for x in per_repeat]),
    }


def bucket_summary(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for r in rows:
        value = r.get(key)
        groups["none" if value is None else str(value)].append(r)
    out = {}
    for k, xs in sorted(groups.items()):
        applicable = [x for x in xs if x.get("teacher_applicable")]
        out[k] = summarize_rows(applicable, total_override=len(xs))
    return out


def summarize_rows(rows: list[dict], total_override: int | None = None) -> dict:
    total = len(rows) if total_override is None else total_override
    if not rows:
        return {"total": total, "teacher_applicable": 0}
    acceptable_values = [r.get("acceptable_agree_rate", 1.0 if r.get("acceptable_agree") else 0.0) for r in rows]
    hard_values = [r.get("hard_top1_agree_rate", 1.0 if r.get("hard_top1_agree") else 0.0) for r in rows]
    regrets = [r.get("regret") for r in rows if r.get("regret") is not None]
    stable = sum(1 for r in rows if r.get("teacher_stability_class") == "stable")
    unstable = sum(1 for r in rows if r.get("teacher_stability_class") == "unstable")
    return {
        "total": total,
        "teacher_applicable": len(rows),
        "teacher_stable": stable,
        "teacher_unstable": unstable,
        "teacher_stable_rate": stable / len(rows),
        "acceptable_agreement_rate": mean(acceptable_values),
        "hard_top1_agreement_rate": mean(hard_values),
        "mean_regret": mean(regrets),
        "p90_regret": percentile(regrets, 0.90),
        "high_regret_count": sum(1 for r in rows if (r.get("regret") or 0.0) >= 1000.0),
    }


def percentile(xs: list[float], q: float) -> float | None:
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def add_pre_post_first_disagreement(rows: list[dict]) -> None:
    first_bad_by_game = {}
    for r in rows:
        if not r.get("teacher_applicable"):
            continue
        if r.get("acceptable_agree"):
            continue
        first_bad_by_game.setdefault(r["game"], r["call"])
    for r in rows:
        first_bad = first_bad_by_game.get(r["game"])
        if first_bad is None:
            r["phase_vs_first_unacceptable"] = "no_unacceptable_in_game"
        elif r["call"] < first_bad:
            r["phase_vs_first_unacceptable"] = "before"
        elif r["call"] == first_bad:
            r["phase_vs_first_unacceptable"] = "first_unacceptable"
        else:
            r["phase_vs_first_unacceptable"] = "after"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="rank", choices=list(A.AGENTS))
    ap.add_argument("--opponent", default="heuristic", choices=list(A.AGENTS))
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--progress", type=int, default=10)
    ap.add_argument("--n-determ", type=int, default=4)
    ap.add_argument("--time-budget", type=float, default=5.0)
    ap.add_argument("--teacher-repeats", type=int, default=3)
    ap.add_argument("--min-teacher-agreement", type=float, default=1.0)
    ap.add_argument("--min-acceptable-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--accept-z", type=float, default=1.0)
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--reference-partition", default="train", choices=["train", "val", "test"])
    ap.add_argument("--reference-max-files", type=int, default=80)
    ap.add_argument("--reference-max-decisions", type=int, default=2000)
    ap.add_argument("--max-teacher-decisions", type=int, default=0,
                    help="cap teacher-labelled decisions for smoke runs; 0 means all traced decisions")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    refs = build_reference_roots(
        args.replay_dir,
        args.split,
        args.reference_partition,
        args.reference_max_files,
        args.reference_max_decisions,
    )
    print(f"reference roots: {len(refs)} from {args.reference_partition}", flush=True)
    decisions, game_summary = run_traced_games(args.student, args.opponent, args.games, args.progress)
    if args.max_teacher_decisions:
        decisions = decisions[:args.max_teacher_decisions]
    print(f"teacher-querying {len(decisions)} visited student decisions", flush=True)

    rows = []
    t0 = time.time()
    for i, decision in enumerate(decisions):
        rows.append(teacher_compare(
            decision,
            refs,
            n_determ=args.n_determ,
            time_budget=args.time_budget,
            seed=args.seed + i * max(1, args.teacher_repeats),
            accept_z=args.accept_z,
            repeats=args.teacher_repeats,
            min_teacher_agreement=args.min_teacher_agreement,
            min_acceptable_rate=args.min_acceptable_rate,
        ))
        if args.progress and ((i + 1) % args.progress == 0 or i + 1 == len(decisions)):
            print(f"  teacher {i + 1}/{len(decisions)} | {time.time() - t0:.0f}s", flush=True)
    add_pre_post_first_disagreement(rows)
    applicable = [r for r in rows if r.get("teacher_applicable")]
    summary = {
        "audit_version": "branch_b_b1_3.0",
        "config": {
            "student": args.student,
            "opponent": args.opponent,
            "games": args.games,
            "n_determ": args.n_determ,
            "time_budget": args.time_budget,
            "teacher_repeats": args.teacher_repeats,
            "min_teacher_agreement": args.min_teacher_agreement,
            "min_acceptable_rate": args.min_acceptable_rate,
            "seed": args.seed,
            "accept_z": args.accept_z,
            "reference_partition": args.reference_partition,
            "reference_roots": len(refs),
        },
        "game_summary": game_summary,
        "overall": summarize_rows(applicable, total_override=len(rows)),
        "by_phase": bucket_summary(rows, "phase_vs_first_unacceptable"),
        "by_turn": bucket_summary(rows, "turn"),
        "by_action_type": bucket_summary(rows, "student_action_type"),
        "by_margin_bin": bucket_summary(rows, "margin_bin"),
        "by_distance_bin": bucket_summary(rows, "distance_bin"),
        "by_teacher_stability": bucket_summary(rows, "teacher_stability_class"),
        "high_regret_examples": sorted(
            [r for r in applicable if (r.get("regret") or 0.0) >= 1000.0],
            key=lambda r: -(r.get("regret") or 0.0),
        )[:20],
        "rows": rows,
        "not_applicable_teacher_decisions": len(rows) - len(applicable),
        "not_applicable_teacher_calls": sum(r.get("teacher_repeat_not_applicable_count", 0) for r in rows),
    }
    print("Branch B / B1.3 on-policy shift diagnostic")
    print(f"  student/opponent: {args.student} vs {args.opponent}")
    print(f"  games           : {args.games}")
    print(f"  visited decisions: {len(rows)}")
    print(f"  applicable teacher decisions: {len(applicable)}")
    print(f"  not-applicable teacher decisions: {summary['not_applicable_teacher_decisions']}")
    print(f"  teacher stability: {summary['by_teacher_stability']}")
    print(f"  acceptable agreement: {summary['overall'].get('acceptable_agreement_rate')}")
    print(f"  hard top-1 agreement: {summary['overall'].get('hard_top1_agreement_rate')}")
    print(f"  mean regret: {summary['overall'].get('mean_regret')}")
    if args.output:
        out = args.output if args.output.is_absolute() else ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote JSON -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
