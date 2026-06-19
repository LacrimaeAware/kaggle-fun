"""Targeted residual/risk labels for Branch B's risk-only follow-up request.

Consumes `teacher_v2_risk_label_request_for_A.json`, which names two exact seed
states and sampling criteria for targeted enrichment. This is deliberately not a
generic larger Teacher V2 batch: it keeps the seed states, excludes the prior
input artifact for additional candidates, and retains only decisions that match
the requested risk patterns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_teacher_stability as A2  # noqa: E402
import main as M  # noqa: E402
import teacher_api_v2 as T2  # noqa: E402

MAN = ROOT / "data" / "manifests"
SPLITS = ROOT / "data" / "splits"
REPLAY_DIR = ROOT / "data" / "external" / "replays"
DEFAULT_REQUEST = MAN / "teacher_v2_risk_label_request_for_A.json"
DEFAULT_OUT = MAN / "teacher_v2_risk_labels_for_B_request.jsonl"
DEFAULT_SUMMARY = ROOT / "docs" / "workstreams" / "teacher_v2_risk_label_request_summary.md"


def _hash(obs: dict) -> str:
    return hashlib.sha1(json.dumps(obs, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _recover_from_source(src: dict, obs_hash: str | None, cache: dict) -> tuple[dict | None, list[int] | None, int | None]:
    fn, step = (src or {}).get("file"), (src or {}).get("step")
    if fn is None or step is None:
        return None, None, None
    if fn not in cache:
        try:
            cache[fn] = json.load((REPLAY_DIR / fn).open(encoding="utf-8"))
        except Exception:
            cache[fn] = None
    replay = cache.get(fn)
    if not replay or step >= len(replay.get("steps", [])):
        return None, None, None

    row = replay["steps"][step]
    obs, player = None, src.get("player")
    for ai in (0, 1):
        ag = row[ai] if ai < len(row) and isinstance(row[ai], dict) else None
        cand = (ag or {}).get("observation") or {}
        if cand and (obs_hash is None or _hash(cand) == obs_hash):
            obs, player = cand, ai
            if obs_hash is not None:
                break

    deck = None
    if player is not None:
        for s in replay.get("steps", []):
            action = s[player].get("action") if player < len(s) and isinstance(s[player], dict) else None
            if isinstance(action, list) and len(action) == 60:
                deck = action
                break
    return obs, deck, player


def _option_by_index(label: dict, index: int | None) -> dict | None:
    if index is None:
        return None
    for opt in label.get("options", []):
        if opt.get("index") == index:
            return opt
    return None


def _hand_outcome_disagree(label: dict) -> bool:
    candidates = [
        (opt["index"], opt["outcome_winrate"])
        for opt in label.get("options", [])
        if opt.get("outcome_winrate") is not None
    ]
    if not candidates:
        return False
    outcome_best = max(candidates, key=lambda x: x[1])[0]
    return outcome_best != label.get("stronger_argmax_option")


def _add_aliases(label: dict) -> None:
    opts = label.get("options", [])
    if not opts:
        return
    best = max((float(opt.get("stronger_value", 0.0)) for opt in opts), default=0.0)
    values = [float(opt.get("stronger_value", 0.0)) for opt in opts]
    spread = max(values) - min(values) if values else 0.0
    denom = spread if spread > 1e-9 else 1.0
    for opt in opts:
        stronger = float(opt.get("stronger_value", 0.0))
        opt["hand_norm_advantage"] = round((stronger - best) / denom, 6)
        opt["hand_value_variance"] = opt.get("value_variance")


def _annotate_label(label: dict, d: dict, args, *, request_reason=None, selection_reason=None) -> dict:
    opts = label.get("options", [])
    all_done = int(
        bool(opts)
        and all(
            opt.get("completed_determinizations", 0) >= args.n_strong
            and opt.get("outcome_playouts", 0) >= args.k_outcome
            for opt in opts
        )
    )
    obs = d["obs"]
    label.update(
        {
            "decision_id": d.get("decision_id") or f"{d.get('file')}:{d.get('step')}",
            "obs_hash": _hash(obs),
            "observation": obs,
            "legal_options": (obs.get("select") or {}).get("option") or [],
            "source": {"file": d.get("file"), "step": d.get("step"), "player": d.get("player")},
            "state_tag": d.get("tag"),
            "request_reason": request_reason,
            "selection_reason": selection_reason,
            "coverage": {"all_siblings_completed": all_done},
            "timing": {"label_time_s": d.get("label_time_s")},
        }
    )
    _add_aliases(label)
    return label


def _label_decision(d: dict, args, *, request_reason=None, selection_reason=None) -> tuple[dict | None, str | None]:
    ts = time.time()
    deck = d.get("deck") or list(M.DECK)
    try:
        label = T2.residual_risk_label(
            d["obs"],
            deck,
            n_strong=args.n_strong,
            n_live=args.n_live,
            k_outcome=args.k_outcome,
            high_regret_thresh=args.high_regret_thresh,
            seed=args.seed,
        )
    except Exception as exc:
        return None, f"label_exception: {exc}"
    if not label.get("applicable"):
        return None, "not_applicable"
    d = dict(d)
    d["label_time_s"] = round(time.time() - ts, 2)
    return _annotate_label(label, d, args, request_reason=request_reason, selection_reason=selection_reason), None


def _classify_requested_pattern(label: dict) -> list[str]:
    opts = label.get("options", [])
    if not opts:
        return []
    reasons = []
    selected = _option_by_index(label, label.get("search_selected_option"))
    if selected and selected.get("high_regret_flag"):
        reasons.append("search_selected_high_regret_analogs")
    if selected and not selected.get("high_regret_flag") and not selected.get("unacceptable_flag") and selected.get("regret", 0.0) <= 25:
        if any(opt.get("unacceptable_flag") or opt.get("high_regret_flag") for opt in opts if opt.get("index") != selected.get("index")):
            reasons.append("safe_search_choice_false_positive_analogs")
    if any(opt.get("high_regret_flag") for opt in opts):
        reasons.append("near_miss_risk_boundary")
    max_abs_delta = max(abs(float(opt.get("delta_to_search_norm", 0.0))) for opt in opts)
    max_se = max(float(opt.get("value_se", 0.0)) for opt in opts)
    if max_abs_delta >= args_near_delta_threshold or max_se >= args_near_se_threshold or _hand_outcome_disagree(label):
        reasons.append("near_miss_risk_boundary")
    return sorted(set(reasons))


args_near_delta_threshold = 1000.0
args_near_se_threshold = 1000.0


def _seed_decisions(request: dict, cache: dict, prior_by_id: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    out, failures = [], []
    for seed in request.get("seed_examples", []):
        obs = seed.get("observation")
        deck = seed.get("deck")
        src = seed.get("source") or {}
        expected_hash = seed.get("obs_hash")
        if obs is None or deck is None:
            rec = prior_by_id.get(seed.get("decision_id"))
            obs = obs or (rec or {}).get("observation")
            deck = deck or (rec or {}).get("deck")
        if obs is None or deck is None:
            r_obs, r_deck, player = _recover_from_source(src, expected_hash, cache)
            obs = obs or r_obs
            deck = deck or r_deck
            if player is not None:
                src = dict(src)
                src["player"] = player
        if obs is None:
            failures.append({"decision_id": seed.get("decision_id"), "reason": "seed_observation_unrecoverable"})
            continue
        actual_hash = _hash(obs)
        if expected_hash and actual_hash != expected_hash:
            failures.append(
                {
                    "decision_id": seed.get("decision_id"),
                    "reason": "seed_obs_hash_mismatch",
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            )
            continue
        out.append(
            {
                "file": src.get("file"),
                "step": src.get("step"),
                "player": src.get("player"),
                "obs": obs,
                "deck": deck or list(M.DECK),
                "decision_id": seed.get("decision_id"),
                "tag": "B_risk_seed",
                "request_reason": seed.get("why_requested"),
            }
        )
    return out, failures


def _candidate_decisions(args, exclude_hashes: set[str]) -> list[dict]:
    manifest = json.load((MAN / args.snapshot).open(encoding="utf-8"))
    split = json.load((SPLITS / args.split).open(encoding="utf-8"))
    sampled = A2.sample_decisions(manifest, split, args.candidates, verify=False)
    ranked = sorted(sampled, key=lambda d: (-T2.criticality_score(d["obs"])["score"], d.get("file") or "", d.get("step") or 0))
    out = []
    seen = set(exclude_hashes)
    for d in ranked:
        h = _hash(d["obs"])
        if h in seen:
            continue
        seen.add(h)
        d = dict(d)
        d["tag"] = "B_risk_request_targeted"
        out.append(d)
    return out


def _summarize(labels: list[dict], failures: list[dict], request: dict, args, elapsed: float) -> dict:
    opts = [opt for row in labels for opt in row.get("options", [])]
    high = sum(int(opt.get("high_regret_flag", 0)) for opt in opts)
    unacc = sum(int(opt.get("unacceptable_flag", 0)) for opt in opts)
    selected_high = sum(
        int((_option_by_index(row, row.get("search_selected_option")) or {}).get("high_regret_flag", 0))
        for row in labels
    )
    reasons = {}
    for row in labels:
        for reason in row.get("selection_reason") or []:
            reasons[reason] = reasons.get(reason, 0) + 1
    seed_ids = [seed.get("decision_id") for seed in request.get("seed_examples", [])]
    labeled_ids = {row.get("decision_id") for row in labels}
    deltas = [float(opt.get("delta_to_search", 0.0)) for opt in opts]
    outcome_se = [float(opt["outcome_se"]) for opt in opts if opt.get("outcome_se") is not None]
    class_balance = {
        "high_regret_positive_options": high,
        "high_regret_negative_options": len(opts) - high,
        "unacceptable_positive_options": unacc,
        "unacceptable_negative_options": len(opts) - unacc,
        "search_selected_high_regret_decisions": selected_high,
    }
    return {
        "artifact_version": "teacher_v2_risk_labels_for_B_request.summary.v1",
        "request_file": str(args.request),
        "output": str(args.out),
        "requested_seed_count": len(seed_ids),
        "target_count": args.target,
        "labeled_count": len(labels),
        "option_count": len(opts),
        "failures": failures,
        "seed_examples_included": {seed_id: seed_id in labeled_ids for seed_id in seed_ids},
        "selection_reasons": reasons,
        "class_balance": class_balance,
        "high_regret_options": high,
        "unacceptable_options": unacc,
        "densifies_sparse_high_regret_class": high > 13,
        "residual_delta": {
            "mean": round(statistics.fmean(deltas), 2) if deltas else None,
            "p50": round(sorted(deltas)[len(deltas) // 2], 2) if deltas else None,
            "abs_mean": round(statistics.fmean(abs(x) for x in deltas), 2) if deltas else None,
        },
        "mean_outcome_se": round(statistics.fmean(outcome_se), 4) if outcome_se else None,
        "all_siblings_completed": f"{sum(row.get('coverage', {}).get('all_siblings_completed', 0) for row in labels)}/{len(labels)}",
        "elapsed_s": round(elapsed, 1),
        "recommendation_for_B": (
            "Use this as a targeted high-regret recall calibration batch; verify the two seed cases first, "
            "then retrain one high-regret-primary risk-only model with threshold calibration."
        ),
    }


def _write_markdown(path: Path, summary: dict, source_path: Path, b_path: Path | None = None) -> None:
    cb = summary["class_balance"]
    seed_lines = "\n".join(f"- {k}: {str(v).lower()}" for k, v in summary["seed_examples_included"].items())
    reason_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(summary["selection_reasons"].items())) or "- none"
    text = f"""# Teacher V2 Risk Label Request Summary

Status: targeted residual/risk labels for Branch B. No live agent change and no arena screen.

## Artifact

- Source path: `{source_path}`
- B access path: `{b_path if b_path else 'not copied yet'}`
- Labeled decisions: {summary['labeled_count']}
- Options: {summary['option_count']}
- Failed/unrecoverable states: {len(summary['failures'])}
- All siblings completed: {summary['all_siblings_completed']}

## Seed Coverage

{seed_lines}

## Selection Reasons

{reason_lines}

## Class Balance

- High-regret positives: {cb['high_regret_positive_options']}
- High-regret negatives: {cb['high_regret_negative_options']}
- Unacceptable positives: {cb['unacceptable_positive_options']}
- Unacceptable negatives: {cb['unacceptable_negative_options']}
- Search-selected high-regret decisions: {cb['search_selected_high_regret_decisions']}
- Densifies sparse high-regret class versus prior 13 positives: {str(summary['densifies_sparse_high_regret_class']).lower()}

## Recommendation For B

{summary['recommendation_for_B']}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    global args_near_delta_threshold, args_near_se_threshold

    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--snapshot", default="replays_20260618.json")
    ap.add_argument("--split", default="replays_20260618_split.json")
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--candidates", type=int, default=320)
    ap.add_argument("--max-labels", type=int, default=90)
    ap.add_argument("--n-strong", type=int, default=32)
    ap.add_argument("--n-live", type=int, default=8)
    ap.add_argument("--k-outcome", type=int, default=16)
    ap.add_argument("--high-regret-thresh", type=float, default=5000.0)
    ap.add_argument("--near-delta-threshold", type=float, default=1000.0)
    ap.add_argument("--near-se-threshold", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    args.request = _resolve(args.request)
    args.out = _resolve(args.out)
    args.summary_md = _resolve(args.summary_md)
    args_near_delta_threshold = args.near_delta_threshold
    args_near_se_threshold = args.near_se_threshold

    request = json.load(args.request.open(encoding="utf-8"))
    prior_path = _resolve(request.get("current_input_artifact", "data/manifests/teacher_v2_residual_risk_labels.jsonl"))
    prior = _load_jsonl(prior_path)
    prior_by_id = {row.get("decision_id"): row for row in prior}
    prior_hashes = {row.get("obs_hash") for row in prior if row.get("obs_hash")}

    cache: dict = {}
    seed_decs, failures = _seed_decisions(request, cache, prior_by_id)
    print(f"[risk-request] seed states: {len(seed_decs)}; prior exclusions: {len(prior_hashes)}", flush=True)

    labels: list[dict] = []
    seen_hashes: set[str] = set()
    t0 = time.time()
    for d in seed_decs:
        lab, failure = _label_decision(
            d,
            args,
            request_reason=d.get("request_reason"),
            selection_reason=["seed_example"],
        )
        if lab:
            labels.append(lab)
            seen_hashes.add(lab["obs_hash"])
            print(f"  seed labeled {lab['decision_id']} options={len(lab['options'])}", flush=True)
        else:
            failures.append({"decision_id": d.get("decision_id"), "reason": failure})

    safe_kept = 0
    near_kept = 0
    labeled_candidates = 0
    candidates = _candidate_decisions(args, prior_hashes | seen_hashes)
    print(f"[risk-request] targeted candidates after prior exclusion: {len(candidates)}", flush=True)
    for d in candidates:
        if len(labels) >= args.target or labeled_candidates >= args.max_labels:
            break
        lab, failure = _label_decision(d, args)
        labeled_candidates += 1
        if not lab:
            failures.append({"decision_id": d.get("decision_id") or f"{d.get('file')}:{d.get('step')}", "reason": failure})
            continue
        reasons = _classify_requested_pattern(lab)
        keep = False
        if "search_selected_high_regret_analogs" in reasons:
            keep = True
        elif any(opt.get("high_regret_flag") for opt in lab.get("options", [])):
            keep = True
        elif "safe_search_choice_false_positive_analogs" in reasons and safe_kept < max(12, args.target // 3):
            keep = True
            safe_kept += 1
        elif "near_miss_risk_boundary" in reasons and near_kept < max(12, args.target // 3):
            keep = True
            near_kept += 1
        if not keep:
            continue
        lab["selection_reason"] = reasons
        labels.append(lab)
        seen_hashes.add(lab["obs_hash"])
        if len(labels) % 10 == 0:
            print(
                f"  kept {len(labels)}/{args.target}; labeled candidates={labeled_candidates}; "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    _write_jsonl(args.out, labels)
    summary = _summarize(labels, failures, request, args, time.time() - t0)
    json_summary = args.out.with_name(args.out.stem + "_summary.json")
    json_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(args.summary_md, summary, args.out)

    print("\n=== RISK REQUEST SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out.relative_to(ROOT)}")
    print(f"-> {args.summary_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
