"""Evaluate the revised Teacher V2 contextual-ranker objective pass.

This is offline-only. It compares the revised model, the previous Teacher V2
model, old-ranker/option-0 baselines, and component ablations on the requested
slices. It does not touch agent_search or run an arena screen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import contextual_ranker as CR  # noqa: E402
import train_contextual_action_ranker as TCR  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["decisions"]


def load_model(path: Path):
    blob = json.loads(path.read_text(encoding="utf-8"))
    model = TCR.ContextualNet(
        len(blob["card_ids"]),
        int(blob["dense_dim"]),
        emb=int(blob.get("emb", 24)),
        hidden=int(blob.get("hidden", 192)),
        use_emb=bool(blob.get("use_emb", True)),
    )
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in blob["state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return blob, model


def compact(metrics: dict) -> dict:
    keys = [
        "n",
        "top1",
        "top2",
        "top3",
        "acceptable_agreement",
        "mean_regret",
        "p95_regret",
        "high_regret_count",
        "pairwise_accuracy",
        "mrr",
        "ndcg",
    ]
    return {k: metrics.get(k) for k in keys if k in metrics}


def eval_model(blob: dict, model, rows: list[dict], *, ablate: dict | None = None,
               zero_embedding: bool = False) -> dict:
    if not rows:
        return {"n": 0}
    mean = torch.tensor(blob["mean"], dtype=torch.float32)
    std = torch.tensor(blob["std"], dtype=torch.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    emb_dim = int(blob.get("emb", 24))
    clip_z = float(blob.get("clip_z", 0.0) or 0.0)
    if zero_embedding and getattr(model, "use_emb", False):
        original = model.emb.weight.detach().clone()
        with torch.no_grad():
            model.emb.weight.zero_()
        metrics = TCR.eval_model(model, rows, mean, std, id2ix, emb_dim, ablate=ablate, clip_z=clip_z)["overall"]
        with torch.no_grad():
            model.emb.weight.copy_(original)
        return compact(metrics)
    return compact(TCR.eval_model(model, rows, mean, std, id2ix, emb_dim, ablate=ablate, clip_z=clip_z)["overall"])


def eval_baseline(rows: list[dict], field: str) -> dict:
    if not rows:
        return {"n": 0}
    return compact(TCR.eval_baseline(rows, field)["overall"])


def adv_spread(row: dict) -> float:
    vals = [float(v) for v in (row.get("adv") or {}).values()]
    return max(vals) - min(vals) if vals else 0.0


def outcome_best(row: dict) -> int | None:
    vals = {int(k): float(v) for k, v in (row.get("outcome_winrate") or {}).items()}
    return max(vals, key=lambda k: (vals[k], -k)) if vals else None


def hand_best(row: dict) -> int | None:
    vals = {int(k): float(v) for k, v in (row.get("adv") or {}).items()}
    return max(vals, key=lambda k: (vals[k], -k)) if vals else None


def hand_outcome_disagree(row: dict) -> bool:
    if row.get("hand_outcome_argmax_disagree") is not None:
        return bool(row.get("hand_outcome_argmax_disagree"))
    ob = outcome_best(row)
    hb = hand_best(row)
    return ob is not None and hb is not None and ob != hb


def variance_threshold(rows: list[dict], q: float) -> float:
    vals = sorted(float(r.get("value_variance_mean") or 0.0) for r in rows if r.get("value_variance_mean") is not None)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def build_slices(rows: list[dict]) -> dict[str, list[dict]]:
    var_med = variance_threshold(rows, 0.50)
    var_p75 = variance_threshold(rows, 0.75)
    return {
        "all_heldout_mixed_test": [r for r in rows if r.get("partition") == "test"],
        "teacher_v2_targeted_failure_states": [r for r in rows if r.get("teacher_v2_overlay")],
        "high_criticality_states": [
            r for r in rows
            if (r.get("criticality_score") or 0.0) >= 0.3
            or adv_spread(r) >= 1000.0
            or abs(float(r.get("top_two_margin") or 0.0)) >= 500.0
            or r.get("high_regret")
        ],
        "high_regret_tail": [r for r in rows if r.get("high_regret")],
        "recovery_states": [r for r in rows if r.get("partition") == "recovery"],
        "hand_outcome_disagreement_states": [r for r in rows if hand_outcome_disagree(r)],
        "stable_low_variance_labels": [
            r for r in rows
            if r.get("teacher_stability") != "unstable"
            and float(r.get("value_variance_mean") or 0.0) <= var_med
            and adv_spread(r) >= 25.0
        ],
        "noisy_labels": [
            r for r in rows
            if r.get("teacher_stability") == "unstable"
            or float(r.get("value_variance_mean") or 0.0) >= var_p75
            or adv_spread(r) < 25.0
            or hand_outcome_disagree(r)
        ],
    }


def model_choice(blob: dict, model, row: dict) -> dict:
    mean = torch.tensor(blob["mean"], dtype=torch.float32)
    std = torch.tensor(blob["std"], dtype=torch.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    emb_dim = int(blob.get("emb", 24))
    ids, logits = TCR.score_decision(model, row, mean, std, id2ix, emb_dim, clip_z=float(blob.get("clip_z", 0.0) or 0.0))
    scores = logits.detach().cpu().numpy().astype(float).tolist()
    pred = ids[max(range(len(ids)), key=lambda i: (scores[i], -ids[i]))]
    return choice_record(row, pred)


def choice_record(row: dict, eq: int | None) -> dict:
    adv = {int(k): float(v) for k, v in (row.get("adv") or {}).items()}
    acc = {int(k): float(v) for k, v in (row.get("acceptable") or {}).items()}
    if not adv or eq is None:
        return {"eq": eq, "regret": None, "acceptable": False}
    best = max(adv, key=lambda k: (adv[k], -k))
    selected = adv.get(int(eq), min(adv.values()))
    return {
        "eq": int(eq),
        "best_eq": int(best),
        "regret": adv[best] - selected,
        "acceptable": acc.get(int(eq), 0.0) >= 0.5,
    }


def decision_id(row: dict) -> str:
    return (
        row.get("teacher_v2_overlay_decision_id")
        or row.get("decision_id")
        or f"{row.get('game_file')}:{row.get('call', row.get('step'))}:{row.get('obs_hash')}"
    )


def high_regret_examples(rows: list[dict], revised_blob: dict, revised_model, previous_blob: dict, previous_model) -> list[dict]:
    out = []
    for row in rows:
        revised = model_choice(revised_blob, revised_model, row)
        previous = model_choice(previous_blob, previous_model, row)
        old = choice_record(row, row.get("old_ranker_eq"))
        opt0 = choice_record(row, row.get("option0_eq"))
        out.append({
            "decision_id": decision_id(row),
            "source": row.get("source"),
            "partition": row.get("partition"),
            "teacher_v2_overlay": bool(row.get("teacher_v2_overlay")),
            "hand_best_eq": hand_best(row),
            "outcome_best_eq": outcome_best(row),
            "hand_outcome_disagree": hand_outcome_disagree(row),
            "criticality_score": row.get("criticality_score"),
            "value_variance_mean": row.get("value_variance_mean"),
            "top_two_margin": row.get("top_two_margin"),
            "revised": revised,
            "previous_teacher_v2": previous,
            "old_ranker": old,
            "option0": opt0,
        })
    out.sort(key=lambda x: float(x["revised"].get("regret") or 0.0), reverse=True)
    return out[:20]


def decision_rule(all_test: dict) -> dict:
    revised = all_test["revised_full"]
    old = all_test["old_ranker"]
    opt0 = all_test["option0"]
    safety_beats_old = (
        (revised.get("acceptable_agreement") or 0.0) >= (old.get("acceptable_agreement") or 0.0)
        and (revised.get("mean_regret") or 1e30) <= (old.get("mean_regret") or 1e30)
        and (revised.get("p95_regret") or 1e30) <= (old.get("p95_regret") or 1e30)
        and (revised.get("high_regret_count") or 0) <= (old.get("high_regret_count") or 0)
    )
    safety_beats_opt0 = (
        (revised.get("acceptable_agreement") or 0.0) >= (opt0.get("acceptable_agreement") or 0.0)
        and (revised.get("mean_regret") or 1e30) <= (opt0.get("mean_regret") or 1e30)
        and (revised.get("p95_regret") or 1e30) <= (opt0.get("p95_regret") or 1e30)
        and (revised.get("high_regret_count") or 0) <= (opt0.get("high_regret_count") or 0)
    )
    if safety_beats_old or safety_beats_opt0:
        return {
            "choice": "A",
            "title": "offline improved enough to justify a small agent_search_ctx_v2 screen",
            "rationale": "Revised model meets the offline safety comparison on acceptable agreement, regret tail, and high-regret count.",
        }
    return {
        "choice": "D",
        "title": "offline did not improve; pause Teacher V2 contextual-ranker path",
        "rationale": (
            "The revised model does not beat old ranker or option-0 on the required safety metrics. "
            "More objective tuning would be another blind pass, and the labels are no longer the blocking issue."
        ),
    }


def md_table(rows: list[list]) -> str:
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, row in enumerate(rows):
        out.append("| " + " | ".join(str(v).ljust(widths[j]) for j, v in enumerate(row)) + " |")
        if i == 0:
            out.append("| " + " | ".join("-" * widths[j] for j in range(len(row))) + " |")
    return "\n".join(out)


def write_markdown(path: Path, report: dict) -> None:
    rows = [["model", "top1", "top3", "acceptable", "mean regret", "p95 regret", "hi-regret", "pairwise", "mrr"]]
    for name, metrics in report["slice_metrics"]["all_heldout_mixed_test"].items():
        rows.append([
            name,
            round(float(metrics.get("top1") or 0.0), 3),
            round(float(metrics.get("top3") or 0.0), 3),
            round(float(metrics.get("acceptable_agreement") or 0.0), 3),
            round(float(metrics.get("mean_regret") or 0.0), 2),
            round(float(metrics.get("p95_regret") or 0.0), 2),
            metrics.get("high_regret_count"),
            round(float(metrics.get("pairwise_accuracy") or 0.0), 3),
            round(float(metrics.get("mrr") or 0.0), 3),
        ])
    slice_rows = [[
        "slice",
        "n",
        "rev acc",
        "rev mean",
        "rev p95",
        "prev mean",
        "old mean",
        "opt0 mean",
    ]]
    for slice_name, models in report["slice_metrics"].items():
        revised = models["revised_full"]
        previous = models["previous_teacher_v2_model"]
        old = models["old_ranker"]
        opt0 = models["option0"]
        slice_rows.append([
            slice_name,
            revised.get("n", 0),
            round(float(revised.get("acceptable_agreement") or 0.0), 3),
            round(float(revised.get("mean_regret") or 0.0), 2),
            round(float(revised.get("p95_regret") or 0.0), 2),
            round(float(previous.get("mean_regret") or 0.0), 2),
            round(float(old.get("mean_regret") or 0.0), 2),
            round(float(opt0.get("mean_regret") or 0.0), 2),
        ])
    text = f"""# Teacher V2 Objective V2 Offline Evaluation

Status: offline only. No live arena/screen and no `agent_search` change.

Decision: **{report['decision']['choice']}** - {report['decision']['title']}

{report['decision']['rationale']}

## Held-Out Mixed Test

{md_table(rows)}

## Slice Summary

{md_table(slice_rows)}

## Notes

- Targeted Teacher V2 overlays applied to held-out replay-test rows: {report['overlay_rows']}
- Revised model artifact: `{report['inputs']['revised_model']}`
- Previous Teacher V2 model: `{report['inputs']['previous_model']}`
- High-regret tail report: `{report['inputs']['tail_report']}`
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--revised-model", type=Path, required=True)
    ap.add_argument("--previous-model", type=Path, default=ROOT / "agent" / "contextual_ranker_teacher_v2.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tail-output", type=Path, required=True)
    ap.add_argument("--summary-md", type=Path, required=True)
    args = ap.parse_args()

    dataset_path = resolve(args.dataset)
    rows = load_dataset(dataset_path)
    revised_blob, revised_model = load_model(resolve(args.revised_model))
    previous_blob, previous_model = load_model(resolve(args.previous_model))
    slices = build_slices(rows)

    metrics = {}
    for name, subset in slices.items():
        metrics[name] = {
            "revised_full": eval_model(revised_blob, revised_model, subset),
            "previous_teacher_v2_model": eval_model(previous_blob, previous_model, subset),
            "old_ranker": eval_baseline(subset, "old_ranker_eq"),
            "option0": eval_baseline(subset, "option0_eq"),
            "no_decoded_effects": eval_model(revised_blob, revised_model, subset, ablate={"effects": True}),
            "no_card_embedding": eval_model(revised_blob, revised_model, subset, zero_embedding=True),
            "no_option_deltas": eval_model(revised_blob, revised_model, subset, ablate={"deltas": True}),
        }

    decision = decision_rule(metrics["all_heldout_mixed_test"])
    tail_rows = high_regret_examples(slices["all_heldout_mixed_test"] + slices["high_regret_tail"], revised_blob, revised_model, previous_blob, previous_model)

    tail_path = resolve(args.tail_output)
    out_path = resolve(args.output)
    md_path = resolve(args.summary_md)
    report = {
        "artifact_version": "contextual_action_ranker.teacher_v2_objective_v2.offline_eval",
        "branch": "exp/robust-learner-v2",
        "live_agent_consumed": "none",
        "arena_screen": "not run",
        "inputs": {
            "dataset": str(dataset_path),
            "revised_model": str(resolve(args.revised_model)),
            "previous_model": str(resolve(args.previous_model)),
            "tail_report": str(tail_path),
        },
        "overlay_rows": sum(1 for r in rows if r.get("teacher_v2_overlay")),
        "slice_sizes": {name: len(subset) for name, subset in slices.items()},
        "slice_metrics": metrics,
        "decision": decision,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tail_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tail_path.write_text(json.dumps({
        "artifact_version": "contextual_action_ranker.teacher_v2_objective_v2.high_regret_tail",
        "branch": "exp/robust-learner-v2",
        "examples": tail_rows,
    }, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({
        "decision": decision,
        "all_heldout_mixed_test": metrics["all_heldout_mixed_test"],
        "slice_sizes": report["slice_sizes"],
        "output": str(out_path.relative_to(ROOT)),
        "tail_output": str(tail_path.relative_to(ROOT)),
        "summary_md": str(md_path.relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
