"""Analyze Teacher V2 contextual-ranker failures without retraining.

Compares the saved Teacher V2 contextual model against old-ranker and option-0
baselines on the held-out mixed test split. It also produces a targeted Model A
label request for test states that currently lack Teacher V2 labels.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import contextual_ranker as CR  # noqa: E402
import state_action_schema_v2 as SCH  # noqa: E402
import train_contextual_action_ranker as TCR  # noqa: E402

DEFAULT_DATASET = ROOT / "docs" / "workstreams" / "contextual_action_ranker_v1_teacher_v2_mixed_dataset.json"
DEFAULT_MODEL = ROOT / "agent" / "contextual_ranker_teacher_v2.json"
DEFAULT_TEACHER_V2 = ROOT / "data" / "manifests" / "teacher_v2_labels_scaled.jsonl"
DEFAULT_JSON = ROOT / "docs" / "workstreams" / "teacher_v2_failure_analysis.json"
DEFAULT_MD = ROOT / "docs" / "workstreams" / "teacher_v2_failure_analysis.md"
DEFAULT_REQUEST = ROOT / "data" / "manifests" / "teacher_v2_label_request_for_A.json"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def canonical_json(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_float_map(d: dict | None) -> dict[int, float]:
    return {int(k): float(v) for k, v in (d or {}).items()}


def class_ids(eqs: list[int]) -> list[int]:
    out = []
    for e in eqs:
        e = int(e)
        if e not in out:
            out.append(e)
    return out


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


def option_logits(model, row: dict, blob: dict, ablate: dict | None = None) -> list[float]:
    dense_np = np.array(row["dense"], dtype=np.float32)
    mean = np.array(blob["mean"], dtype=np.float32)
    std = np.array(blob["std"], dtype=np.float32)
    dense_np = (dense_np - mean) / std
    clip_z = float(blob.get("clip_z", 0.0) or 0.0)
    if clip_z > 0:
        dense_np = np.clip(dense_np, -clip_z, clip_z)
    dense_np = CR.apply_ablation(dense_np, ablate)
    dense = torch.tensor(dense_np, dtype=torch.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    emb_dim = int(blob.get("emb", 24))
    idxs = [id2ix.get(int(cid), -1) for cid in row["cids"]]
    with torch.no_grad():
        if not model.use_emb:
            cidx = torch.tensor([i if i >= 0 else 0 for i in idxs], dtype=torch.long)
            logits = model(cidx, dense)
        else:
            emb_rows = [
                model.emb.weight[ix] if ix >= 0 else torch.zeros(emb_dim)
                for ix in idxs
            ]
            x = torch.cat([torch.stack(emb_rows), dense], dim=-1)
            logits = model.net(x).squeeze(-1)
    return logits.detach().cpu().numpy().astype(float).tolist()


def class_scores_from_option_logits(logits: list[float], eqs: list[int]) -> dict[int, float]:
    out = {}
    for eq in class_ids(eqs):
        vals = [float(logits[i]) for i, e in enumerate(eqs) if int(e) == int(eq)]
        m = max(vals)
        out[int(eq)] = float(m + math.log(sum(math.exp(v - m) for v in vals)))
    return out


def choice_from_eq(row: dict, eq: int | None, name: str, option_scores: list[float] | None = None) -> dict:
    if eq is None:
        return {"name": name, "eq_class": None, "option_index": None}
    members = [i for i, e in enumerate(row["eq"]) if int(e) == int(eq)]
    if not members:
        return {"name": name, "eq_class": int(eq), "option_index": None}
    if option_scores:
        option_index = max(members, key=lambda i: (option_scores[i], -i))
    else:
        option_index = members[0]
    key = row["keys"][option_index]
    adv = to_float_map(row.get("adv"))
    acceptable = to_float_map(row.get("acceptable"))
    best_adv = max(adv.values()) if adv else 0.0
    selected_adv = adv.get(int(eq), min(adv.values()) if adv else 0.0)
    return {
        "name": name,
        "eq_class": int(eq),
        "option_index": int(option_index),
        "semantic_action_key": key,
        "action_type": key[0] if key else None,
        "acceptable": acceptable.get(int(eq), 0.0) >= 0.5,
        "acceptable_score": acceptable.get(int(eq), 0.0),
        "correct": int(eq) == best_eq(row),
        "advantage": selected_adv,
        "regret": best_adv - selected_adv,
    }


def best_eq(row: dict) -> int:
    adv = to_float_map(row.get("adv"))
    return max(adv, key=lambda k: (adv[k], -k))


def best_choice(row: dict, teacher_label: dict | None) -> dict:
    if teacher_label:
        opts = teacher_label.get("options") or []
        if opts:
            best_opt = max(opts, key=lambda o: float(o.get("hand_norm_advantage") or -1e30))
            idx = int(best_opt["index"])
            eq = int(row["eq"][idx]) if 0 <= idx < len(row["eq"]) else int(best_opt["eq_class"])
            choice = choice_from_eq(row, eq, "teacher_best_hand_norm_advantage")
            choice["option_index"] = idx
            choice["semantic_action_key"] = best_opt.get("semantic_action_key")
            choice["action_type"] = (best_opt.get("semantic_action_key") or [None])[0]
            choice["teacher_option_eq_class"] = int(best_opt["eq_class"])
            choice["hand_norm_advantage"] = float(best_opt["hand_norm_advantage"])
            return choice
    return choice_from_eq(row, best_eq(row), "teacher_best_current_label")


def top_margin(values: dict[int, float]) -> float | None:
    if len(values) < 2:
        return None
    ordered = sorted(values.values(), reverse=True)
    return ordered[0] - ordered[1]


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


def teacher_label_for_row(row: dict, labels: list[dict]) -> dict | None:
    did = row.get("decision_id")
    if did:
        for lab in labels:
            if lab.get("decision_id") == did:
                return lab
    tv_hash = row.get("teacher_v2_obs_hash")
    if tv_hash:
        for lab in labels:
            if lab.get("obs_hash") == tv_hash:
                return lab
    row_hash = row.get("obs_hash")
    if row_hash:
        for lab in labels:
            obs = lab.get("observation")
            if isinstance(obs, dict) and TCR.obs_hash(obs) == row_hash:
                return lab
    idx = row.get("teacher_v2_label_index")
    if isinstance(idx, int) and 0 <= idx < len(labels):
        return labels[idx]
    return None


def row_with_teacher_targets(row: dict, label: dict | None) -> dict:
    if not label:
        return row
    opts = label.get("options") or []
    if not opts:
        return row
    out = dict(row)
    adv_acc = defaultdict(list)
    variance = []
    completed = []
    outcome = defaultdict(list)
    outcome_se = defaultdict(list)
    teacher_to_feature_eq = {}
    for opt in opts:
        try:
            idx = int(opt.get("index"))
        except Exception:
            continue
        if idx < 0 or idx >= len(row.get("eq") or []):
            continue
        if list(opt.get("semantic_action_key") or []) != list((row.get("keys") or [])[idx]):
            continue
        feq = int(row["eq"][idx])
        teq = int(opt.get("eq_class", feq))
        teacher_to_feature_eq[teq] = feq
        if opt.get("hand_norm_advantage") is not None:
            adv_acc[feq].append(float(opt["hand_norm_advantage"]))
        if opt.get("hand_value_variance") is not None:
            variance.append(float(opt["hand_value_variance"]))
        if opt.get("completed_determinizations") is not None:
            completed.append(float(opt["completed_determinizations"]))
        if opt.get("outcome_winrate") is not None:
            outcome[feq].append(float(opt["outcome_winrate"]))
        if opt.get("outcome_se") is not None:
            outcome_se[feq].append(float(opt["outcome_se"]))
    if not adv_acc:
        return row
    acceptable = {}
    for teq in label.get("acceptable_action_set") or []:
        try:
            feq = teacher_to_feature_eq[int(teq)]
        except Exception:
            continue
        acceptable[feq] = 1.0
    out["adv"] = {str(k): sum(v) / len(v) for k, v in adv_acc.items()}
    out["acceptable"] = {str(k): float(v) for k, v in acceptable.items()}
    out["outcome_winrate"] = {str(k): sum(v) / len(v) for k, v in outcome.items()}
    out["outcome_se"] = {str(k): sum(v) / len(v) for k, v in outcome_se.items()}
    out["value_variance_mean"] = sum(variance) / len(variance) if variance else row.get("value_variance_mean")
    out["completed_determinizations_mean"] = (
        sum(completed) / len(completed) if completed else row.get("completed_determinizations_mean")
    )
    out["top_two_margin"] = label.get("top_two_margin", row.get("top_two_margin"))
    out["criticality_score"] = (label.get("criticality") or {}).get("score", row.get("criticality_score"))
    out["teacher_stability"] = "teacher_v2_available"
    out["teacher_confidence"] = 1.0
    return out


def option_diag_from_teacher(label: dict | None, option_index: int | None) -> dict:
    if not label or option_index is None:
        return {}
    for opt in label.get("options") or []:
        if int(opt.get("index", -1)) == int(option_index):
            return {
                "hand_mean_value": opt.get("hand_mean_value"),
                "hand_norm_advantage": opt.get("hand_norm_advantage"),
                "hand_value_variance": opt.get("hand_value_variance"),
                "completed_determinizations": opt.get("completed_determinizations"),
                "outcome_winrate": opt.get("outcome_winrate"),
                "outcome_playouts": opt.get("outcome_playouts"),
                "outcome_se": opt.get("outcome_se"),
            }
    return {}


def outcome_margin(label: dict | None) -> float | None:
    if not label:
        return None
    vals = [float(o["outcome_winrate"]) for o in label.get("options") or [] if o.get("outcome_winrate") is not None]
    if len(vals) < 2:
        return None
    vals.sort(reverse=True)
    return vals[0] - vals[1]


def top_outcome_se(label: dict | None) -> float | None:
    if not label:
        return None
    opts = [o for o in label.get("options") or [] if o.get("outcome_winrate") is not None]
    if not opts:
        return None
    best = max(opts, key=lambda o: float(o["outcome_winrate"]))
    return best.get("outcome_se")


def label_ambiguity(row: dict, label: dict | None) -> list[str]:
    reasons = []
    adv = to_float_map(row.get("adv"))
    margin = top_margin(adv)
    if margin is not None and margin < 50.0:
        reasons.append("small_hand_advantage_margin")
    acceptable = [k for k, v in to_float_map(row.get("acceptable")).items() if v >= 0.5]
    if len(acceptable) > 1:
        reasons.append("multiple_acceptable_actions")
    if row.get("teacher_stability") == "unstable" or float(row.get("teacher_confidence") or 1.0) < 0.3:
        reasons.append("low_stability_or_confidence")
    if label and label.get("hand_outcome_agree") is False:
        reasons.append("hand_outcome_argmax_disagree")
    om = outcome_margin(label)
    ose = top_outcome_se(label)
    if om is not None and ose is not None and om <= 2.0 * float(ose):
        reasons.append("outcome_near_tie_vs_se")
    if float(row.get("value_variance_mean") or 0.0) > 1e8:
        reasons.append("very_high_hand_value_variance")
    return reasons


def classify(row: dict, model: dict, old: dict, option0: dict, label: dict | None) -> list[str]:
    classes = []
    if model.get("correct") and not old.get("correct"):
        classes.append("teacher_v2_model_correct_old_ranker_wrong")
    if old.get("correct") and not model.get("correct"):
        classes.append("old_ranker_correct_teacher_v2_model_wrong")
    if option0.get("correct") and not model.get("correct"):
        classes.append("option0_correct_teacher_v2_model_wrong")
    if not model.get("correct") and not old.get("correct") and not option0.get("correct"):
        classes.append("all_wrong")
    if model.get("acceptable") and old.get("acceptable") and option0.get("acceptable"):
        classes.append("all_acceptable")
    if label_ambiguity(row, label):
        classes.append("teacher_v2_label_or_current_label_ambiguous_noisy")
    return classes


def likely_causes(row: dict, model: dict, old: dict, option0: dict, label: dict | None,
                  no_effects: dict, no_deltas: dict) -> list[str]:
    causes = []
    ambiguous = label_ambiguity(row, label)
    if ambiguous:
        causes.append("label noise")
    if label is None:
        causes.append("old-ranker teacher-alignment issue")
    if not model.get("correct") and no_deltas.get("correct"):
        causes.append("option-delta calibration")
    if not model.get("correct") and no_effects.get("correct"):
        causes.append("decoded-effect / interaction overreaction")
    if not model.get("correct") and option0.get("correct"):
        causes.append("option-0 prior issue")
    if label is not None and not model.get("correct") and not ambiguous:
        causes.append("objective weighting")
    if label is not None and not model.get("correct"):
        causes.append("overfitting small n")
    if not causes:
        causes.append("no clear failure; acceptable or correct under current label")
    return causes


def load_replay(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_observation(row: dict, replay_dir: Path) -> dict | None:
    game_file = row.get("game_file")
    if not game_file:
        return None
    game = load_replay(replay_dir / game_file)
    if not game:
        return None
    target_hash = row.get("obs_hash")
    for step_i, step in enumerate(game.get("steps") or []):
        if not isinstance(step, list):
            continue
        for player, rec in enumerate(step):
            if not isinstance(rec, dict):
                continue
            obs = rec.get("observation") or {}
            if not SCH.is_single_pick_decision(obs):
                continue
            if TCR.obs_hash(obs) == target_hash:
                me = (obs.get("current") or {}).get("yourIndex", player)
                deck = TCR.player_deck(game, me) or TCR.player_deck(game, player)
                return {"game": game, "step": step_i, "player": player, "me": me, "obs": obs, "deck": deck}
    return None


def make_request_entry(row: dict, analysis: dict, replay_dir: Path) -> dict | None:
    found = find_observation(row, replay_dir)
    if not found:
        return None
    obs = found["obs"]
    decision_id = f"{row.get('game_file')}:{found['step']}:{found['player']}"
    why = [
        "held_out_mixed_test_state_without_teacher_v2_label",
        *analysis.get("classifications", []),
    ]
    return {
        "request_id": f"teacher_v2_failure_test_{int(analysis['index']):03d}",
        "decision_id": decision_id,
        "source": {
            "file": row.get("game_file"),
            "step": found["step"],
            "player": found["player"],
            "deck_n": len(found["deck"] or []),
        },
        "obs_hash": TCR.obs_hash(obs),
        "branch_b_dataset_obs_hash": row.get("obs_hash"),
        "deck_hash": TCR.deck_hash(found["deck"]),
        "deck": found["deck"],
        "observation": obs,
        "legal_options": copy.deepcopy((obs.get("select") or {}).get("option") or []),
        "current_label_source": row.get("source"),
        "why_requested": sorted(set(why)),
        "branch_b_current_analysis": {
            "model_eq": analysis["choices"]["teacher_v2_model"]["eq_class"],
            "old_ranker_eq": analysis["choices"]["old_ranker"]["eq_class"],
            "option0_eq": analysis["choices"]["option0"]["eq_class"],
            "current_best_eq": analysis["teacher_best"]["eq_class"],
            "model_regret_under_current_label": analysis["choices"]["teacher_v2_model"].get("regret"),
            "old_ranker_regret_under_current_label": analysis["choices"]["old_ranker"].get("regret"),
            "option0_regret_under_current_label": analysis["choices"]["option0"].get("regret"),
        },
    }


def summarize(decisions: list[dict]) -> dict:
    counts = Counter()
    cause_counts = Counter()
    source_counts = Counter(d["source"] for d in decisions)
    for d in decisions:
        for cls in d["classifications"]:
            counts[cls] += 1
        for cause in d["likely_causes"]:
            cause_counts[cause] += 1
    return {
        "n_test_decisions": len(decisions),
        "by_source": dict(sorted(source_counts.items())),
        "classification_counts": dict(sorted(counts.items())),
        "likely_cause_counts": dict(sorted(cause_counts.items())),
        "teacher_v2_labels_available": sum(1 for d in decisions if d["teacher_v2_label_available"]),
        "teacher_v2_labels_missing": sum(1 for d in decisions if not d["teacher_v2_label_available"]),
        "replay_test_rows_reinterpreted_with_teacher_v2": sum(
            1 for d in decisions if d["source"] == "replay_test" and d["teacher_v2_label_available"]
        ),
        "hand_outcome_disagreement": sum(
            1 for d in decisions if d["label_diagnostics"]["hand_outcome_argmax_disagree"] is True
        ),
    }


def choice_metrics(decisions: list[dict], choice_name: str) -> dict:
    choices = [d["choices"][choice_name] for d in decisions]
    regrets = [float(c.get("regret") or 0.0) for c in choices]
    return {
        "n": len(choices),
        "top1": sum(1 for c in choices if c.get("correct")) / len(choices) if choices else None,
        "acceptable_agreement": (
            sum(1 for c in choices if c.get("acceptable")) / len(choices) if choices else None
        ),
        "mean_regret": sum(regrets) / len(regrets) if regrets else None,
        "p90_regret": percentile(regrets, 0.90),
        "p95_regret": percentile(regrets, 0.95),
        "high_regret_count_ge_100": sum(1 for r in regrets if r >= 100.0),
        "high_regret_count_ge_1000": sum(1 for r in regrets if r >= 1000.0),
    }


def performance_summary(decisions: list[dict]) -> dict:
    names = [
        "teacher_v2_model",
        "old_ranker",
        "option0",
        "full_model_zero_effects",
        "full_model_zero_deltas",
    ]
    overall = {name: choice_metrics(decisions, name) for name in names}
    by_source = {}
    for source in sorted(set(d["source"] for d in decisions)):
        subset = [d for d in decisions if d["source"] == source]
        by_source[source] = {
            name: choice_metrics(subset, name)
            for name in ("teacher_v2_model", "old_ranker", "option0")
        }
    return {"overall": overall, "by_source": by_source}


def make_recommendation(summary: dict, performance: dict) -> dict:
    if summary["teacher_v2_labels_missing"]:
        return {
            "choice": "B",
            "title": "request more targeted labels from Model A",
            "rationale": (
                f"{summary['teacher_v2_labels_missing']}/{summary['n_test_decisions']} held-out rows still "
                "lack Teacher V2 labels, so another model change would still confound label-source mismatch "
                "with training behavior."
            ),
        }
    full = performance["overall"]["teacher_v2_model"]
    old = performance["overall"]["old_ranker"]
    opt0 = performance["overall"]["option0"]
    no_effects = performance["overall"]["full_model_zero_effects"]
    full_under_old = (
        (full["top1"] or 0.0) < (old["top1"] or 0.0)
        and (full["acceptable_agreement"] or 0.0) < (old["acceptable_agreement"] or 0.0)
        and (full["mean_regret"] or 0.0) > (old["mean_regret"] or 0.0)
    )
    zero_effects_better = (
        (no_effects["top1"] or 0.0) > (full["top1"] or 0.0)
        and (no_effects["mean_regret"] or 0.0) < (full["mean_regret"] or 0.0)
    )
    if full_under_old or zero_effects_better:
        return {
            "choice": "C",
            "title": "revise objective/weighting before retraining",
            "rationale": (
                "All held-out rows now have Teacher V2 labels, so the failure no longer looks like a "
                "label-source mismatch. The saved full model still trails old-ranker on top-1, acceptable "
                "agreement, and mean regret, while the zero-effects ablation is much better on regret. "
                "The next change should recalibrate the objective/weights around hand_norm_advantage and "
                "regularize decoded-effect/delta influence before another retrain."
            ),
        }
    if (full["mean_regret"] or 0.0) <= (old["mean_regret"] or 0.0) and (
        full["mean_regret"] or 0.0
    ) <= (opt0["mean_regret"] or 0.0):
        return {
            "choice": "A",
            "title": "retrain using the targeted labels included",
            "rationale": (
                "The targeted labels are aligned and the existing model is not clearly worse under Teacher V2 "
                "targets, so one narrow retrain with the targeted labels included is justified."
            ),
        }
    return {
        "choice": "D",
        "title": "pause Teacher V2 path because targeted labels still do not clarify the failure",
        "rationale": (
            "Teacher V2 labels are present, but the comparison does not identify a single actionable training "
            "change with enough confidence."
        ),
    }


def markdown_table(rows: list[list]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return "\n".join(out)


def write_markdown(path: Path, report: dict) -> None:
    s = report["summary"]
    perf = report["performance"]["overall"]
    rows = [["class", "count"]] + [[k, v] for k, v in s["classification_counts"].items()]
    cause_rows = [["likely cause", "count"]] + [[k, v] for k, v in s["likely_cause_counts"].items()]
    perf_rows = [[
        "model",
        "top1",
        "acceptable",
        "mean regret",
        "p90 regret",
        "p95 regret",
        ">=100 regret",
    ]]
    for name in ("teacher_v2_model", "old_ranker", "option0", "full_model_zero_effects", "full_model_zero_deltas"):
        m = perf[name]
        perf_rows.append([
            name,
            round(float(m["top1"] or 0.0), 3),
            round(float(m["acceptable_agreement"] or 0.0), 3),
            round(float(m["mean_regret"] or 0.0), 2),
            round(float(m["p90_regret"] or 0.0), 2),
            round(float(m["p95_regret"] or 0.0), 2),
            m["high_regret_count_ge_100"],
        ])
    decision_rows = [[
        "decision",
        "src",
        "classes",
        "model",
        "old",
        "opt0",
        "best",
        "model regret",
        "causes",
    ]]
    for d in report["decisions"]:
        decision_rows.append([
            d["decision_id"],
            d["source"],
            ",".join(d["classifications"]) or "-",
            d["choices"]["teacher_v2_model"]["eq_class"],
            d["choices"]["old_ranker"]["eq_class"],
            d["choices"]["option0"]["eq_class"],
            d["teacher_best"]["eq_class"],
            round(float(d["choices"]["teacher_v2_model"].get("regret") or 0.0), 2),
            ",".join(d["likely_causes"][:3]),
        ])

    text = f"""# Teacher V2 Failure Analysis

Branch: `exp/robust-learner-v2`

Status: analysis only. No retrain and no arena screen.

## Summary

- Held-out mixed test decisions: {s['n_test_decisions']}
- Teacher V2 labels available on test rows: {s['teacher_v2_labels_available']}
- Teacher V2 labels missing on test rows: {s['teacher_v2_labels_missing']}
- Replay-test rows reinterpreted with targeted Teacher V2 labels: {s['replay_test_rows_reinterpreted_with_teacher_v2']}
- Hand/outcome argmax disagreement on labelled test rows: {s['hand_outcome_disagreement']}/{s['n_test_decisions']}
- Recommendation: **{report['recommendation']['choice']}** - {report['recommendation']['title']}

{report['recommendation']['rationale']}

## Performance

{markdown_table(perf_rows)}

## Classifications

{markdown_table(rows)}

## Likely Causes

{markdown_table(cause_rows)}

## Decision Table

{markdown_table(decision_rows)}

## Request For Model A

Request file: `{report['request_for_A']['path']}`

Requested states: {report['request_for_A']['n_requested']}

Reason: {report['request_for_A']['criteria']}.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--teacher-v2", type=Path, nargs="+", default=[DEFAULT_TEACHER_V2])
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    ap.add_argument("--request-out", type=Path, default=DEFAULT_REQUEST)
    args = ap.parse_args()

    args.dataset = resolve(args.dataset)
    args.model = resolve(args.model)
    args.teacher_v2 = [resolve(path) for path in args.teacher_v2]
    args.replay_dir = resolve(args.replay_dir)
    args.json_out = resolve(args.json_out)
    args.md_out = resolve(args.md_out)
    args.request_out = resolve(args.request_out)

    dataset_payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = [r for r in dataset_payload["decisions"] if r["partition"] == "test"]
    labels = []
    for path in args.teacher_v2:
        labels.extend(load_jsonl(path))
    blob, model = load_model(args.model)

    decisions = []
    request_entries = []
    for idx, row in enumerate(rows):
        label = teacher_label_for_row(row, labels)
        eval_row = row_with_teacher_targets(row, label)
        logits = option_logits(model, row, blob)
        scores = class_scores_from_option_logits(logits, eval_row["eq"])
        model_eq = max(scores, key=lambda k: (scores[k], -k))
        no_effects_scores = class_scores_from_option_logits(option_logits(model, row, blob, {"effects": True}), eval_row["eq"])
        no_deltas_scores = class_scores_from_option_logits(option_logits(model, row, blob, {"deltas": True}), eval_row["eq"])
        no_effects_eq = max(no_effects_scores, key=lambda k: (no_effects_scores[k], -k))
        no_deltas_eq = max(no_deltas_scores, key=lambda k: (no_deltas_scores[k], -k))

        model_choice = choice_from_eq(eval_row, model_eq, "teacher_v2_model", logits)
        old_choice = choice_from_eq(eval_row, row.get("old_ranker_eq"), "old_ranker")
        option0_choice = choice_from_eq(eval_row, row.get("option0_eq"), "option0")
        no_effects_choice = choice_from_eq(eval_row, no_effects_eq, "full_model_zero_effects")
        no_deltas_choice = choice_from_eq(eval_row, no_deltas_eq, "full_model_zero_deltas")
        teacher_best = best_choice(eval_row, label)

        for choice in (model_choice, old_choice, option0_choice, no_effects_choice, no_deltas_choice, teacher_best):
            choice.update(option_diag_from_teacher(label, choice.get("option_index")))

        adv = to_float_map(eval_row.get("adv"))
        out_margin = outcome_margin(label)
        analysis = {
            "index": idx,
            "decision_id": (
                label.get("decision_id") if label
                else f"{row.get('game_file')}:{row.get('obs_hash')}"
            ),
            "obs_hash": row.get("obs_hash"),
            "teacher_v2_obs_hash": row.get("teacher_v2_obs_hash") or (label.get("obs_hash") if label else None),
            "source": row.get("source"),
            "partition": row.get("partition"),
            "game_file": row.get("game_file"),
            "step": row.get("step", row.get("call")),
            "player": row.get("player"),
            "teacher_v2_label_available": bool(label),
            "current_label_source": "teacher_v2" if label else "existing_contextual_dataset_label",
            "teacher_best": teacher_best,
            "choices": {
                "teacher_v2_model": model_choice,
                "old_ranker": old_choice,
                "option0": option0_choice,
                "full_model_zero_effects": no_effects_choice,
                "full_model_zero_deltas": no_deltas_choice,
            },
            "label_diagnostics": {
                "criticality": label.get("criticality") if label else row.get("criticality_score"),
                "hand_advantage_margin": top_margin(adv),
                "hand_value_variance_mean": eval_row.get("value_variance_mean"),
                "outcome_winrate_margin": out_margin,
                "top_outcome_se": top_outcome_se(label),
                "hand_outcome_argmax_disagree": (label.get("hand_outcome_agree") is False) if label else None,
                "acceptable_eqs": [k for k, v in to_float_map(eval_row.get("acceptable")).items() if v >= 0.5],
                "teacher_stability": eval_row.get("teacher_stability"),
                "teacher_confidence": eval_row.get("teacher_confidence"),
                "top_two_margin": eval_row.get("top_two_margin"),
                "ambiguity_reasons": label_ambiguity(eval_row, label),
            },
        }
        analysis["classifications"] = classify(eval_row, model_choice, old_choice, option0_choice, label)
        analysis["likely_causes"] = likely_causes(
            eval_row, model_choice, old_choice, option0_choice, label, no_effects_choice, no_deltas_choice
        )
        decisions.append(analysis)

        if not label:
            req = make_request_entry(row, analysis, args.replay_dir)
            if req:
                request_entries.append(req)

    unique_request_entries = []
    seen_request_roots = set()
    for req in request_entries:
        key = req.get("branch_b_dataset_obs_hash") or req.get("decision_id")
        if key in seen_request_roots:
            continue
        seen_request_roots.add(key)
        unique_request_entries.append(req)
    request_entries = unique_request_entries

    summary = summarize(decisions)
    performance = performance_summary(decisions)
    recommendation = make_recommendation(summary, performance)
    report = {
        "artifact_version": "teacher_v2_failure_analysis.v1",
        "branch": "exp/robust-learner-v2",
        "inputs": {
            "dataset": str(args.dataset),
            "model": str(args.model),
            "teacher_v2": [str(path) for path in args.teacher_v2],
            "replay_dir": str(args.replay_dir),
        },
        "live_agent_consumed": "none",
        "arena_screen": "not run",
        "summary": summary,
        "performance": performance,
        "recommendation": recommendation,
        "decisions": decisions,
        "failures": [d for d in decisions if not d["choices"]["teacher_v2_model"].get("correct")],
        "request_for_A": {
            "path": str(args.request_out),
            "n_requested": len(request_entries),
            "criteria": (
                "unique held-out replay-test roots lacking Teacher V2 labels"
                if request_entries
                else "none; targeted labels cover all held-out mixed-test rows in this analysis"
            ),
        },
    }
    request = {
        "artifact_version": "teacher_v2_label_request_for_A.v1",
        "branch": "exp/robust-learner-v2",
        "purpose": "Targeted labels for Branch B Teacher V2 retrain failure analysis; not a generic larger batch.",
        "requested_output": "data/manifests/teacher_v2_labels_for_B_failures.jsonl",
        "labeling_requirements": [
            "observation",
            "legal_options",
            "decision_id",
            "obs_hash",
            "option index",
            "semantic_action_key",
            "eq_class",
            "hand_norm_advantage",
            "hand_value_variance",
            "criticality",
            "outcome_winrate",
            "outcome_se",
            "coverage",
            "timing",
            "seed / paired metadata",
        ],
        "requests": request_entries,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.request_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.request_out.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.md_out, report)

    print(json.dumps({
        "json": str(args.json_out.relative_to(ROOT)),
        "markdown": str(args.md_out.relative_to(ROOT)),
        "request": str(args.request_out.relative_to(ROOT)),
        "summary": report["summary"],
        "recommendation": report["recommendation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
