"""Search Metadata Dominance Audit V1.

This is a bounded follow-up to Continuous Terrain Representation V1.  It asks
whether the strong R1 search-metadata result is a deployable live-search signal
or an artifact of feature/label leakage from the stronger teacher computation.

The audit intentionally does not train another terrain model, does not modify
agent_search, and does not run arena games.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import train_continuous_terrain_v1 as TERRAIN  # noqa: E402


DEFAULT_LABELS = ROOT / "data" / "manifests" / "continuous_terrain_v1.jsonl"
DEFAULT_METADATA = ROOT / "agent" / "continuous_terrain_encoder_v1.json"
DEFAULT_EVAL = ROOT / "docs" / "workstreams" / "continuous_terrain_representation_v1_eval.json"
DEFAULT_OUT_JSON = ROOT / "docs" / "workstreams" / "search_metadata_dominance_audit_v1.json"
DEFAULT_OUT_MD = ROOT / "docs" / "workstreams" / "SEARCH_METADATA_DOMINANCE_AUDIT_V1.md"


@dataclass
class Probe:
    w: torch.Tensor
    b: torch.Tensor
    median: np.ndarray
    scale: np.ndarray


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-6, scale, x.std(axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    return med, scale


def apply_stats(x: np.ndarray, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((x - med) / scale, -8.0, 8.0).astype(np.float32)


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Probe:
    if len(np.unique(y_train >= 0.5)) < 2:
        med, scale = robust_stats(x_train)
        w = torch.zeros(x_train.shape[1], 1)
        b = torch.tensor([float(np.mean(y_train))])
        return Probe(w=w, b=b, median=med, scale=scale)
    med, scale = robust_stats(x_train)
    xz = apply_stats(x_train, med, scale)
    torch.manual_seed(seed)
    w = torch.zeros(xz.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=0.04, weight_decay=1e-3)
    xx = torch.tensor(xz, dtype=torch.float32)
    yy = torch.tensor((y_train >= 0.5).astype(np.float32), dtype=torch.float32)
    pos = yy.sum()
    neg = len(yy) - pos
    pos_weight = torch.tensor([min(25.0, max(1.0, float(neg / max(1.0, pos))))])
    for _ in range(160):
        opt.zero_grad(set_to_none=True)
        logits = (xx @ w).squeeze(-1) + b
        loss = F.binary_cross_entropy_with_logits(logits, yy, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    return Probe(w=w.detach(), b=b.detach(), median=med, scale=scale)


def probe_predict(probe: Probe, x: np.ndarray) -> np.ndarray:
    xz = apply_stats(x, probe.median, probe.scale)
    if probe.w.abs().sum().item() == 0.0 and 0.0 <= float(probe.b.item()) <= 1.0:
        return np.full(x.shape[0], float(probe.b.item()), dtype=np.float32)
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(xz, dtype=torch.float32) @ probe.w + probe.b).squeeze(-1).numpy()


def threshold_at_train_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    yy = (np.asarray(y) >= 0.5).astype(np.int32)
    ss = np.asarray(score, dtype=np.float64)
    neg = int((yy == 0).sum())
    if neg == 0:
        return float(np.min(ss) - 1e-9)
    candidates = sorted(set(float(s) for s in ss), reverse=True)
    best = candidates[-1] + 1e-9
    for thr in candidates:
        pred = ss >= thr
        fpr = int(((pred == 1) & (yy == 0)).sum()) / max(1, neg)
        if fpr <= max_fpr + 1e-12:
            best = thr
    return float(best)


def metric_block(y: np.ndarray, score: np.ndarray) -> dict:
    return TERRAIN.metric_block(np.asarray(y, dtype=np.float32).tolist(), np.asarray(score, dtype=np.float32).tolist())


def option_alias(opt: dict, *names: str, default=0.0):
    for name in names:
        if name in opt and opt[name] is not None:
            return opt[name]
    return default


def live_distribution(label: dict, n: int, selected: int | None) -> list[float]:
    raw = label.get("live_selected_action_distribution") or label.get("live_selected_distribution") or {}
    if isinstance(raw, list):
        vals = [float(raw[i]) if i < len(raw) else 0.0 for i in range(n)]
    elif isinstance(raw, dict):
        vals = [float(raw.get(str(i), raw.get(i, 0.0)) or 0.0) for i in range(n)]
    else:
        vals = [0.0] * n
    total = sum(vals)
    if total > 0:
        return [v / total for v in vals]
    out = [0.0] * n
    if selected is not None and 0 <= selected < n:
        out[selected] = 1.0
    return out


def entropy(probs: list[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    return float(-sum(p * math.log(max(p, 1e-9)) for p in probs) / math.log(max(2, len(probs))))


def label_lookup(labels: list[dict]) -> dict[str, dict]:
    return {str(l.get("decision_id")): l for l in labels}


def row_live_context(label: dict, option_index: int) -> dict:
    options = {int(o["index"]): o for o in (label.get("options") or [])}
    opt = options[option_index]
    n = max(options) + 1 if options else 0
    selected_raw = label.get("search_selected_option")
    try:
        selected = int(selected_raw)
    except Exception:
        selected = None
    values = [
        float(option_alias(options[i], "mean_live_value", "mean_live_search_value", "live_search_value_mean", "current_search_value", default=0.0))
        for i in range(n)
    ]
    probs = live_distribution(label, n, selected)
    order = np.argsort(-np.asarray(values))
    ranks = {int(idx): int(rank) for rank, idx in enumerate(order)}
    sorted_values = sorted(values, reverse=True)
    margin = sorted_values[0] - sorted_values[1] if len(sorted_values) > 1 else 0.0
    spread = float(np.std(values)) if values else 0.0
    live_value = values[option_index]
    best_live = sorted_values[0] if sorted_values else live_value
    live_var = float(option_alias(opt, "live_value_variance", "live_search_value_variance", default=0.0))
    live_se = float(option_alias(opt, "live_value_se", "live_search_value_se", default=0.0))
    completed_live = float(option_alias(opt, "live_completed_determinizations", "completed_determinizations", default=0.0))
    ent = float(label.get("live_action_entropy") if label.get("live_action_entropy") is not None else entropy(probs))
    modal = float(label.get("modal_action_stability") if label.get("modal_action_stability") is not None else (max(probs) if probs else 0.0))
    return {
        "live_value": live_value,
        "live_centered": live_value - float(np.mean(values)) if values else 0.0,
        "live_gap_to_best": best_live - live_value,
        "live_rank": ranks.get(option_index, n - 1) / max(1.0, n - 1.0),
        "live_margin": margin,
        "live_spread": spread,
        "live_variance": live_var,
        "live_se": live_se,
        "live_completed": completed_live,
        "live_prob": probs[option_index] if option_index < len(probs) else 0.0,
        "live_entropy": ent,
        "live_modal": modal,
        "selected_bit": 1.0 if selected == option_index else 0.0,
        "n_options": n / 20.0,
    }


def build_feature_matrices(examples: list[dict], labels_by_decision: dict[str, dict]) -> dict[str, np.ndarray]:
    rows = {
        "previous_R1_suspect": [],
        "live_n8_values_only": [],
        "live_n8_uncertainty_only": [],
        "live_n8_strict_prechoice": [],
        "live_n8_strict_postchoice": [],
        "leakage_control_stronger": [],
    }
    for e in examples:
        label = labels_by_decision[str(e["decision_id"])]
        ctx = row_live_context(label, int(e["option_index"]))
        opt = {int(o["index"]): o for o in (label.get("options") or [])}[int(e["option_index"])]
        rows["previous_R1_suspect"].append(np.asarray(e["search_metadata_only"], dtype=np.float32))
        rows["live_n8_values_only"].append(np.asarray([
            ctx["live_value"],
            ctx["live_centered"],
            ctx["live_gap_to_best"],
            ctx["live_rank"],
            ctx["live_margin"],
            ctx["live_spread"],
            ctx["n_options"],
        ], dtype=np.float32))
        rows["live_n8_uncertainty_only"].append(np.asarray([
            ctx["live_variance"],
            ctx["live_se"],
            ctx["live_completed"],
            ctx["live_prob"],
            ctx["live_entropy"],
            ctx["live_modal"],
            ctx["n_options"],
        ], dtype=np.float32))
        rows["live_n8_strict_prechoice"].append(np.asarray([
            ctx["live_value"],
            ctx["live_centered"],
            ctx["live_gap_to_best"],
            ctx["live_rank"],
            ctx["live_margin"],
            ctx["live_spread"],
            ctx["live_variance"],
            ctx["live_se"],
            ctx["live_completed"],
            ctx["live_prob"],
            ctx["live_entropy"],
            ctx["live_modal"],
            ctx["n_options"],
        ], dtype=np.float32))
        rows["live_n8_strict_postchoice"].append(np.asarray([
            ctx["live_value"],
            ctx["live_centered"],
            ctx["live_gap_to_best"],
            ctx["live_rank"],
            ctx["live_margin"],
            ctx["live_spread"],
            ctx["live_variance"],
            ctx["live_se"],
            ctx["live_completed"],
            ctx["live_prob"],
            ctx["live_entropy"],
            ctx["live_modal"],
            ctx["selected_bit"],
            ctx["n_options"],
        ], dtype=np.float32))
        rows["leakage_control_stronger"].append(np.asarray([
            float(option_alias(opt, "mean_stronger_value", "stronger_value", default=0.0)),
            float(option_alias(opt, "delta_to_search", default=0.0)),
            float(option_alias(opt, "delta_to_search_norm", default=0.0)),
            float(option_alias(opt, "hand_norm_advantage", default=0.0)),
            float(option_alias(opt, "high_regret_prob", default=0.0)),
            float(option_alias(opt, "unacceptable_prob", default=0.0)),
            float(option_alias(opt, "acceptable_prob", default=0.0)),
        ], dtype=np.float32))
    return {name: np.stack(vals) for name, vals in rows.items()}


def probe_metrics(examples: list[dict], matrices: dict[str, np.ndarray], seed: int) -> dict:
    train_idx = [i for i, e in enumerate(examples) if e["partition"] == "train"]
    test_idx = [i for i, e in enumerate(examples) if e["partition"] == "test"]
    targets = {
        "high_regret": lambda e: e.get("high_regret"),
        "unacceptable": lambda e: e.get("unacceptable"),
        "selected_high_regret": lambda e: e.get("selected_high_regret"),
    }
    out = {}
    for target, getter in targets.items():
        out[target] = {}
        kept_train = [i for i in train_idx if getter(examples[i]) is not None]
        kept_test = [i for i in test_idx if getter(examples[i]) is not None]
        y_train = np.asarray([float(getter(examples[i])) for i in kept_train], dtype=np.float32)
        y_test = np.asarray([float(getter(examples[i])) for i in kept_test], dtype=np.float32)
        for name, mat in matrices.items():
            probe = fit_probe(mat[kept_train], y_train, seed + len(target) * 17 + len(name))
            score = probe_predict(probe, mat[kept_test])
            out[target][name] = metric_block(y_test, score)
    return out


def decision_records(labels: list[dict], examples: list[dict]) -> list[dict]:
    by_row = {(str(e["decision_id"]), int(e["option_index"])): e for e in examples}
    records = []
    for label in labels:
        if label.get("eval_only"):
            continue
        did = str(label.get("decision_id"))
        options = {int(o["index"]): o for o in (label.get("options") or [])}
        if not options:
            continue
        first = by_row.get((did, min(options)))
        if not first or first.get("partition") != "test":
            continue
        try:
            selected = int(label.get("search_selected_option"))
        except Exception:
            selected = max(options, key=lambda i: float(option_alias(options[i], "mean_live_value", default=0.0)))
        if selected not in options:
            continue
        stronger_values = {
            i: float(option_alias(opt, "mean_stronger_value", "stronger_value", default=0.0))
            for i, opt in options.items()
        }
        live_values = {
            i: float(option_alias(opt, "mean_live_value", "current_search_value", default=0.0))
            for i, opt in options.items()
        }
        best_stronger = max(stronger_values, key=stronger_values.get)
        best_value = stronger_values[best_stronger]
        records.append({
            "decision_id": did,
            "obs_hash": label.get("obs_hash"),
            "partition": first.get("partition"),
            "selected": selected,
            "best_stronger": best_stronger,
            "options": sorted(options),
            "stronger_values": stronger_values,
            "live_values": live_values,
            "selected_high_regret": float(option_alias(options[selected], "high_regret_prob", default=0.0)),
            "selected_unacceptable": float(option_alias(options[selected], "unacceptable_prob", default=0.0)),
            "selected_regret": max(0.0, best_value - stronger_values[selected]),
            "best_high_regret": float(option_alias(options[best_stronger], "high_regret_prob", default=0.0)),
            "best_unacceptable": float(option_alias(options[best_stronger], "unacceptable_prob", default=0.0)),
            "best_regret": 0.0,
        })
    return records


def offline_trigger_eval(
    examples: list[dict],
    labels: list[dict],
    matrices: dict[str, np.ndarray],
    labels_by_decision: dict[str, dict],
    seed: int,
) -> dict:
    row_by_key = {(str(e["decision_id"]), int(e["option_index"])): i for i, e in enumerate(examples)}
    train_sel = [i for i, e in enumerate(examples) if e["partition"] == "train" and e.get("selected_high_regret") is not None]
    y_train = np.asarray([float(examples[i]["selected_high_regret"]) for i in train_sel], dtype=np.float32)
    records = decision_records(labels, examples)
    out = {}
    for name in ("previous_R1_suspect", "live_n8_strict_prechoice", "live_n8_strict_postchoice"):
        probe = fit_probe(matrices[name][train_sel], y_train, seed + len(name) + 911)
        train_scores = probe_predict(probe, matrices[name][train_sel])
        threshold = threshold_at_train_fpr(y_train, train_scores, 0.10)
        all_scores = probe_predict(probe, matrices[name])
        triggered = []
        before = []
        oracle = []
        live_filter = []
        false_positive_safe = 0
        caught_bad = 0
        missed_bad = 0
        for rec in records:
            selected_idx = row_by_key[(rec["decision_id"], rec["selected"])]
            score = float(all_scores[selected_idx])
            is_trigger = score >= threshold
            triggered.append(is_trigger)
            selected_bad = rec["selected_high_regret"] >= 0.5
            if is_trigger and selected_bad:
                caught_bad += 1
            if (not is_trigger) and selected_bad:
                missed_bad += 1
            if is_trigger and not selected_bad:
                false_positive_safe += 1
            before.append({
                "high_regret": rec["selected_high_regret"],
                "unacceptable": rec["selected_unacceptable"],
                "regret": rec["selected_regret"],
            })
            oracle_choice = rec["best_stronger"] if is_trigger else rec["selected"]
            oracle_opt = labels_by_decision[rec["decision_id"]]["options"][oracle_choice]
            oracle.append({
                "high_regret": float(option_alias(oracle_opt, "high_regret_prob", default=0.0)),
                "unacceptable": float(option_alias(oracle_opt, "unacceptable_prob", default=0.0)),
                "regret": max(0.0, rec["stronger_values"][rec["best_stronger"]] - rec["stronger_values"][oracle_choice]),
            })
            if is_trigger:
                candidates = []
                for oi in rec["options"]:
                    ri = row_by_key[(rec["decision_id"], oi)]
                    if float(all_scores[ri]) < threshold:
                        candidates.append(oi)
                if candidates:
                    live_choice = max(candidates, key=lambda oi: rec["live_values"][oi])
                else:
                    live_choice = rec["selected"]
            else:
                live_choice = rec["selected"]
            live_opt = labels_by_decision[rec["decision_id"]]["options"][live_choice]
            live_filter.append({
                "high_regret": float(option_alias(live_opt, "high_regret_prob", default=0.0)),
                "unacceptable": float(option_alias(live_opt, "unacceptable_prob", default=0.0)),
                "regret": max(0.0, rec["stronger_values"][rec["best_stronger"]] - rec["stronger_values"][live_choice]),
            })
        def summarize(rows: list[dict]) -> dict:
            return {
                "high_regret_count": int(sum(1 for r in rows if r["high_regret"] >= 0.5)),
                "unacceptable_count": int(sum(1 for r in rows if r["unacceptable"] >= 0.5)),
                "mean_regret": float(np.mean([r["regret"] for r in rows])) if rows else 0.0,
                "p95_regret": float(np.percentile([r["regret"] for r in rows], 95)) if rows else 0.0,
            }
        out[name] = {
            "threshold_train_fpr_10": threshold,
            "test_decisions": len(records),
            "trigger_count": int(sum(triggered)),
            "trigger_rate": float(np.mean(triggered)) if triggered else 0.0,
            "caught_selected_high_regret": caught_bad,
            "missed_selected_high_regret": missed_bad,
            "false_positive_safe_selected": false_positive_safe,
            "before_live_selected": summarize(before),
            "after_oracle_extra_search_upper_bound": summarize(oracle),
            "after_live_next_unflagged_proxy": summarize(live_filter),
        }
    return out


def leakage_and_metric_integrity(labels: list[dict], examples: list[dict], previous_eval: dict) -> dict:
    hi = np.asarray([float(e["high_regret"]) >= 0.5 for e in examples])
    un = np.asarray([float(e["unacceptable"]) >= 0.5 for e in examples])
    test = np.asarray([e["partition"] == "test" for e in examples])
    live_missing_stronger_present = 0
    value_se_present = 0
    for label in labels:
        for opt in label.get("options") or []:
            if "live_value_variance" not in opt and "stronger_value_variance" in opt:
                live_missing_stronger_present += 1
            if "value_se" in opt or "live_value_se" in opt or "stronger_value_se" in opt:
                value_se_present += 1
    prev_un = (previous_eval.get("predictive_metrics") or {}).get("unacceptable") or {}
    prev_hi = (previous_eval.get("predictive_metrics") or {}).get("high_regret") or {}
    return {
        "previous_r1_feature_audit": {
            "fields": [
                "live value margin",
                "live value spread",
                "option value variance through alias",
                "option value SE through alias",
                "live action entropy",
                "modal action stability",
                "criticality score",
                "selected option bit",
            ],
            "direct_stronger_value_fields_used": False,
            "live_variance_missing_but_stronger_variance_present_options": live_missing_stronger_present,
            "value_se_present_options": value_se_present,
            "non_deployable_or_suspect_fields": [
                "criticality score is dataset-selection/label metadata, not an N=8 live-search output",
                "value_se provenance is not explicit enough to keep in strict live-only R1",
            ],
        },
        "semantic_feature_leakage": {
            "policy_target_in_action_scalars": True,
            "why_it_matters": "semantic_action_scalars appends policy_prob from stronger_soft_policy/policy_from_label, so R3/R4 are not leakage-clean deployable feature conclusions.",
        },
        "target_construction_coupling": {
            "high_regret_uses_live_vs_stronger_residual": True,
            "why_it_matters": "Live value features can predict residual/high-regret labels partly because live value is one input to the label definition; this is not direct stronger-label leakage, but it is not independent causal evidence either.",
        },
        "r4_fusion_audit": {
            "metadata_branch_contains_criticality_and_coverage": True,
            "action_scalars_contain_teacher_policy_target": True,
            "interpretation": "R4 fusion metrics should be treated as diagnostic only; they are not a live-safe fusion of semantic plus search metadata.",
        },
        "unacceptable_table_audit": {
            "score_key_bug_found": False,
            "all_rows_high_regret_equals_unacceptable": int(np.sum(hi == un)),
            "all_rows_total": int(len(examples)),
            "test_rows_high_regret_equals_unacceptable": int(np.sum((hi == un) & test)),
            "test_rows_total": int(np.sum(test)),
            "high_regret_positive_count": int(np.sum(hi)),
            "unacceptable_positive_count": int(np.sum(un)),
            "previous_high_regret_R4_ap": (prev_hi.get("R4_semantic_plus_search") or {}).get("average_precision"),
            "previous_unacceptable_R4_ap": (prev_un.get("R4_semantic_plus_search") or {}).get("average_precision"),
            "interpretation": "The odd unacceptable table is mostly label identity/near-identity, not a learned-score key mismatch.",
        },
    }


def write_markdown(path: Path, report: dict) -> None:
    pm = report["probe_metrics"]
    trig = report["offline_trigger_eval"]
    integ = report["leakage_and_metric_integrity"]
    lines = [
        "# Search Metadata Dominance Audit V1",
        "",
        "## Verdict",
        "",
        f"**{report['decision']['verdict']}**",
        "",
        report["decision"]["summary"],
        "",
        "## Dataset",
        "",
        f"- Input: `{report['input']}`",
        f"- Decisions/options/games: {report['dataset']['decisions']} / {report['dataset']['options']} / {report['dataset']['games']}",
        f"- Test decisions/options: {report['dataset']['test_decisions']} / {report['dataset']['test_options']}",
        "- Arena/live screen: not run",
        "- `agent_search`: not modified",
        "",
        "## High-Regret Prediction",
        "",
        "| feature set | AP | AUROC | recall@FPR10 | positives/test rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(pm["high_regret"].items()):
        lines.append(
            f"| `{name}` | {fmt(metrics.get('average_precision'))} | {fmt(metrics.get('auroc'))} | "
            f"{fmt(metrics.get('recall_at_fpr_10'))} | {metrics.get('positives')}/{metrics.get('n')} |"
        )
    lines.extend([
        "",
        "## Selected-Action Trigger Proxy",
        "",
        "Thresholds are calibrated on train selected rows for <=10% safe-action FPR, then evaluated on held-out test decisions.",
        "",
        "| feature set | trigger rate | caught bad | missed bad | false safe triggers | before high/p95 regret | oracle extra-search high/p95 | live-next-unflagged high/p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, row in sorted(trig.items()):
        before = row["before_live_selected"]
        oracle = row["after_oracle_extra_search_upper_bound"]
        live = row["after_live_next_unflagged_proxy"]
        lines.append(
            f"| `{name}` | {fmt(row['trigger_rate'])} | {row['caught_selected_high_regret']} | "
            f"{row['missed_selected_high_regret']} | {row['false_positive_safe_selected']} | "
            f"{before['high_regret_count']}/{fmt(before['p95_regret'])} | "
            f"{oracle['high_regret_count']}/{fmt(oracle['p95_regret'])} | "
            f"{live['high_regret_count']}/{fmt(live['p95_regret'])} |"
        )
    lines.extend([
        "",
        "## Integrity Findings",
        "",
        "- Previous R1 did not use direct stronger values, but it was not strict live-only: it included dataset criticality metadata and ambiguous value-SE provenance.",
        "- Strict live N=8 values still predict high-regret strongly, but high-regret is a residual-style label built from live-vs-stronger values, so this is target-coupled evidence.",
        "- Strict live-only N=8 metadata must exclude stronger values, deltas, hand advantages, high-regret/unacceptable probabilities, criticality metadata, and teacher policy targets.",
        "- The semantic/R4 path leaks `policy_prob` from the stronger soft policy through `action_scalars`, so the previous R3/R4 conclusions are diagnostic only.",
        f"- High-regret and unacceptable binary labels are identical on {integ['unacceptable_table_audit']['all_rows_high_regret_equals_unacceptable']}/{integ['unacceptable_table_audit']['all_rows_total']} rows.",
        "- I did not find a score-key bug in the unacceptable table; the oddness is label identity/near-identity plus fusion leakage, not a table indexing error.",
        "",
        "## Recommendation",
        "",
        report["decision"]["recommendation"],
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(x) -> str:
    if x is None:
        return "-"
    return f"{float(x):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--model-metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--previous-eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--seed", type=int, default=61357)
    ap.add_argument("--max-entities", type=int, default=36)
    ap.add_argument("--high-regret-threshold", type=float, default=5000.0)
    ap.add_argument("--residual-clip", type=float, default=50000.0)
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    args = ap.parse_args()
    for key in ("labels", "model_metadata", "previous_eval", "out_json", "out_md", "replay_dir"):
        val = getattr(args, key)
        setattr(args, key, val if val.is_absolute() else ROOT / val)

    labels = TERRAIN.load_jsonl(args.labels)
    examples, failures = TERRAIN.build_option_examples(labels, args)
    metadata = load_json(args.model_metadata)
    split = metadata.get("split") or TERRAIN.group_split(examples, args.seed)
    TERRAIN.assign_partition(examples, split)
    labels_by_decision = label_lookup(labels)
    matrices = build_feature_matrices(examples, labels_by_decision)
    pm = probe_metrics(examples, matrices, args.seed)
    triggers = offline_trigger_eval(examples, labels, matrices, labels_by_decision, args.seed)
    previous_eval = load_json(args.previous_eval)
    integrity = leakage_and_metric_integrity(labels, examples, previous_eval)

    test_decisions = {e["decision_id"] for e in examples if e["partition"] == "test"}
    high = pm["high_regret"]
    strict = high["live_n8_strict_postchoice"]
    prev = high["previous_R1_suspect"]
    values = high["live_n8_values_only"]
    verdict = "B. LIVE N8 METADATA HAS SIGNAL BUT DEPLOYABLE USE IS NOT VALIDATED"
    summary = (
        "Strict live-available N=8 metadata still predicts high-regret better than chance, "
        "but high-regret labels are target-coupled to live values and the deployable proxy did "
        "not improve selected-action safety."
    )
    recommendation = (
        "Next, do not run a large representation experiment or live screen. First build a small offline "
        "extra-search simulator for triggered states, or request targeted higher-k labels on triggered "
        "false positives/false negatives, after removing teacher policy targets from all model inputs."
    )
    if (strict.get("average_precision") or 0.0) < max(0.05, (values.get("positive_rate") or 0.0) * 1.5):
        verdict = "D. LIVE N8 METADATA NOT VALIDATED"
        summary = "After leakage cleanup, strict live N=8 metadata does not show enough high-regret signal."
    else:
        trigger = triggers.get("live_n8_strict_postchoice") or {}
        before = trigger.get("before_live_selected") or {}
        live_after = trigger.get("after_live_next_unflagged_proxy") or {}
        oracle_after = trigger.get("after_oracle_extra_search_upper_bound") or {}
        live_proxy_safe = (
            live_after.get("high_regret_count", 10**9) <= before.get("high_regret_count", -1)
            and live_after.get("mean_regret", float("inf")) <= before.get("mean_regret", -1)
        )
        oracle_promising = (
            oracle_after.get("high_regret_count", 10**9) < before.get("high_regret_count", -1)
            or oracle_after.get("mean_regret", float("inf")) < before.get("mean_regret", -1)
        )
        if live_proxy_safe and oracle_promising:
            verdict = "A. LIVE N8 EXTRA-SEARCH TRIGGER CANDIDATE"
            summary = "Strict live N=8 metadata retains strong signal and the offline trigger proxy improves selected-action safety."
            recommendation = (
                "Consider one tiny live screen only after implementing a conservative extra-search trigger "
                "that leaves normal search authority intact."
            )
        elif oracle_promising:
            verdict = "B. LIVE N8 WARNING SIGNAL; EXTRA-SEARCH MECHANISM STILL UNPROVEN"
            summary = (
                "Strict live N=8 metadata retains strong high-regret signal and an oracle stronger-search "
                "upper bound improves, but the deployable live-next-unflagged proxy gets worse."
            )
            recommendation = (
                "Next, build a tiny offline extra-search simulator for only the triggered test states, or "
                "ask Model A for higher-k labels on the triggered false positives/false negatives. Do not "
                "promote a live risk rule from this audit alone."
            )

    report = {
        "artifact_version": "search_metadata_dominance_audit_v1",
        "branch": "exp/robust-learner-v2",
        "input": display(args.labels),
        "agent_search_modified": False,
        "arena_screen": "not run",
        "dataset": {
            "decisions": len({e["decision_id"] for e in examples}),
            "options": len(examples),
            "games": len({e["group_id"] for e in examples}),
            "test_decisions": len(test_decisions),
            "test_options": sum(1 for e in examples if e["partition"] == "test"),
            "failures": failures,
        },
        "feature_sets": {
            "previous_R1_suspect": "Original R1 vector from the prior run; includes criticality and ambiguous SE.",
            "live_n8_values_only": "Only live N=8 sibling values/rank/margin/spread/count.",
            "live_n8_uncertainty_only": "Only live N=8 variance/SE/completion/distribution stability/count.",
            "live_n8_strict_prechoice": "Live N=8 values plus uncertainty and stability, no selected bit.",
            "live_n8_strict_postchoice": "Strict prechoice plus the N=8 selected-option bit.",
            "leakage_control_stronger": "Known label-derived stronger/value/residual/flag fields; upper-bound leakage control only.",
        },
        "probe_metrics": pm,
        "offline_trigger_eval": triggers,
        "leakage_and_metric_integrity": integrity,
        "decision": {
            "verdict": verdict,
            "summary": summary,
            "recommendation": recommendation,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.out_md, report)
    print(json.dumps({
        "verdict": verdict,
        "outputs": {"json": display(args.out_json), "markdown": display(args.out_md)},
        "strict_postchoice_high_regret_ap": strict.get("average_precision"),
        "previous_r1_high_regret_ap": prev.get("average_precision"),
        "agent_search_modified": False,
        "arena_screen": "not run",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
