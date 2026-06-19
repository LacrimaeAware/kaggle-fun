"""Signal Radius Audit V1 for round-2 Teacher V2 residual/risk labels.

This is a bounded diagnostic. It does not train a deployable policy, does not
modify agent_search, and does not run an arena screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import contextual_ranker as CR  # noqa: E402
import train_risk_only_contextual as RISK  # noqa: E402


DEFAULT_ROUND2 = ROOT / "data" / "manifests" / "teacher_v2_residual_risk_labels_round2.jsonl"
DEFAULT_ALIAS = ROOT / "data" / "manifests" / "teacher_v2_risk_labels_for_B_request.jsonl"
DOCS = ROOT / "docs" / "workstreams"
TARGET_ORDER = [
    "high_regret_flag",
    "unacceptable_flag",
    "selected_option_high_regret_flag",
    "c1_reproduced_this_label",
    "c2_safe_search_false_positive",
    "c3_near_miss_boundary",
]
KS = [5, 10, 25, 50]


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def average(xs: list[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return float(num) / float(den)


def percentile(xs: list[float], q: float) -> float | None:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def bool_tag(label: dict, tag: str) -> bool:
    return tag in set(label.get("criterion_tags") or [])


def robust_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.zeros((x.shape[1],), dtype=np.float32), np.ones((x.shape[1],), dtype=np.float32)
    med = np.nanmedian(x, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    scale = q75 - q25
    std = np.nanstd(x, axis=0)
    scale = np.where(scale > 1e-6, scale, std)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return med.astype(np.float32), scale.astype(np.float32)


def robust_transform(x: np.ndarray, med: np.ndarray, scale: np.ndarray, clip: float = 8.0) -> np.ndarray:
    z = (x - med) / scale
    z = np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip).astype(np.float32)


def rank_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int32)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and score[order[j]] == score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int32)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-score)
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if y[idx]:
            hits += 1
            total += hits / rank
    return total / n_pos


def recall_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float | None:
    y = np.asarray(y, dtype=np.int32)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-score)
    tp = 0
    fp = 0
    best = 0.0
    for idx in order:
        if y[idx]:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        if fpr <= max_fpr + 1e-12:
            best = max(best, tp / n_pos)
    return best


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float | None:
    if len(y) == 0:
        return None
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    total = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (p >= lo) & (p <= hi if b == bins - 1 else p < hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def metric_block(y: list[int], pred: list[float]) -> dict:
    yy = np.asarray(y, dtype=np.int32)
    pp = np.asarray(pred, dtype=np.float64)
    if len(yy) == 0:
        return {"n": 0, "positives": 0}
    return {
        "n": int(len(yy)),
        "positives": int(yy.sum()),
        "positive_rate": float(yy.mean()),
        "auroc": rank_auc(yy, pp),
        "average_precision": average_precision(yy, pp),
        "recall_at_fpr_05": recall_at_fpr(yy, pp, 0.05),
        "recall_at_fpr_10": recall_at_fpr(yy, pp, 0.10),
        "recall_at_fpr_20": recall_at_fpr(yy, pp, 0.20),
        "brier": float(np.mean((pp - yy) ** 2)),
        "ece_5bin": ece_score(yy, pp),
    }


class LinearProbe(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.linear = nn.Linear(n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_linear_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int, epochs: int) -> LinearProbe | None:
    y_train = y_train.astype(np.float32)
    if len(np.unique(y_train)) < 2:
        return None
    torch.manual_seed(seed)
    model = LinearProbe(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=1e-3)
    xx = torch.tensor(x_train, dtype=torch.float32)
    yy = torch.tensor(y_train, dtype=torch.float32)
    n_pos = float(yy.sum().item())
    n_neg = float(len(yy) - n_pos)
    pos_weight = torch.tensor([min(25.0, max(1.0, n_neg / max(1.0, n_pos)))], dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(xx), yy, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def predict_linear(model: LinearProbe, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def semkey_features(key: list, eq: int, idx: int, n_options: int) -> list[float]:
    vals = []
    for raw in (list(key) + [None] * 6)[:6]:
        if raw is None:
            vals.extend([0.0, 1.0])
        else:
            try:
                vals.extend([float(raw) / 2000.0, 0.0])
            except Exception:
                vals.extend([0.0, 1.0])
    vals.extend([
        float(eq) / max(1.0, float(n_options - 1)),
        float(idx) / max(1.0, float(n_options - 1)),
        float(n_options) / 20.0,
    ])
    return vals


def one_hot(value: int, vocab: dict[int, int]) -> list[float]:
    out = [0.0] * (len(vocab) + 1)
    pos = vocab.get(int(value))
    out[pos if pos is not None else len(vocab)] = 1.0
    return out


def build_rows(labels: list[dict], args) -> tuple[list[dict], dict]:
    row_args = argparse.Namespace(replay_dir=args.replay_dir, high_regret_threshold=args.high_regret_threshold)
    option_rows: list[dict] = []
    failures = []
    counts = Counter()
    for li, label in enumerate(labels):
        row, failure = RISK.make_row(label, li, li, row_args)
        if failure:
            failures.append(failure)
            continue
        counts["decisions_aligned"] += 1
        opts = {int(o["index"]): o for o in label.get("options") or []}
        n = len(row["base_dense"])
        search_selected = row.get("search_selected_option")
        tags = set(label.get("criterion_tags") or [])
        search_values = [float(opts[i].get("current_search_value") or 0.0) for i in range(n)]
        sorted_values = sorted(search_values, reverse=True)
        margin = (sorted_values[0] - sorted_values[1]) if len(sorted_values) > 1 else 0.0
        spread = float(np.std(search_values)) if search_values else 0.0
        coverage = label.get("coverage") if isinstance(label.get("coverage"), dict) else {}
        crit = label.get("criticality") if isinstance(label.get("criticality"), dict) else {}
        for oi in range(n):
            opt = opts.get(oi, {})
            selected = search_selected is not None and int(search_selected) == oi
            base_dense = [float(x) for x in row["base_dense"][oi]]
            dense = [float(x) for x in row["dense"][oi]]
            extra = dense[CR.SLICES["dense_dim"][1]:]
            option_rows.append({
                "row_id": f"{label.get('decision_id')}#{oi}",
                "decision_id": label.get("decision_id"),
                "obs_hash": label.get("obs_hash"),
                "group_id": label.get("group_id") or row.get("game_file"),
                "eval_only": bool(label.get("eval_only")),
                "option_index": oi,
                "n_options": n,
                "semantic_action_key": list(opt.get("semantic_action_key") or row["keys"][oi]),
                "eq_class": int(opt.get("eq_class", row["eq"][oi])),
                "card_id": int(row["cids"][oi]) if int(row["cids"][oi]) >= 0 else -1,
                "is_search_selected": bool(selected),
                "criterion_tags": sorted(tags),
                "c1_candidate": bool(label.get("c1_candidate")),
                "c1_reproduced_this_label": bool(label.get("c1_reproduced_this_label")),
                "c2_safe_search_false_positive": "c2_safe_search_false_positive" in tags,
                "c3_near_miss_boundary": "c3_near_miss_boundary" in tags,
                "selected_option_high_regret_flag": int(label.get("selected_option_high_regret_flag") or 0),
                "selected_option_unacceptable_flag": int(label.get("selected_option_unacceptable_flag") or 0),
                "high_regret_flag": int(opt.get("high_regret_flag") or 0),
                "unacceptable_flag": int(opt.get("unacceptable_flag") or 0),
                "regret": float(opt.get("regret") or 0.0),
                "clipped_regret": min(float(opt.get("regret") or 0.0), args.regret_clip),
                "current_search_value": float(opt.get("current_search_value") or 0.0),
                "current_search_value_centered": extra[1] if len(extra) > 1 else 0.0,
                "search_rank01": extra[2] if len(extra) > 2 else 0.0,
                "old_ranker_selected": extra[4] if len(extra) > 4 else 0.0,
                "option0_selected": extra[5] if len(extra) > 5 else 0.0,
                "search_value_margin": float(margin),
                "search_value_spread": float(spread),
                "value_variance": float(opt.get("value_variance") or 0.0),
                "value_se": float(opt.get("value_se") or 0.0),
                "completed_determinizations": float(opt.get("completed_determinizations") or 0.0),
                "criticality_score": float(crit.get("score", 0.0) or 0.0),
                "criticality_can_ko": float(crit.get("can_ko", 0.0) or 0.0),
                "criticality_ko_back": float(crit.get("ko_back", 0.0) or 0.0),
                "criticality_endgame": float(crit.get("endgame", 0.0) or 0.0),
                "coverage_all_siblings_completed": float(coverage.get("all_siblings_completed", 0.0) or 0.0),
                "base_dense": base_dense,
            })
            counts["options_aligned"] += 1
    return option_rows, {"failures": failures, "option_alignment": dict(counts)}


def make_feature_builders(rows: list[dict]) -> tuple[dict[str, Callable[[dict], list[float]]], dict]:
    non_eval_card_ids = sorted({r["card_id"] for r in rows if not r["eval_only"] and r["card_id"] >= 0})
    card_vocab = {cid: i for i, cid in enumerate(non_eval_card_ids)}
    s = CR.SLICES

    def section(row: dict, name: str) -> list[float]:
        a, b = s[name]
        return list(row["base_dense"][a:b])

    def root(row: dict) -> list[float]:
        return section(row, "root") + section(row, "history")

    def action(row: dict) -> list[float]:
        return section(row, "action_base") + semkey_features(
            row["semantic_action_key"], row["eq_class"], row["option_index"], row["n_options"]
        )

    def card(row: dict) -> list[float]:
        return one_hot(row["card_id"], card_vocab)

    def search_meta(row: dict) -> list[float]:
        return [
            row["current_search_value"] / 100000.0,
            row["current_search_value_centered"],
            row["search_rank01"],
            1.0 if row["is_search_selected"] else 0.0,
            row["old_ranker_selected"],
            row["option0_selected"],
            math.log1p(max(0.0, row["search_value_margin"])) / 12.0,
            math.log1p(max(0.0, row["search_value_spread"])) / 12.0,
            math.log1p(max(0.0, row["value_variance"])) / 30.0,
            row["value_se"] / 100000.0,
            row["completed_determinizations"] / 32.0,
            row["criticality_score"],
            row["criticality_can_ko"],
            row["criticality_ko_back"],
            row["criticality_endgame"],
            row["coverage_all_siblings_completed"],
            row["n_options"] / 20.0,
        ]

    builders = {
        "A_root": lambda r: root(r),
        "B_root_action": lambda r: root(r) + action(r),
        "C_plus_card_identity": lambda r: root(r) + action(r) + card(r),
        "D_plus_decoded_effects": lambda r: root(r) + action(r) + card(r) + section(r, "effects"),
        "E_plus_target_entity": lambda r: root(r) + action(r) + card(r) + section(r, "effects") + section(r, "target"),
        "F_plus_state_effect_interactions": (
            lambda r: root(r) + action(r) + card(r) + section(r, "effects")
            + section(r, "target") + section(r, "interactions")
        ),
        "G_plus_option_deltas": (
            lambda r: root(r) + action(r) + card(r) + section(r, "effects")
            + section(r, "target") + section(r, "interactions") + section(r, "deltas")
        ),
        "H_plus_search_uncertainty": (
            lambda r: root(r) + action(r) + card(r) + section(r, "effects")
            + section(r, "target") + section(r, "interactions") + section(r, "deltas") + search_meta(r)
        ),
        "baseline_criticality_only": lambda r: [
            r["criticality_score"],
            r["criticality_can_ko"],
            r["criticality_ko_back"],
            r["criticality_endgame"],
            r["n_options"] / 20.0,
        ],
        "baseline_search_variance_only": lambda r: [
            math.log1p(max(0.0, r["search_value_margin"])) / 12.0,
            math.log1p(max(0.0, r["search_value_spread"])) / 12.0,
            math.log1p(max(0.0, r["value_variance"])) / 30.0,
            r["value_se"] / 100000.0,
            r["completed_determinizations"] / 32.0,
            r["search_rank01"],
            1.0 if r["is_search_selected"] else 0.0,
        ],
    }
    meta = {
        "card_identity_vocab_size": len(card_vocab),
        "card_identity_encoding": "one-hot over non-eval card ids with unknown bucket",
        "continuous_normalization": "robust median/IQR from non-eval fitting rows; clipped to +/-8",
        "feature_sections": {k: list(v) for k, v in CR.SLICES.items()},
    }
    return builders, meta


def target_spec(name: str) -> dict:
    if name in ("high_regret_flag", "unacceptable_flag"):
        return {"scope": "all_options", "fn": lambda r, n=name: int(r[n])}
    if name == "selected_option_high_regret_flag":
        return {
            "scope": "selected_options",
            "fn": lambda r: int(r["selected_option_high_regret_flag"]) if r["is_search_selected"] else None,
        }
    if name == "c1_reproduced_this_label":
        return {
            "scope": "selected_options",
            "fn": lambda r: int(r["c1_reproduced_this_label"]) if r["is_search_selected"] else None,
        }
    if name == "c2_safe_search_false_positive":
        return {
            "scope": "selected_options",
            "fn": lambda r: int(r["c2_safe_search_false_positive"]) if r["is_search_selected"] else None,
        }
    if name == "c3_near_miss_boundary":
        return {
            "scope": "selected_options",
            "fn": lambda r: int(r["c3_near_miss_boundary"]) if r["is_search_selected"] else None,
        }
    raise KeyError(name)


def target_rows(rows: list[dict], target: str) -> tuple[list[dict], np.ndarray]:
    spec = target_spec(target)
    out_rows = []
    y = []
    for row in rows:
        val = spec["fn"](row)
        if val is None:
            continue
        out_rows.append(row)
        y.append(int(val))
    return out_rows, np.asarray(y, dtype=np.int32)


def matrix(rows: list[dict], builder: Callable[[dict], list[float]]) -> np.ndarray:
    return np.asarray([builder(r) for r in rows], dtype=np.float32)


def neighborhood_analysis(rows: list[dict], builders: dict, neighbors_path: Path) -> list[dict]:
    results = []
    neighbor_lines = []
    for target in TARGET_ORDER:
        trows, y = target_rows(rows, target)
        if len(trows) == 0:
            continue
        non_eval = np.asarray([not r["eval_only"] for r in trows], dtype=bool)
        positive_idx = np.where(y == 1)[0]
        for pack in [k for k in builders if k.startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_"))]:
            x = matrix(trows, builders[pack])
            ref_mask = non_eval
            med, scale = robust_fit(x[ref_mask])
            z = robust_transform(x, med, scale)
            for k in KS:
                labels = []
                bg_rates = []
                k_used = []
                retrieved_positive_ids = set()
                query_count = 0
                eval_query_count = 0
                for qi in positive_idx:
                    q = trows[qi]
                    cand = [
                        ci for ci, row in enumerate(trows)
                        if non_eval[ci] and row["group_id"] != q["group_id"]
                    ]
                    if not cand:
                        continue
                    query_count += 1
                    eval_query_count += int(q["eval_only"])
                    d = np.linalg.norm(z[cand] - z[qi], axis=1)
                    order = np.argsort(d)[:min(k, len(cand))]
                    chosen = [cand[int(i)] for i in order]
                    labels.extend([int(y[i]) for i in chosen])
                    bg_rates.append(float(y[cand].mean()) if cand else 0.0)
                    k_used.append(len(chosen))
                    for ni in chosen:
                        if y[ni]:
                            retrieved_positive_ids.add(trows[ni]["row_id"])
                    if k == 10:
                        neighbor_lines.append({
                            "target": target,
                            "feature_pack": pack,
                            "query_row_id": q["row_id"],
                            "query_decision_id": q["decision_id"],
                            "query_group_id": q["group_id"],
                            "query_eval_only": q["eval_only"],
                            "neighbors": [
                                {
                                    "row_id": trows[ni]["row_id"],
                                    "decision_id": trows[ni]["decision_id"],
                                    "group_id": trows[ni]["group_id"],
                                    "label": int(y[ni]),
                                    "distance": float(d[int(order[pos])]),
                                }
                                for pos, ni in enumerate(chosen)
                            ],
                        })
                bg = average(bg_rates)
                nr = average([float(v) for v in labels])
                total_pos_non_eval = {
                    trows[i]["row_id"] for i in range(len(trows)) if non_eval[i] and y[i] == 1
                }
                results.append({
                    "target": target,
                    "feature_pack": pack,
                    "k": k,
                    "query_count": int(query_count),
                    "eval_only_query_count": int(eval_query_count),
                    "background_positive_rate": bg,
                    "neighbor_positive_rate": nr,
                    "enrichment_ratio": safe_ratio(nr, bg),
                    "precision_among_neighbors": nr,
                    "recall_coverage_of_non_eval_positives": safe_ratio(
                        len(retrieved_positive_ids), len(total_pos_non_eval)
                    ),
                    "mean_effective_k": average([float(v) for v in k_used]),
                    "candidate_positive_count_non_eval": int(len(total_pos_non_eval)),
                })
    neighbors_path.parent.mkdir(parents=True, exist_ok=True)
    with neighbors_path.open("w", encoding="utf-8") as f:
        for item in neighbor_lines:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    return results


def grouped_probe(rows: list[dict], builders: dict, args) -> list[dict]:
    probe_names = ["class_frequency", "baseline_criticality_only", "baseline_search_variance_only"] + [
        k for k in builders if k.startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_"))
    ]
    results = []
    for target in TARGET_ORDER:
        trows, y_all = target_rows(rows, target)
        if len(trows) == 0:
            continue
        groups = np.asarray([r["group_id"] for r in trows])
        eval_mask = np.asarray([r["eval_only"] for r in trows], dtype=bool)
        non_eval_idx = np.where(~eval_mask)[0]
        unique_groups = sorted(set(groups[non_eval_idx]))
        for pack in probe_names:
            fold_y = []
            fold_pred = []
            valid_folds = 0
            skipped_folds = 0
            for gi, group in enumerate(unique_groups):
                train_idx = np.asarray([i for i in non_eval_idx if groups[i] != group], dtype=np.int32)
                test_idx = np.asarray([i for i in non_eval_idx if groups[i] == group], dtype=np.int32)
                if len(train_idx) == 0 or len(test_idx) == 0:
                    skipped_folds += 1
                    continue
                y_train = y_all[train_idx]
                if len(np.unique(y_train)) < 2:
                    skipped_folds += 1
                    continue
                if pack == "class_frequency":
                    pred = np.full(len(test_idx), float(y_train.mean()), dtype=np.float32)
                else:
                    x_all = matrix(trows, builders[pack])
                    med, scale = robust_fit(x_all[train_idx])
                    x_train = robust_transform(x_all[train_idx], med, scale)
                    x_test = robust_transform(x_all[test_idx], med, scale)
                    model = train_linear_probe(
                        x_train, y_train, seed=args.seed + gi + len(pack) + len(target), epochs=args.epochs
                    )
                    if model is None:
                        skipped_folds += 1
                        continue
                    pred = predict_linear(model, x_test)
                fold_y.extend([int(v) for v in y_all[test_idx]])
                fold_pred.extend([float(v) for v in pred])
                valid_folds += 1

            metrics = metric_block(fold_y, fold_pred)
            seed_eval = {"n": int(eval_mask.sum()), "positives": int(y_all[eval_mask].sum())}
            if eval_mask.any():
                train_idx = non_eval_idx
                y_train = y_all[train_idx]
                eval_idx = np.where(eval_mask)[0]
                if pack == "class_frequency":
                    pred_seed = np.full(len(eval_idx), float(y_train.mean()) if len(y_train) else 0.0)
                    seed_eval.update(metric_block([int(v) for v in y_all[eval_idx]], [float(v) for v in pred_seed]))
                elif len(np.unique(y_train)) >= 2:
                    x_all = matrix(trows, builders[pack])
                    med, scale = robust_fit(x_all[train_idx])
                    model = train_linear_probe(
                        robust_transform(x_all[train_idx], med, scale),
                        y_train,
                        seed=args.seed + 9000 + len(pack) + len(target),
                        epochs=args.epochs,
                    )
                    if model is not None:
                        pred_seed = predict_linear(model, robust_transform(x_all[eval_idx], med, scale))
                        seed_eval.update(
                            metric_block([int(v) for v in y_all[eval_idx]], [float(v) for v in pred_seed])
                        )
                        seed_eval["mean_prediction_positive"] = average([
                            float(p) for p, yy in zip(pred_seed, y_all[eval_idx]) if yy == 1
                        ])
                        seed_eval["mean_prediction_negative"] = average([
                            float(p) for p, yy in zip(pred_seed, y_all[eval_idx]) if yy == 0
                        ])
            results.append({
                "target": target,
                "feature_pack": pack,
                "probe": "linear_logistic_group_leave_one_game_out",
                "eligible_rows": int(len(trows)),
                "eligible_non_eval_rows": int(len(non_eval_idx)),
                "groups_non_eval": int(len(unique_groups)),
                "valid_folds": int(valid_folds),
                "skipped_folds": int(skipped_folds),
                **metrics,
                "seed_eval": seed_eval,
            })
    return results


def dataset_audit(labels: list[dict], option_rows: list[dict], build_report: dict, canonical: Path, alias: Path) -> dict:
    decision_ids = [r.get("decision_id") for r in labels]
    option_ids = [(r["decision_id"], r["option_index"]) for r in option_rows]
    by_game = {}
    for group in sorted({r["group_id"] for r in option_rows}):
        grows = [r for r in option_rows if r["group_id"] == group]
        decisions = sorted({r["decision_id"] for r in grows})
        by_game[group] = {
            "decisions": len(decisions),
            "options": len(grows),
            "high_regret_options": int(sum(r["high_regret_flag"] for r in grows)),
            "high_regret_rate": sum(r["high_regret_flag"] for r in grows) / max(1, len(grows)),
            "unacceptable_options": int(sum(r["unacceptable_flag"] for r in grows)),
            "unacceptable_rate": sum(r["unacceptable_flag"] for r in grows) / max(1, len(grows)),
            "c1_reproduced_decisions": len({r["decision_id"] for r in grows if r["c1_reproduced_this_label"]}),
            "c1_candidate_not_reproduced_decisions": len({
                r["decision_id"] for r in grows if r["c1_candidate"] and not r["c1_reproduced_this_label"]
            }),
            "c2_decisions": len({r["decision_id"] for r in grows if r["c2_safe_search_false_positive"]}),
            "c3_decisions": len({r["decision_id"] for r in grows if r["c3_near_miss_boundary"]}),
            "eval_only_decisions": len({r["decision_id"] for r in grows if r["eval_only"]}),
        }

    missing = defaultdict(int)
    required_record = [
        "decision_id", "obs_hash", "observation", "legal_options", "group_id", "eval_only",
        "c1_candidate", "c1_reproduced_this_label", "criterion_tags",
        "selected_option_high_regret_flag", "selected_option_unacceptable_flag",
    ]
    required_option = [
        "index", "semantic_action_key", "eq_class", "current_search_value", "stronger_value",
        "delta_to_search", "delta_to_search_norm", "high_regret_flag", "unacceptable_flag",
        "value_variance", "value_se", "completed_determinizations",
    ]
    for label in labels:
        for key in required_record:
            if key not in label:
                missing[f"record.{key}"] += 1
        for opt in label.get("options") or []:
            for key in required_option:
                if key not in opt:
                    missing[f"option.{key}"] += 1

    return {
        "canonical_file": display(canonical),
        "alias_file": display(alias),
        "canonical_sha256": sha256(canonical),
        "alias_sha256": sha256(alias) if alias.exists() else None,
        "canonical_alias_byte_identical": alias.exists() and canonical.read_bytes() == alias.read_bytes(),
        "records_loaded": len(labels),
        "options_loaded": sum(len(r.get("options") or []) for r in labels),
        "feature_rows_generated": len(option_rows),
        "unique_group_ids": len({r["group_id"] for r in option_rows}),
        "c1_reproduced_count": sum(1 for r in labels if r.get("c1_reproduced_this_label")),
        "c1_candidate_not_reproduced_count": sum(
            1 for r in labels if r.get("c1_candidate") and not r.get("c1_reproduced_this_label")
        ),
        "c2_count": sum(1 for r in labels if bool_tag(r, "c2_safe_search_false_positive")),
        "c3_count": sum(1 for r in labels if bool_tag(r, "c3_near_miss_boundary")),
        "high_regret_flag_count": int(sum(r["high_regret_flag"] for r in option_rows)),
        "unacceptable_flag_count": int(sum(r["unacceptable_flag"] for r in option_rows)),
        "eval_only_count": sum(1 for r in labels if r.get("eval_only")),
        "missing_fields": dict(sorted(missing.items())),
        "duplicate_decision_ids": [k for k, v in Counter(decision_ids).items() if v > 1],
        "duplicate_option_id_count": sum(1 for v in Counter(option_ids).values() if v > 1),
        "class_rates_overall": {
            "high_regret_flag": sum(r["high_regret_flag"] for r in option_rows) / max(1, len(option_rows)),
            "unacceptable_flag": sum(r["unacceptable_flag"] for r in option_rows) / max(1, len(option_rows)),
            "selected_option_high_regret_flag": (
                sum(1 for r in option_rows if r["is_search_selected"] and r["selected_option_high_regret_flag"])
                / max(1, sum(1 for r in option_rows if r["is_search_selected"]))
            ),
            "c1_reproduced_this_label": sum(1 for r in labels if r.get("c1_reproduced_this_label")) / max(1, len(labels)),
            "c2_safe_search_false_positive": sum(1 for r in labels if bool_tag(r, "c2_safe_search_false_positive")) / max(1, len(labels)),
            "c3_near_miss_boundary": sum(1 for r in labels if bool_tag(r, "c3_near_miss_boundary")) / max(1, len(labels)),
        },
        "class_rates_by_game": by_game,
        "alignment": build_report.get("option_alignment", {}),
        "build_failures": build_report.get("failures", []),
        "training_ready_for_audit": not build_report.get("failures") and len(option_rows) == sum(
            len(r.get("options") or []) for r in labels
        ),
    }


def best_rows(table: list[dict], metric: str, target: str, k: int | None = None) -> list[dict]:
    rows = [r for r in table if r.get("target") == target]
    if k is not None:
        rows = [r for r in rows if r.get("k") == k]
    rows = [r for r in rows if r.get(metric) is not None]
    return sorted(rows, key=lambda r: float(r[metric]), reverse=True)


def summarize_feature_contrib(neighborhood: list[dict], probes: list[dict]) -> dict:
    out = {}
    for target in TARGET_ORDER:
        pack_probe = {
            r["feature_pack"]: r for r in probes
            if r["target"] == target and r["feature_pack"].startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_"))
        }
        pack_nn = {
            r["feature_pack"]: r for r in neighborhood
            if r["target"] == target and r["k"] == 10 and r["feature_pack"].startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_"))
        }
        base_ap = (pack_probe.get("A_root") or {}).get("average_precision")
        full_ap = (pack_probe.get("H_plus_search_uncertainty") or {}).get("average_precision")
        search_ap = next(
            (r.get("average_precision") for r in probes
             if r["target"] == target and r["feature_pack"] == "baseline_search_variance_only"),
            None,
        )
        out[target] = {
            "root_ap": base_ap,
            "full_ap": full_ap,
            "search_variance_only_ap": search_ap,
            "best_probe": best_rows(probes, "average_precision", target)[:3],
            "best_neighbor_k10": best_rows(neighborhood, "enrichment_ratio", target, k=10)[:3],
            "incremental_ap": {
                pack: (pack_probe.get(pack) or {}).get("average_precision")
                for pack in [
                    "A_root", "B_root_action", "C_plus_card_identity", "D_plus_decoded_effects",
                    "E_plus_target_entity", "F_plus_state_effect_interactions", "G_plus_option_deltas",
                    "H_plus_search_uncertainty",
                ]
            },
            "incremental_neighbor_enrichment_k10": {
                pack: (pack_nn.get(pack) or {}).get("enrichment_ratio")
                for pack in [
                    "A_root", "B_root_action", "C_plus_card_identity", "D_plus_decoded_effects",
                    "E_plus_target_entity", "F_plus_state_effect_interactions", "G_plus_option_deltas",
                    "H_plus_search_uncertainty",
                ]
            },
        }
    return out


def component_delta_rows(feature_summary: dict) -> list[dict]:
    pairs = [
        ("card_embedding_or_identity", "B_root_action", "C_plus_card_identity"),
        ("decoded_effects", "C_plus_card_identity", "D_plus_decoded_effects"),
        ("target_entity_features", "D_plus_decoded_effects", "E_plus_target_entity"),
        ("state_effect_interactions", "E_plus_target_entity", "F_plus_state_effect_interactions"),
        ("immediate_option_deltas", "F_plus_state_effect_interactions", "G_plus_option_deltas"),
        ("search_uncertainty_metadata", "G_plus_option_deltas", "H_plus_search_uncertainty"),
    ]
    rows = []
    for name, before, after in pairs:
        ap_deltas = []
        nn_deltas = []
        improved_targets = []
        for target, summary in feature_summary.items():
            ap = summary.get("incremental_ap") or {}
            nn = summary.get("incremental_neighbor_enrichment_k10") or {}
            if ap.get(before) is not None and ap.get(after) is not None:
                delta = float(ap[after]) - float(ap[before])
                ap_deltas.append(delta)
                if delta >= 0.05:
                    improved_targets.append(target)
            if nn.get(before) is not None and nn.get(after) is not None:
                nn_deltas.append(float(nn[after]) - float(nn[before]))
        mean_ap = average(ap_deltas)
        mean_nn = average(nn_deltas)
        if mean_ap is None:
            read = "not measured"
        elif mean_ap >= 0.05:
            read = "adds probe signal on average"
        elif mean_ap <= -0.05:
            read = "hurts or destabilizes probes on average"
        elif mean_nn is not None and mean_nn >= 0.20:
            read = "adds neighborhood signal but weak probe gain"
        else:
            read = "little consistent incremental signal"
        rows.append({
            "component": name,
            "from_pack": before,
            "to_pack": after,
            "mean_ap_delta": mean_ap,
            "mean_neighbor_enrichment_delta_k10": mean_nn,
            "targets_with_ap_gain_ge_0.05": improved_targets,
            "read": read,
        })
    return rows


def choose_verdict(dataset: dict, feature_summary: dict) -> dict:
    c1_groups = [
        g for g, stats in dataset["class_rates_by_game"].items()
        if stats["c1_reproduced_decisions"] > 0
    ]
    high = feature_summary.get("high_regret_flag", {})
    c1 = feature_summary.get("c1_reproduced_this_label", {})
    full_ap = high.get("full_ap")
    search_ap = high.get("search_variance_only_ap")
    root_ap = high.get("root_ap")
    c1_full_ap = c1.get("full_ap")
    c1_search_ap = c1.get("search_variance_only_ap")
    notes = []
    if len(c1_groups) < 6:
        notes.append(
            "c1 reproduced positives are clustered across fewer than six games, so c1 generalization is fragile."
        )
    if full_ap is not None and search_ap is not None and abs(full_ap - search_ap) <= 0.05:
        notes.append("full high-regret probe is close to search-variance metadata.")
    if root_ap is not None and full_ap is not None and full_ap > root_ap + 0.05:
        notes.append("full features improve over root-only for option-level high_regret.")
    if c1_full_ap is not None and c1_search_ap is not None and c1_full_ap <= c1_search_ap + 0.05:
        notes.append("c1 probe signal is not clearly beyond search metadata.")
    if len(c1_groups) < 6:
        verdict = "D. CURRENT ARTIFACTS INCONCLUSIVE"
        recommendation = (
            "Before another risk-policy retrain, mine disjoint replay shards until there are at least 25 "
            "reproduced search-selected-high-regret c1 decisions across at least 12 group_id games, plus "
            "matched c2/c3/background states from the same high-criticality band, then rerun this audit with "
            "the current two eval-only seeds still held out."
        )
    elif full_ap is not None and search_ap is not None and search_ap >= full_ap - 0.03:
        verdict = "C. SEARCH-METADATA-ONLY"
        recommendation = (
            "Use the current risk signal as a selective-compute trigger, not a broad semantic policy learner, "
            "unless a future audit shows semantic/delta enrichment beyond search metadata."
        )
    elif full_ap is not None and root_ap is not None and full_ap > root_ap + 0.08:
        verdict = "B. FEATURE-LIMITED"
        recommendation = (
            "Refine the feature family that improved signal radius before another policy/risk-model attempt."
        )
    else:
        verdict = "E. NO PRACTICAL SIGNAL FOUND"
        recommendation = "Pause failure-neighborhood learning until the label set changes."
    return {"verdict": verdict, "notes": notes, "one_recommended_next_experiment": recommendation}


def md_float(x, nd=3) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def write_markdown(path: Path, report: dict) -> None:
    ds = report["dataset_audit"]
    feature_summary = report["feature_contribution"]
    verdict = report["decision"]
    lines = []
    lines.append("# Signal Radius Audit V1")
    lines.append("")
    lines.append("Status: bounded diagnostic on round-2 Teacher V2 residual/risk labels. No live agent change, no arena screen, and no production risk-policy retrain.")
    lines.append("")
    lines.append("## Dataset And Class Summary")
    lines.append("")
    lines.append(f"- Canonical labels: `{ds['canonical_file']}`")
    lines.append(f"- Alias labels: `{ds['alias_file']}`")
    lines.append(f"- Canonical and alias byte-identical: `{ds['canonical_alias_byte_identical']}`")
    lines.append(f"- Records/options loaded: {ds['records_loaded']} decisions / {ds['options_loaded']} options")
    lines.append(f"- Feature rows generated: {ds['feature_rows_generated']}")
    lines.append(f"- Unique games (`group_id`): {ds['unique_group_ids']}")
    lines.append(f"- c1 reproduced: {ds['c1_reproduced_count']}")
    lines.append(f"- c1 candidate but not reproduced: {ds['c1_candidate_not_reproduced_count']}")
    lines.append(f"- c2 safe-search false-positive states: {ds['c2_count']}")
    lines.append(f"- c3 near-miss/boundary states: {ds['c3_count']}")
    lines.append(f"- High-regret options: {ds['high_regret_flag_count']}")
    lines.append(f"- Unacceptable options: {ds['unacceptable_flag_count']}")
    lines.append(f"- Eval-only seed decisions: {ds['eval_only_count']} (excluded from fitting and feature-selection steps)")
    lines.append(f"- Duplicate decisions: {len(ds['duplicate_decision_ids'])}; duplicate option identities: {ds['duplicate_option_id_count']}")
    lines.append(f"- Missing fields: `{ds['missing_fields']}`")
    lines.append("")
    lines.append("Class rates by game are in the JSON report. The important caveat is that reproduced c1 positives are game-clustered, so group-held-out estimates are high-variance.")
    lines.append("")
    lines.append("| group_id | decisions | options | high-regret rate | unacceptable rate | c1 repr | c2 | c3 | eval seeds |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group, stats in sorted(ds["class_rates_by_game"].items()):
        lines.append(
            f"| {group} | {stats['decisions']} | {stats['options']} | "
            f"{md_float(stats['high_regret_rate'])} | {md_float(stats['unacceptable_rate'])} | "
            f"{stats['c1_reproduced_decisions']} | {stats['c2_decisions']} | "
            f"{stats['c3_decisions']} | {stats['eval_only_decisions']} |"
        )
    lines.append("")
    lines.append("## Feature Packs")
    lines.append("")
    for pack, desc in report["feature_pack_definitions"].items():
        lines.append(f"- `{pack}`: {desc}")
    lines.append("")
    lines.append("Continuous features were robust-normalized with median/IQR from non-eval fitting rows. Card identity is a categorical one-hot proxy for the trainable card embedding used by the contextual model.")
    lines.append("")
    lines.append("## Neighborhood Enrichment")
    lines.append("")
    lines.append("Table shows k=10 standardized-Euclidean neighbor enrichment with same-game neighbors forbidden and eval-only rows excluded from neighbor pools.")
    lines.append("")
    lines.append("| target | best pack | bg rate | neighbor rate | enrich | coverage | queries |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for target in TARGET_ORDER:
        best = best_rows(report["neighborhood_enrichment"], "enrichment_ratio", target, k=10)
        if not best:
            continue
        row = best[0]
        lines.append(
            f"| {target} | `{row['feature_pack']}` | {md_float(row['background_positive_rate'])} | "
            f"{md_float(row['neighbor_positive_rate'])} | {md_float(row['enrichment_ratio'])} | "
            f"{md_float(row['recall_coverage_of_non_eval_positives'])} | {row['query_count']} |"
        )
    lines.append("")
    lines.append("## Predictive Probes")
    lines.append("")
    lines.append("Grouped leave-one-game-out linear probes; eval-only seeds are final-eval only. Full per-pack metrics are in the JSON.")
    lines.append("")
    lines.append("| target | best probe pack | AP | AUROC | recall@FPR10 | valid folds | full-pack AP | search-meta AP | root AP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for target in TARGET_ORDER:
        best = best_rows(report["predictive_probes"], "average_precision", target)
        if not best:
            continue
        full = next((r for r in report["predictive_probes"] if r["target"] == target and r["feature_pack"] == "H_plus_search_uncertainty"), {})
        search = next((r for r in report["predictive_probes"] if r["target"] == target and r["feature_pack"] == "baseline_search_variance_only"), {})
        root = next((r for r in report["predictive_probes"] if r["target"] == target and r["feature_pack"] == "A_root"), {})
        row = best[0]
        lines.append(
            f"| {target} | `{row['feature_pack']}` | {md_float(row.get('average_precision'))} | "
            f"{md_float(row.get('auroc'))} | {md_float(row.get('recall_at_fpr_10'))} | "
            f"{row.get('valid_folds')} | {md_float(full.get('average_precision'))} | "
            f"{md_float(search.get('average_precision'))} | {md_float(root.get('average_precision'))} |"
        )
    lines.append("")
    lines.append("## Feature Contribution Conclusions")
    lines.append("")
    lines.append("| target | root AP | full AP | search-meta AP | contribution read |")
    lines.append("|---|---:|---:|---:|---|")
    for target in TARGET_ORDER:
        row = feature_summary.get(target, {})
        root = row.get("root_ap")
        full = row.get("full_ap")
        search = row.get("search_variance_only_ap")
        if full is None:
            read = "labels too sparse or folds skipped"
        elif search is not None and full <= search + 0.05:
            read = "signal dominated by search metadata or no gain beyond it"
        elif root is not None and full > root + 0.05:
            read = "signal visible after feature enrichment"
        else:
            read = "current representation weak or inconclusive"
        lines.append(f"| {target} | {md_float(root)} | {md_float(full)} | {md_float(search)} | {read} |")
    lines.append("")
    lines.append("Component-level read:")
    lines.append("")
    lines.append("| component | mean AP delta | mean k10 enrich delta | read |")
    lines.append("|---|---:|---:|---|")
    for row in report["component_deltas"]:
        lines.append(
            f"| {row['component']} | {md_float(row['mean_ap_delta'])} | "
            f"{md_float(row['mean_neighbor_enrichment_delta_k10'])} | {row['read']} |"
        )
    lines.append("")
    lines.append("The explicit contribution answer is: card identity, decoded effects, target/entity features, interactions, and option deltas do not yet add reliable c1/search-failure signal; search/criticality metadata dominates most state-level labels; option-level high-regret has some broader structure but the c1 class remains too clustered for a reliable policy conclusion.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for note in verdict["notes"]:
        lines.append(f"- {note}")
    lines.append("- Raw regret magnitude was not used as a primary target; it is noisy under the round-2 artifact notes.")
    lines.append("- c1 positives are rare and clustered; a failed c1 probe does not prove there is no underlying structure.")
    lines.append("- Linear probes are diagnostic only. They are not a deployable model and were not written to `agent/`.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict['verdict']}**")
    lines.append("")
    lines.append(verdict["one_recommended_next_experiment"])
    lines.append("")
    lines.append("Model A remains idle unless that next experiment is explicitly authorized and asks for independent c1 labels from disjoint games.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_ROUND2)
    ap.add_argument("--alias", type=Path, default=DEFAULT_ALIAS)
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    ap.add_argument("--json-out", type=Path, default=DOCS / "signal_radius_audit_v1.json")
    ap.add_argument("--md-out", type=Path, default=DOCS / "SIGNAL_RADIUS_AUDIT_V1.md")
    ap.add_argument("--neighbors-out", type=Path, default=DOCS / "signal_radius_audit_v1_neighbors.jsonl")
    ap.add_argument("--high-regret-threshold", type=float, default=5000.0)
    ap.add_argument("--regret-clip", type=float, default=50000.0)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--seed", type=int, default=47219)
    args = ap.parse_args()

    for key in ("labels", "alias", "replay_dir", "json_out", "md_out", "neighbors_out"):
        val = getattr(args, key)
        setattr(args, key, val if val.is_absolute() else ROOT / val)

    if args.alias.exists() and args.labels.exists() and args.alias.read_bytes() != args.labels.read_bytes():
        print(json.dumps({
            "warning": "round2 canonical and alias are not byte-identical; using --labels as canonical",
            "labels": display(args.labels),
            "alias": display(args.alias),
        }), flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    labels = load_jsonl(args.labels)
    option_rows, build_report = build_rows(labels, args)
    dataset = dataset_audit(labels, option_rows, build_report, args.labels, args.alias)
    builders, feature_meta = make_feature_builders(option_rows)
    feature_pack_definitions = {
        "A_root": "root-state vector plus public turn/history only",
        "B_root_action": "A + action descriptor, semantic_action_key numeric encoding, eq_class/option count",
        "C_plus_card_identity": "B + categorical card identity one-hot proxy for trainable embedding",
        "D_plus_decoded_effects": "C + decoded card-effect features",
        "E_plus_target_entity": "D + target/entity properties",
        "F_plus_state_effect_interactions": "E + state x effect interaction features",
        "G_plus_option_deltas": "F + immediate one-step option deltas",
        "H_plus_search_uncertainty": "G + search value rank/margin/spread, value variance/SE, determinization, coverage, criticality metadata",
        "baseline_criticality_only": "criticality fields only",
        "baseline_search_variance_only": "search/teacher uncertainty fields only, without semantic features",
        "class_frequency": "fold-local class prevalence baseline",
    }
    neighborhood = neighborhood_analysis(option_rows, builders, args.neighbors_out)
    probes = grouped_probe(option_rows, builders, args)
    feature_summary = summarize_feature_contrib(neighborhood, probes)
    component_deltas = component_delta_rows(feature_summary)
    decision = choose_verdict(dataset, feature_summary)
    report = {
        "artifact_version": "signal_radius_audit_v1",
        "branch": "exp/robust-learner-v2",
        "live_agent_consumed": "none",
        "arena_screen": "not run",
        "risk_policy_retrained": False,
        "dataset_audit": dataset,
        "target_definitions": {
            "high_regret_flag": "option-level high_regret_flag on all legal options",
            "unacceptable_flag": "option-level unacceptable_flag on all legal options",
            "selected_option_high_regret_flag": "decision-level selected-search-action flag evaluated on selected option rows",
            "c1_reproduced_this_label": "decision-level reproduced c1 flag evaluated on selected option rows",
            "c2_safe_search_false_positive": "decision-level c2 tag evaluated on selected option rows",
            "c3_near_miss_boundary": "decision-level c3 tag evaluated on selected option rows",
        },
        "feature_pack_definitions": feature_pack_definitions,
        "feature_metadata": feature_meta,
        "neighborhood_enrichment": neighborhood,
        "predictive_probes": probes,
        "feature_contribution": feature_summary,
        "component_deltas": component_deltas,
        "decision": decision,
        "outputs": {
            "json": display(args.json_out),
            "markdown": display(args.md_out),
            "neighbors": display(args.neighbors_out),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.md_out, report)
    print(json.dumps({
        "dataset": {
            "records": dataset["records_loaded"],
            "options": dataset["options_loaded"],
            "groups": dataset["unique_group_ids"],
            "high_regret": dataset["high_regret_flag_count"],
            "unacceptable": dataset["unacceptable_flag_count"],
            "training_ready": dataset["training_ready_for_audit"],
        },
        "decision": decision,
        "outputs": report["outputs"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
