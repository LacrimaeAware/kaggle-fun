"""Train/evaluate a risk-only contextual model from Teacher V2 residual/risk labels.

The model predicts whether each legal sibling action is dangerous
(`high_regret_flag` or `unacceptable_flag`). It does not learn broad residual
corrections and does not replace search ranking.

Offline integration rule:
    let agent_search choose normally;
    if the chosen raw option is predicted high-risk, choose the first lower-risk
    option in current search-value order; otherwise keep the search choice.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import build_teacher_v2_contextual_rows as BLD  # noqa: E402
import contextual_ranker as CR  # noqa: E402
import train_contextual_action_ranker as TCR  # noqa: E402
import train_residual_risk_contextual as BRISK  # noqa: E402

DEFAULT_LABELS = ROOT / "data" / "manifests" / "teacher_v2_residual_risk_labels.jsonl"
DEFAULT_PREVIOUS_FULL = ROOT / "agent" / "contextual_ranker_teacher_v2.json"
DEFAULT_B_BOOTSTRAP = ROOT / "agent" / "contextual_residual_risk_v1.json"
EXTRA_FEATURE_NAMES = [
    "current_search_value",
    "current_search_centered",
    "search_rank01",
    "search_selected",
    "old_ranker_selected",
    "option0_selected",
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def average(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


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


def partition_for(label: dict, highcrit_index: int) -> str:
    if label.get("state_tag") == "B_failure_state":
        return "test"
    return "test" if highcrit_index % 5 == 0 else "train"


def option_lookup(label: dict) -> dict[int, dict]:
    return {int(o["index"]): o for o in (label.get("options") or [])}


def add_search_features(base_dense: list[list[float]], options: dict[int, dict],
                        search_selected: int | None, old_eq: int | None,
                        option0: int, feat_eq: list[int]) -> list[list[float]]:
    values = {idx: float(opt.get("current_search_value") or 0.0) for idx, opt in options.items()}
    mean_v = average(list(values.values())) if values else 0.0
    ranked = sorted(values, key=lambda i: values[i], reverse=True)
    rank = {idx: r for r, idx in enumerate(ranked)}
    denom = max(1, len(ranked) - 1)
    out = []
    for idx, row in enumerate(base_dense):
        eq = int(feat_eq[idx])
        value = values.get(idx, 0.0)
        extra = [
            value / 1000.0,
            (value - mean_v) / 1000.0,
            float(rank.get(idx, len(ranked))) / denom,
            1.0 if search_selected is not None and idx == search_selected else 0.0,
            1.0 if old_eq is not None and eq == old_eq else 0.0,
            1.0 if idx == option0 else 0.0,
        ]
        out.append([float(x) for x in row] + extra)
    return out


def raw_option_from_eq(row: dict, eq: int | None) -> int | None:
    if eq is None:
        return None
    members = [i for i, e in enumerate(row["eq"]) if int(e) == int(eq)]
    if not members:
        return None
    search_values = row.get("current_search_value") or []
    return max(members, key=lambda i: (float(search_values[i]), -i))


def make_row(label: dict, label_index: int, highcrit_index: int, args) -> tuple[dict | None, dict | None]:
    reconstructed, attempts = BLD.find_reconstructed(label, args.replay_dir)
    if not reconstructed:
        return None, {"decision_id": label.get("decision_id"), "failure": "reconstruction_failed", "attempts": attempts}
    feat = reconstructed["feat"]
    if len(feat["dense"]) != len(label.get("options") or []):
        return None, {
            "decision_id": label.get("decision_id"),
            "failure": "option_count_mismatch",
            "feature_options": len(feat["dense"]),
            "label_options": len(label.get("options") or []),
        }
    opts = option_lookup(label)
    mismatches = []
    for idx, key in enumerate(feat["keys"]):
        opt = opts.get(idx)
        if opt is None or list(opt.get("semantic_action_key") or []) != list(key):
            mismatches.append({"index": idx, "feature_key": list(key), "label_key": opt.get("semantic_action_key") if opt else None})
    if mismatches:
        return None, {"decision_id": label.get("decision_id"), "failure": "semantic_key_mismatch", "sample": mismatches[:3]}

    obs = reconstructed["obs"]
    deck = reconstructed["deck"]
    old_eq = TCR.old_ranker_eq(obs, deck, feat)
    option0 = 0
    search_selected = label.get("search_selected_option", label.get("search_argmax_option"))
    try:
        search_selected = int(search_selected)
    except Exception:
        search_selected = None
    dense = add_search_features(feat["dense"], opts, search_selected, old_eq, option0, feat["eq"])
    stronger_values = [float(opts[i].get("stronger_value") or 0.0) for i in range(len(feat["eq"]))]
    best_idx = max(range(len(stronger_values)), key=lambda i: (stronger_values[i], -i))
    best_value = stronger_values[best_idx]
    return {
        "source": label.get("state_tag") or "unknown",
        "partition": partition_for(label, highcrit_index),
        "decision_id": label.get("decision_id") or f"{reconstructed.get('file')}:{reconstructed.get('step')}",
        "obs_hash": label.get("obs_hash") or TCR.obs_hash(obs),
        "game_file": reconstructed.get("file"),
        "step": reconstructed.get("step"),
        "player": reconstructed.get("player"),
        "deck_hash": TCR.deck_hash(deck),
        "turn": (obs.get("current") or {}).get("turn"),
        "turn_action_count": (obs.get("current") or {}).get("turnActionCount"),
        "cids": [int(x) for x in feat["cids"]],
        "base_dense": feat["dense"],
        "dense": dense,
        "eq": [int(x) for x in feat["eq"]],
        "keys": feat["keys"],
        "search_selected_option": search_selected,
        "old_ranker_option": raw_option_from_eq({"eq": feat["eq"], "current_search_value": [opts[i]["current_search_value"] for i in range(len(feat["eq"]))]}, old_eq),
        "option0_option": option0,
        "stronger_best_option": best_idx,
        "stronger_value": stronger_values,
        "current_search_value": [float(opts[i].get("current_search_value") or 0.0) for i in range(len(feat["eq"]))],
        "regret": [float(opts[i].get("regret") or max(0.0, best_value - stronger_values[i])) for i in range(len(feat["eq"]))],
        "high_regret_flag": [int(opts[i].get("high_regret_flag") or 0) for i in range(len(feat["eq"]))],
        "unacceptable_flag": [int(opts[i].get("unacceptable_flag") or 0) for i in range(len(feat["eq"]))],
        "outcome_winrate": [opts[i].get("outcome_winrate") for i in range(len(feat["eq"]))],
        "outcome_se": [opts[i].get("outcome_se") for i in range(len(feat["eq"]))],
        "value_variance": [opts[i].get("value_variance") for i in range(len(feat["eq"]))],
        "value_se": [opts[i].get("value_se") for i in range(len(feat["eq"]))],
        "high_regret_thresh": float(label.get("high_regret_thresh", args.high_regret_threshold)),
        "criticality_score": float((label.get("criticality") or {}).get("score", 0.0) or 0.0),
        "hand_outcome_disagree": label.get("stronger_argmax_option") != label.get("search_argmax_option")
        if label.get("stronger_argmax_option") is not None else None,
        "coverage": label.get("coverage"),
        "timing": label.get("timing"),
        "config": label.get("config"),
    }, None


def dataset_summary(rows: list[dict]) -> dict:
    return {
        "decisions": len(rows),
        "options": sum(len(r["eq"]) for r in rows),
        "by_partition": dict(sorted(Counter(r["partition"] for r in rows).items())),
        "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
        "high_regret_options": sum(sum(r["high_regret_flag"]) for r in rows),
        "unacceptable_options": sum(sum(r["unacceptable_flag"]) for r in rows),
    }


def build_dataset(args) -> dict:
    labels = load_jsonl(args.labels)
    rows = []
    failures = []
    option_alignment = Counter()
    highcrit_i = 0
    for i, label in enumerate(labels):
        current_highcrit_i = highcrit_i
        if label.get("state_tag") == "high_criticality":
            highcrit_i += 1
        row, failure = make_row(label, i, current_highcrit_i, args)
        if row:
            rows.append(row)
            option_alignment["decisions_aligned"] += 1
            option_alignment["options_aligned"] += len(row["eq"])
        else:
            failures.append(failure)
    report = {
        "artifact_version": "teacher_v2_residual_risk_labels.risk_only_features",
        "branch": "exp/robust-learner-v2",
        "input": display(args.labels),
        "summary": dataset_summary(rows),
        "labels_loaded": len(labels),
        "options_loaded": sum(len(r.get("options") or []) for r in labels),
        "feature_rows_generated": sum(len(r["eq"]) for r in rows),
        "option_alignment": dict(option_alignment),
        "missing_fields": [],
        "failures": failures,
        "training_ready": len(rows) == len(labels) and not failures,
        "decisions": rows,
    }
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


class RiskNet(nn.Module):
    def __init__(self, n_cards: int, dense_dim: int, emb: int = 24, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(n_cards, emb)
        self.net = nn.Sequential(
            nn.Linear(emb + dense_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.high_regret = nn.Linear(hidden, 1)
        self.unacceptable = nn.Linear(hidden, 1)

    def forward(self, cidx, dense):
        h = self.net(torch.cat([self.emb(cidx), dense], dim=-1))
        return self.high_regret(h).squeeze(-1), self.unacceptable(h).squeeze(-1)


def ablated_dense(dense: list[list[float]], ablate: dict | None) -> np.ndarray:
    arr = np.array(dense, dtype=np.float32)
    if not ablate:
        return arr
    base_dim = CR.SLICES["dense_dim"][1]
    base = CR.apply_ablation(arr[:, :base_dim], ablate)
    return np.concatenate([base, arr[:, base_dim:]], axis=1)


def dense_stats(rows: list[dict], ablate: dict | None, min_std: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.concatenate([ablated_dense(r["dense"], ablate) for r in rows], axis=0)
    return arr.mean(axis=0), np.maximum(arr.std(axis=0), min_std)


def tensor_inputs(row: dict, mean: np.ndarray, std: np.ndarray, id2ix: dict[int, int],
                  clip_z: float, ablate: dict | None):
    arr = ablated_dense(row["dense"], ablate)
    arr = (arr - mean) / std
    if clip_z > 0:
        arr = np.clip(arr, -clip_z, clip_z)
    dense = torch.tensor(arr, dtype=torch.float32)
    cidx = torch.tensor([id2ix.get(int(c), 0) for c in row["cids"]], dtype=torch.long)
    return cidx, dense


def target_tensors(row: dict):
    hi = torch.tensor([float(x) for x in row["high_regret_flag"]], dtype=torch.float32)
    unacc = torch.tensor([float(x) for x in row["unacceptable_flag"]], dtype=torch.float32)
    return hi, unacc


def train_one(name: str, train: list[dict], rows: list[dict], args, ablate: dict | None = None) -> tuple[RiskNet, dict]:
    seed_offsets = {
        "new_a_label_risk_only": 0,
        "no_effects_risk_only": 101,
        "no_deltas_risk_only": 202,
    }
    torch.manual_seed(args.seed + seed_offsets.get(name, 303))
    mean, std = dense_stats(train, ablate, args.min_std)
    card_ids = sorted({int(c) for r in rows for c in r["cids"] if int(c) >= 0}) or [0]
    id2ix = {c: i for i, c in enumerate(card_ids)}
    model = RiskNet(len(card_ids), len(mean), emb=args.emb_dim, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hi_pos = sum(sum(r["high_regret_flag"]) for r in train)
    hi_total = sum(len(r["eq"]) for r in train)
    un_pos = sum(sum(r["unacceptable_flag"]) for r in train)
    un_total = hi_total
    hi_pos_weight = min(args.max_pos_weight, max(1.0, (hi_total - hi_pos) / max(1, hi_pos)))
    un_pos_weight = min(args.max_pos_weight, max(1.0, (un_total - un_pos) / max(1, un_pos)))
    order = list(range(len(train)))
    losses = []
    for ep in range(args.epochs):
        model.train()
        rng = np.random.default_rng(args.seed + ep)
        rng.shuffle(order)
        total = 0.0
        for i in order:
            row = train[i]
            cidx, dense = tensor_inputs(row, mean, std, id2ix, args.clip_z, ablate)
            pred_hi, pred_un = model(cidx, dense)
            hi, un = target_tensors(row)
            hi_loss = F.binary_cross_entropy_with_logits(
                pred_hi, hi, pos_weight=torch.tensor(float(hi_pos_weight))
            )
            un_loss = F.binary_cross_entropy_with_logits(
                pred_un, un, pos_weight=torch.tensor(float(un_pos_weight))
            )
            loss = args.lam_high_regret * hi_loss + args.lam_unacceptable * un_loss
            opt.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            total += float(loss.detach())
        losses.append(total / max(1, len(order)))
        print(f"  {name} epoch {ep + 1}/{args.epochs} loss={losses[-1]:.4f}", flush=True)
    model.eval()
    blob = {
        "artifact_version": "contextual_risk_only.v1",
        "branch": "exp/robust-learner-v2",
        "state_dict": {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()},
        "card_ids": card_ids,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "dense_dim": int(len(mean)),
        "base_feature_dim": CR.SLICES["dense_dim"][1],
        "extra_feature_names": EXTRA_FEATURE_NAMES,
        "emb": args.emb_dim,
        "hidden": args.hidden,
        "clip_z": args.clip_z,
        "ablate": ablate or {},
        "target": "high_regret_flag_and_unacceptable_flag",
        "integration": "agent_search chooses normally; risk model only intervenes if selected option is high-risk",
        "threshold": args.risk_threshold,
        "train_rows": len(train),
        "epoch_losses": losses,
        "pos_weight": {
            "high_regret": hi_pos_weight,
            "unacceptable": un_pos_weight,
        },
        "live_agent_consumed": "none",
    }
    return model, blob


def save_model(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, separators=(",", ":")), encoding="utf-8")


def risk_probs(model: RiskNet, blob: dict, row: dict) -> tuple[list[float], list[float], list[float]]:
    mean = np.array(blob["mean"], dtype=np.float32)
    std = np.array(blob["std"], dtype=np.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    cidx, dense = tensor_inputs(row, mean, std, id2ix, float(blob.get("clip_z", 6.0)), blob.get("ablate") or None)
    with torch.no_grad():
        hi, un = model(cidx, dense)
    hi_p = torch.sigmoid(hi).detach().cpu().numpy().astype(float).tolist()
    un_p = torch.sigmoid(un).detach().cpu().numpy().astype(float).tolist()
    risk = [max(a, b) for a, b in zip(hi_p, un_p)]
    return risk, hi_p, un_p


def choose_by_risk(row: dict, risk: list[float], threshold: float) -> int:
    selected = row.get("search_selected_option")
    if selected is None or selected < 0 or selected >= len(row["eq"]):
        selected = max(range(len(row["eq"])), key=lambda i: (row["current_search_value"][i], -i))
    if risk[selected] < threshold:
        return int(selected)
    order = sorted(range(len(row["eq"])), key=lambda i: (row["current_search_value"][i], i), reverse=True)
    for idx in order:
        if risk[idx] < threshold:
            return int(idx)
    return int(selected)


def previous_full_choice(row: dict, model_path: Path, ablate: dict | None = None) -> int:
    blob = json.loads(model_path.read_text(encoding="utf-8"))
    model = TCR.ContextualNet(
        len(blob["card_ids"]),
        int(blob["dense_dim"]),
        emb=int(blob.get("emb", 24)),
        hidden=int(blob.get("hidden", 192)),
        use_emb=bool(blob.get("use_emb", True)),
    )
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in blob["state_dict"].items()})
    model.eval()
    dense = np.array(row["base_dense"], dtype=np.float32)
    mean = np.array(blob["mean"], dtype=np.float32)
    std = np.array(blob["std"], dtype=np.float32)
    dense = (dense - mean) / std
    clip_z = float(blob.get("clip_z", 0.0) or 0.0)
    if clip_z > 0:
        dense = np.clip(dense, -clip_z, clip_z)
    if ablate:
        dense = CR.apply_ablation(dense, ablate)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    emb_dim = int(blob.get("emb", 24))
    idxs = [id2ix.get(int(cid), -1) for cid in row["cids"]]
    with torch.no_grad():
        emb_rows = [model.emb.weight[ix] if ix >= 0 else torch.zeros(emb_dim) for ix in idxs]
        x = torch.cat([torch.stack(emb_rows), torch.tensor(dense, dtype=torch.float32)], dim=-1)
        logits = model.net(x).squeeze(-1).detach().cpu().numpy().astype(float).tolist()
    return int(max(range(len(logits)), key=lambda i: (logits[i], -i)))


def load_b_bootstrap(path: Path):
    blob = json.loads(path.read_text(encoding="utf-8"))
    model = BRISK.ResidualRiskNet(
        len(blob["card_ids"]),
        int(blob["dense_dim"]),
        emb=int(blob.get("emb", 24)),
        hidden=int(blob.get("hidden", 128)),
    )
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in blob["state_dict"].items()})
    model.eval()
    return model, blob


def b_bootstrap_choice(model, blob: dict, row: dict, threshold: float) -> int:
    dense_dim = int(blob["dense_dim"])
    b_row = dict(row)
    b_row["dense"] = [list(x[:dense_dim]) for x in row["dense"]]
    pred, rr = BRISK.predict_row(model, b_row, blob, "risk", 0.45)
    return choose_by_risk(row, rr["option_risk_probs"], threshold)


def selection_metrics(rows: list[dict], choices: dict[str, list[int]]) -> dict:
    out = {}
    for name, preds in choices.items():
        regrets = []
        acceptable = []
        high = []
        top1 = []
        for row, pred in zip(rows, preds):
            pred = int(pred)
            regrets.append(float(row["regret"][pred]))
            acceptable.append(int(row["unacceptable_flag"][pred]) == 0)
            high.append(int(row["high_regret_flag"][pred]) == 1)
            top1.append(pred == int(row["stronger_best_option"]))
        out[name] = {
            "n": len(rows),
            "mean_regret": average(regrets) if regrets else None,
            "p95_regret": percentile(regrets, 0.95),
            "p99_regret": percentile(regrets, 0.99),
            "high_regret_count": sum(high),
            "acceptable_action_rate": sum(acceptable) / len(acceptable) if acceptable else None,
            "top1": sum(top1) / len(top1) if top1 else None,
        }
    return out


def detection_metrics(rows: list[dict], risk_by_row: list[list[float]], threshold: float) -> dict:
    hi_y = []
    un_y = []
    safe_y = []
    pred = []
    for row, risks in zip(rows, risk_by_row):
        for i, p in enumerate(risks):
            hi = int(row["high_regret_flag"][i]) == 1
            un = int(row["unacceptable_flag"][i]) == 1
            hi_y.append(hi)
            un_y.append(un)
            safe_y.append(not hi and not un)
            pred.append(float(p) >= threshold)
    hi_recall = sum(1 for y, p in zip(hi_y, pred) if y and p) / max(1, sum(hi_y))
    un_recall = sum(1 for y, p in zip(un_y, pred) if y and p) / max(1, sum(un_y))
    fp = sum(1 for safe, p in zip(safe_y, pred) if safe and p)
    tn = sum(1 for safe, p in zip(safe_y, pred) if safe and not p)
    return {
        "high_regret_recall": hi_recall,
        "unacceptable_action_recall": un_recall,
        "false_positive_risk_rate": fp / max(1, fp + tn),
        "high_regret_options": sum(hi_y),
        "unacceptable_options": sum(un_y),
        "safe_options": sum(safe_y),
    }


def evaluate(rows: list[dict], models: dict[str, tuple[RiskNet, dict]], args) -> dict:
    prev_full_path = resolve(args.previous_full_model)
    b_model, b_blob = load_b_bootstrap(resolve(args.b_bootstrap_model))
    subsets = {
        "all": rows,
        "test": [r for r in rows if r["partition"] == "test"],
        "b_failure": [r for r in rows if r["source"] == "B_failure_state"],
        "high_criticality": [r for r in rows if r["source"] == "high_criticality"],
        "high_regret_decisions": [r for r in rows if any(r["high_regret_flag"])],
    }
    report = {}
    risk_reports = {}
    for subset_name, subset in subsets.items():
        if not subset:
            continue
        choices = {
            "agent_search": [r.get("search_selected_option") if r.get("search_selected_option") is not None else 0 for r in subset],
            "old_ranker": [r.get("old_ranker_option") if r.get("old_ranker_option") is not None else 0 for r in subset],
            "previous_full_teacher_v2": [previous_full_choice(r, prev_full_path, None) for r in subset],
            "previous_b_bootstrap_risk_only": [b_bootstrap_choice(b_model, b_blob, r, args.risk_threshold) for r in subset],
        }
        for name, (model, blob) in models.items():
            risk_rows = [risk_probs(model, blob, r)[0] for r in subset]
            choices[name] = [choose_by_risk(r, risk, args.risk_threshold) for r, risk in zip(subset, risk_rows)]
            risk_reports[f"{subset_name}:{name}"] = detection_metrics(subset, risk_rows, args.risk_threshold)
        report[subset_name] = selection_metrics(subset, choices)
    return {"selection": report, "detection": risk_reports}


def decide(eval_report: dict) -> dict:
    test = eval_report["selection"]["test"]
    base = test["agent_search"]
    candidate = test["new_a_label_risk_only"]
    old = test["old_ranker"]
    previous_b = test["previous_b_bootstrap_risk_only"]
    improves = (
        (candidate["mean_regret"] or 1e30) < (base["mean_regret"] or 1e30)
        and (candidate["p95_regret"] or 1e30) <= (base["p95_regret"] or 1e30)
        and (candidate["p99_regret"] or 1e30) <= (base["p99_regret"] or 1e30)
        and (candidate["high_regret_count"] or 0) <= (base["high_regret_count"] or 0)
        and (candidate["acceptable_action_rate"] or 0.0) >= (base["acceptable_action_rate"] or 0.0)
    )
    not_bad = (
        (candidate["mean_regret"] or 1e30) <= min(old["mean_regret"] or 1e30, previous_b["mean_regret"] or 1e30)
        or (candidate["acceptable_action_rate"] or 0.0) >= max(old["acceptable_action_rate"] or 0.0, previous_b["acceptable_action_rate"] or 0.0)
    )
    if improves and not_bad:
        return {
            "choice": "A",
            "title": "risk-only offline improves enough to justify a small live screen",
            "rationale": (
                "The A-label risk model improves agent_search safety metrics on the held-out test split. "
                "Use only a conservative risk gate; do not promote from offline evidence."
            ),
        }
    if improves:
        return {
            "choice": "B",
            "title": "risk-only improves offline but needs more labels before live screen",
            "rationale": "The model improves agent_search but does not compare cleanly against prior risk/old-ranker safety baselines.",
        }
    return {
        "choice": "C",
        "title": "risk-only fails; request more/different labels from Model A",
        "rationale": "The A-label risk-only model did not improve the required offline safety metrics over agent_search.",
    }


def markdown_table(rows: list[list]) -> str:
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, row in enumerate(rows):
        out.append("| " + " | ".join(str(v).ljust(widths[j]) for j, v in enumerate(row)) + " |")
        if i == 0:
            out.append("| " + " | ".join("-" * widths[j] for j in range(len(row))) + " |")
    return "\n".join(out)


def write_summary(path: Path, report: dict) -> None:
    ingest = report["label_ingest"]
    test = report["offline_eval"]["selection"]["test"]
    rows = [["model", "mean regret", "p95", "p99", "hi-regret", "acceptable", "top1"]]
    for name in [
        "agent_search",
        "old_ranker",
        "previous_full_teacher_v2",
        "previous_b_bootstrap_risk_only",
        "new_a_label_risk_only",
        "no_effects_risk_only",
        "no_deltas_risk_only",
    ]:
        m = test[name]
        rows.append([
            name,
            round(float(m.get("mean_regret") or 0.0), 2),
            round(float(m.get("p95_regret") or 0.0), 2),
            round(float(m.get("p99_regret") or 0.0), 2),
            m.get("high_regret_count"),
            round(float(m.get("acceptable_action_rate") or 0.0), 3),
            round(float(m.get("top1") or 0.0), 3),
        ])
    det = report["offline_eval"]["detection"]["test:new_a_label_risk_only"]
    if report["decision"]["choice"] == "A":
        integration = (
            "`agent_search_risk`: run `agent_search` normally. If the chosen raw option is "
            "predicted high-risk above the calibrated threshold, do one conservative fallback "
            "only: choose the first lower-risk sibling in current search-value order, or trigger "
            "extra search if an integration budget is available. The risk model must not freely "
            "reorder all actions and must never replace search's scoring."
        )
        label_request = ""
    else:
        integration = (
            "No `agent_search_risk` integration is recommended from this run. The classifier has "
            "detection signal, but the gated intervention did not improve selected-action safety "
            "over plain `agent_search`."
        )
        label_request = (
            "\n## Label Request\n\n"
            "Decision C is packaged as `data/manifests/teacher_v2_risk_label_request_for_A.json`. "
            "The request asks for targeted residual/risk enrichment around search-selected "
            "high-regret misses and safe search-choice false positives, not a generic larger batch.\n"
        )
    text = f"""# A-Label Risk-Only Contextual Model

Status: offline only. No live screen, no promotion, and no `agent_search` modification.

Decision: **{report['decision']['choice']}** - {report['decision']['title']}

{report['decision']['rationale']}

## Ingest And Alignment

- Decisions loaded: {ingest['decisions_loaded']}
- Options loaded: {ingest['options_loaded']}
- Feature rows generated: {ingest['feature_rows_generated']}
- Option alignment: {ingest['option_alignment']['options_aligned']} / {ingest['options_loaded']}
- Missing fields: {ingest['missing_fields']}
- Training-ready: {str(ingest['training_ready']).lower()}

## Held-Out Test Safety

{markdown_table(rows)}

## Risk Detection

- High-regret recall: {det['high_regret_recall']:.3f}
- Unacceptable-action recall: {det['unacceptable_action_recall']:.3f}
- False-positive risk rate: {det['false_positive_risk_rate']:.3f}

## Conservative Integration Proposal

{integration}
{label_request}
"""
    path.write_text(text, encoding="utf-8")


def tail_report(rows: list[dict], models: dict[str, tuple[RiskNet, dict]], args) -> dict:
    model, blob = models["new_a_label_risk_only"]
    examples = []
    for row in [r for r in rows if r["partition"] == "test"]:
        risks, hi, un = risk_probs(model, blob, row)
        choice = choose_by_risk(row, risks, args.risk_threshold)
        search = row.get("search_selected_option") if row.get("search_selected_option") is not None else 0
        examples.append({
            "decision_id": row["decision_id"],
            "source": row["source"],
            "search_option": search,
            "risk_choice": choice,
            "best_option": row["stronger_best_option"],
            "search_regret": row["regret"][search],
            "risk_choice_regret": row["regret"][choice],
            "search_unacceptable": row["unacceptable_flag"][search],
            "risk_choice_unacceptable": row["unacceptable_flag"][choice],
            "search_high_regret": row["high_regret_flag"][search],
            "risk_choice_high_regret": row["high_regret_flag"][choice],
            "search_predicted_risk": risks[search],
            "risk_choice_predicted_risk": risks[choice],
            "max_predicted_risk": max(risks),
            "criticality_score": row.get("criticality_score"),
        })
    examples.sort(key=lambda e: (e["search_regret"] - e["risk_choice_regret"]), reverse=True)
    return {
        "artifact_version": "contextual_risk_only_v1.tail_report",
        "branch": "exp/robust-learner-v2",
        "examples": examples[:25],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    ap.add_argument("--dataset-out", type=Path, default=ROOT / "docs" / "workstreams" / "contextual_risk_only_v1_dataset.json")
    ap.add_argument("--model-out", type=Path, default=ROOT / "agent" / "contextual_risk_only_v1.json")
    ap.add_argument("--no-effects-model-out", type=Path, default=ROOT / "agent" / "contextual_risk_only_v1_no_effects.json")
    ap.add_argument("--no-deltas-model-out", type=Path, default=ROOT / "agent" / "contextual_risk_only_v1_no_deltas.json")
    ap.add_argument("--eval-out", type=Path, default=ROOT / "docs" / "workstreams" / "contextual_risk_only_v1_eval.json")
    ap.add_argument("--tail-out", type=Path, default=ROOT / "docs" / "workstreams" / "contextual_risk_only_v1_tail_report.json")
    ap.add_argument("--summary-out", type=Path, default=ROOT / "docs" / "workstreams" / "contextual_risk_only_v1_summary.md")
    ap.add_argument("--previous-full-model", type=Path, default=DEFAULT_PREVIOUS_FULL)
    ap.add_argument("--b-bootstrap-model", type=Path, default=DEFAULT_B_BOOTSTRAP)
    ap.add_argument("--high-regret-threshold", type=float, default=5000.0)
    ap.add_argument("--risk-threshold", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=24)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=6e-5)
    ap.add_argument("--min-std", type=float, default=0.1)
    ap.add_argument("--clip-z", type=float, default=6.0)
    ap.add_argument("--lam-high-regret", type=float, default=2.0)
    ap.add_argument("--lam-unacceptable", type=float, default=1.0)
    ap.add_argument("--max-pos-weight", type=float, default=30.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=31057)
    args = ap.parse_args()

    for key in [
        "labels", "replay_dir", "dataset_out", "model_out", "no_effects_model_out", "no_deltas_model_out",
        "eval_out", "tail_out", "summary_out", "previous_full_model", "b_bootstrap_model",
    ]:
        setattr(args, key, resolve(getattr(args, key)))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = build_dataset(args)
    if not dataset["training_ready"]:
        raise SystemExit("Risk-only dataset is not training-ready; see failures in dataset artifact.")
    rows = dataset["decisions"]
    train = [r for r in rows if r["partition"] == "train"]
    print(json.dumps({
        "decisions_loaded": dataset["labels_loaded"],
        "options_loaded": dataset["options_loaded"],
        "feature_rows_generated": dataset["feature_rows_generated"],
        "training_ready": dataset["training_ready"],
        "summary": dataset["summary"],
    }, indent=2), flush=True)

    full_model, full_blob = train_one("new_a_label_risk_only", train, rows, args, None)
    no_effects_model, no_effects_blob = train_one("no_effects_risk_only", train, rows, args, {"effects": True})
    no_deltas_model, no_deltas_blob = train_one("no_deltas_risk_only", train, rows, args, {"deltas": True})
    save_model(args.model_out, full_blob)
    save_model(args.no_effects_model_out, no_effects_blob)
    save_model(args.no_deltas_model_out, no_deltas_blob)
    models = {
        "new_a_label_risk_only": (full_model, full_blob),
        "no_effects_risk_only": (no_effects_model, no_effects_blob),
        "no_deltas_risk_only": (no_deltas_model, no_deltas_blob),
    }
    offline_eval = evaluate(rows, models, args)
    decision = decide(offline_eval)
    report = {
        "artifact_version": "contextual_risk_only_v1.eval",
        "branch": "exp/robust-learner-v2",
        "live_agent_consumed": "none",
        "arena_screen": "not run",
        "inputs": {
            "labels": display(args.labels),
            "dataset": display(args.dataset_out),
            "model": display(args.model_out),
            "no_effects_model": display(args.no_effects_model_out),
            "no_deltas_model": display(args.no_deltas_model_out),
            "previous_full_model": display(args.previous_full_model),
            "b_bootstrap_model": display(args.b_bootstrap_model),
        },
        "label_ingest": {
            "decisions_loaded": dataset["labels_loaded"],
            "options_loaded": dataset["options_loaded"],
            "feature_rows_generated": dataset["feature_rows_generated"],
            "option_alignment": dataset["option_alignment"],
            "missing_fields": dataset["missing_fields"],
            "training_ready": dataset["training_ready"],
            "failures": dataset["failures"],
        },
        "dataset_summary": dataset["summary"],
        "offline_eval": offline_eval,
        "decision": decision,
    }
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.tail_out.write_text(json.dumps(tail_report(rows, models, args), indent=2, sort_keys=True), encoding="utf-8")
    write_summary(args.summary_out, report)
    print(json.dumps({
        "decision": decision,
        "test": offline_eval["selection"]["test"],
        "detection": offline_eval["detection"]["test:new_a_label_risk_only"],
        "eval": display(args.eval_out),
        "summary": display(args.summary_out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
