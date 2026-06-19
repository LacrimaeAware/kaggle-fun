"""Train/evaluate a residual+risk contextual model relative to agent_search.

This is Branch B's narrower follow-up after the full Teacher V2 scorer failed:
the model predicts residual correction to current search and catastrophic-risk
signals, rather than replacing the whole action ranking.

The script can bootstrap labels from existing self-contained Teacher V2 labels
by querying the deployed Teacher V1/search estimate for the same roots. It does
not modify agent_search and does not run live arena screens.
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
import main as M  # noqa: E402
import teacher_api_v1 as TV1  # noqa: E402
import train_contextual_action_ranker as TCR  # noqa: E402


DEFAULT_LABELS = [
    ROOT / "data" / "manifests" / "teacher_v2_labels_scaled.jsonl",
    ROOT / "data" / "manifests" / "teacher_v2_labels_for_B_failures.jsonl",
]
EXTRA_FEATURE_NAMES = [
    "search_value",
    "search_norm_advantage",
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


def class_ids(eqs: list[int]) -> list[int]:
    out = []
    for e in eqs:
        e = int(e)
        if e not in out:
            out.append(e)
    return out


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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_source_name(path: Path) -> str:
    if "for_B_failures" in path.name:
        return "teacher_v2_targeted_failures"
    if "scaled" in path.name:
        return "teacher_v2_scaled"
    return path.stem


def partition_for(source: str, index: int) -> str:
    if source == "teacher_v2_targeted_failures":
        return "test"
    mod = index % 10
    if mod == 8:
        return "val"
    if mod == 9:
        return "test"
    return "train"


def option_maps(label: dict, feat: dict) -> dict:
    hand_value = {}
    hand_adv = {}
    variance = {}
    completed = {}
    outcome = {}
    outcome_se = {}
    for opt in label.get("options") or []:
        idx = int(opt["index"])
        if idx < 0 or idx >= len(feat["eq"]):
            continue
        if list(opt.get("semantic_action_key") or []) != list(feat["keys"][idx]):
            continue
        eq = int(feat["eq"][idx])
        if opt.get("hand_mean_value") is not None:
            hand_value.setdefault(eq, []).append(float(opt["hand_mean_value"]))
        if opt.get("hand_norm_advantage") is not None:
            hand_adv.setdefault(eq, []).append(float(opt["hand_norm_advantage"]))
        if opt.get("hand_value_variance") is not None:
            variance.setdefault(eq, []).append(float(opt["hand_value_variance"]))
        if opt.get("completed_determinizations") is not None:
            completed.setdefault(eq, []).append(float(opt["completed_determinizations"]))
        if opt.get("outcome_winrate") is not None:
            outcome.setdefault(eq, []).append(float(opt["outcome_winrate"]))
        if opt.get("outcome_se") is not None:
            outcome_se.setdefault(eq, []).append(float(opt["outcome_se"]))
    return {
        "hand_value": {k: average(v) for k, v in hand_value.items()},
        "hand_adv": {k: average(v) for k, v in hand_adv.items()},
        "variance": {k: average(v) for k, v in variance.items()},
        "completed": {k: average(v) for k, v in completed.items()},
        "outcome": {k: average(v) for k, v in outcome.items()},
        "outcome_se": {k: average(v) for k, v in outcome_se.items()},
    }


def search_maps(result: dict, feat: dict) -> dict | None:
    if not result.get("applicable"):
        return None
    vals = {}
    adv = {}
    counts = {}
    for opt in result.get("options") or []:
        idx = int(opt["index"])
        if idx < 0 or idx >= len(feat["eq"]):
            continue
        if list(opt.get("semantic_action_key") or []) != list(feat["keys"][idx]):
            continue
        eq = int(feat["eq"][idx])
        if opt.get("mean_value") is not None:
            vals.setdefault(eq, []).append(float(opt["mean_value"]))
        if opt.get("normalized_advantage") is not None:
            adv.setdefault(eq, []).append(float(opt["normalized_advantage"]))
        counts.setdefault(eq, []).append(float(opt.get("completed_determinizations") or 0))
    if not vals or not adv:
        return None
    chosen = result.get("chosen_option")
    chosen_eq = None
    if isinstance(chosen, int) and 0 <= chosen < len(feat["eq"]):
        chosen_eq = int(feat["eq"][chosen])
    return {
        "value": {k: average(v) for k, v in vals.items()},
        "adv": {k: average(v) for k, v in adv.items()},
        "counts": {k: average(v) for k, v in counts.items()},
        "chosen_eq": chosen_eq,
        "acceptable": [int(x) for x in (result.get("acceptable_action_set") or [])],
    }


def append_search_features(base_dense: list[list[float]], eqs: list[int], search_adv: dict[int, float],
                           search_value: dict[int, float], search_eq: int | None,
                           old_eq: int | None, option0_eq: int | None) -> list[list[float]]:
    ordered = sorted(search_adv, key=lambda k: search_adv[k], reverse=True)
    rank = {eq: i for i, eq in enumerate(ordered)}
    denom = max(1, len(ordered) - 1)
    out = []
    for row, eq_raw in zip(base_dense, eqs):
        eq = int(eq_raw)
        extra = [
            float(search_value.get(eq, 0.0)) / 1000.0,
            float(search_adv.get(eq, 0.0)) / 1000.0,
            float(rank.get(eq, len(ordered))) / denom,
            1.0 if search_eq is not None and eq == search_eq else 0.0,
            1.0 if old_eq is not None and eq == old_eq else 0.0,
            1.0 if option0_eq is not None and eq == option0_eq else 0.0,
        ]
        out.append([float(x) for x in row] + extra)
    return out


def make_row(label: dict, source_name: str, source_index: int, args) -> tuple[dict | None, dict | None]:
    reconstructed, attempts = BLD.find_reconstructed(label, args.replay_dir)
    if not reconstructed:
        return None, {"decision_id": label.get("decision_id"), "failure": "reconstruction_failed", "attempts": attempts}
    obs = reconstructed["obs"]
    deck = reconstructed["deck"] or list(M.DECK)
    feat = reconstructed["feat"]
    maps = option_maps(label, feat)
    if not maps["hand_adv"]:
        return None, {"decision_id": label.get("decision_id"), "failure": "missing_hand_targets"}
    try:
        search = TV1.query(
            obs,
            deck,
            n_determ=args.search_n_determ,
            time_budget=args.search_time_budget,
            leaf_mode="hand",
            seed=int(args.seed + source_index),
        )
    except Exception as exc:
        return None, {"decision_id": label.get("decision_id"), "failure": f"search_query_failed:{exc}"}
    smaps = search_maps(search, feat)
    if not smaps:
        return None, {"decision_id": label.get("decision_id"), "failure": "search_not_applicable_or_unaligned"}

    old_eq = TCR.old_ranker_eq(obs, deck, feat)
    option0_eq = int(feat["eq"][0])
    dense = append_search_features(feat["dense"], feat["eq"], smaps["adv"], smaps["value"], smaps["chosen_eq"], old_eq, option0_eq)
    ids = class_ids(feat["eq"])
    best_adv = max(maps["hand_adv"].values())
    deltas = {eq: maps["hand_adv"].get(eq, 0.0) - smaps["adv"].get(eq, 0.0) for eq in ids}
    max_abs_delta = max(1.0, max(abs(v) for v in deltas.values()))
    residual = {eq: deltas[eq] / max_abs_delta for eq in ids}
    regret = {eq: best_adv - maps["hand_adv"].get(eq, min(maps["hand_adv"].values())) for eq in ids}

    acceptable_teacher = set(int(x) for x in (label.get("acceptable_action_set") or []))
    # Teacher eq classes match feature eqs for these self-contained labels after alignment.
    risk = {
        eq: 1.0 if regret[eq] >= args.high_regret_threshold or eq not in acceptable_teacher else 0.0
        for eq in ids
    }
    strong_best_eq = max(maps["hand_adv"], key=lambda k: (maps["hand_adv"][k], -k))
    search_regret = regret.get(smaps["chosen_eq"], None)
    old_regret = regret.get(old_eq, None) if old_eq is not None else None
    opt0_regret = regret.get(option0_eq, None)
    return {
        "source": source_name,
        "partition": partition_for(source_name, source_index),
        "decision_id": label.get("decision_id") or f"{reconstructed.get('file')}:{reconstructed.get('step')}",
        "obs_hash": TCR.obs_hash(obs),
        "game_file": reconstructed.get("file"),
        "step": reconstructed.get("step"),
        "player": reconstructed.get("player"),
        "deck_hash": TCR.deck_hash(deck),
        "turn": (obs.get("current") or {}).get("turn"),
        "turn_action_count": (obs.get("current") or {}).get("turnActionCount"),
        "cids": [int(x) for x in feat["cids"]],
        "dense": dense,
        "base_dense": feat["dense"],
        "eq": [int(x) for x in feat["eq"]],
        "keys": feat["keys"],
        "search_value": {str(k): float(v) for k, v in smaps["value"].items()},
        "search_adv": {str(k): float(v) for k, v in smaps["adv"].items()},
        "search_eq": smaps["chosen_eq"],
        "old_ranker_eq": old_eq,
        "option0_eq": option0_eq,
        "strong_adv": {str(k): float(v) for k, v in maps["hand_adv"].items()},
        "strong_value": {str(k): float(v) for k, v in maps["hand_value"].items()},
        "residual": {str(k): float(v) for k, v in residual.items()},
        "risk": {str(k): float(v) for k, v in risk.items()},
        "regret": {str(k): float(v) for k, v in regret.items()},
        "acceptable": {str(k): 1.0 for k in acceptable_teacher},
        "strong_best_eq": strong_best_eq,
        "search_regret": search_regret,
        "old_ranker_regret": old_regret,
        "option0_regret": opt0_regret,
        "high_regret_threshold": args.high_regret_threshold,
        "outcome_winrate": {str(k): float(v) for k, v in maps["outcome"].items()},
        "outcome_se": {str(k): float(v) for k, v in maps["outcome_se"].items()},
        "hand_outcome_agree": label.get("hand_outcome_agree"),
        "criticality_score": BLD.criticality_score(label),
        "coverage_weight": BLD.coverage_weight(label),
        "value_variance_mean": average(list(maps["variance"].values())) if maps["variance"] else None,
        "completed_determinizations_mean": average(list(maps["completed"].values())) if maps["completed"] else None,
        "search_completed_determinizations_mean": average(list(smaps["counts"].values())) if smaps["counts"] else None,
        "teacher_v2_label_source": source_name,
        "teacher_v2_config": label.get("config"),
        "baseline_search_config": search.get("config"),
    }, None


def build_dataset(args) -> dict:
    rows = []
    failures = []
    label_counts = Counter()
    for path in args.teacher_v2_labels:
        source = label_source_name(path)
        labels = load_jsonl(path)
        label_counts[source] += len(labels)
        for idx, label in enumerate(labels):
            row, failure = make_row(label, source, idx, args)
            if row:
                rows.append(row)
            else:
                failures.append(failure)
    high_risk = sum(sum(1 for v in (r.get("risk") or {}).values() if v >= 0.5) for r in rows)
    residuals = [float(v) for r in rows for v in (r.get("residual") or {}).values()]
    report = {
        "artifact_version": "teacher_v2_residual_risk_labels.bootstrap_from_existing_v2",
        "branch": "exp/robust-learner-v2",
        "note": (
            "B-side bootstrap labels from existing Teacher V2 scaled/failure artifacts plus local "
            "Teacher V1 deployed-search estimates. This is not Model A's new high-compute residual-risk artifact."
        ),
        "inputs": [display(p) for p in args.teacher_v2_labels],
        "config": {
            "search_n_determ": args.search_n_determ,
            "search_time_budget": args.search_time_budget,
            "high_regret_threshold": args.high_regret_threshold,
            "seed": args.seed,
        },
        "summary": dataset_summary(rows),
        "source_label_counts": dict(label_counts),
        "failures": failures,
        "residual_distribution": {
            "n": len(residuals),
            "mean": average(residuals) if residuals else None,
            "p05": percentile(residuals, 0.05),
            "p50": percentile(residuals, 0.50),
            "p95": percentile(residuals, 0.95),
        },
        "high_regret_action_count": high_risk,
        "hand_outcome_disagreement_rate": (
            sum(1 for r in rows if r.get("hand_outcome_agree") is False) / len(rows) if rows else None
        ),
        "labels_complete_enough_for_B": len(rows) >= 20 and not failures,
        "decisions": rows,
    }
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def dataset_summary(rows: list[dict]) -> dict:
    return {
        "total_decisions": len(rows),
        "total_options": sum(len(r["eq"]) for r in rows),
        "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
        "by_partition": dict(sorted(Counter(r["partition"] for r in rows).items())),
        "search_high_regret_decisions": sum(
            1 for r in rows if r.get("search_regret") is not None and r["search_regret"] >= r["high_regret_threshold"]
        ),
    }


class ResidualRiskNet(nn.Module):
    def __init__(self, n_cards: int, dense_dim: int, emb: int = 24, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(n_cards, emb)
        self.shared = nn.Sequential(
            nn.Linear(emb + dense_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.residual = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, 1)
        self.gate = nn.Linear(hidden, 1)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, cidx, dense):
        h = self.shared(torch.cat([self.emb(cidx), dense], dim=-1))
        return (
            self.residual(h).squeeze(-1),
            self.risk(h).squeeze(-1),
            self.gate(h).squeeze(-1),
        )


def normalize_dense(train: list[dict], min_std: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array([x for row in train for x in row["dense"]], dtype=np.float32)
    return arr.mean(axis=0), np.maximum(arr.std(axis=0), min_std)


def tensor_inputs(row: dict, mean: np.ndarray, std: np.ndarray, id2ix: dict[int, int], clip_z: float):
    dense_np = np.array(row["dense"], dtype=np.float32)
    dense_np = (dense_np - mean) / std
    if clip_z > 0:
        dense_np = np.clip(dense_np, -clip_z, clip_z)
    dense = torch.tensor(dense_np, dtype=torch.float32)
    cidx = torch.tensor([id2ix.get(int(c), 0) for c in row["cids"]], dtype=torch.long)
    return cidx, dense


def to_map(row: dict, key: str) -> dict[int, float]:
    return {int(k): float(v) for k, v in (row.get(key) or {}).items()}


def option_targets(row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = to_map(row, "residual")
    risk = to_map(row, "risk")
    regret = to_map(row, "regret")
    res = torch.tensor([residual.get(int(e), 0.0) for e in row["eq"]], dtype=torch.float32)
    ris = torch.tensor([risk.get(int(e), 0.0) for e in row["eq"]], dtype=torch.float32)
    gate_target = torch.tensor([
        1.0 if abs(residual.get(int(e), 0.0)) >= 0.15 or regret.get(int(e), 0.0) >= row["high_regret_threshold"] else 0.0
        for e in row["eq"]
    ], dtype=torch.float32)
    return res, ris, gate_target


def train_model(rows: list[dict], args) -> tuple[ResidualRiskNet, dict]:
    train = [r for r in rows if r["partition"] == "train"]
    val = [r for r in rows if r["partition"] == "val"]
    if not train:
        raise SystemExit("No train rows for residual/risk model.")
    mean, std = normalize_dense(train, args.min_std)
    card_ids = sorted({int(c) for r in rows for c in r["cids"] if int(c) >= 0}) or [0]
    id2ix = {c: i for i, c in enumerate(card_ids)}
    model = ResidualRiskNet(len(card_ids), len(mean), emb=args.emb_dim, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    order = list(range(len(train)))
    for ep in range(args.epochs):
        model.train()
        rng = np.random.default_rng(args.seed + ep)
        rng.shuffle(order)
        total = 0.0
        for i in order:
            row = train[i]
            cidx, dense = tensor_inputs(row, mean, std, id2ix, args.clip_z)
            pred_res, pred_risk, pred_gate = model(cidx, dense)
            target_res, target_risk, target_gate = option_targets(row)
            weight = 1.0 + float(row.get("criticality_score") or 0.0)
            res_loss = F.smooth_l1_loss(pred_res, target_res)
            risk_loss = F.binary_cross_entropy_with_logits(pred_risk, target_risk)
            gate_loss = F.binary_cross_entropy_with_logits(pred_gate, target_gate)
            gate_small = torch.sigmoid(pred_gate).mean()
            loss = weight * (args.lam_residual * res_loss + args.lam_risk * risk_loss + args.lam_gate * gate_loss)
            loss = loss + args.lam_gate_small * gate_small
            opt.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            total += float(loss.detach())
        losses.append(total / max(1, len(order)))
        print(f"  residual-risk epoch {ep + 1}/{args.epochs} loss={losses[-1]:.4f}", flush=True)
    model.eval()
    return model, {
        "epoch_losses": losses,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "card_ids": card_ids,
        "dense_dim": int(len(mean)),
        "emb_dim": args.emb_dim,
        "hidden": args.hidden,
        "val_rows": len(val),
    }


def class_scores(values: list[float], eqs: list[int], mode: str = "max") -> dict[int, float]:
    out = {}
    for eq in class_ids(eqs):
        vals = [float(values[i]) for i, e in enumerate(eqs) if int(e) == int(eq)]
        out[eq] = max(vals) if mode == "max" else average(vals)
    return out


def predict_row(model: ResidualRiskNet, row: dict, blob: dict, mode: str, risk_penalty: float) -> tuple[int, dict]:
    mean = np.array(blob["mean"], dtype=np.float32)
    std = np.array(blob["std"], dtype=np.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    cidx, dense = tensor_inputs(row, mean, std, id2ix, float(blob.get("clip_z", 6.0)))
    with torch.no_grad():
        res, risk_logit, gate_logit = model(cidx, dense)
    residual = res.detach().cpu().numpy().astype(float).tolist()
    risk = torch.sigmoid(risk_logit).detach().cpu().numpy().astype(float).tolist()
    gate = torch.sigmoid(gate_logit).detach().cpu().numpy().astype(float).tolist()
    search = [to_map(row, "search_adv").get(int(e), 0.0) / 1000.0 for e in row["eq"]]
    if mode == "residual":
        scores = [s + r for s, r in zip(search, residual)]
    elif mode == "risk":
        scores = [s - risk_penalty * p for s, p in zip(search, risk)]
    else:
        scores = [s + g * r - risk_penalty * p for s, g, r, p in zip(search, gate, residual, risk)]
    cls = class_scores(scores, row["eq"])
    pred = max(cls, key=lambda k: (cls[k], -k))
    return pred, {
        "option_risk_probs": risk,
        "option_gate": gate,
        "option_residual": residual,
    }


def previous_model_choice(row: dict, model_path: Path, ablate_effects: bool = False) -> int | None:
    blob = json.loads(model_path.read_text(encoding="utf-8"))
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
    eval_row = dict(row)
    eval_row["dense"] = row["base_dense"]
    mean = torch.tensor(blob["mean"], dtype=torch.float32)
    std = torch.tensor(blob["std"], dtype=torch.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    emb_dim = int(blob.get("emb", 24))
    ablate = {"effects": True} if ablate_effects else None
    with torch.no_grad():
        ids, logits = TCR.score_decision(model, eval_row, mean, std, id2ix, emb_dim, ablate=ablate, clip_z=float(blob.get("clip_z", 0.0) or 0.0))
    scores = logits.detach().cpu().numpy().astype(float).tolist()
    return ids[max(range(len(ids)), key=lambda i: (scores[i], -ids[i]))]


def selected_metrics(rows: list[dict], choices: dict[str, list[int | None]], risk_records: list[dict] | None = None) -> dict:
    out = {}
    for name, preds in choices.items():
        recs = []
        for row, pred in zip(rows, preds):
            adv = to_map(row, "strong_adv")
            regret = to_map(row, "regret")
            acceptable = to_map(row, "acceptable")
            best = max(adv, key=lambda k: (adv[k], -k))
            if pred is None:
                pred = row.get("option0_eq")
            recs.append({
                "top1": int(pred) == int(best),
                "acceptable": acceptable.get(int(pred), 0.0) >= 0.5,
                "regret": regret.get(int(pred), max(regret.values()) if regret else 0.0),
            })
        regrets = [r["regret"] for r in recs]
        out[name] = {
            "n": len(rows),
            "top1": sum(1 for r in recs if r["top1"]) / len(recs) if recs else None,
            "acceptable_action_recall": sum(1 for r in recs if r["acceptable"]) / len(recs) if recs else None,
            "mean_regret": average(regrets) if regrets else None,
            "p95_regret": percentile(regrets, 0.95),
            "p99_regret": percentile(regrets, 0.99),
            "high_regret_count": sum(1 for r in regrets if r >= (rows[0]["high_regret_threshold"] if rows else 100.0)),
        }
    if risk_records:
        labels = []
        probs = []
        for row, rr in zip(rows, risk_records):
            risk = to_map(row, "risk")
            for i, eq in enumerate(row["eq"]):
                labels.append(risk.get(int(eq), 0.0) >= 0.5)
                probs.append(float(rr["option_risk_probs"][i]))
        tp = sum(1 for y, p in zip(labels, probs) if y and p >= 0.5)
        fn = sum(1 for y, p in zip(labels, probs) if y and p < 0.5)
        fp = sum(1 for y, p in zip(labels, probs) if not y and p >= 0.5)
        tn = sum(1 for y, p in zip(labels, probs) if not y and p < 0.5)
        out["risk_detection"] = {
            "catastrophic_risk_recall": tp / max(1, tp + fn),
            "false_positive_risk_rate": fp / max(1, fp + tn),
            "risk_positive_count": tp + fn,
            "risk_negative_count": fp + tn,
        }
    return out


def eval_model(rows: list[dict], model: ResidualRiskNet, blob: dict, args) -> dict:
    previous_path = resolve(args.previous_teacher_v2_model)
    choices = {
        "agent_search": [r.get("search_eq") for r in rows],
        "old_ranker": [r.get("old_ranker_eq") for r in rows],
        "option0": [r.get("option0_eq") for r in rows],
        "previous_full_teacher_v2": [previous_model_choice(r, previous_path, False) for r in rows],
        "previous_no_effects": [previous_model_choice(r, previous_path, True) for r in rows],
        "residual_only": [],
        "risk_only": [],
        "residual_plus_risk": [],
    }
    risk_records = []
    for row in rows:
        pred, rr = predict_row(model, row, blob, "residual", args.risk_penalty)
        choices["residual_only"].append(pred)
        pred, rr = predict_row(model, row, blob, "risk", args.risk_penalty)
        choices["risk_only"].append(pred)
        pred, rr = predict_row(model, row, blob, "both", args.risk_penalty)
        choices["residual_plus_risk"].append(pred)
        risk_records.append(rr)
    metrics = selected_metrics(rows, choices, risk_records)
    return metrics


def save_model(model: ResidualRiskNet, blob: dict, path: Path, args) -> None:
    out = {
        "artifact_version": "contextual_residual_risk.v1",
        "branch": "exp/robust-learner-v2",
        "state_dict": {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()},
        "card_ids": [int(x) for x in blob["card_ids"]],
        "mean": blob["mean"],
        "std": blob["std"],
        "dense_dim": blob["dense_dim"],
        "base_feature_dim": int(blob["dense_dim"]) - len(EXTRA_FEATURE_NAMES),
        "extra_feature_names": EXTRA_FEATURE_NAMES,
        "emb": args.emb_dim,
        "hidden": args.hidden,
        "clip_z": args.clip_z,
        "target": "search_residual_and_catastrophic_risk",
        "integration": "final_score = search_score + gate * predicted_residual; risk can down-prioritize/trigger extra search",
        "live_agent_consumed": "none",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")


def write_summary(path: Path, report: dict) -> None:
    test = report["eval"]["test"]
    rows = [["model", "mean regret", "p95", "p99", "hi-regret", "acceptable", "top1"]]
    for name in [
        "agent_search",
        "old_ranker",
        "option0",
        "previous_full_teacher_v2",
        "previous_no_effects",
        "residual_only",
        "risk_only",
        "residual_plus_risk",
    ]:
        m = test[name]
        rows.append([
            name,
            round(float(m.get("mean_regret") or 0.0), 2),
            round(float(m.get("p95_regret") or 0.0), 2),
            round(float(m.get("p99_regret") or 0.0), 2),
            m.get("high_regret_count"),
            round(float(m.get("acceptable_action_recall") or 0.0), 3),
            round(float(m.get("top1") or 0.0), 3),
        ])
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    table = []
    for i, row in enumerate(rows):
        table.append("| " + " | ".join(str(v).ljust(widths[j]) for j, v in enumerate(row)) + " |")
        if i == 0:
            table.append("| " + " | ".join("-" * widths[j] for j in range(len(row))) + " |")
    text = f"""# Residual/Risk Contextual Prototype

Status: offline only. No live screen and no `agent_search` modification.

Decision: **{report['decision']['choice']}** - {report['decision']['title']}

{report['decision']['rationale']}

Important caveat: labels are a Branch B bootstrap from existing Teacher V2 artifacts plus local deployed-search estimates, not Model A's new high-compute residual/risk artifact.

## Test Metrics

{chr(10).join(table)}

Risk detection: catastrophic-risk recall {test['risk_detection']['catastrophic_risk_recall']:.3f}, false-positive rate {test['risk_detection']['false_positive_risk_rate']:.3f}.
"""
    path.write_text(text, encoding="utf-8")


def decide(test: dict) -> dict:
    base = test["agent_search"]
    old = test["old_ranker"]
    opt0 = test["option0"]
    improved = []
    for name in ("residual_only", "risk_only", "residual_plus_risk"):
        candidate = test[name]
        beats_search = (
            (candidate["mean_regret"] or 1e30) < (base["mean_regret"] or 1e30)
            and (candidate["p95_regret"] or 1e30) <= (base["p95_regret"] or 1e30)
            and (candidate["p99_regret"] or 1e30) <= (base["p99_regret"] or 1e30)
            and (candidate["high_regret_count"] or 0) <= (base["high_regret_count"] or 0)
            and (candidate["acceptable_action_recall"] or 0.0) >= (base["acceptable_action_recall"] or 0.0)
        )
        not_worse_than_baselines = (
            (candidate["mean_regret"] or 1e30) <= min(old["mean_regret"] or 1e30, opt0["mean_regret"] or 1e30)
            and (candidate["p95_regret"] or 1e30) <= min(old["p95_regret"] or 1e30, opt0["p95_regret"] or 1e30)
            and (candidate["high_regret_count"] or 0) <= min(old["high_regret_count"] or 0, opt0["high_regret_count"] or 0)
        )
        if beats_search and not_worse_than_baselines:
            improved.append(name)
    if improved:
        return {
            "choice": "A",
            "title": "offline improved enough to justify a small agent_search_residual screen",
            "rationale": (
                f"{', '.join(improved)} improved mean regret, p95/p99 regret, high-regret count, and "
                "acceptable-action recall against agent_search on the held-out bootstrap test without losing "
                "to old-ranker/option-0 safety metrics. Treat this as screen-eligible only with the caveat "
                "that these are B-bootstrap labels, not Model A's dedicated high-compute residual/risk artifact."
            ),
        }
    return {
        "choice": "C",
        "title": "offline did not improve; request more/specific residual-risk labels from Model A",
        "rationale": (
            "The prototype uses only a small B-bootstrap label set. It does not clear the offline gate, "
            "so the next useful step is the dedicated Model A residual/risk artifact rather than another "
            "B-side proxy pass."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-v2-labels", type=Path, nargs="*", default=DEFAULT_LABELS)
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    ap.add_argument("--dataset-out", type=Path, default=ROOT / "data" / "manifests" / "teacher_v2_residual_risk_labels_B_bootstrap.json")
    ap.add_argument("--model-out", type=Path, default=ROOT / "agent" / "contextual_residual_risk_v1.json")
    ap.add_argument("--report-out", type=Path, default=ROOT / "docs" / "workstreams" / "contextual_residual_risk_v1_eval.json")
    ap.add_argument("--summary-out", type=Path, default=ROOT / "docs" / "workstreams" / "contextual_residual_risk_v1_summary.md")
    ap.add_argument("--previous-teacher-v2-model", type=Path, default=ROOT / "agent" / "contextual_ranker_teacher_v2.json")
    ap.add_argument("--search-n-determ", type=int, default=8)
    ap.add_argument("--search-time-budget", type=float, default=0.6)
    ap.add_argument("--high-regret-threshold", type=float, default=100.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=24)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-5)
    ap.add_argument("--min-std", type=float, default=0.1)
    ap.add_argument("--clip-z", type=float, default=6.0)
    ap.add_argument("--lam-residual", type=float, default=1.0)
    ap.add_argument("--lam-risk", type=float, default=0.7)
    ap.add_argument("--lam-gate", type=float, default=0.1)
    ap.add_argument("--lam-gate-small", type=float, default=0.03)
    ap.add_argument("--risk-penalty", type=float, default=0.45)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=27181)
    args = ap.parse_args()

    args.teacher_v2_labels = [resolve(p) for p in args.teacher_v2_labels]
    args.replay_dir = resolve(args.replay_dir)
    args.dataset_out = resolve(args.dataset_out)
    args.model_out = resolve(args.model_out)
    args.report_out = resolve(args.report_out)
    args.summary_out = resolve(args.summary_out)
    args.previous_teacher_v2_model = resolve(args.previous_teacher_v2_model)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = build_dataset(args)
    rows = dataset["decisions"]
    model, train_blob = train_model(rows, args)
    save_model(model, train_blob, args.model_out, args)

    eval_sets = {
        "train": [r for r in rows if r["partition"] == "train"],
        "val": [r for r in rows if r["partition"] == "val"],
        "test": [r for r in rows if r["partition"] == "test"],
        "targeted_failures": [r for r in rows if r["source"] == "teacher_v2_targeted_failures"],
        "hand_outcome_disagreement": [r for r in rows if r.get("hand_outcome_agree") is False],
        "search_high_regret": [
            r for r in rows
            if r.get("search_regret") is not None and r["search_regret"] >= r["high_regret_threshold"]
        ],
    }
    eval_report = {name: eval_model(subset, model, train_blob, args) for name, subset in eval_sets.items() if subset}
    decision = decide(eval_report["test"])
    report = {
        "artifact_version": "contextual_residual_risk_v1.eval",
        "branch": "exp/robust-learner-v2",
        "live_agent_consumed": "none",
        "arena_screen": "not run",
        "inputs": {
            "bootstrap_labels": display(args.dataset_out),
            "model": display(args.model_out),
            "teacher_v2_labels": [display(p) for p in args.teacher_v2_labels],
            "previous_teacher_v2_model": display(args.previous_teacher_v2_model),
        },
        "config": {
            "high_regret_threshold": args.high_regret_threshold,
            "risk_penalty": args.risk_penalty,
            "search_n_determ": args.search_n_determ,
            "search_time_budget": args.search_time_budget,
        },
        "dataset_summary": dataset["summary"],
        "train": train_blob,
        "eval": eval_report,
        "decision": decision,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(args.summary_out, report)
    print(json.dumps({
        "dataset": dataset["summary"],
        "decision": decision,
        "test": eval_report["test"],
        "model": display(args.model_out),
        "report": display(args.report_out),
        "summary": display(args.summary_out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
