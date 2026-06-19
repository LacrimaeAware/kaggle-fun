"""Calibration diagnostics for Contextual Action Ranker V1.

This script is read-only over an existing dataset/model pair. It reports the
feature-scale path, slice metrics, baselines, and full-model sensitivity to
feature families so calibration notes are reproducible.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import contextual_ranker as CR  # noqa: E402
import features as FT  # noqa: E402
import train_contextual_action_ranker as TCR  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def to_float(x):
    if x is None:
        return None
    return float(x)


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


def feature_names() -> list[str]:
    names = []
    names += [f"action_type_{t}" for t in CR.OTYPES]
    names += [
        "card_is_pokemon", "card_is_trainer", "card_is_energy", "card_is_basic", "card_is_evolution",
        "card_is_ex_or_mega", "card_hp", "card_best_damage", "attack_damage", "attack_cost",
        "target_has_inplay_index", "target_is_opp",
    ]
    names += [f"effect_{k}" for k in CR.EFFECT_KEYS]
    names += [
        "target_remaining_hp", "target_koable_now", "target_prize_value", "target_energy_attached",
        "target_future_threat", "target_engine_role", "target_retreat_strand", "target_is_basic",
        "target_is_evolution", "target_is_ex_or_mega", "target_is_opp", "target_is_active",
        "target_is_bench", "target_is_retreat",
    ]
    names += [f"delta_{k}" for k in CR.DELTA_KEYS]
    names += [f"root_{k}" for k in FT.FEATURE_KEYS]
    names += [f"interaction_{e}_x_{c}" for e in CR.EFFECT_KEYS for c in CR.CTX_KEYS]
    names += ["hist_turn", "hist_turn_action_count", "hist_supporter", "hist_stadium", "hist_energy", "hist_retreat"]
    return names


def section_for_index(idx: int) -> str:
    for name, (a, b) in CR.SLICES.items():
        if name != "dense_dim" and a <= idx < b:
            return name
    return "unknown"


def normalization_diagnostics(rows: list[dict], blob: dict, train_partitions: set[str]) -> dict:
    allx = np.array([x for d in rows for x in d["dense"]], dtype=np.float32)
    train = [d for d in rows if d["partition"] in train_partitions]
    trainx = np.array([x for d in train for x in d["dense"]], dtype=np.float32)
    mean = np.array(blob["mean"], dtype=np.float32)
    std = np.array(blob["std"], dtype=np.float32)
    names = feature_names()
    out = {
        "dense_dim": int(allx.shape[1]),
        "n_option_rows": int(allx.shape[0]),
        "std_min": float(std.min()),
        "std_le_1e_5": int((std <= 1e-5).sum()),
        "std_le_1e_3": int((std <= 1e-3).sum()),
        "std_le_0_01": int((std <= 0.01).sum()),
        "std_floor": blob.get("min_std"),
        "clip_z": blob.get("clip_z"),
        "by_section": {},
        "tiny_std_but_varies": [],
    }
    for name, (a, b) in CR.SLICES.items():
        if name == "dense_dim":
            continue
        sx = std[a:b]
        vals = trainx[:, a:b]
        allvals = allx[:, a:b]
        z = np.abs((allvals - mean[a:b]) / sx)
        out["by_section"][name] = {
            "dim": b - a,
            "std_min": float(sx.min()),
            "std_le_1e_5": int((sx <= 1e-5).sum()),
            "std_le_1e_3": int((sx <= 1e-3).sum()),
            "train_nonzero_rate": float((vals != 0).mean()),
            "all_nonzero_rate": float((allvals != 0).mean()),
            "raw_max_abs": float(np.abs(allvals).max()),
            "z_p99_abs": float(np.percentile(z, 99)),
            "z_max_abs": float(z.max()),
        }
    for j in range(len(std)):
        if std[j] <= 1e-5 and np.max(np.abs(allx[:, j] - mean[j])) > 1e-8:
            out["tiny_std_but_varies"].append({
                "index": j,
                "section": section_for_index(j),
                "name": names[j] if j < len(names) else f"feature_{j}",
                "mean": float(mean[j]),
                "std": float(std[j]),
                "min": float(allx[:, j].min()),
                "max": float(allx[:, j].max()),
            })
    return out


def class_ids(d: dict) -> set[int]:
    return {int(x) for x in d["eq"]}


def adv_spread(d: dict) -> float:
    vals = [float(v) for v in (d.get("adv") or {}).values()]
    return max(vals) - min(vals) if vals else 0.0


def action_types(d: dict) -> set[int]:
    out = set()
    for k in d.get("keys") or []:
        if k and isinstance(k[0], int):
            out.add(k[0])
    return out


def is_mixed_strategic(d: dict) -> bool:
    return len(class_ids(d)) >= 3 and len(action_types(d)) >= 2


def is_high_criticality(d: dict, spread_threshold: float, margin_threshold: float) -> bool:
    margin = abs(float(d.get("top_two_margin") or 0.0))
    return bool(d.get("high_regret")) or adv_spread(d) >= spread_threshold or margin >= margin_threshold


def compact_metrics(metrics: dict) -> dict:
    keys = [
        "n", "top1", "top2", "top3", "pairwise_accuracy", "mrr", "ndcg", "acceptable_agreement",
        "mean_regret", "p90_regret", "p95_regret", "high_regret_count", "mean_entropy", "mean_score_margin",
    ]
    return {k: to_float(metrics.get(k)) for k in keys if k in metrics}


def eval_slices(model, rows: list[dict], blob: dict, args) -> dict:
    mean = torch.tensor(blob["mean"], dtype=torch.float32)
    std = torch.tensor(blob["std"], dtype=torch.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    emb_dim = int(blob.get("emb", 24))
    clip_z = float(blob.get("clip_z", 0.0) or 0.0)

    slices = {
        "all": rows,
        "train": [d for d in rows if d["partition"] == "train"],
        "val": [d for d in rows if d["partition"] == "val"],
        "test": [d for d in rows if d["partition"] == "test"],
        "recovery": [d for d in rows if d["partition"] == "recovery"],
        "stable": [d for d in rows if d.get("teacher_stability") == "stable"],
        "unstable": [d for d in rows if d.get("teacher_stability") == "unstable"],
        "high_criticality": [d for d in rows if is_high_criticality(d, args.criticality_spread, args.criticality_margin)],
        "high_regret_recovery": [d for d in rows if d.get("high_regret")],
        "mixed_strategic": [d for d in rows if is_mixed_strategic(d)],
    }

    out = {}
    for name, subset in slices.items():
        if not subset:
            out[name] = {"n": 0}
            continue
        full = TCR.eval_model(model, subset, mean, std, id2ix, emb_dim, clip_z=clip_z)["overall"]
        old = TCR.eval_baseline(subset, "old_ranker_eq")["overall"]
        opt0 = TCR.eval_baseline(subset, "option0_eq")["overall"]
        out[name] = {
            "full": compact_metrics(full),
            "old_ranker": compact_metrics(old),
            "option0": compact_metrics(opt0),
        }

    by_player = Counter(str(d.get("player")) for d in rows)
    out["held_out_player_like"] = {}
    for player, n in by_player.items():
        if n < args.min_group:
            continue
        subset = [d for d in rows if str(d.get("player")) == player]
        out["held_out_player_like"][player] = {
            "full": compact_metrics(TCR.eval_model(model, subset, mean, std, id2ix, emb_dim, clip_z=clip_z)["overall"]),
            "old_ranker": compact_metrics(TCR.eval_baseline(subset, "old_ranker_eq")["overall"]),
            "option0": compact_metrics(TCR.eval_baseline(subset, "option0_eq")["overall"]),
        }

    by_deck = Counter(str(d.get("deck_hash")) for d in rows)
    out["held_out_deck_like"] = {}
    for deck, n in by_deck.items():
        if n < args.min_group:
            continue
        subset = [d for d in rows if str(d.get("deck_hash")) == deck]
        out["held_out_deck_like"][deck] = {
            "full": compact_metrics(TCR.eval_model(model, subset, mean, std, id2ix, emb_dim, clip_z=clip_z)["overall"]),
            "old_ranker": compact_metrics(TCR.eval_baseline(subset, "old_ranker_eq")["overall"]),
            "option0": compact_metrics(TCR.eval_baseline(subset, "option0_eq")["overall"]),
        }
    return out


def sensitivity(model, rows: list[dict], blob: dict) -> dict:
    mean = torch.tensor(blob["mean"], dtype=torch.float32)
    std = torch.tensor(blob["std"], dtype=torch.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    emb_dim = int(blob.get("emb", 24))
    clip_z = float(blob.get("clip_z", 0.0) or 0.0)

    out = {}
    for name, ablate in {
        "zero_decoded_effects_and_interactions": {"effects": True},
        "zero_option_deltas": {"deltas": True},
        "zero_target_entity": {"target": True},
        "zero_history": {"history": True},
    }.items():
        out[name] = compact_metrics(
            TCR.eval_model(model, rows, mean, std, id2ix, emb_dim, ablate=ablate, clip_z=clip_z)["overall"]
        )

    original = model.emb.weight.detach().clone()
    with torch.no_grad():
        model.emb.weight.zero_()
    out["zero_card_embedding"] = compact_metrics(
        TCR.eval_model(model, rows, mean, std, id2ix, emb_dim, clip_z=clip_z)["overall"]
    )
    with torch.no_grad():
        model.emb.weight.copy_(original)
    return out


def dataset_summary(rows: list[dict]) -> dict:
    return {
        "total_decisions": len(rows),
        "by_source": dict(sorted(Counter(d["source"] for d in rows).items())),
        "by_partition": dict(sorted(Counter(d["partition"] for d in rows).items())),
        "by_teacher_stability": dict(sorted(Counter(d.get("teacher_stability", "unknown") for d in rows).items())),
        "high_regret": sum(1 for d in rows if d.get("high_regret")),
        "mixed_strategic": sum(1 for d in rows if is_mixed_strategic(d)),
        "players": len({d.get("player") for d in rows if d.get("player")}),
        "decks": len({d.get("deck_hash") for d in rows if d.get("deck_hash")}),
    }


def trainable_vs_fixed(blob: dict) -> dict:
    return {
        "decoded_effects": "trainable dense inputs plus trainable interactions; not fixed final bonuses",
        "tactic_features": "no Tactic Miner features are consumed by V1; target_engine_role is a fixed engineered scalar input, not a final score",
        "option_deltas": "trainable dense inputs from immediate one-step consequences; not fixed final bonuses",
        "state_effect_interactions": "hand-constructed products of decoded effects and root context, then trainable MLP inputs",
        "same_effect_can_vary_by_state": True,
        "manual_final_score_weights": False,
        "manual_preprocessing_weights": [
            "feature scaling constants such as /300, /3, /5",
            "target_engine_role combines decoded effects with fixed coefficients before the trainable MLP",
            "production forced-move floor still precedes search for lethal/go-first decisions",
        ],
        "live_authority": "contextual model orders candidates and tie-breaks equal search values; forward-model search remains final authority",
        "model_target": blob.get("target"),
        "clip_z": blob.get("clip_z"),
        "min_std": blob.get("min_std"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=ROOT / "docs" / "workstreams" / "contextual_action_ranker_v1_dataset.json")
    ap.add_argument("--model", type=Path, default=ROOT / "agent" / "contextual_ranker_v1.json")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "workstreams" / "contextual_action_ranker_v1_calibration_diagnostics.json")
    ap.add_argument("--criticality-spread", type=float, default=1000.0)
    ap.add_argument("--criticality-margin", type=float, default=500.0)
    ap.add_argument("--min-group", type=int, default=8)
    args = ap.parse_args()

    rows = load_dataset(resolve(args.dataset))
    blob, model = load_model(resolve(args.model))
    report = {
        "artifact_version": "contextual_action_ranker_v1.calibration_diagnostics",
        "dataset": str(resolve(args.dataset)),
        "model": str(resolve(args.model)),
        "trainable_vs_fixed": trainable_vs_fixed(blob),
        "dataset_summary": dataset_summary(rows),
        "normalization": normalization_diagnostics(rows, blob, {"train", "recovery"}),
        "slice_metrics": eval_slices(model, rows, blob, args),
        "full_model_sensitivity_all": sensitivity(model, rows, blob),
    }
    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "dataset": report["dataset_summary"],
        "normalization": {
            "std_min": report["normalization"]["std_min"],
            "std_le_1e_5": report["normalization"]["std_le_1e_5"],
            "tiny_std_but_varies": len(report["normalization"]["tiny_std_but_varies"]),
        },
        "all": report["slice_metrics"]["all"],
    }, indent=2, sort_keys=True))
    print(f"wrote -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
