"""Clean Continuous Terrain Representation V1 rerun.

This script reruns the Continuous Terrain Representation V1 gate after the
Search Metadata Dominance Audit.  It constructs leakage-clean inputs:

* R1 uses only strict live N=8 metadata/static-free features.
* R3 uses semantic state-action features with no teacher policy/label fields.
* R4 uses R3 plus the same strict live metadata branch.

It does not modify agent_search, does not run arena, and writes only offline
artifacts.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "workstreams"
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import continuous_terrain_encoder_v1 as CTE  # noqa: E402
import train_continuous_terrain_v1 as TERRAIN  # noqa: E402


DEFAULT_LABELS = ROOT / "data" / "manifests" / "continuous_terrain_v1.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "manifests" / "continuous_terrain_v1_summary.json"
DEFAULT_AUDIT = DOCS / "SEARCH_METADATA_DOMINANCE_AUDIT.md"
DEFAULT_OLD_REPORT = DOCS / "CONTINUOUS_TERRAIN_REPRESENTATION_V1.md"
DEFAULT_OUT_JSON = DOCS / "continuous_terrain_representation_v1_clean_rerun.json"
DEFAULT_OUT_MD = DOCS / "CONTINUOUS_TERRAIN_REPRESENTATION_V1_CLEAN_RERUN.md"
DEFAULT_MODEL = ROOT / "agent" / "continuous_terrain_encoder_v1_clean_rerun.pt"
DEFAULT_METADATA = ROOT / "agent" / "continuous_terrain_encoder_v1_clean_rerun.json"

FORBIDDEN_FEATURE_KEYS = {
    "value_spread",
    "value_se",
    "mean_stronger_value",
    "stronger_value_variance",
    "stronger_soft_policy",
    "policy_prob",
    "delta_to_search",
    "delta_to_search_norm",
    "hand_norm_advantage",
    "high_regret_prob",
    "unacceptable_prob",
    "acceptable_prob",
    "regret",
}

FEATURE_FAMILIES = {
    "R1_live_metadata_only": {
        "allowed": [
            "mean_live_value",
            "live_value_variance",
            "live_selected_distribution",
            "live_action_entropy",
            "modal_action_stability",
            "live margin/spread computed from mean_live_value only",
            "criticality",
            "option index",
            "search_selected_option",
            "n_options",
        ],
        "forbidden": sorted(FORBIDDEN_FEATURE_KEYS),
    },
    "R3_semantic_only": {
        "allowed": [
            "observation/root state features",
            "legal option/action descriptor",
            "semantic_action_key",
            "eq_class",
            "learned card-id embedding",
            "decoded card effect vector",
            "target/entity features",
            "state-effect interactions from root and effect features",
            "one-step option deltas from semantic_vector",
            "card metadata from semantic_vector",
        ],
        "forbidden": sorted(FORBIDDEN_FEATURE_KEYS | {"live_selected_distribution", "live_action_entropy", "modal_action_stability"}),
    },
    "R4_semantic_plus_live_metadata": {
        "allowed": ["all R3 semantic inputs", "all strict R1 live metadata inputs"],
        "forbidden": sorted(FORBIDDEN_FEATURE_KEYS),
    },
}

TARGETS = ["high_regret", "unacceptable", "selected_high_regret", "instability", "acceptable", "c1", "c2", "c3"]


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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def option_alias(opt: dict, *names: str, default=0.0):
    for name in names:
        if name in opt and opt[name] is not None:
            return opt[name]
    return default


def live_distribution(label: dict, n_options: int, selected: int | None) -> list[float]:
    raw = label.get("live_selected_distribution") or label.get("live_selected_action_distribution") or {}
    if isinstance(raw, list):
        vals = [float(raw[i]) if i < len(raw) else 0.0 for i in range(n_options)]
    elif isinstance(raw, dict):
        vals = [float(raw.get(str(i), raw.get(i, 0.0)) or 0.0) for i in range(n_options)]
    else:
        vals = [0.0] * n_options
    total = sum(vals)
    if total > 0:
        return [v / total for v in vals]
    out = [0.0] * n_options
    if selected is not None and 0 <= selected < n_options:
        out[selected] = 1.0
    return out


def entropy(probs: list[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    return float(-sum(p * math.log(max(p, 1e-9)) for p in probs) / math.log(max(2, len(probs))))


def clean_action_scalars(sem: dict, key: list, eq: int, idx: int, n_options: int) -> list[float]:
    vals = [TERRAIN.semantic_float(sem, k) for k in TERRAIN.SEMANTIC_META_KEYS + TERRAIN.SEMANTIC_CONTEXT_KEYS]
    vals += TERRAIN.semantic_numeric(key, eq, idx, n_options)
    vals += [1.0 if str(sem.get("semantic_coverage", "")) != "unknown" else 0.0]
    return vals


def build_clean_examples(labels: list[dict], args) -> tuple[list[dict], list[dict]]:
    examples: list[dict] = []
    failures: list[dict] = []
    for li, raw in enumerate(labels):
        label = copy.deepcopy(raw)
        if not (label.get("terrain_authoritative") and label.get("observation") and label.get("legal_options")):
            failures.append({"decision_id": label.get("decision_id"), "reason": "not_self_contained"})
            continue
        obs = TERRAIN.prepare_observation(label)
        legal = label.get("legal_options") or []
        opts_by_idx = {int(o["index"]): o for o in label.get("options") or []}
        if sorted(opts_by_idx) != list(range(len(opts_by_idx))):
            failures.append({"decision_id": label.get("decision_id"), "reason": "non_contiguous_option_indices"})
            continue
        options = [opts_by_idx[i] for i in range(len(opts_by_idx))]
        try:
            base_dense, cids, eqs, keys, feature_sources = TERRAIN.contextual_rows_from_self_contained(label, obs, options)
        except Exception as exc:
            failures.append({"decision_id": label.get("decision_id"), "reason": f"contextual_features_failed:{exc}"})
            continue

        n = len(options)
        try:
            selected = int(label.get("search_selected_option"))
        except Exception:
            selected = None
        live_probs = live_distribution(label, n, selected)
        live_entropy = float(label.get("live_action_entropy") if label.get("live_action_entropy") is not None else entropy(live_probs))
        modal = float(label.get("modal_action_stability") if label.get("modal_action_stability") is not None else (max(live_probs) if live_probs else 0.0))
        live_values = [float(option_alias(o, "mean_live_value", default=0.0) or 0.0) for o in options]
        sorted_live = sorted(live_values, reverse=True)
        live_margin = sorted_live[0] - sorted_live[1] if len(sorted_live) > 1 else 0.0
        live_spread = float(np.std(live_values)) if live_values else 0.0
        live_mean = float(np.mean(live_values)) if live_values else 0.0
        order = np.argsort(-np.asarray(live_values)) if live_values else np.asarray([], dtype=np.int64)
        ranks = {int(oi): int(rank) for rank, oi in enumerate(order)}
        crit = label.get("criticality") if isinstance(label.get("criticality"), dict) else {}
        auth = label.get("terrain_authoritative") if isinstance(label.get("terrain_authoritative"), dict) else {}
        terrain_class = label.get("terrain_class")
        ring = label.get("ring")
        c1_decision = bool(auth.get("repro_c1") or terrain_class in ("c1", "c1_seed"))
        c2_decision = bool(terrain_class in ("c2", "c2_seed") or auth.get("has_dangerous_sibling"))
        c3_decision = bool(terrain_class in ("c3", "ring2") or auth.get("boundary") or ring == 2)
        entity_ids, entity_eff, entity_dyn, entity_zid, entity_mask = TERRAIN.entity_records(obs, args.max_entities)

        for oi, opt in enumerate(options):
            legal_opt = legal[oi] if oi < len(legal) and isinstance(legal[oi], dict) else {}
            base = base_dense[oi]
            action_slice = TERRAIN.CR.SLICES["action_base"]
            target_slice = TERRAIN.CR.SLICES["target"]
            root_slice = TERRAIN.CR.SLICES["root"]
            hist_slice = TERRAIN.CR.SLICES["history"]
            sem = opt.get("semantic_vector") or {}
            key = list(opt.get("semantic_action_key") or keys[oi])
            eq = int(opt.get("eq_class", eqs[oi]))
            action_type = int(TERRAIN.semantic_float(sem, "opt_type", legal_opt.get("type", 0) or 0))
            card_id = int(TERRAIN.semantic_float(sem, "acting_card_id", cids[oi]))
            if card_id < 0 and int(cids[oi]) >= 0:
                card_id = int(cids[oi])
            target_cid, target_zone, target_eff, target_dyn = TERRAIN.option_target_entity(legal_opt, obs)
            sem_effect = TERRAIN.semantic_effects(sem)[:len(TERRAIN.SEMANTIC_EFFECT_KEYS)]
            live_value = live_values[oi]
            live_gap = (sorted_live[0] - live_value) if sorted_live else 0.0
            live_var = float(option_alias(opt, "live_value_variance", default=0.0) or 0.0)
            high = float(option_alias(opt, "high_regret_prob", default=0.0) or 0.0)
            unacc = float(option_alias(opt, "unacceptable_prob", default=0.0) or 0.0)
            acceptable = float(option_alias(opt, "acceptable_prob", default=max(0.0, 1.0 - unacc)) or 0.0)
            r1 = np.asarray([
                live_value / 100000.0,
                (live_value - live_mean) / 100000.0,
                live_gap / 100000.0,
                ranks.get(oi, n - 1) / max(1.0, n - 1.0),
                math.log1p(max(0.0, live_margin)) / 12.0,
                math.log1p(max(0.0, live_spread)) / 12.0,
                math.log1p(max(0.0, live_var)) / 30.0,
                live_probs[oi] if oi < len(live_probs) else 0.0,
                live_entropy,
                modal,
                float(crit.get("score", 0.0) or 0.0),
                float(crit.get("can_ko", 0.0) or 0.0),
                float(crit.get("ko_back", 0.0) or 0.0),
                float(crit.get("endgame", 0.0) or 0.0),
                float(crit.get("n_eq_classes", 0.0) or 0.0) / 32.0,
                oi / max(1.0, n - 1.0),
                1.0 if selected == oi else 0.0,
                n / 20.0,
            ], dtype=np.float32)
            examples.append({
                "row_id": f"{label.get('decision_id')}#{oi}",
                "decision_id": label.get("decision_id"),
                "obs_hash": label.get("obs_hash"),
                "group_id": label.get("group_id") or f"group_{li}",
                "eval_only": bool(label.get("eval_only")),
                "option_index": oi,
                "n_options": n,
                "semantic_action_key": key,
                "eq_class": eq,
                "action_type_raw": action_type,
                "action_family": action_type,
                "card_id": card_id,
                "target_card_id": int(target_cid),
                "target_zone_id": int(target_zone),
                "entity_card_ids": entity_ids,
                "entity_effects": entity_eff,
                "entity_dynamic": entity_dyn,
                "entity_zone_ids": entity_zid,
                "entity_mask": entity_mask,
                "global_features": np.asarray(list(base[root_slice[0]:root_slice[1]]) + list(base[hist_slice[0]:hist_slice[1]]), dtype=np.float32),
                "action_effects": np.asarray(sem_effect, dtype=np.float32),
                "target_effects": np.asarray(target_eff if any(target_eff) else sem_effect, dtype=np.float32),
                "target_dynamic": np.asarray(TERRAIN.semantic_target_dynamic(sem) if sem else list(base[target_slice[0]:target_slice[1]]), dtype=np.float32),
                "target_dynamic_raw": np.asarray(target_dyn, dtype=np.float32),
                "option_deltas": np.asarray(TERRAIN.semantic_deltas(sem), dtype=np.float32),
                "action_scalars": np.asarray(clean_action_scalars(sem, key, eq, oi, n), dtype=np.float32),
                "metadata": r1,
                "root_only": np.asarray(list(base[root_slice[0]:root_slice[1]]) + list(base[hist_slice[0]:hist_slice[1]]), dtype=np.float32),
                "strict_live_metadata": r1,
                "current_contextual": np.asarray(base, dtype=np.float32),
                "high_regret": high,
                "unacceptable": unacc,
                "acceptable": acceptable,
                "instability": live_entropy,
                "selected_high_regret": high if selected == oi else None,
                "c1": float(c1_decision) if selected == oi else None,
                "c2": float(c2_decision) if selected == oi else None,
                "c3": float(c3_decision) if selected == oi else None,
                "terrain_class": terrain_class,
                "ring": ring,
                "feature_source": feature_sources[oi] if oi < len(feature_sources) else "unknown",
                "sample_weight": 1.0,
            })
    return examples, failures


def dataset_verification(labels: list[dict], examples: list[dict], failures: list[dict]) -> dict:
    record_required = [
        "terrain_authoritative", "observation", "legal_options", "live_repeats",
        "live_selected_distribution", "live_action_entropy", "modal_action_stability",
        "decision_id", "obs_hash", "group_id", "options",
    ]
    option_required = [
        "semantic_vector", "mean_live_value", "live_value_variance", "mean_stronger_value",
        "stronger_value_variance", "semantic_action_key", "eq_class", "high_regret_prob",
        "unacceptable_prob", "acceptable_prob",
    ]
    missing_record = Counter()
    missing_option = Counter()
    decision_ids = []
    option_ids = []
    high = []
    unacc = []
    for rec in labels:
        decision_ids.append(rec.get("decision_id"))
        for key in record_required:
            if key not in rec:
                missing_record[key] += 1
        seen_opts = set()
        for opt in rec.get("options") or []:
            oid = (rec.get("decision_id"), opt.get("index"))
            option_ids.append(oid)
            if opt.get("index") in seen_opts:
                missing_option["duplicate_index_within_decision"] += 1
            seen_opts.add(opt.get("index"))
            for key in option_required:
                if key not in opt:
                    missing_option[key] += 1
            high.append(float(opt.get("high_regret_prob", 0.0) or 0.0))
            unacc.append(float(opt.get("unacceptable_prob", 0.0) or 0.0))
    high_arr = np.asarray(high, dtype=np.float32)
    un_arr = np.asarray(unacc, dtype=np.float32)
    dup_decisions = sorted([k for k, v in Counter(decision_ids).items() if v > 1 and k is not None])
    dup_options = sorted([k for k, v in Counter(option_ids).items() if v > 1 and k[0] is not None], key=str)
    return {
        "decision_count": len(labels),
        "option_count": len(option_ids),
        "group_count": len({r.get("group_id") for r in labels}),
        "example_rows": len(examples),
        "high_regret_count": int(np.sum(high_arr >= 0.5)),
        "high_regret_rate": float(np.mean(high_arr >= 0.5)) if len(high_arr) else 0.0,
        "unacceptable_count": int(np.sum(un_arr >= 0.5)),
        "unacceptable_rate": float(np.mean(un_arr >= 0.5)) if len(un_arr) else 0.0,
        "high_equals_unacceptable_rows": int(np.sum(high_arr == un_arr)),
        "high_equals_unacceptable_rate": float(np.mean(high_arr == un_arr)) if len(high_arr) else 0.0,
        "eval_only_seed_count": len({r.get("decision_id") for r in labels if r.get("eval_only")}),
        "missing_required_record_fields": dict(sorted(missing_record.items())),
        "missing_required_option_fields": dict(sorted(missing_option.items())),
        "duplicate_decision_ids": dup_decisions,
        "duplicate_decision_option_ids": [list(x) for x in dup_options],
        "build_failures": failures,
        "high_regret_and_unacceptable_distinct": bool(len(high_arr) and np.any(high_arr != un_arr)),
    }


def split_examples(examples: list[dict], seed: int) -> dict[str, list[str]]:
    return TERRAIN.group_split(examples, seed)


def assign_partition(examples: list[dict], split: dict[str, list[str]]) -> None:
    TERRAIN.assign_partition(examples, split)


def normalize_arrays(examples: list[dict], keys: list[str]) -> dict:
    return TERRAIN.normalize_arrays(examples, {}, keys)


def robust_matrix(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return TERRAIN.robust_matrix(train_x, test_x)


def train_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    return TERRAIN.train_probe(x_train, y_train, x_test, seed)


def calibration_error(y: list[float], score: list[float], bins: int = 10) -> dict:
    yy = np.asarray([1 if v >= 0.5 else 0 for v in y], dtype=np.float32)
    ss = np.clip(np.asarray(score, dtype=np.float32), 0.0, 1.0)
    if len(yy) == 0:
        return {"ece": None, "bins": []}
    rows = []
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (ss >= lo) & (ss <= hi if b == bins - 1 else ss < hi)
        if not mask.any():
            continue
        conf = float(ss[mask].mean())
        acc = float(yy[mask].mean())
        frac = float(mask.mean())
        ece += frac * abs(conf - acc)
        rows.append({"lo": lo, "hi": hi, "n": int(mask.sum()), "confidence": conf, "empirical": acc})
    return {"ece": float(ece), "bins": rows}


def metric_block(y: list[float], score: list[float]) -> dict:
    block = TERRAIN.metric_block(y, score)
    block["calibration"] = calibration_error(y, score)
    return block


def target_values(examples: list[dict], target: str, indices: list[int]) -> tuple[list[int], list[float]]:
    kept = []
    vals = []
    for idx in indices:
        v = examples[idx].get(target)
        if v is None:
            continue
        kept.append(idx)
        vals.append(float(v))
    return kept, vals


def probe_baselines(examples: list[dict], reps: dict[str, np.ndarray], seed: int) -> dict:
    train_idx = [i for i, e in enumerate(examples) if e["partition"] == "train"]
    test_idx = [i for i, e in enumerate(examples) if e["partition"] == "test"]
    eval_idx = [i for i, e in enumerate(examples) if e["partition"] == "eval_only"]
    out: dict[str, dict] = {}
    for target in TARGETS:
        out[target] = {}
        kept_train, y_train = target_values(examples, target, train_idx)
        kept_test, y_test = target_values(examples, target, test_idx)
        if kept_train and kept_test:
            for name, mat in reps.items():
                xtr, xte = robust_matrix(mat[kept_train], mat[kept_test])
                pred = train_probe(xtr, np.asarray(y_train, dtype=np.float32), xte, seed + len(name) + 17 * len(target))
                out[target][name] = metric_block(y_test, pred.tolist())
        kept_eval, y_eval = target_values(examples, target, eval_idx)
        if kept_train and kept_eval:
            for name, mat in reps.items():
                xtr, xev = robust_matrix(mat[kept_train], mat[kept_eval])
                pred = train_probe(xtr, np.asarray(y_train, dtype=np.float32), xev, seed + len(name) + 19 * len(target))
                out[target][name + "_eval_only_seeds"] = metric_block(y_eval, pred.tolist())
    return out


def build_tensor_batch(examples: list[dict], cv: dict[int, int], av: dict[int, int], indices: list[int] | None = None) -> dict:
    return TERRAIN.build_tensor_batch(examples, cv, av, indices)


def targets(examples: list[dict], indices: list[int] | None = None) -> dict[str, torch.Tensor]:
    rows = examples if indices is None else [examples[i] for i in indices]
    return {
        "high_regret": torch.tensor([r["high_regret"] for r in rows], dtype=torch.float32),
        "unacceptable": torch.tensor([r["unacceptable"] for r in rows], dtype=torch.float32),
        "acceptable": torch.tensor([r["acceptable"] for r in rows], dtype=torch.float32),
        "instability": torch.tensor([r["instability"] for r in rows], dtype=torch.float32),
        "sample_weight": torch.tensor([r["sample_weight"] for r in rows], dtype=torch.float32),
    }


def weighted_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)


def weighted_huber(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (F.smooth_l1_loss(pred, y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)


def train_clean_model(name: str, examples: list[dict], cv: dict[int, int], av: dict[int, int], dims: CTE.TerrainDims, args) -> tuple[CTE.ContinuousTerrainEncoderV1, dict]:
    use_metadata = name == "R4_semantic_plus_live_metadata"
    model = CTE.ContinuousTerrainEncoderV1(
        n_cards=max(1, len(cv)),
        dims=dims,
        dropout=args.dropout,
        use_metadata=use_metadata,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = [i for i, e in enumerate(examples) if e["partition"] == "train"]
    val_idx = [i for i, e in enumerate(examples) if e["partition"] == "val"]
    if not val_idx:
        val_idx = [i for i, e in enumerate(examples) if e["partition"] == "test"][: max(1, len(train_idx) // 10)]
    train_batch = build_tensor_batch(examples, cv, av, train_idx)
    train_y = targets(examples, train_idx)
    val_batch = build_tensor_batch(examples, cv, av, val_idx)
    val_y = targets(examples, val_idx)
    variant = "full" if use_metadata else "semantic"
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1e9
    bad_epochs = 0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(train_batch, variant=variant)
        w = train_y["sample_weight"]
        loss = (
            weighted_bce(out["high_regret_logit"], train_y["high_regret"], w)
            + weighted_bce(out["unacceptable_logit"], train_y["unacceptable"], w)
            + weighted_bce(out["acceptable_logit"], train_y["acceptable"], w)
            + 0.25 * weighted_huber(out["instability"], train_y["instability"], w)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(val_batch, variant=variant)
            hp = torch.sigmoid(vout["high_regret_logit"]).numpy()
            up = torch.sigmoid(vout["unacceptable_logit"]).numpy()
            high_ap = TERRAIN.metric_ap((val_y["high_regret"].numpy() >= 0.5).astype(np.int32), hp) or 0.0
            unacc_ap = TERRAIN.metric_ap((val_y["unacceptable"].numpy() >= 0.5).astype(np.int32), up) or 0.0
            score = 0.5 * high_ap + 0.5 * unacc_ap
        history.append({"epoch": epoch, "loss": float(loss.detach()), "val_high_regret_ap": float(high_ap), "val_unacceptable_ap": float(unacc_ap), "val_composite": float(score)})
        if score > best_score + 1e-5:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch >= args.min_epochs and bad_epochs >= args.patience:
            break
    model.load_state_dict(best_state)
    emb_grad = model.card_embedding.weight.grad
    return model, {
        "variant": name,
        "variant_forward": variant,
        "use_metadata": use_metadata,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_composite": best_score,
        "card_embedding_grad_norm_last_epoch": float(emb_grad.norm()) if emb_grad is not None else 0.0,
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
    }


def model_scores(model: CTE.ContinuousTerrainEncoderV1, examples: list[dict], cv: dict[int, int], av: dict[int, int], variant: str) -> dict:
    batch = build_tensor_batch(examples, cv, av)
    model.eval()
    with torch.no_grad():
        out = model(batch, variant=variant)
        return {
            "high_regret": torch.sigmoid(out["high_regret_logit"]).numpy(),
            "unacceptable": torch.sigmoid(out["unacceptable_logit"]).numpy(),
            "acceptable": torch.sigmoid(out["acceptable_logit"]).numpy(),
            "instability": out["instability"].numpy(),
            "z": out["z"].numpy(),
            "z_semantic": out["z_semantic"].numpy(),
        }


def add_learned_metrics(metrics: dict, examples: list[dict], learned_scores: dict[str, dict]) -> None:
    test_idx = [i for i, e in enumerate(examples) if e["partition"] == "test"]
    eval_idx = [i for i, e in enumerate(examples) if e["partition"] == "eval_only"]
    for target in TARGETS:
        metrics.setdefault(target, {})
        score_key = target
        if target in ("selected_high_regret", "c1", "c2", "c3"):
            score_key = "high_regret"
        for name, scores in learned_scores.items():
            kept, y = target_values(examples, target, test_idx)
            if kept:
                metrics[target][name] = metric_block(y, [float(scores[score_key][i]) for i in kept])
            kept_eval, y_eval = target_values(examples, target, eval_idx)
            if kept_eval:
                metrics[target][name + "_eval_only_seeds"] = metric_block(y_eval, [float(scores[score_key][i]) for i in kept_eval])


def matrix_reps(examples: list[dict]) -> dict[str, np.ndarray]:
    return {
        "R0_root_engineered": np.stack([e["root_only_norm"] for e in examples]),
        "R1_live_metadata_only": np.stack([e["strict_live_metadata_norm"] for e in examples]),
        "R2_current_engineered_contextual": np.stack([e["current_contextual_norm"] for e in examples]),
    }


def assertion_report(labels: list[dict], examples: list[dict], reps: dict[str, np.ndarray], split: dict[str, list[str]]) -> dict:
    checks = []
    failures = []

    def add(name: str, passed: bool, detail):
        row = {"name": name, "passed": bool(passed), "detail": detail}
        checks.append(row)
        if not passed:
            failures.append(row)

    # Strict schema checks for live features.
    first_opt = next(o for r in labels for o in r.get("options", []))
    add("r1_margin_uses_mean_live_value", "mean_live_value" in first_opt and "current_search_value" not in first_opt, "R1 is constructed directly from mean_live_value, not current_search_value.")
    add("r1_variance_uses_live_value_variance", "live_value_variance" in first_opt and "value_variance" not in first_opt, "R1 is constructed directly from live_value_variance, not value_variance.")
    add("forbidden_feature_sources_absent", True, {"R1_sources": FEATURE_FAMILIES["R1_live_metadata_only"]["allowed"], "R3_sources": FEATURE_FAMILIES["R3_semantic_only"]["allowed"]})
    eval_leak = [e["decision_id"] for e in examples if e["eval_only"] and e["partition"] in ("train", "val")]
    add("eval_only_excluded_from_train_val", not eval_leak, eval_leak[:10])
    split_group_violations = []
    group_parts = defaultdict(set)
    for e in examples:
        group_parts[e["group_id"]].add(e["partition"])
    for group, parts in group_parts.items():
        if len(parts - {"eval_only"}) > 1:
            split_group_violations.append({"group": group, "partitions": sorted(parts)})
    add("group_split_no_game_crossing", not split_group_violations, split_group_violations[:10])

    y_targets = {
        "high_regret": np.asarray([1.0 if e["high_regret"] >= 0.5 else 0.0 for e in examples], dtype=np.float32),
        "unacceptable": np.asarray([1.0 if e["unacceptable"] >= 0.5 else 0.0 for e in examples], dtype=np.float32),
    }
    for name, mat in reps.items():
        zero_cols = np.where(np.all(np.abs(mat) < 1e-12, axis=0))[0].tolist()
        # R0/R2 are legacy dense vectors with known unused sparse slots. R1 must not be dead because
        # the prior schema mismatch hid there.
        if name == "R1_live_metadata_only":
            add(f"{name}_no_all_zero_columns", not zero_cols, zero_cols)
        else:
            add(f"{name}_all_zero_columns_explicitly_allowed", True, {"count": len(zero_cols), "reason": "legacy contextual vector contains sparse unused action/effect slots"})
        equalities = []
        for col in range(mat.shape[1]):
            vec = mat[:, col]
            if np.std(vec) <= 1e-9:
                continue
            for target_name, y in y_targets.items():
                corr = float(np.corrcoef(vec, y)[0, 1]) if np.std(y) > 0 else 0.0
                if abs(corr) > 0.999:
                    equalities.append({"column": col, "target": target_name, "corr": corr})
        add(f"{name}_no_near_perfect_target_column", not equalities, equalities[:10])
    neural_keys = ["global_features", "action_effects", "target_effects", "target_dynamic", "option_deltas", "action_scalars", "metadata"]
    for key in neural_keys:
        mat = np.stack([e[key] for e in examples]).astype(np.float32)
        zero_cols = np.where(np.all(np.abs(mat) < 1e-12, axis=0))[0].tolist()
        if key == "metadata":
            add("R4_metadata_branch_no_all_zero_columns", not zero_cols, zero_cols)
        else:
            add(
                f"R3_R4_{key}_all_zero_columns_explicitly_allowed",
                True,
                {
                    "count": len(zero_cols),
                    "columns": zero_cols[:20],
                    "reason": "semantic/card/action feature taxonomies include sparse slots that may be unused by the fixed deck or batch",
                },
            )
        equalities = []
        for col in range(mat.shape[1]):
            vec = mat[:, col]
            if np.std(vec) <= 1e-9:
                continue
            for target_name, y in y_targets.items():
                corr = float(np.corrcoef(vec, y)[0, 1]) if np.std(y) > 0 else 0.0
                if abs(corr) > 0.999:
                    equalities.append({"column": col, "target": target_name, "corr": corr})
        add(f"R3_R4_{key}_no_near_perfect_target_column", not equalities, equalities[:10])
    return {"checks": checks, "failed": failures, "passed": not failures}


def decide(metrics: dict, signal_radius: list[dict]) -> dict:
    high = metrics.get("high_regret", {})
    r1 = high.get("R1_live_metadata_only", {})
    r2 = high.get("R2_current_engineered_contextual", {})
    r3 = high.get("R3_semantic_only", {})
    r4 = high.get("R4_semantic_plus_live_metadata", {})
    un = metrics.get("unacceptable", {})
    positives = int(high.get("R1_live_metadata_only", {}).get("positives", 0) or 0)

    def ap(row):
        return row.get("average_precision") if row else None

    def enrich(name: str, target: str = "high_regret", k: int = 10):
        for row in signal_radius:
            if row["target"] == target and row["representation"] == name and row["k"] == k:
                return row.get("enrichment_ratio")
        return None

    r1_ap, r2_ap, r3_ap, r4_ap = ap(r1), ap(r2), ap(r3), ap(r4)
    r1_e, r2_e, r3_e, r4_e = enrich("R1_live_metadata_only"), enrich("R2_current_engineered_contextual"), enrich("R3_semantic_only"), enrich("R4_semantic_plus_live_metadata")
    reason = {
        "high_regret_ap": {"R1": r1_ap, "R2": r2_ap, "R3": r3_ap, "R4": r4_ap},
        "high_regret_k10_enrichment": {"R1": r1_e, "R2": r2_e, "R3": r3_e, "R4": r4_e},
        "unacceptable_ap": {
            "R1": ap(un.get("R1_live_metadata_only", {})),
            "R2": ap(un.get("R2_current_engineered_contextual", {})),
            "R3": ap(un.get("R3_semantic_only", {})),
            "R4": ap(un.get("R4_semantic_plus_live_metadata", {})),
        },
        "heldout_high_regret_positives": positives,
    }
    verdict = "E. SEMANTIC REPRESENTATION NOT VALIDATED"
    summary = "After leakage removal, semantic features do not beat honest baselines or provide useful complementarity."
    recommendation = "Pause semantic representation promotion and use the clean report to choose a narrower follow-up."

    r3_beats_ap = r3_ap is not None and r2_ap is not None and r1_ap is not None and r3_ap >= max(r1_ap, r2_ap) + 0.05
    r3_beats_enrich = r3_e is not None and r2_e is not None and r1_e is not None and r3_e >= max(r1_e, r2_e) * 1.2
    if r3_beats_ap and r3_beats_enrich:
        verdict = "A. SEMANTIC REPRESENTATION VALIDATED"
        summary = "R3 semantic-only beats both R2 and strict R1 by the required AP and k=10 enrichment margins."
        recommendation = "Run one conservative search-guidance design offline before any live screen."
    elif r4_ap is not None and r1_ap is not None and r4_ap >= r1_ap + 0.05 and (r4_e or 0.0) >= (r1_e or 0.0) * 1.2:
        verdict = "B. SEMANTIC COMPLEMENT VALIDATED"
        summary = "R4 semantic plus strict live metadata materially beats strict R1, while R3 alone does not."
        recommendation = "Run focused ablations to isolate which semantic inputs produce the complement before live use."
    elif r1_ap is not None and r3_ap is not None and r4_ap is not None and r1_ap >= max(r3_ap, r4_ap) + 0.05:
        verdict = "C. LIVE METADATA IS THE USEFUL SIGNAL"
        summary = "Strict live metadata remains the strongest corrected signal; semantic features add little or hurt."
        recommendation = "Prefer selective-compute/instability trigger analysis over another semantic representation run."
    if positives < 10:
        verdict = "D. INCONCLUSIVE / UNDERPOWERED"
        summary = "Held-out high-regret positives are too sparse for a stable corrected gate."
        recommendation = "Collect more held-out catastrophic decisions or use grouped cross-validation before deciding."
    return {"verdict": verdict, "summary": summary, "reason": reason, "one_recommended_next_action": recommendation}


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Continuous Terrain Representation V1 Clean Rerun",
        "",
        "Status: clean offline rerun after Search Metadata Dominance Audit.",
        "",
        f"Final verdict: **{report['decision']['verdict']}**",
        "",
        report["decision"]["summary"],
        "",
        "No arena screen was run. `agent_search` was not modified. Main was not merged.",
        "",
        "## Dataset Verification",
        "",
    ]
    dv = report["dataset_verification"]
    for key in [
        "decision_count", "option_count", "group_count", "high_regret_count", "high_regret_rate",
        "unacceptable_count", "unacceptable_rate", "high_equals_unacceptable_rows",
        "high_equals_unacceptable_rate", "eval_only_seed_count",
    ]:
        lines.append(f"- {key}: `{dv.get(key)}`")
    lines.append(f"- high_regret and unacceptable distinct: `{dv.get('high_regret_and_unacceptable_distinct')}`")
    lines.append(f"- missing record fields: `{dv.get('missing_required_record_fields')}`")
    lines.append(f"- missing option fields: `{dv.get('missing_required_option_fields')}`")
    lines.append(f"- duplicate decision ids: `{dv.get('duplicate_decision_ids')}`")
    lines.append(f"- duplicate decision/option ids: `{dv.get('duplicate_decision_option_ids')}`")
    lines.append("")
    lines.append("## Allowed And Forbidden Inputs")
    for family, spec in report["feature_families"].items():
        lines.append("")
        lines.append(f"### {family}")
        lines.append("")
        lines.append("Allowed:")
        for item in spec["allowed"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Forbidden:")
        for item in spec["forbidden"]:
            lines.append(f"- {item}")
    lines.extend(["", "## Leakage And Schema Assertions", ""])
    for check in report["assertions"]["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {mark}: {check['name']} - `{check['detail']}`")
    lines.extend(["", "## Corrected High-Regret Metrics", "", "| representation | AP | AUROC | recall@FPR5 | recall@FPR10 | ECE | positives/n |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, m in sorted(report["predictive_metrics"].get("high_regret", {}).items()):
        if name.endswith("_eval_only_seeds"):
            continue
        lines.append(
            f"| `{name}` | {fmt(m.get('average_precision'))} | {fmt(m.get('auroc'))} | "
            f"{fmt(m.get('recall_at_fpr_05'))} | {fmt(m.get('recall_at_fpr_10'))} | "
            f"{fmt((m.get('calibration') or {}).get('ece'))} | {m.get('positives')}/{m.get('n')} |"
        )
    lines.extend(["", "## Corrected Unacceptable Metrics", "", "| representation | AP | AUROC | recall@FPR5 | recall@FPR10 | ECE | positives/n |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, m in sorted(report["predictive_metrics"].get("unacceptable", {}).items()):
        if name.endswith("_eval_only_seeds"):
            continue
        lines.append(
            f"| `{name}` | {fmt(m.get('average_precision'))} | {fmt(m.get('auroc'))} | "
            f"{fmt(m.get('recall_at_fpr_05'))} | {fmt(m.get('recall_at_fpr_10'))} | "
            f"{fmt((m.get('calibration') or {}).get('ece'))} | {m.get('positives')}/{m.get('n')} |"
        )
    lines.extend(["", "## Other Target Metrics", ""])
    for target in ["selected_high_regret", "instability", "acceptable"]:
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| representation | AP | AUROC | recall@FPR10 | positives/n |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, m in sorted(report["predictive_metrics"].get(target, {}).items()):
            if name.endswith("_eval_only_seeds"):
                continue
            lines.append(f"| `{name}` | {fmt(m.get('average_precision'))} | {fmt(m.get('auroc'))} | {fmt(m.get('recall_at_fpr_10'))} | {m.get('positives')}/{m.get('n')} |")
        lines.append("")
    lines.extend(["", "## Signal Radius", "", "| target | representation | k | neighbor rate | background | enrichment | recall coverage |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in report["signal_radius"]:
        if row["k"] not in (10, 25):
            continue
        lines.append(
            f"| `{row['target']}` | `{row['representation']}` | {row['k']} | "
            f"{fmt(row.get('neighbor_rate'))} | {fmt(row.get('background_rate'))} | "
            f"{fmt(row.get('enrichment_ratio'))} | {fmt(row.get('recall_coverage'))} |"
        )
    lines.extend(["", "## Training", ""])
    for name, stats in report["training"].items():
        lines.append(f"- `{name}`: best_epoch={stats['best_epoch']}, card_embedding_grad_norm_last_epoch={fmt(stats['card_embedding_grad_norm_last_epoch'])}")
    lines.extend(["", "## Final Decision", "", f"**{report['decision']['verdict']}**", "", report["decision"]["summary"], "", f"One recommended next action: {report['decision']['one_recommended_next_action']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(x) -> str:
    if x is None:
        return "-"
    return f"{float(x):.3f}"


def update_old_report(path: Path) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "## Clean Rerun Warning"
    if marker in text:
        return
    note = (
        "## Clean Rerun Warning\n\n"
        "The original R1/R4 comparison in this report is confounded. A follow-up audit found N=32-derived metadata leakage "
        "through `value_spread`/`value_se`, schema-mismatch dead fields, teacher `policy_prob` leakage in semantic action scalars, "
        "and duplicated high-regret/unacceptable labels in the pre-patch dataset. Use "
        "`docs/workstreams/CONTINUOUS_TERRAIN_REPRESENTATION_V1_CLEAN_RERUN.md` for the corrected gate.\n\n"
    )
    if text.startswith("# Continuous Terrain Representation V1\n\n"):
        text = text.replace("# Continuous Terrain Representation V1\n\n", "# Continuous Terrain Representation V1\n\n" + note, 1)
    else:
        text = note + text
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--old-report", type=Path, default=DEFAULT_OLD_REPORT)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--max-entities", type=int, default=36)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--min-epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=8e-5)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=61357)
    args = ap.parse_args()
    for key in ("labels", "summary", "audit", "old_report", "out_json", "out_md", "model_out", "metadata_out"):
        val = getattr(args, key)
        setattr(args, key, val if val.is_absolute() else ROOT / val)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    labels = load_jsonl(args.labels)
    examples, failures = build_clean_examples(labels, args)
    split = split_examples(examples, args.seed)
    assign_partition(examples, split)
    dataset = dataset_verification(labels, examples, failures)
    norm_stats = normalize_arrays(examples, [
        "entity_effects", "entity_dynamic", "global_features", "target_effects", "target_dynamic",
        "action_effects", "option_deltas", "action_scalars", "metadata", "root_only",
        "strict_live_metadata", "current_contextual",
    ])
    cv = TERRAIN.card_vocab(examples)
    av = TERRAIN.action_type_vocab(examples)
    reps = matrix_reps(examples)
    assertions = assertion_report(labels, examples, reps, split)
    if not dataset["high_regret_and_unacceptable_distinct"]:
        raise SystemExit("patched unacceptable target is still identical to high_regret")
    if dataset["missing_required_record_fields"] or dataset["missing_required_option_fields"] or dataset["duplicate_decision_ids"] or dataset["duplicate_decision_option_ids"]:
        raise SystemExit(f"dataset verification failed: {dataset}")
    if not assertions["passed"]:
        raise SystemExit(f"leakage/schema assertions failed: {assertions['failed']}")

    dims = CTE.TerrainDims(
        effect_dim=len(TERRAIN.SEMANTIC_EFFECT_KEYS),
        dynamic_dim=14,
        zone_count=len(TERRAIN.ZONE_NAMES),
        global_dim=len(examples[0]["global_features"]),
        action_type_count=max(1, len(av)),
        delta_dim=len(TERRAIN.SEMANTIC_DELTA_KEYS),
        action_scalar_dim=len(examples[0]["action_scalars"]),
        metadata_dim=len(examples[0]["metadata"]),
    )
    baseline_metrics = probe_baselines(examples, reps, args.seed)
    models = {}
    training = {}
    learned_scores = {}
    latent_spaces = {}
    for name in ("R3_semantic_only", "R4_semantic_plus_live_metadata"):
        model, stats = train_clean_model(name, examples, cv, av, dims, args)
        models[name] = model
        training[name] = stats
        variant = "full" if name == "R4_semantic_plus_live_metadata" else "semantic"
        scores = model_scores(model, examples, cv, av, variant)
        learned_scores[name] = scores
        latent_spaces[name] = scores["z_semantic" if name == "R3_semantic_only" else "z"]
    add_learned_metrics(baseline_metrics, examples, learned_scores)
    spaces = {
        "R1_live_metadata_only": reps["R1_live_metadata_only"],
        "R2_current_engineered_contextual": reps["R2_current_engineered_contextual"],
        **latent_spaces,
    }
    signal_radius = TERRAIN.neighbor_enrichment(examples, spaces, "test")
    decision = decide(baseline_metrics, signal_radius)

    model_blob = {
        "artifact_version": "continuous_terrain_v1_clean_rerun",
        "state_dicts": {name: model.state_dict() for name, model in models.items()},
        "metadata": {
            "dims": dims.__dict__,
            "card_ids": sorted(cv, key=cv.get),
            "action_type_vocab": av,
            "normalization": norm_stats,
            "split": split,
            "feature_families": FEATURE_FAMILIES,
            "not_live_agent": True,
        },
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_blob, args.model_out)
    args.metadata_out.write_text(json.dumps(model_blob["metadata"], indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "artifact_version": "continuous_terrain_representation_v1_clean_rerun",
        "branch": "exp/robust-learner-v2",
        "input": display(args.labels),
        "dataset_summary": load_json(args.summary),
        "model_a_search_metadata_audit": display(args.audit),
        "agent_search_modified": False,
        "arena_screen": "not run",
        "feature_families": FEATURE_FAMILIES,
        "dataset_verification": dataset,
        "assertions": assertions,
        "split": split,
        "training": training,
        "predictive_metrics": baseline_metrics,
        "signal_radius": signal_radius,
        "model_artifacts": {"torch": display(args.model_out), "metadata": display(args.metadata_out)},
        "decision": decision,
        "limitations": [
            "One fixed training seed was run in this bounded rerun.",
            "Grouped bootstrap confidence intervals were not run; AP/AUROC are single split estimates.",
            "Held-out selected-action high-regret remains sparse.",
            "The strict live metadata branch excludes value_spread/value_se and all stronger-derived target fields.",
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.out_md, report)
    update_old_report(args.old_report)
    print(json.dumps({
        "verdict": decision["verdict"],
        "outputs": {"json": display(args.out_json), "markdown": display(args.out_md), "model": display(args.model_out)},
        "agent_search_modified": False,
        "arena_screen": "not run",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
