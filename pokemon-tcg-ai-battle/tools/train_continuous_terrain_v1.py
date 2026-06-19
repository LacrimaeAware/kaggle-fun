"""Train Continuous Terrain Representation V1.

Final mode consumes Model A's `data/manifests/continuous_terrain_v1.jsonl`.
Until that artifact exists, `--smoke-test-round2` exercises the exact same
model/training/evaluation path on the old round-2 risk artifact. Smoke-test
results are diagnostic only and must not be used as the final verdict.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import continuous_terrain_encoder_v1 as CTE  # noqa: E402
import contextual_ranker as CR  # noqa: E402
import features as FT  # noqa: E402
import state_action_schema_v2 as SCH  # noqa: E402
import train_risk_only_contextual as RISK  # noqa: E402


DEFAULT_FINAL = ROOT / "data" / "manifests" / "continuous_terrain_v1.jsonl"
DEFAULT_ROUND2 = ROOT / "data" / "manifests" / "teacher_v2_residual_risk_labels_round2.jsonl"
DOCS = ROOT / "docs" / "workstreams"
ZONE_NAMES = ["hand", "active", "bench", "discard", "opp_active", "opp_bench", "opp_discard"]
STAGE_KEYS = ["basic", "stage1", "stage2"]
MODEL_VARIANTS = {
    "R3_semantic": {"use_metadata": False, "forward_variant": "semantic", "use_contrastive": True, "use_ranking": True},
    "R4_semantic_plus_search": {"use_metadata": True, "forward_variant": "full", "use_contrastive": True, "use_ranking": True},
    "R4_no_card_embedding": {"use_metadata": True, "use_card_embedding": False, "forward_variant": "full", "use_contrastive": True, "use_ranking": True},
    "R4_no_decoded_effects": {"use_metadata": True, "use_effects": False, "forward_variant": "full", "use_contrastive": True, "use_ranking": True},
    "R4_no_target_entity": {"use_metadata": True, "use_target_entity": False, "forward_variant": "full", "use_contrastive": True, "use_ranking": True},
    "R4_no_option_deltas": {"use_metadata": True, "use_deltas": False, "forward_variant": "full", "use_contrastive": True, "use_ranking": True},
    "R4_no_contrastive": {"use_metadata": True, "forward_variant": "full", "use_contrastive": False, "use_ranking": True},
    "R4_no_ranking": {"use_metadata": True, "forward_variant": "full", "use_contrastive": True, "use_ranking": False},
}
TARGETS = ["high_regret", "unacceptable", "selected_high_regret", "c1", "c2", "c3"]


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


CF = load_json(ROOT / "agent" / "card_features.json")
CE = load_json(ROOT / "agent" / "card_effects.json")


def card_effect_vector(cid: int | None) -> list[float]:
    ce = CE.get(str(cid), {}) if cid is not None and cid >= 0 else {}
    return [float(ce.get(k, 0.0) or 0.0) for k in CR.EFFECT_KEYS]


def card_dynamic_vector(ent, *, side: float = 0.0) -> list[float]:
    if isinstance(ent, dict):
        cid = ent.get("id")
        hp = float(ent.get("hp", ent.get("hp_left", 0.0)) or 0.0)
        energy = len(ent.get("energyCards") or ent.get("energies") or [])
    else:
        cid = ent if isinstance(ent, int) else None
        hp = 0.0
        energy = 0
    cf = CF.get(str(cid), {}) if cid is not None else {}
    stage = str(cf.get("stage") or "")
    return [
        hp / 300.0,
        min(1.0, energy / 6.0),
        float(cf.get("hp", 0.0) or 0.0) / 300.0,
        float(cf.get("best_dmg", 0.0) or 0.0) / 300.0,
        1.0 if cf.get("ct") == 0 else 0.0,
        1.0 if cf.get("ct") in (1, 2, 3, 4) else 0.0,
        1.0 if cf.get("ct") in (5, 6) else 0.0,
        1.0 if stage == "basic" else 0.0,
        1.0 if stage in ("stage1", "stage2") else 0.0,
        1.0 if (cf.get("ex") or cf.get("mega")) else 0.0,
        float(cf.get("retreat", 0.0) or 0.0) / 4.0,
        float(cf.get("prize", 0.0) or (2.0 if cf.get("ex") else 1.0 if cid is not None else 0.0)) / 3.0,
        float(side),
        1.0 if cid is not None else 0.0,
    ]


def slot_id(x) -> int:
    if isinstance(x, dict):
        return int(x.get("id", -1) or -1)
    try:
        return int(x)
    except Exception:
        return -1


def player(cur: dict, idx: int) -> dict:
    players = cur.get("players") or []
    return players[idx] if 0 <= idx < len(players) and isinstance(players[idx], dict) else {}


def active_list(p: dict) -> list:
    a = p.get("active") or []
    return [x for x in a if x is not None]


def entity_records(obs: dict, max_entities: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cur = obs.get("current") or {}
    me_i = int(cur.get("yourIndex", 0) or 0)
    me = player(cur, me_i)
    opp = player(cur, 1 - me_i)
    zones = [
        (0, me.get("hand") or [], 0.0),
        (1, active_list(me), 0.0),
        (2, me.get("bench") or [], 0.0),
        (3, me.get("discard") or [], 0.0),
        (4, active_list(opp), 1.0),
        (5, opp.get("bench") or [], 1.0),
        (6, opp.get("discard") or [], 1.0),
    ]
    ids, eff, dyn, zid, mask = [], [], [], [], []
    for zone_id, cards, side in zones:
        for ent in cards:
            if len(ids) >= max_entities:
                break
            cid = slot_id(ent)
            ids.append(cid)
            eff.append(card_effect_vector(cid))
            dyn.append(card_dynamic_vector(ent, side=side))
            zid.append(zone_id)
            mask.append(1.0)
    while len(ids) < max_entities:
        ids.append(-1)
        eff.append([0.0] * len(CR.EFFECT_KEYS))
        dyn.append([0.0] * 14)
        zid.append(0)
        mask.append(0.0)
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(eff, dtype=np.float32),
        np.asarray(dyn, dtype=np.float32),
        np.asarray(zid, dtype=np.int64),
        np.asarray(mask, dtype=np.float32),
    )


def option_target_entity(opt: dict, obs: dict) -> tuple[int, int, list[float], list[float]]:
    cur = obs.get("current") or {}
    me = int(cur.get("yourIndex", 0) or 0)
    t = opt.get("type")
    zone_id = 0
    ent = None
    side = 0.0
    if t == 13:
        ent = active_list(player(cur, 1 - me))[0] if active_list(player(cur, 1 - me)) else None
        zone_id = 4
        side = 1.0
    else:
        pi = opt.get("playerIndex")
        if pi is None:
            pi = me
        pi = int(pi)
        side = 1.0 if pi != me else 0.0
        p = player(cur, pi)
        area = opt.get("inPlayArea")
        if area == 4:
            ent = active_list(p)[0] if active_list(p) else None
            zone_id = 4 if side else 1
        elif area == 5:
            bench = p.get("bench") or []
            idx = opt.get("inPlayIndex")
            ent = bench[idx] if isinstance(idx, int) and 0 <= idx < len(bench) else None
            zone_id = 5 if side else 2
    cid = slot_id(ent)
    return cid, zone_id, card_effect_vector(cid), card_dynamic_vector(ent, side=side)


def semantic_numeric(key: list, eq: int, idx: int, n_options: int) -> list[float]:
    out = []
    for raw in (list(key) + [None] * 6)[:6]:
        if raw is None:
            out.extend([0.0, 1.0])
        else:
            try:
                out.extend([float(raw) / 2000.0, 0.0])
            except Exception:
                out.extend([0.0, 1.0])
    out.extend([idx / max(1.0, n_options - 1.0), eq / max(1.0, n_options - 1.0), n_options / 20.0])
    return out


def option_alias(opt: dict, *names, default=None):
    for name in names:
        if name in opt and opt[name] is not None:
            return opt[name]
    return default


def normalize_label(label: dict) -> dict:
    lab = copy.deepcopy(label)
    for opt in lab.get("options") or []:
        if "current_search_value" not in opt:
            opt["current_search_value"] = option_alias(
                opt, "mean_live_search_value", "live_search_value_mean", "live_value", default=0.0
            )
        if "stronger_value" not in opt:
            opt["stronger_value"] = option_alias(
                opt, "mean_stronger_teacher_value", "stronger_teacher_value_mean", "teacher_value", default=opt["current_search_value"]
            )
        if "delta_to_search" not in opt:
            opt["delta_to_search"] = float(opt["stronger_value"] or 0.0) - float(opt["current_search_value"] or 0.0)
        if "delta_to_search_norm" not in opt:
            opt["delta_to_search_norm"] = option_alias(opt, "normalized_residual", "clipped_normalized_delta_to_search", default=opt["delta_to_search"])
        if "value_variance" not in opt:
            opt["value_variance"] = option_alias(opt, "live_search_value_variance", "stronger_teacher_value_variance", default=0.0)
        if "value_se" not in opt:
            opt["value_se"] = option_alias(opt, "live_search_value_se", "stronger_teacher_value_se", default=0.0)
        if "completed_determinizations" not in opt:
            opt["completed_determinizations"] = option_alias(opt, "completed_determinizations_mean", default=0)
        if "high_regret_flag" not in opt:
            opt["high_regret_flag"] = int(float(option_alias(opt, "high_regret_probability", default=0.0)) >= 0.5)
        if "unacceptable_flag" not in opt:
            opt["unacceptable_flag"] = int(float(option_alias(opt, "unacceptable_probability", default=0.0)) >= 0.5)
    return lab


def build_option_examples(labels: list[dict], args) -> tuple[list[dict], list[dict]]:
    row_args = argparse.Namespace(replay_dir=args.replay_dir, high_regret_threshold=args.high_regret_threshold)
    examples = []
    failures = []
    for li, raw_label in enumerate(labels):
        label = normalize_label(raw_label)
        row, failure = RISK.make_row(label, li, li, row_args)
        if failure:
            failures.append(failure)
            continue
        obs = label.get("observation") or {}
        legal = label.get("legal_options") or (obs.get("select") or {}).get("option") or []
        opts = {int(o["index"]): o for o in label.get("options") or []}
        n = len(row["base_dense"])
        selected = row.get("search_selected_option")
        group_id = label.get("group_id") or row.get("game_file") or f"group_{li}"
        tags = set(label.get("criterion_tags") or label.get("terrain_tags") or [])
        live_dist = label.get("live_selected_action_distribution") or label.get("live_selected_distribution") or {}
        if isinstance(live_dist, list):
            live_probs = [float(x) for x in live_dist]
        elif isinstance(live_dist, dict):
            total = sum(float(v) for v in live_dist.values()) or 1.0
            live_probs = [float(live_dist.get(str(i), live_dist.get(i, 0.0))) / total for i in range(n)]
        else:
            live_probs = [0.0] * n
            if selected is not None and 0 <= int(selected) < n:
                live_probs[int(selected)] = 1.0
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in live_probs) / math.log(max(2, n))
        modal = max(live_probs) if live_probs else 0.0
        search_values = [float(opts[i].get("current_search_value") or 0.0) for i in range(n)]
        stronger_values = [float(opts[i].get("stronger_value") or 0.0) for i in range(n)]
        centered = np.asarray(stronger_values, dtype=np.float32)
        scale = float(np.std(centered)) or 1.0
        policy_logits = np.clip((centered - float(centered.mean())) / scale, -8.0, 8.0)
        expv = np.exp(policy_logits - np.max(policy_logits))
        policy_target = expv / max(1e-9, float(expv.sum()))
        sorted_values = sorted(search_values, reverse=True)
        margin = sorted_values[0] - sorted_values[1] if len(sorted_values) > 1 else 0.0
        spread = float(np.std(search_values)) if search_values else 0.0
        crit = label.get("criticality") if isinstance(label.get("criticality"), dict) else {}
        coverage = label.get("coverage") if isinstance(label.get("coverage"), dict) else {}
        entity_ids, entity_eff, entity_dyn, entity_zid, entity_mask = entity_records(obs, args.max_entities)
        for oi in range(n):
            opt = opts.get(oi, {})
            legal_opt = legal[oi] if oi < len(legal) and isinstance(legal[oi], dict) else {}
            base = row["base_dense"][oi]
            action_slice = CR.SLICES["action_base"]
            eff_slice = CR.SLICES["effects"]
            target_slice = CR.SLICES["target"]
            delta_slice = CR.SLICES["deltas"]
            root_slice = CR.SLICES["root"]
            hist_slice = CR.SLICES["history"]
            action_type = legal_opt.get("type", 0)
            target_cid, target_zone, target_eff, target_dyn = option_target_entity(legal_opt, obs)
            c1 = bool(label.get("c1_reproduced_this_label") or "c1_search_selected_high_regret" in tags)
            c2 = "c2_safe_search_false_positive" in tags
            c3 = "c3_near_miss_boundary" in tags
            high_prob = float(option_alias(opt, "high_regret_probability", default=opt.get("high_regret_flag", 0.0)) or 0.0)
            unacc_prob = float(option_alias(opt, "unacceptable_probability", default=opt.get("unacceptable_flag", 0.0)) or 0.0)
            acceptable_prob = float(option_alias(opt, "acceptable_probability", default=1.0 - unacc_prob))
            value_var = float(opt.get("value_variance") or 0.0)
            value_se = float(opt.get("value_se") or 0.0)
            residual = float(opt.get("delta_to_search_norm", opt.get("delta_to_search", 0.0)) or 0.0)
            examples.append({
                "row_id": f"{label.get('decision_id')}#{oi}",
                "decision_id": label.get("decision_id"),
                "obs_hash": label.get("obs_hash"),
                "group_id": group_id,
                "eval_only": bool(label.get("eval_only")),
                "option_index": oi,
                "n_options": n,
                "semantic_action_key": list(opt.get("semantic_action_key") or row["keys"][oi]),
                "eq_class": int(opt.get("eq_class", row["eq"][oi])),
                "action_type_raw": int(action_type or 0),
                "action_family": int(action_type or 0),
                "card_id": int(row["cids"][oi]) if int(row["cids"][oi]) >= 0 else -1,
                "target_card_id": int(target_cid),
                "target_zone_id": int(target_zone),
                "entity_card_ids": entity_ids,
                "entity_effects": entity_eff,
                "entity_dynamic": entity_dyn,
                "entity_zone_ids": entity_zid,
                "entity_mask": entity_mask,
                "global_features": np.asarray(list(base[root_slice[0]:root_slice[1]]) + list(base[hist_slice[0]:hist_slice[1]]), dtype=np.float32),
                "action_effects": np.asarray(list(base[eff_slice[0]:eff_slice[1]]), dtype=np.float32),
                "target_effects": np.asarray(target_eff, dtype=np.float32),
                "target_dynamic": np.asarray(list(base[target_slice[0]:target_slice[1]]), dtype=np.float32),
                "target_dynamic_raw": np.asarray(target_dyn, dtype=np.float32),
                "option_deltas": np.asarray(list(base[delta_slice[0]:delta_slice[1]]), dtype=np.float32),
                "action_scalars": np.asarray(
                    list(base[action_slice[0]:action_slice[1]])
                    + semantic_numeric(list(opt.get("semantic_action_key") or row["keys"][oi]), int(opt.get("eq_class", row["eq"][oi])), oi, n)
                    + [1.0 if selected == oi else 0.0, policy_target[oi], live_probs[oi] if oi < len(live_probs) else 0.0],
                    dtype=np.float32,
                ),
                "metadata": np.asarray([
                    float(search_values[oi]) / 100000.0,
                    float(search_values[oi] - np.mean(search_values)) / 100000.0 if search_values else 0.0,
                    oi / max(1.0, n - 1.0),
                    1.0 if selected == oi else 0.0,
                    math.log1p(max(0.0, margin)) / 12.0,
                    math.log1p(max(0.0, spread)) / 12.0,
                    math.log1p(max(0.0, value_var)) / 30.0,
                    value_se / 100000.0,
                    float(opt.get("completed_determinizations") or 0.0) / 32.0,
                    float(crit.get("score", 0.0) or 0.0),
                    float(crit.get("can_ko", 0.0) or 0.0),
                    float(crit.get("ko_back", 0.0) or 0.0),
                    float(crit.get("endgame", 0.0) or 0.0),
                    float(coverage.get("all_siblings_completed", 0.0) or 0.0),
                    entropy,
                    modal,
                    n / 20.0,
                ], dtype=np.float32),
                "current_contextual": np.asarray(base, dtype=np.float32),
                "root_only": np.asarray(list(base[root_slice[0]:root_slice[1]]) + list(base[hist_slice[0]:hist_slice[1]]), dtype=np.float32),
                "search_metadata_only": np.asarray([
                    math.log1p(max(0.0, margin)) / 12.0,
                    math.log1p(max(0.0, spread)) / 12.0,
                    math.log1p(max(0.0, value_var)) / 30.0,
                    value_se / 100000.0,
                    entropy,
                    modal,
                    float(crit.get("score", 0.0) or 0.0),
                    1.0 if selected == oi else 0.0,
                ], dtype=np.float32),
                "policy_target": float(policy_target[oi]),
                "high_regret": high_prob,
                "unacceptable": unacc_prob,
                "acceptable": acceptable_prob,
                "instability": entropy,
                "residual": max(-1.0, min(1.0, residual / args.residual_clip)),
                "regret": float(opt.get("regret") or 0.0),
                "selected_high_regret": float(label.get("selected_option_high_regret_flag") or 0.0) if selected == oi else None,
                "c1": float(c1) if selected == oi else None,
                "c2": float(c2) if selected == oi else None,
                "c3": float(c3) if selected == oi else None,
                "sample_weight": float(max(0.05, coverage.get("all_siblings_completed", 1.0) or 1.0)) * (
                    1.0 / max(1.0, math.sqrt(max(0.0, value_var)) / 10000.0)
                ),
            })
    return examples, failures


def group_split(examples: list[dict], seed: int) -> dict[str, list[str]]:
    groups = sorted({e["group_id"] for e in examples if not e["eval_only"]})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15))) if n >= 3 else 0
    if n_train + n_val >= n and n >= 2:
        n_train = max(1, n - 1)
        n_val = 0
    return {
        "train": sorted(groups[:n_train]),
        "val": sorted(groups[n_train:n_train + n_val]),
        "test": sorted(groups[n_train + n_val:]),
        "eval_only": sorted({e["group_id"] for e in examples if e["eval_only"]}),
    }


def assign_partition(examples: list[dict], split: dict[str, list[str]]) -> None:
    for e in examples:
        if e["eval_only"]:
            e["partition"] = "eval_only"
        elif e["group_id"] in split["train"]:
            e["partition"] = "train"
        elif e["group_id"] in split["val"]:
            e["partition"] = "val"
        elif e["group_id"] in split["test"]:
            e["partition"] = "test"
        else:
            e["partition"] = "test"


def card_vocab(examples: list[dict]) -> dict[int, int]:
    ids = sorted({
        int(v)
        for e in examples
        for v in [e["card_id"], e["target_card_id"], *[int(x) for x in e["entity_card_ids"] if int(x) >= 0]]
        if int(v) >= 0
    })
    return {cid: i for i, cid in enumerate(ids)}


def action_type_vocab(examples: list[dict]) -> dict[int, int]:
    vals = sorted({int(e["action_type_raw"]) for e in examples})
    return {v: i for i, v in enumerate(vals)}


def normalize_arrays(examples: list[dict], split: dict[str, list[str]], keys: list[str]) -> dict:
    train = [e for e in examples if e["partition"] == "train"]
    stats = {}
    for key in keys:
        arr = np.stack([e[key] for e in train]).astype(np.float32)
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        scale = q75 - q25
        std = arr.std(axis=0)
        scale = np.where(scale > 1e-6, scale, std)
        scale = np.where(scale > 1e-6, scale, 1.0)
        stats[key] = {"median": med, "scale": scale}
    for e in examples:
        for key, st in stats.items():
            e[key + "_norm"] = np.clip((e[key] - st["median"]) / st["scale"], -8.0, 8.0).astype(np.float32)
    return {
        key: {"median": stats[key]["median"].tolist(), "scale": stats[key]["scale"].tolist()}
        for key in keys
    }


def build_tensor_batch(examples: list[dict], cv: dict[int, int], av: dict[int, int], indices: list[int] | None = None) -> dict[str, torch.Tensor]:
    rows = examples if indices is None else [examples[i] for i in indices]

    def arr(key, dtype=np.float32):
        return np.stack([r[key] for r in rows]).astype(dtype)

    def card_idx(v):
        return cv.get(int(v), -1)

    return {
        "entity_card_ids": torch.tensor([[card_idx(x) for x in r["entity_card_ids"]] for r in rows], dtype=torch.long),
        "entity_effects": torch.tensor(arr("entity_effects_norm"), dtype=torch.float32),
        "entity_dynamic": torch.tensor(arr("entity_dynamic_norm"), dtype=torch.float32),
        "entity_zone_ids": torch.tensor(arr("entity_zone_ids", np.int64), dtype=torch.long),
        "entity_mask": torch.tensor(arr("entity_mask"), dtype=torch.float32),
        "global_features": torch.tensor(arr("global_features_norm"), dtype=torch.float32),
        "action_type": torch.tensor([av.get(int(r["action_type_raw"]), 0) for r in rows], dtype=torch.long),
        "action_card_id": torch.tensor([card_idx(r["card_id"]) for r in rows], dtype=torch.long),
        "target_card_id": torch.tensor([card_idx(r["target_card_id"]) for r in rows], dtype=torch.long),
        "target_zone_id": torch.tensor([int(r["target_zone_id"]) for r in rows], dtype=torch.long),
        "target_effects": torch.tensor(arr("target_effects_norm"), dtype=torch.float32),
        "target_dynamic": torch.tensor(arr("target_dynamic_norm"), dtype=torch.float32),
        "action_effects": torch.tensor(arr("action_effects_norm"), dtype=torch.float32),
        "option_deltas": torch.tensor(arr("option_deltas_norm"), dtype=torch.float32),
        "action_scalars": torch.tensor(arr("action_scalars_norm"), dtype=torch.float32),
        "metadata": torch.tensor(arr("metadata_norm"), dtype=torch.float32),
    }


def targets(examples: list[dict], indices: list[int] | None = None) -> dict[str, torch.Tensor]:
    rows = examples if indices is None else [examples[i] for i in indices]
    return {
        "high_regret": torch.tensor([r["high_regret"] for r in rows], dtype=torch.float32),
        "unacceptable": torch.tensor([r["unacceptable"] for r in rows], dtype=torch.float32),
        "acceptable": torch.tensor([r["acceptable"] for r in rows], dtype=torch.float32),
        "instability": torch.tensor([r["instability"] for r in rows], dtype=torch.float32),
        "residual": torch.tensor([r["residual"] for r in rows], dtype=torch.float32),
        "sample_weight": torch.tensor([r["sample_weight"] for r in rows], dtype=torch.float32),
        "policy_target": torch.tensor([r["policy_target"] for r in rows], dtype=torch.float32),
        "action_family": torch.tensor([r["action_family"] for r in rows], dtype=torch.long),
    }


def decision_index(examples: list[dict], partition: str) -> list[list[int]]:
    buckets = defaultdict(list)
    for i, e in enumerate(examples):
        if e["partition"] == partition:
            buckets[e["decision_id"]].append(i)
    return list(buckets.values())


def weighted_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)


def weighted_mse(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (((pred - y) ** 2) * w).sum() / w.sum().clamp_min(1.0)


def weighted_huber(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (F.smooth_l1_loss(pred, y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)


def ranking_loss(outputs: dict, y: dict, decisions: list[list[int]], row_to_pos: dict[int, int]) -> torch.Tensor:
    parts = []
    logits = outputs["policy_logit"]
    targets_t = y["policy_target"]
    for inds in decisions:
        pos = [row_to_pos[i] for i in inds if i in row_to_pos]
        if len(pos) < 2:
            continue
        p = targets_t[pos]
        p = p / p.sum().clamp_min(1e-6)
        parts.append(-(p * F.log_softmax(logits[pos], dim=0)).sum())
    if not parts:
        return logits.sum() * 0.0
    return torch.stack(parts).mean()


def metric_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int32)
    score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score))
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def metric_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int32)
    pos = int(y.sum())
    if pos == 0:
        return None
    order = np.argsort(-np.asarray(score))
    hit = 0
    total = 0.0
    for rank, idx in enumerate(order, 1):
        if y[idx]:
            hit += 1
            total += hit / rank
    return float(total / pos)


def recall_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float | None:
    y = np.asarray(y, dtype=np.int32)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(-np.asarray(score))
    tp = fp = 0
    best = 0.0
    for idx in order:
        if y[idx]:
            tp += 1
        else:
            fp += 1
        if fp / neg <= max_fpr + 1e-12:
            best = max(best, tp / pos)
    return float(best)


def metric_block(y: list[float], score: list[float]) -> dict:
    yy = np.asarray([1 if v >= 0.5 else 0 for v in y], dtype=np.int32)
    ss = np.asarray(score, dtype=np.float64)
    if len(yy) == 0:
        return {"n": 0, "positives": 0}
    return {
        "n": int(len(yy)),
        "positives": int(yy.sum()),
        "positive_rate": float(yy.mean()),
        "average_precision": metric_ap(yy, ss),
        "auroc": metric_auc(yy, ss),
        "recall_at_fpr_05": recall_at_fpr(yy, ss, 0.05),
        "recall_at_fpr_10": recall_at_fpr(yy, ss, 0.10),
        "brier": float(np.mean((ss - yy) ** 2)),
    }


def train_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    if len(np.unique(y_train >= 0.5)) < 2:
        return np.full(x_test.shape[0], float(np.mean(y_train)), dtype=np.float32)
    torch.manual_seed(seed)
    w = torch.zeros(x_train.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=0.04, weight_decay=1e-3)
    xx = torch.tensor(x_train, dtype=torch.float32)
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
    with torch.no_grad():
        pred = torch.sigmoid(torch.tensor(x_test, dtype=torch.float32) @ w + b).squeeze(-1).numpy()
    return pred


def robust_matrix(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(train_x, axis=0)
    q25 = np.percentile(train_x, 25, axis=0)
    q75 = np.percentile(train_x, 75, axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-6, scale, train_x.std(axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (
        np.clip((train_x - med) / scale, -8.0, 8.0).astype(np.float32),
        np.clip((test_x - med) / scale, -8.0, 8.0).astype(np.float32),
    )


def train_one_variant(name: str, examples: list[dict], cv: dict[int, int], av: dict[int, int], dims: CTE.TerrainDims, args) -> tuple[CTE.ContinuousTerrainEncoderV1, dict]:
    cfg = MODEL_VARIANTS[name]
    model = CTE.ContinuousTerrainEncoderV1(
        n_cards=max(1, len(cv)),
        dims=dims,
        dropout=args.dropout,
        use_card_embedding=cfg.get("use_card_embedding", True),
        use_effects=cfg.get("use_effects", True),
        use_target_entity=cfg.get("use_target_entity", True),
        use_deltas=cfg.get("use_deltas", True),
        use_metadata=cfg.get("use_metadata", True),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = [i for i, e in enumerate(examples) if e["partition"] == "train"]
    val_idx = [i for i, e in enumerate(examples) if e["partition"] == "val"]
    train_decisions = decision_index(examples, "train")
    batch = build_tensor_batch(examples, cv, av, train_idx)
    y = targets(examples, train_idx)
    group_vocab = {g: i for i, g in enumerate(sorted({examples[i]["group_id"] for i in train_idx}))}
    group_ids = torch.tensor([group_vocab[examples[i]["group_id"]] for i in train_idx], dtype=torch.long)
    profile = torch.stack([y["high_regret"], y["unacceptable"], y["instability"], (y["residual"] + 1.0) / 2.0], dim=1)
    row_to_pos = {idx: pos for pos, idx in enumerate(train_idx)}
    history = []
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(batch, variant=cfg.get("forward_variant", "full"))
        w = y["sample_weight"]
        losses = {
            "ranking": ranking_loss(out, y, train_decisions, row_to_pos) if cfg.get("use_ranking", True) else out["policy_logit"].sum() * 0.0,
            "high_regret": weighted_bce(out["high_regret_logit"], y["high_regret"], w),
            "unacceptable": weighted_bce(out["unacceptable_logit"], y["unacceptable"], w),
            "acceptable": weighted_bce(out["acceptable_logit"], y["acceptable"], w),
            "instability": weighted_mse(out["instability"], y["instability"], w),
            "residual": weighted_huber(out["residual"], y["residual"], w),
            "contrastive": CTE.supervised_contrastive_loss(out["z_semantic"], y["action_family"], profile, group_ids)
            if cfg.get("use_contrastive", True) else out["z_semantic"].sum() * 0.0,
        }
        total = sum(model.weighted_loss(k, v) for k, v in losses.items())
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        for p in model.log_sigma.values():
            p.data.clamp_(-4.0, 4.0)
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 4) == 0:
            history.append({
                "epoch": epoch + 1,
                "total_loss": float(total.detach()),
                "grad_norm": float(grad_norm),
                **{f"loss_{k}": float(v.detach()) for k, v in losses.items()},
            })
    emb_grad = model.card_embedding.weight.grad
    return model, {
        "variant_config": cfg,
        "history": history,
        "learned_task_weights": {
            k: float(torch.exp(-v.detach().clamp(-4.0, 4.0))) for k, v in model.log_sigma.items()
        },
        "log_sigma": {k: float(v.detach()) for k, v in model.log_sigma.items()},
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
            "residual": out["residual"].numpy(),
            "z": out["z"].numpy(),
            "z_semantic": out["z_semantic"].numpy(),
        }


def baseline_representations(examples: list[dict]) -> dict[str, np.ndarray]:
    return {
        "R0_root_engineered": np.stack([e["root_only_norm"] for e in examples]),
        "R1_search_metadata_only": np.stack([e["search_metadata_only_norm"] for e in examples]),
        "R2_current_engineered_contextual": np.stack([e["current_contextual_norm"] for e in examples]),
    }


def target_values(examples: list[dict], target: str, indices: list[int]) -> tuple[list[int], list[float]]:
    y = []
    kept = []
    for idx in indices:
        v = examples[idx].get(target)
        if v is None:
            continue
        kept.append(idx)
        y.append(float(v))
    return kept, y


def predictive_eval(
    examples: list[dict],
    learned: dict[str, dict],
    reps: dict[str, np.ndarray],
    args,
) -> dict:
    train_idx = [i for i, e in enumerate(examples) if e["partition"] == "train"]
    test_idx = [i for i, e in enumerate(examples) if e["partition"] == "test"]
    eval_idx = [i for i, e in enumerate(examples) if e["partition"] == "eval_only"]
    out = {}
    for target in TARGETS:
        out[target] = {}
        for name, mat in reps.items():
            kept_train, y_train = target_values(examples, target, train_idx)
            kept_test, y_test = target_values(examples, target, test_idx)
            if not kept_test or not kept_train:
                continue
            xtr, xte = robust_matrix(mat[kept_train], mat[kept_test])
            pred = train_probe(xtr, np.asarray(y_train), xte, args.seed + len(name) + len(target))
            out[target][name] = metric_block(y_test, pred.tolist())
        for name, scores in learned.items():
            score_key = "high_regret" if target in ("high_regret", "selected_high_regret", "c1", "c2", "c3") else target
            if target == "acceptable":
                score_key = "acceptable"
            if target == "unacceptable":
                score_key = "unacceptable"
            kept_test, y_test = target_values(examples, target, test_idx)
            if kept_test:
                out[target][name] = metric_block(y_test, [float(scores[score_key][i]) for i in kept_test])
            kept_eval, y_eval = target_values(examples, target, eval_idx)
            if kept_eval:
                out[target][name + "_eval_only_seeds"] = metric_block(y_eval, [float(scores[score_key][i]) for i in kept_eval])
    return out


def neighbor_enrichment(examples: list[dict], spaces: dict[str, np.ndarray], partition: str = "test") -> list[dict]:
    rows = []
    idx_all = [i for i, e in enumerate(examples) if e["partition"] == partition]
    if len(idx_all) < 3:
        return rows
    for target in TARGETS:
        valid = [i for i in idx_all if examples[i].get(target) is not None]
        positives = [i for i in valid if float(examples[i][target]) >= 0.5]
        if not positives:
            continue
        for name, mat in spaces.items():
            x = mat[valid]
            med = np.median(x, axis=0)
            scale = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
            scale = np.where(scale > 1e-6, scale, x.std(axis=0))
            scale = np.where(scale > 1e-6, scale, 1.0)
            z = np.clip((x - med) / scale, -8.0, 8.0)
            pos_to_local = {idx: valid.index(idx) for idx in positives}
            for k in [5, 10, 25, 50]:
                labels = []
                bg_rates = []
                retrieved = set()
                q_count = 0
                for qi in positives:
                    qgroup = examples[qi]["group_id"]
                    cand = [ci for ci in valid if ci != qi and examples[ci]["group_id"] != qgroup]
                    if not cand:
                        continue
                    q_count += 1
                    cand_local = [valid.index(ci) for ci in cand]
                    d = np.linalg.norm(z[cand_local] - z[pos_to_local[qi]], axis=1)
                    order = np.argsort(d)[:min(k, len(cand))]
                    chosen = [cand[int(i)] for i in order]
                    labs = [1 if float(examples[i][target]) >= 0.5 else 0 for i in chosen]
                    labels.extend(labs)
                    bg_rates.append(sum(1 if float(examples[i][target]) >= 0.5 else 0 for i in cand) / max(1, len(cand)))
                    for ci, lab in zip(chosen, labs):
                        if lab:
                            retrieved.add(ci)
                bg = sum(bg_rates) / len(bg_rates) if bg_rates else None
                nr = sum(labels) / len(labels) if labels else None
                rows.append({
                    "partition": partition,
                    "target": target,
                    "representation": name,
                    "k": k,
                    "query_count": q_count,
                    "background_rate": bg,
                    "neighbor_rate": nr,
                    "enrichment_ratio": (nr / bg) if bg and nr is not None else None,
                    "recall_coverage": len(retrieved) / max(1, len(positives)),
                })
    return rows


def summarize_decision(predictive: dict, neighborhood: list[dict], smoke: bool) -> dict:
    if smoke:
        return {
            "verdict": "D. CURRENT DATA UNDERPOWERED",
            "note": (
                "Smoke-test only: final verdict requires Model A continuous_terrain_v1.jsonl. "
                "The round-2 artifact is too small and c1-clustered for the required gate."
            ),
        }
    high = predictive.get("high_regret", {})
    sem = high.get("R3_semantic", {}).get("average_precision")
    search = high.get("R1_search_metadata_only", {}).get("average_precision")
    engineered = high.get("R2_current_engineered_contextual", {}).get("average_precision")
    full = high.get("R4_semantic_plus_search", {}).get("average_precision")
    sem_nn = max([
        r.get("enrichment_ratio") or 0.0 for r in neighborhood
        if r["target"] == "high_regret" and r["representation"] == "R3_semantic" and r["k"] == 10
    ] or [0.0])
    eng_nn = max([
        r.get("enrichment_ratio") or 0.0 for r in neighborhood
        if r["target"] == "high_regret" and r["representation"] == "R2_current_engineered_contextual" and r["k"] == 10
    ] or [0.0])
    if sem is not None and engineered is not None and search is not None and sem >= max(engineered, search) + 0.05 and sem_nn >= 1.2 * max(eng_nn, 1e-9):
        return {"verdict": "A. BROADER STRUCTURE SUPPORTED"}
    if full is not None and search is not None and full > search + 0.03 and (sem is None or sem <= search + 0.03):
        return {"verdict": "C. SEARCH-METADATA-DOMINATED"}
    if sem is not None and engineered is not None and sem > engineered + 0.02:
        return {"verdict": "B. FEATURE-LIMITED BUT IMPROVED"}
    return {"verdict": "E. CURRENT SEMANTIC REPRESENTATION NOT VALIDATED"}


def write_summary(path: Path, report: dict) -> None:
    lines = [
        "# Continuous Terrain Representation V1",
        "",
        f"Status: {'round-2 smoke test only' if report['mode'] == 'smoke_round2' else 'final terrain experiment'}.",
        "",
        "No live agent was modified and no arena screen was run.",
        "",
        "## Dataset",
        "",
        f"- Input: `{report['input']}`",
        f"- Decisions/options/games: {report['dataset']['decisions']} / {report['dataset']['options']} / {report['dataset']['games']}",
        f"- Eval-only decisions: {report['dataset']['eval_only_decisions']}",
        f"- Failures: {len(report['dataset']['failures'])}",
        f"- Split: train games {report['split']['train']}; val games {report['split']['val']}; test games {report['split']['test']}; eval-only {report['split']['eval_only']}",
        "",
        "## Architecture",
        "",
        "- Learned card-id embedding dimension: 32.",
        "- Effect vector MLP, dynamic entity MLP, DeepSets zone pooling, 128-d state encoder.",
        "- Action encoder combines action type, acting card embedding, target entity, decoded effects, option deltas, and action scalars.",
        "- Semantic latent is separated from the search-metadata branch; R3 uses semantic only, R4 adds metadata.",
        "- Homoscedastic task weights are learned for ranking, risk, acceptability, instability, residual, and contrastive losses.",
        "",
        "## Trainable Embedding Check",
        "",
    ]
    for name, stats in report["training"].items():
        lines.append(f"- `{name}` card embedding grad norm, last epoch: {stats['card_embedding_grad_norm_last_epoch']:.6f}")
    lines.extend(["", "## Learned Task Weights", "", "| variant | ranking | high_regret | unacceptable | acceptable | instability | residual | contrastive |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, stats in report["training"].items():
        w = stats["learned_task_weights"]
        lines.append(
            f"| `{name}` | {w['ranking']:.3f} | {w['high_regret']:.3f} | {w['unacceptable']:.3f} | "
            f"{w['acceptable']:.3f} | {w['instability']:.3f} | {w['residual']:.3f} | {w['contrastive']:.3f} |"
        )
    lines.extend(["", "## Predictive Metrics", "", "High-regret AP/AUROC on held-out test games:", "", "| representation | AP | AUROC | recall@FPR10 |", "|---|---:|---:|---:|"])
    for name, metrics in sorted((report["predictive_metrics"].get("high_regret") or {}).items()):
        if name.endswith("_eval_only_seeds"):
            continue
        lines.append(
            f"| `{name}` | {fmt(metrics.get('average_precision'))} | {fmt(metrics.get('auroc'))} | {fmt(metrics.get('recall_at_fpr_10'))} |"
        )
    lines.extend(["", "## Signal Radius", "", "k=10 held-out-game high-regret enrichment:", "", "| representation | bg rate | neighbor rate | enrich | queries |", "|---|---:|---:|---:|---:|"])
    for row in report["signal_radius"]:
        if row["target"] == "high_regret" and row["k"] == 10:
            lines.append(
                f"| `{row['representation']}` | {fmt(row.get('background_rate'))} | {fmt(row.get('neighbor_rate'))} | "
                f"{fmt(row.get('enrichment_ratio'))} | {row['query_count']} |"
            )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Verdict", "", f"**{report['decision']['verdict']}**", ""])
    if report["decision"].get("note"):
        lines.append(report["decision"]["note"])
    lines.extend(["", f"One next experiment: {report['one_next_experiment']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(x) -> str:
    if x is None:
        return "-"
    return f"{float(x):.3f}"


def save_model(path: Path, model: CTE.ContinuousTerrainEncoderV1, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_FINAL)
    ap.add_argument("--smoke-test-round2", action="store_true")
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "data" / "external" / "replays")
    ap.add_argument("--eval-out", type=Path, default=DOCS / "continuous_terrain_representation_v1_eval.json")
    ap.add_argument("--summary-out", type=Path, default=DOCS / "CONTINUOUS_TERRAIN_REPRESENTATION_V1.md")
    ap.add_argument("--model-out", type=Path, default=ROOT / "agent" / "continuous_terrain_encoder_v1.pt")
    ap.add_argument("--metadata-out", type=Path, default=ROOT / "agent" / "continuous_terrain_encoder_v1.json")
    ap.add_argument("--max-entities", type=int, default=36)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=8e-5)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=61357)
    ap.add_argument("--high-regret-threshold", type=float, default=5000.0)
    ap.add_argument("--residual-clip", type=float, default=50000.0)
    args = ap.parse_args()

    for key in ("labels", "replay_dir", "eval_out", "summary_out", "model_out", "metadata_out"):
        val = getattr(args, key)
        setattr(args, key, val if val.is_absolute() else ROOT / val)
    if args.smoke_test_round2:
        args.labels = DEFAULT_ROUND2
        args.model_out = ROOT / "agent" / "continuous_terrain_encoder_v1_smoke.pt"
        args.metadata_out = ROOT / "agent" / "continuous_terrain_encoder_v1_smoke.json"
    elif not args.labels.exists():
        raise SystemExit(
            f"Missing final terrain artifact {display(args.labels)}. Use --smoke-test-round2 only for non-final pipeline testing."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels = load_jsonl(args.labels)
    examples, failures = build_option_examples(labels, args)
    split = group_split(examples, args.seed)
    assign_partition(examples, split)
    cv = card_vocab(examples)
    av = action_type_vocab(examples)
    norm_stats = normalize_arrays(examples, split, [
        "entity_effects", "entity_dynamic", "global_features", "target_effects", "target_dynamic",
        "action_effects", "option_deltas", "action_scalars", "metadata", "root_only",
        "search_metadata_only", "current_contextual",
    ])
    dims = CTE.TerrainDims(
        effect_dim=len(CR.EFFECT_KEYS),
        dynamic_dim=14,
        zone_count=len(ZONE_NAMES),
        global_dim=len(examples[0]["global_features"]),
        action_type_count=max(1, len(av)),
        delta_dim=len(CR.DELTA_KEYS),
        action_scalar_dim=len(examples[0]["action_scalars"]),
        metadata_dim=len(examples[0]["metadata"]),
    )
    training = {}
    learned_scores = {}
    latent_spaces = {}
    models = {}
    for variant in MODEL_VARIANTS:
        model, stats = train_one_variant(variant, examples, cv, av, dims, args)
        training[variant] = stats
        models[variant] = model
        scores = model_scores(model, examples, cv, av, MODEL_VARIANTS[variant].get("forward_variant", "full"))
        learned_scores[variant] = scores
        latent_spaces[variant] = scores["z_semantic" if variant == "R3_semantic" else "z"]
    baseline_spaces = baseline_representations(examples)
    spaces = {**baseline_spaces, **latent_spaces}
    predictive = predictive_eval(examples, learned_scores, baseline_spaces, args)
    # Add learned representation head metrics under their variant names.
    for target in TARGETS:
        predictive.setdefault(target, {})
        for variant, scores in learned_scores.items():
            score_key = "high_regret" if target in ("high_regret", "selected_high_regret", "c1", "c2", "c3") else target
            test_idx = [i for i, e in enumerate(examples) if e["partition"] == "test" and e.get(target) is not None]
            if test_idx:
                predictive[target][variant] = metric_block(
                    [examples[i][target] for i in test_idx],
                    [float(scores[score_key][i]) for i in test_idx],
                )
            eval_idx = [i for i, e in enumerate(examples) if e["partition"] == "eval_only" and e.get(target) is not None]
            if eval_idx:
                predictive[target][variant + "_eval_only_seeds"] = metric_block(
                    [examples[i][target] for i in eval_idx],
                    [float(scores[score_key][i]) for i in eval_idx],
                )
    signal_radius = neighbor_enrichment(examples, spaces, "test")
    decision = summarize_decision(predictive, signal_radius, args.smoke_test_round2)
    final_model = models["R4_semantic_plus_search"]
    metadata = {
        "artifact_version": "continuous_terrain_encoder_v1",
        "mode": "smoke_round2" if args.smoke_test_round2 else "final",
        "dims": dims.__dict__,
        "card_ids": sorted(cv, key=cv.get),
        "action_type_vocab": av,
        "normalization": norm_stats,
        "split": split,
        "not_live_agent": True,
    }
    save_model(args.model_out, final_model, metadata)
    args.metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    dataset = {
        "decisions": len({e["decision_id"] for e in examples}),
        "options": len(examples),
        "games": len({e["group_id"] for e in examples}),
        "eval_only_decisions": len({e["decision_id"] for e in examples if e["eval_only"]}),
        "class_balance": {
            "high_regret_options": sum(1 for e in examples if e["high_regret"] >= 0.5),
            "unacceptable_options": sum(1 for e in examples if e["unacceptable"] >= 0.5),
            "c1_selected_rows": sum(1 for e in examples if e.get("c1") is not None and e["c1"] >= 0.5),
            "c2_selected_rows": sum(1 for e in examples if e.get("c2") is not None and e["c2"] >= 0.5),
            "c3_selected_rows": sum(1 for e in examples if e.get("c3") is not None and e["c3"] >= 0.5),
        },
        "failures": failures,
    }
    limitations = [
        "Round-2 smoke-test data is not the expanded continuous terrain dataset.",
        "Only the final A artifact may be used for the true representation verdict.",
        "The old round-2 c1 class remains sparse and game-clustered.",
        "Smoke mode reconstructs some features from replays; final A data is expected to be self-contained.",
    ] if args.smoke_test_round2 else [
        "No live agent was built unless the representation gate passes in a later explicit step.",
    ]
    report = {
        "artifact_version": "continuous_terrain_representation_v1",
        "branch": "exp/robust-learner-v2",
        "mode": "smoke_round2" if args.smoke_test_round2 else "final",
        "input": display(args.labels),
        "agent_search_modified": False,
        "arena_screen": "not run",
        "dataset": dataset,
        "split": split,
        "architecture": {
            "card_embedding_dim": 32,
            "state_embedding_dim": 128,
            "action_embedding_dim": 64,
            "semantic_latent_dim": 64,
            "search_metadata_separate_branch": True,
            "homoscedastic_task_weighting": True,
        },
        "training": training,
        "predictive_metrics": predictive,
        "signal_radius": signal_radius,
        "model_artifacts": {
            "torch": display(args.model_out),
            "metadata": display(args.metadata_out),
        },
        "limitations": limitations,
        "decision": decision,
        "one_next_experiment": (
            "Run this same pipeline on Model A's expanded continuous_terrain_v1.jsonl once available."
            if args.smoke_test_round2 else
            "If the gate passes, build one conservative search-guidance candidate; otherwise request only the missing terrain slice identified by this report."
        ),
    }
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(args.summary_out, report)
    print(json.dumps({
        "mode": report["mode"],
        "dataset": dataset,
        "decision": decision,
        "outputs": {
            "eval": display(args.eval_out),
            "summary": display(args.summary_out),
            "model": display(args.model_out),
            "metadata": display(args.metadata_out),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
