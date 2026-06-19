"""Branch B / Contextual Action Ranker V1.

Train an integrated sibling-action model from full root states and candidate
actions, then evaluate it against Teacher V1 soft policies/advantages and the
old ranker baselines. This is deliberately not another global state-value
model and not another hard argmax imitation run.

The train/serve feature path is agent/contextual_ranker.py:

    root state + action descriptor + acting card embedding + decoded effects
    + target/entity features + one-step option deltas + state/effect
    interactions + short public history -> grouped sibling logits

Recovery data is recollected from the Round 1 and Round 2 deployed ranker
artifacts because the prior on-policy JSON reports did not retain raw
observations.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

logging.disable(logging.CRITICAL)

import contextual_ranker as CR  # noqa: E402
import main as M  # noqa: E402
import ranker as OLD_RANKER  # noqa: E402
import state_action_schema_v2 as SCH  # noqa: E402
import teacher_api_v1 as T  # noqa: E402

ARENA = None


def arena_module():
    global ARENA
    if ARENA is None:
        import cabt_arena as arena  # noqa: WPS433
        ARENA = arena
    return ARENA

DEFAULT_SPLIT = ROOT / "data" / "splits" / "replays_20260618_split.json"
DEFAULT_REPLAY_DIR = ROOT / "data" / "external" / "replays"
DOCS = ROOT / "docs" / "workstreams"


def canonical_json(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def obs_hash(obs: dict) -> str:
    payload = {"current": obs.get("current"), "select": obs.get("select")}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def deck_hash(deck: list[int] | None) -> str | None:
    if not deck:
        return None
    return hashlib.sha1(",".join(str(x) for x in deck).encode("utf-8")).hexdigest()[:16]


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def player_deck(game: dict, player: int) -> list[int] | None:
    for step in game.get("steps", []):
        if player < len(step) and isinstance(step[player], dict):
            action = step[player].get("action")
            if isinstance(action, list) and len(action) == 60:
                return [int(x) for x in action]
    return None


def player_name(game: dict, player: int) -> str:
    agents = game.get("info", {}).get("Agents", [])
    if player < len(agents) and isinstance(agents[player], dict):
        return str(agents[player].get("Name") or f"seat{player}")
    return f"seat{player}"


def split_names(split_path: Path, partition: str) -> list[str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return list(split.get(partition) or [])


def option_eq_from_move(feat: dict, move) -> int | None:
    if not isinstance(move, list) or len(move) != 1:
        return None
    i = move[0]
    if not isinstance(i, int) or i < 0 or i >= len(feat["eq"]):
        return None
    return int(feat["eq"][i])


def class_ids(eqs: list[int]) -> list[int]:
    out = []
    for e in eqs:
        e = int(e)
        if e not in out:
            out.append(e)
    return out


def old_ranker_eq(obs: dict, deck: list[int], feat: dict) -> int | None:
    """Old-ranker baseline under agent/ranker_model.json."""
    old_env = os.environ.get("CABT_RANKER_MODEL")
    try:
        os.environ["CABT_RANKER_MODEL"] = "ranker_model.json"
        OLD_RANKER._MODEL = None
        OLD_RANKER._MODEL_NAME = None
        mv = OLD_RANKER.predict(obs, deck)
        return option_eq_from_move(feat, mv)
    except Exception:
        return None
    finally:
        if old_env is None:
            os.environ.pop("CABT_RANKER_MODEL", None)
        else:
            os.environ["CABT_RANKER_MODEL"] = old_env
        OLD_RANKER._MODEL = None
        OLD_RANKER._MODEL_NAME = None


def teacher_targets(obs: dict, deck: list[int], feat: dict, args, seed_base: int) -> dict | None:
    """Aggregate repeated Teacher V1 calls onto this script's eq classes."""
    key_to_eq = {tuple(k): int(e) for k, e in zip(feat["keys"], feat["eq"])}
    per = []
    label_counts = Counter()
    not_applicable = 0
    for r in range(args.teacher_repeats):
        result = T.query(
            obs,
            deck,
            n_determ=args.n_determ,
            time_budget=args.time_budget,
            leaf_mode="hand",
            seed=seed_base + r,
            accept_z=args.accept_z,
        )
        if not result.get("applicable"):
            not_applicable += 1
            continue
        chosen = result.get("chosen_option")
        opts = result.get("options") or []
        if chosen is not None and chosen < len(opts):
            label_counts[tuple(opts[chosen]["semantic_action_key"])] += 1
        per.append(result)

    if not per:
        return None

    soft_acc = defaultdict(list)
    adv_acc = defaultdict(list)
    acceptable_acc = defaultdict(list)
    variance_acc = defaultdict(list)
    completed_acc = defaultdict(list)
    margins = []
    for result in per:
        soft = result.get("soft_policy_target") or {}
        values = [
            float(c["mean_value"])
            for c in (result.get("eq_classes") or [])
            if c.get("mean_value") is not None
        ]
        if not values:
            continue
        mean_v = sum(values) / len(values)
        acceptable = set(result.get("acceptable_action_set") or [])
        if result.get("forced_action_flag") and result.get("forced_eq_key") is not None:
            fk = tuple(result["forced_eq_key"])
            if fk in key_to_eq:
                acceptable.add(key_to_eq[fk])
        if result.get("top_two_margin") is not None:
            margins.append(float(result["top_two_margin"]))
        for cls in result.get("eq_classes") or []:
            k = tuple(cls.get("key") or ())
            if k not in key_to_eq or cls.get("mean_value") is None:
                continue
            eq = key_to_eq[k]
            teq = cls["eq_class"]
            soft_acc[eq].append(float(soft.get(teq, soft.get(str(teq), 0.0)) or 0.0))
            adv_acc[eq].append(float(cls["mean_value"]) - mean_v)
            acceptable_acc[eq].append(1.0 if teq in acceptable or eq in acceptable else 0.0)
            variance_acc[eq].append(float(cls.get("value_variance") or 0.0))
            completed_acc[eq].append(float(cls.get("completed_determinizations") or 0.0))

    if not soft_acc:
        return None

    soft = {int(k): sum(v) / len(v) for k, v in soft_acc.items()}
    z = sum(soft.values())
    if z <= 0:
        return None
    soft = {k: v / z for k, v in soft.items()}
    adv = {int(k): sum(v) / len(v) for k, v in adv_acc.items()}
    acceptable = {int(k): sum(v) / len(v) for k, v in acceptable_acc.items()}
    variance = {int(k): sum(v) / len(v) for k, v in variance_acc.items()}
    completed = {int(k): sum(v) / len(v) for k, v in completed_acc.items()}
    agreement = label_counts.most_common(1)[0][1] / len(per) if label_counts else 0.0
    stability = "stable" if agreement >= args.min_teacher_agreement else "unstable"
    confidence = agreement if stability == "stable" else max(0.15, 0.35 * agreement)
    return {
        "soft": soft,
        "adv": adv,
        "acceptable": acceptable,
        "value_variance": variance,
        "completed_determinizations": completed,
        "teacher_label_agreement": agreement,
        "teacher_stability": stability,
        "teacher_confidence": confidence,
        "teacher_applicable_repeats": len(per),
        "teacher_not_applicable_repeats": not_applicable,
        "top_two_margin": sum(margins) / len(margins) if margins else None,
    }


def finalize_decision(obs: dict, deck: list[int], source: str, partition: str, args,
                      seed_base: int, meta: dict, chosen_move=None, student_move=None) -> dict | None:
    if M._forced_move(obs) is not None:
        return None
    feat = CR.decision_features(obs, deck)
    if feat is None:
        return None
    target = teacher_targets(obs, deck, feat, args, seed_base)
    if target is None:
        return None
    chosen_eq = option_eq_from_move(feat, chosen_move)
    student_eq = option_eq_from_move(feat, student_move)
    adv = target["adv"]
    best_adv = max(adv.values()) if adv else 0.0
    student_regret = None
    high_regret = False
    if student_eq is not None and student_eq in adv:
        student_regret = best_adv - float(adv[student_eq])
        high_regret = student_regret >= args.high_regret_threshold
    base_weight = args.replay_weight if source.startswith("replay") else args.recovery_weight
    if target["teacher_stability"] == "unstable":
        base_weight *= args.unstable_weight_scale
    if high_regret:
        base_weight *= args.high_regret_weight_scale
    old_eq = old_ranker_eq(obs, deck, feat) if args.old_ranker_baseline else None
    rec = {
        "source": source,
        "partition": partition,
        "obs_hash": obs_hash(obs),
        "deck_hash": deck_hash(deck),
        "player": meta.get("player"),
        "game_file": meta.get("game_file"),
        "game": meta.get("game"),
        "call": meta.get("call"),
        "turn": (obs.get("current") or {}).get("turn"),
        "turn_action_count": (obs.get("current") or {}).get("turnActionCount"),
        "cids": [int(x) for x in feat["cids"]],
        "dense": feat["dense"],
        "eq": [int(x) for x in feat["eq"]],
        "keys": feat["keys"],
        "soft": {str(k): float(v) for k, v in target["soft"].items()},
        "adv": {str(k): float(v) for k, v in target["adv"].items()},
        "acceptable": {str(k): float(v) for k, v in target["acceptable"].items()},
        "teacher_label_agreement": float(target["teacher_label_agreement"]),
        "teacher_stability": target["teacher_stability"],
        "teacher_confidence": float(target["teacher_confidence"]),
        "teacher_applicable_repeats": int(target["teacher_applicable_repeats"]),
        "teacher_not_applicable_repeats": int(target["teacher_not_applicable_repeats"]),
        "top_two_margin": target["top_two_margin"],
        "value_variance_mean": (
            sum(target["value_variance"].values()) / len(target["value_variance"])
            if target["value_variance"] else None
        ),
        "completed_determinizations_mean": (
            sum(target["completed_determinizations"].values()) / len(target["completed_determinizations"])
            if target["completed_determinizations"] else None
        ),
        "chosen_eq": chosen_eq,
        "student_eq": student_eq,
        "student_regret": student_regret,
        "high_regret": high_regret,
        "old_ranker_eq": old_eq,
        "option0_eq": int(feat["eq"][0]),
        "weight": float(base_weight * max(0.1, target["teacher_confidence"])),
    }
    return rec


def collect_replay_decisions(partition: str, limit: int, args, seed_offset: int) -> tuple[list[dict], dict]:
    out = []
    stats = Counter()
    for file_i, name in enumerate(split_names(args.split, partition)):
        if args.max_replay_files and file_i >= args.max_replay_files:
            break
        path = args.replay_dir / name
        if not path.exists():
            stats["missing_file"] += 1
            continue
        try:
            game = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            stats["bad_json"] += 1
            continue
        rewards = game.get("rewards") or []
        if len(rewards) != 2 or rewards[0] == rewards[1] or None in rewards:
            stats["no_winner"] += 1
            continue
        winner = 0 if rewards[0] > rewards[1] else 1
        deck = player_deck(game, winner)
        if not deck:
            stats["no_deck"] += 1
            continue
        for step_i, step in enumerate(game.get("steps", [])):
            if winner >= len(step) or not isinstance(step[winner], dict):
                continue
            rec = step[winner]
            obs = rec.get("observation") or {}
            action = rec.get("action")
            if not SCH.is_single_pick_decision(obs) or not isinstance(action, list) or len(action) != 1:
                continue
            stats["candidate"] += 1
            meta = {
                "player": player_name(game, winner),
                "game_file": name,
                "step": step_i,
            }
            d = finalize_decision(
                obs,
                deck,
                source=f"replay_{partition}",
                partition=partition,
                args=args,
                seed_base=args.seed + seed_offset + stats["candidate"] * args.teacher_repeats,
                meta=meta,
                chosen_move=action,
            )
            if d is None:
                stats["skipped"] += 1
                continue
            out.append(d)
            stats["kept"] += 1
            if args.progress and len(out) % args.progress == 0:
                print(f"  replay {partition}: kept {len(out)}/{limit} from {stats['candidate']} candidates", flush=True)
            if limit and len(out) >= limit:
                return out, dict(stats)
    return out, dict(stats)


def normalize_obs(obs: dict) -> dict:
    inner = obs.get("observation") if isinstance(obs, dict) else None
    return inner if isinstance(inner, dict) else obs


def make_traced_agent(name: str, fn, records: list[dict], game_id: int, seat: int):
    call_i = 0

    def traced_agent(obs: dict):
        nonlocal call_i
        raw_obs = normalize_obs(obs)
        action = fn(raw_obs)
        call_i += 1
        try:
            if SCH.is_single_pick_decision(raw_obs) and isinstance(action, list) and len(action) == 1:
                records.append({
                    "game": game_id,
                    "seat": seat,
                    "call": call_i,
                    "action": list(action),
                    "obs": copy.deepcopy(raw_obs),
                })
        except Exception:
            pass
        return action

    traced_agent.__name__ = f"traced_{name}"
    return traced_agent


def run_traced_rank_games(model_name: str, games: int, progress: int) -> tuple[list[dict], dict]:
    arena = arena_module()
    old_env = os.environ.get("CABT_RANKER_MODEL")
    os.environ["CABT_RANKER_MODEL"] = model_name
    OLD_RANKER._MODEL = None
    OLD_RANKER._MODEL_NAME = None
    decisions = []
    wins_student = wins_opp = draws = errors = 0
    t0 = time.time()
    try:
        for g in range(games):
            student_seat = g % 2
            game_records = []
            traced = make_traced_agent("rank", arena.AGENTS["rank"], game_records, g, student_seat)
            agents = [traced, arena.AGENTS["heuristic"]] if student_seat == 0 else [arena.AGENTS["heuristic"], traced]
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    env = arena.make("cabt")
                    env.run(agents)
                winner = arena.winner_of(env)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  recovery {model_name} ERROR game {g + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            decisions.extend(game_records)
            if winner is None:
                draws += 1
            elif winner == student_seat:
                wins_student += 1
            else:
                wins_opp += 1
            if progress and ((g + 1) % progress == 0 or g + 1 == games):
                print(f"  recovery {model_name}: {g + 1}/{games} games | decisions={len(decisions)}", flush=True)
    finally:
        if old_env is None:
            os.environ.pop("CABT_RANKER_MODEL", None)
        else:
            os.environ["CABT_RANKER_MODEL"] = old_env
        OLD_RANKER._MODEL = None
        OLD_RANKER._MODEL_NAME = None
    return decisions, {
        "model": model_name,
        "games": games,
        "wins_student": wins_student,
        "wins_opponent": wins_opp,
        "draws": draws,
        "errors": errors,
        "raw_decisions": len(decisions),
        "seconds": time.time() - t0,
    }


def collect_recovery_decisions(model_name: str, source: str, games: int, limit: int,
                               args, seed_offset: int) -> tuple[list[dict], dict]:
    if games <= 0 or limit == 0:
        return [], {"model": model_name, "games": games, "raw_decisions": 0, "label_kept": 0}
    raw, summary = run_traced_rank_games(model_name, games, args.progress)
    out = []
    stats = Counter()
    for i, d in enumerate(raw):
        if limit and len(out) >= limit:
            break
        obs = d["obs"]
        stats["candidate"] += 1
        rec = finalize_decision(
            obs,
            list(M.DECK),
            source=source,
            partition="recovery",
            args=args,
            seed_base=args.seed + seed_offset + i * args.teacher_repeats,
            meta={"game": d["game"], "call": d["call"], "player": source},
            student_move=d["action"],
        )
        if rec is None:
            stats["skipped"] += 1
            continue
        out.append(rec)
        stats["kept"] += 1
        if args.progress and len(out) % args.progress == 0:
            print(f"  {source}: labelled {len(out)}/{limit}", flush=True)
    summary.update({f"label_{k}": v for k, v in stats.items()})
    return out, summary


def to_float_map(d: dict) -> dict[int, float]:
    return {int(k): float(v) for k, v in (d or {}).items()}


class ContextualNet(nn.Module):
    def __init__(self, n_cards: int, dense_dim: int, emb: int = 24, hidden: int = 192, use_emb: bool = True):
        super().__init__()
        self.use_emb = use_emb
        self.emb = nn.Embedding(n_cards, emb)
        inp = (emb if use_emb else 0) + dense_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cidx, dense):
        x = torch.cat([self.emb(cidx), dense], dim=-1) if self.use_emb else dense
        return self.net(x).squeeze(-1)


def class_logits(option_logits: torch.Tensor, eqs: list[int]) -> tuple[list[int], torch.Tensor]:
    ids = class_ids(eqs)
    vals = []
    for e in ids:
        idx = [i for i, x in enumerate(eqs) if int(x) == int(e)]
        vals.append(torch.logsumexp(option_logits[idx], dim=0))
    return ids, torch.stack(vals)


def score_decision(model: ContextualNet, d: dict, mean: torch.Tensor, std: torch.Tensor,
                   id2ix: dict[int, int], emb_dim: int, ablate: dict | None = None) -> tuple[list[int], torch.Tensor]:
    dense_np = np.array(d["dense"], dtype=np.float32)
    if ablate:
        dense_np = CR.apply_ablation(dense_np, ablate)
    dense = (torch.tensor(dense_np, dtype=torch.float32) - mean) / std
    idxs = [id2ix.get(int(cid), -1) for cid in d["cids"]]
    cidx = torch.tensor([i if i >= 0 else 0 for i in idxs], dtype=torch.long)
    if not model.use_emb:
        logits = model(cidx, dense)
    else:
        # Unknown cards get a zero embedding rather than card 0's embedding.
        emb_rows = []
        for cid, ix in zip(d["cids"], idxs):
            emb_rows.append(model.emb.weight[ix] if ix >= 0 else torch.zeros(emb_dim))
        x = torch.cat([torch.stack(emb_rows), dense], dim=-1)
        logits = model.net(x).squeeze(-1)
    return class_logits(logits, d["eq"])


def one_loss(model: ContextualNet, d: dict, mean: torch.Tensor, std: torch.Tensor,
             id2ix: dict[int, int], emb_dim: int, args, ablate: dict | None = None):
    ids, logits = score_decision(model, d, mean, std, id2ix, emb_dim, ablate)
    soft_map = to_float_map(d["soft"])
    soft = torch.tensor([soft_map.get(e, 0.0) for e in ids], dtype=torch.float32)
    if float(soft.sum()) <= 0:
        return None
    soft = soft / soft.sum()
    policy_loss = -(soft * F.log_softmax(logits, dim=0)).sum()

    acc_map = to_float_map(d["acceptable"])
    accept = torch.tensor([acc_map.get(e, 0.0) for e in ids], dtype=torch.float32)
    accept_loss = F.binary_cross_entropy_with_logits(logits, accept)

    adv_map = to_float_map(d["adv"])
    adv = [adv_map.get(e, 0.0) for e in ids]
    max_diff = max(1e-6, max(adv) - min(adv))
    pair_losses = []
    for i in range(len(ids)):
        for j in range(len(ids)):
            if adv[i] <= adv[j] + 1e-9:
                continue
            w = min(1.0, (adv[i] - adv[j]) / max_diff)
            pair_losses.append(w * F.softplus(-(logits[i] - logits[j])))
    rank_loss = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0)

    loss = policy_loss + args.lam_rank * rank_loss + args.lam_accept * accept_loss
    chosen = d.get("chosen_eq")
    if chosen is not None and chosen in ids and args.lam_aux_choice > 0:
        loss = loss + args.lam_aux_choice * F.cross_entropy(logits.unsqueeze(0), torch.tensor([ids.index(chosen)]))
    return float(d.get("weight", 1.0)) * loss


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


def dcg(relevances: list[float]) -> float:
    return sum(max(0.0, float(r)) / math.log2(i + 2.0) for i, r in enumerate(relevances))


def eval_predictions(preds: list[dict]) -> dict:
    if not preds:
        return {"n": 0}
    regrets = [p["regret"] for p in preds if p.get("regret") is not None]
    return {
        "n": len(preds),
        "top1": sum(1 for p in preds if p["top1"]) / len(preds),
        "top2": sum(1 for p in preds if p["topk"][2]) / len(preds),
        "top3": sum(1 for p in preds if p["topk"][3]) / len(preds),
        "acceptable_agreement": sum(1 for p in preds if p["acceptable"]) / len(preds),
        "pairwise_accuracy": sum(p["pairwise_correct"] for p in preds) / max(1, sum(p["pairwise_total"] for p in preds)),
        "mrr": sum(p["rr"] for p in preds) / len(preds),
        "ndcg": sum(p["ndcg"] for p in preds) / len(preds),
        "mean_regret": sum(regrets) / len(regrets) if regrets else None,
        "p90_regret": percentile(regrets, 0.90),
        "p95_regret": percentile(regrets, 0.95),
        "high_regret_count": sum(1 for r in regrets if r >= 1000.0),
        "mean_entropy": sum(p["entropy"] for p in preds) / len(preds),
        "mean_score_margin": sum(p["score_margin"] for p in preds) / len(preds),
    }


def eval_model(model: ContextualNet, decisions: list[dict], mean: torch.Tensor, std: torch.Tensor,
               id2ix: dict[int, int], emb_dim: int, ablate: dict | None = None) -> dict:
    preds = []
    by_source = defaultdict(list)
    by_stability = defaultdict(list)
    by_player = defaultdict(list)
    by_deck = defaultdict(list)
    with torch.no_grad():
        for d in decisions:
            ids, logits = score_decision(model, d, mean, std, id2ix, emb_dim, ablate)
            scores = logits.detach().cpu().numpy().astype(float).tolist()
            order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
            adv = to_float_map(d["adv"])
            acc = to_float_map(d["acceptable"])
            best = max(ids, key=lambda e: adv.get(e, -1e30))
            pred = ids[order[0]]
            best_adv = adv.get(best, 0.0)
            pred_adv = adv.get(pred, min(adv.values()) if adv else 0.0)
            rank = next((r + 1 for r, oi in enumerate(order) if ids[oi] == best), len(order))
            rel_order = [max(0.0, adv.get(ids[i], 0.0) - min(adv.values())) for i in order]
            ideal_order = sorted([max(0.0, adv.get(e, 0.0) - min(adv.values())) for e in ids], reverse=True)
            ndcg_v = dcg(rel_order) / max(1e-9, dcg(ideal_order))
            pw_c = pw_t = 0
            for i in range(len(ids)):
                for j in range(len(ids)):
                    if adv.get(ids[i], 0.0) <= adv.get(ids[j], 0.0) + 1e-9:
                        continue
                    pw_t += 1
                    pw_c += int(scores[i] > scores[j])
            probs = F.softmax(logits, dim=0).detach().cpu().numpy().astype(float).tolist()
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
            margin = scores[order[0]] - scores[order[1]] if len(order) > 1 else 0.0
            rec = {
                "top1": pred == best,
                "topk": {
                    2: any(ids[oi] == best for oi in order[:2]),
                    3: any(ids[oi] == best for oi in order[:3]),
                },
                "acceptable": acc.get(pred, 0.0) >= 0.5,
                "regret": best_adv - pred_adv,
                "pairwise_correct": pw_c,
                "pairwise_total": pw_t,
                "rr": 1.0 / rank,
                "ndcg": ndcg_v,
                "entropy": entropy,
                "score_margin": margin,
            }
            preds.append(rec)
            by_source[d["source"]].append(rec)
            by_stability[d.get("teacher_stability", "unknown")].append(rec)
            by_player[str(d.get("player"))].append(rec)
            by_deck[str(d.get("deck_hash"))].append(rec)
    return {
        "overall": eval_predictions(preds),
        "by_source": {k: eval_predictions(v) for k, v in sorted(by_source.items())},
        "by_teacher_stability": {k: eval_predictions(v) for k, v in sorted(by_stability.items())},
        "held_out_player": {k: eval_predictions(v) for k, v in sorted(by_player.items()) if len(v) >= 3},
        "held_out_deck": {k: eval_predictions(v) for k, v in sorted(by_deck.items()) if len(v) >= 3},
    }


def eval_baseline(decisions: list[dict], field: str) -> dict:
    preds = []
    for d in decisions:
        ids = class_ids(d["eq"])
        pred = d.get(field)
        if pred is None:
            continue
        pred = int(pred)
        adv = to_float_map(d["adv"])
        acc = to_float_map(d["acceptable"])
        best = max(ids, key=lambda e: adv.get(e, -1e30))
        best_adv = adv.get(best, 0.0)
        pred_adv = adv.get(pred, min(adv.values()) if adv else 0.0)
        # Baselines expose one selected class only; pairwise/top-k are still meaningful for top-1,
        # while MRR/NDCG use a degenerate ranking with the selected class first.
        order = [pred] + [e for e in ids if e != pred]
        rank = order.index(best) + 1 if best in order else len(order)
        rel_order = [max(0.0, adv.get(e, 0.0) - min(adv.values())) for e in order]
        ideal_order = sorted([max(0.0, adv.get(e, 0.0) - min(adv.values())) for e in ids], reverse=True)
        pw_t = pw_c = 0
        for e in ids:
            if e == pred:
                continue
            if adv.get(pred, 0.0) > adv.get(e, 0.0):
                pw_c += 1
            if adv.get(best, 0.0) > adv.get(e, 0.0):
                pw_t += 1
        preds.append({
            "top1": pred == best,
            "topk": {2: best in order[:2], 3: best in order[:3]},
            "acceptable": acc.get(pred, 0.0) >= 0.5,
            "regret": best_adv - pred_adv,
            "pairwise_correct": pw_c,
            "pairwise_total": max(1, pw_t),
            "rr": 1.0 / rank,
            "ndcg": dcg(rel_order) / max(1e-9, dcg(ideal_order)),
            "entropy": 0.0,
            "score_margin": 0.0,
        })
    return {"overall": eval_predictions(preds)}


def train_one(name: str, train: list[dict], eval_sets: dict[str, list[dict]], args,
              card_ids: list[int], mean_np: np.ndarray, std_np: np.ndarray,
              use_emb: bool = True, ablate: dict | None = None) -> tuple[ContextualNet, dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dense_dim = len(train[0]["dense"][0])
    model = ContextualNet(len(card_ids), dense_dim, emb=args.emb_dim, hidden=args.hidden, use_emb=use_emb)
    id2ix = {int(c): i for i, c in enumerate(card_ids)}
    mean = torch.tensor(mean_np, dtype=torch.float32)
    std = torch.tensor(std_np, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = list(range(len(train)))
    losses = []
    for ep in range(args.epochs):
        rng = np.random.default_rng(args.seed + ep)
        rng.shuffle(order)
        total = 0.0
        nb = 0
        opt.zero_grad()
        for n, i in enumerate(order, start=1):
            loss = one_loss(model, train[i], mean, std, id2ix, args.emb_dim, args, ablate)
            if loss is None:
                continue
            (loss / args.batch_decisions).backward()
            total += float(loss.detach())
            nb += 1
            if n % args.batch_decisions == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                opt.zero_grad()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        opt.zero_grad()
        losses.append(total / max(1, nb))
        print(f"  {name} epoch {ep + 1}/{args.epochs} loss={losses[-1]:.4f}", flush=True)
    model.eval()
    evals = {
        split: eval_model(model, rows, mean, std, id2ix, args.emb_dim, ablate)
        for split, rows in eval_sets.items()
        if rows
    }
    return model, {"epoch_losses": losses, "eval": evals, "use_emb": use_emb, "ablate": ablate or {}}


def dataset_summary(rows: list[dict]) -> dict:
    by_source = Counter(r["source"] for r in rows)
    by_part = Counter(r["partition"] for r in rows)
    by_stab = Counter(r.get("teacher_stability", "unknown") for r in rows)
    return {
        "total_decisions": len(rows),
        "by_source": dict(sorted(by_source.items())),
        "by_partition": dict(sorted(by_part.items())),
        "by_teacher_stability": dict(sorted(by_stab.items())),
        "high_regret": sum(1 for r in rows if r.get("high_regret")),
        "players": len({r.get("player") for r in rows if r.get("player")}),
        "decks": len({r.get("deck_hash") for r in rows if r.get("deck_hash")}),
    }


def config_dict(args) -> dict:
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def save_dataset(rows: list[dict], path: Path, args, collection: dict) -> None:
    payload = {
        "artifact_version": "contextual_action_ranker_v1.dataset",
        "config": {
            "split": str(args.split),
            "replay_dir": str(args.replay_dir),
            "teacher_repeats": args.teacher_repeats,
            "n_determ": args.n_determ,
            "time_budget": args.time_budget,
            "accept_z": args.accept_z,
            "seed": args.seed,
            "note": "Prior on-policy JSONs do not store raw observations; recovery states are recollected from Round 1/2 models.",
        },
        "collection": collection,
        "summary": dataset_summary(rows),
        "decisions": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_model(model: ContextualNet, path: Path, card_ids: list[int], mean: np.ndarray, std: np.ndarray,
               args, dataset_path: Path) -> None:
    blob = {
        "state_dict": {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()},
        "card_ids": [int(c) for c in card_ids],
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "hidden": args.hidden,
        "emb": args.emb_dim,
        "dense_dim": int(len(mean)),
        "use_emb": True,
        "target": "teacher_soft_policy_advantage_acceptable_confidence",
        "trained": "contextual_action_ranker_v1",
        "dataset": display(dataset_path),
        "feature_sections": {k: list(v) for k, v in CR.SLICES.items()},
        "effect_keys": CR.EFFECT_KEYS,
        "delta_keys": CR.DELTA_KEYS,
        "context_keys": CR.CTX_KEYS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    ap.add_argument("--max-replay-files", type=int, default=80)
    ap.add_argument("--replay-train-limit", type=int, default=72)
    ap.add_argument("--replay-val-limit", type=int, default=24)
    ap.add_argument("--replay-test-limit", type=int, default=24)
    ap.add_argument("--round1-games", type=int, default=4)
    ap.add_argument("--round2-games", type=int, default=4)
    ap.add_argument("--round1-limit", type=int, default=80)
    ap.add_argument("--round2-limit", type=int, default=80)
    ap.add_argument("--teacher-repeats", type=int, default=2)
    ap.add_argument("--min-teacher-agreement", type=float, default=1.0)
    ap.add_argument("--n-determ", type=int, default=2)
    ap.add_argument("--time-budget", type=float, default=1.2)
    ap.add_argument("--accept-z", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-decisions", type=int, default=16)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--emb-dim", type=int, default=24)
    ap.add_argument("--lam-rank", type=float, default=0.25)
    ap.add_argument("--lam-accept", type=float, default=0.12)
    ap.add_argument("--lam-aux-choice", type=float, default=0.04)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--replay-weight", type=float, default=0.9)
    ap.add_argument("--recovery-weight", type=float, default=1.1)
    ap.add_argument("--unstable-weight-scale", type=float, default=0.35)
    ap.add_argument("--high-regret-threshold", type=float, default=1000.0)
    ap.add_argument("--high-regret-weight-scale", type=float, default=1.6)
    ap.add_argument("--old-ranker-baseline", action="store_true", default=True)
    ap.add_argument("--skip-collection", action="store_true")
    ap.add_argument("--dataset-in", type=Path, default=None)
    ap.add_argument("--dataset-out", type=Path, default=DOCS / "contextual_action_ranker_v1_dataset.json")
    ap.add_argument("--model-out", type=Path, default=ROOT / "agent" / "contextual_ranker_v1.json")
    ap.add_argument("--report-out", type=Path, default=DOCS / "contextual_action_ranker_v1_train_eval.json")
    ap.add_argument("--seed", type=int, default=17041)
    ap.add_argument("--progress", type=int, default=20)
    args = ap.parse_args()

    args.split = args.split if args.split.is_absolute() else ROOT / args.split
    args.replay_dir = args.replay_dir if args.replay_dir.is_absolute() else ROOT / args.replay_dir
    args.dataset_out = args.dataset_out if args.dataset_out.is_absolute() else ROOT / args.dataset_out
    args.model_out = args.model_out if args.model_out.is_absolute() else ROOT / args.model_out
    args.report_out = args.report_out if args.report_out.is_absolute() else ROOT / args.report_out
    if args.dataset_in:
        args.dataset_in = args.dataset_in if args.dataset_in.is_absolute() else ROOT / args.dataset_in

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset_in:
        payload = json.loads(args.dataset_in.read_text(encoding="utf-8"))
        rows = payload["decisions"]
        collection = payload.get("collection", {})
    else:
        print("collecting replay teacher decisions", flush=True)
        train_rows, train_stats = collect_replay_decisions("train", args.replay_train_limit, args, 10_000)
        val_rows, val_stats = collect_replay_decisions("val", args.replay_val_limit, args, 200_000)
        test_rows, test_stats = collect_replay_decisions("test", args.replay_test_limit, args, 300_000)
        print("collecting Round 1/2 recovery states", flush=True)
        r1_rows, r1_stats = collect_recovery_decisions(
            "ranker_model_dagger_round1.json", "recovery_round1", args.round1_games, args.round1_limit, args, 400_000
        )
        r2_rows, r2_stats = collect_recovery_decisions(
            "ranker_model_dagger_round2.json", "recovery_round2", args.round2_games, args.round2_limit, args, 500_000
        )
        rows = train_rows + val_rows + test_rows + r1_rows + r2_rows
        collection = {
            "replay_train": train_stats,
            "replay_val": val_stats,
            "replay_test": test_stats,
            "recovery_round1": r1_stats,
            "recovery_round2": r2_stats,
        }
        save_dataset(rows, args.dataset_out, args, collection)
        print(f"saved dataset -> {display(args.dataset_out)}", flush=True)

    if not rows:
        raise SystemExit("No decisions collected; refusing to train.")

    train = [r for r in rows if r["partition"] == "train" or r["partition"] == "recovery"]
    val = [r for r in rows if r["partition"] == "val"]
    test = [r for r in rows if r["partition"] == "test"]
    if not train or not (val or test):
        raise SystemExit("Need non-empty train and validation/test rows.")

    dense_all = np.array([row for d in train for row in d["dense"]], dtype=np.float32)
    mean = dense_all.mean(axis=0)
    std = dense_all.std(axis=0) + 1e-6
    card_ids = sorted({int(c) for d in rows for c in d["cids"] if int(c) >= 0})
    if not card_ids:
        card_ids = [0]

    print("training contextual models", flush=True)
    eval_sets = {"train": train, "val": val, "test": test}
    full_model, full_report = train_one("full", train, eval_sets, args, card_ids, mean, std, use_emb=True)
    no_effects_model, no_effects_report = train_one(
        "no_effects", train, eval_sets, args, card_ids, mean, std, use_emb=True, ablate={"effects": True}
    )
    no_emb_model, no_emb_report = train_one(
        "no_embedding", train, eval_sets, args, card_ids, mean, std, use_emb=False
    )
    no_delta_model, no_delta_report = train_one(
        "no_deltas", train, eval_sets, args, card_ids, mean, std, use_emb=True, ablate={"deltas": True}
    )

    baseline_sets = {"val": val, "test": test}
    baselines = {
        split: {
            "old_ranker": eval_baseline(decisions, "old_ranker_eq"),
            "option0": eval_baseline(decisions, "option0_eq"),
        }
        for split, decisions in baseline_sets.items()
        if decisions
    }

    save_model(full_model, args.model_out, card_ids, mean, std, args, args.dataset_out)
    print(f"saved deploy model -> {display(args.model_out)}", flush=True)

    report = {
        "artifact_version": "contextual_action_ranker_v1.train_eval",
        "branch": "exp/robust-learner-v2",
        "config": config_dict(args),
        "dataset": dataset_summary(rows),
        "collection": collection,
        "model": {
            "dense_dim": int(len(mean)),
            "card_ids": len(card_ids),
            "hidden": args.hidden,
            "emb_dim": args.emb_dim,
            "feature_sections": {k: list(v) for k, v in CR.SLICES.items()},
        },
        "ablations": {
            "full": full_report,
            "no_decoded_effects": no_effects_report,
            "no_card_embedding": no_emb_report,
            "no_option_deltas": no_delta_report,
        },
        "baselines": baselines,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved train/eval report -> {display(args.report_out)}", flush=True)
    print(json.dumps({
        "dataset": report["dataset"],
        "full_test": full_report["eval"].get("test", {}).get("overall"),
        "old_ranker_test": baselines.get("test", {}).get("old_ranker", {}).get("overall"),
        "option0_test": baselines.get("test", {}).get("option0", {}).get("overall"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
