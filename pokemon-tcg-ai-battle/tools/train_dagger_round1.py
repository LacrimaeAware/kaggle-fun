"""Branch B / B2 pilot -- one bounded DAgger round for the existing action ranker.

This deliberately keeps the deployed student architecture fixed:

  card embedding + action/root/delta dense features -> 2-layer MLP -> option score

The round starts from the current `agent/ranker_model.json` weights, mixes the original
fixed-deck distillation rows with learner-visited recovery states, and fine-tunes once using
Teacher V1 soft policy, advantages, acceptable-action sets, and confidence weights.

It is not a new representation, embedding experiment, successor-affordance head, or RL run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))

import build_action_dataset as BAD  # noqa: E402
import features as FT  # noqa: E402
import main as M  # noqa: E402
import measure_on_policy_shift as OPS  # noqa: E402
import search as S  # noqa: E402
import teacher_api_v1 as T  # noqa: E402
from train_action_ranker import Ranker, dense_vec  # noqa: E402


def softmax_from_values(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    best = max(values.values())
    spread = max(values.values()) - min(values.values())
    temp = max(spread, 1e-9)
    exps = {k: math.exp((v - best) / temp) for k, v in values.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def grouped(xs):
    d = defaultdict(list)
    for x in xs:
        d[x["gid"]].append(x)
    return d


def resolve_base_data(path_arg: str | None) -> Path:
    if path_arg:
        p = Path(path_arg)
        return p if p.is_absolute() else ROOT / p
    candidates = [
        ROOT / "data" / "replay_db" / "action_adv.jsonl",
        ROOT.parents[3] / "pokemon-tcg-ai-battle" / "data" / "replay_db" / "action_adv.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("Could not find action_adv.jsonl; pass --base-data explicitly.")


def load_base_distill(path: Path, limit_decisions: int, base_weight: float,
                      acceptable_margin: float) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("has_val"):
                rows.append(r)
    out = []
    for gid, opts in grouped(rows).items():
        if len(opts) < 2:
            continue
        class_vals = defaultdict(list)
        for o in opts:
            class_vals[int(o["eq"])].append(float(o["val"]))
        values = {k: sum(vs) / len(vs) for k, vs in class_vals.items()}
        if len(values) < 2:
            continue
        best = max(values.values())
        mean_v = sum(values.values()) / len(values)
        soft = softmax_from_values(values)
        acceptable = {k: 1.0 if best - v <= acceptable_margin else 0.0 for k, v in values.items()}
        out.append({
            "source": "base_distill",
            "stability": "original",
            "weight": base_weight,
            "cids": [int(o.get("card_id", -1)) for o in opts],
            "dense": [dense_vec(o) for o in opts],
            "eq": [int(o["eq"]) for o in opts],
            "soft": soft,
            "adv": {k: v - mean_v for k, v in values.items()},
            "acceptable": acceptable,
        })
        if limit_decisions and len(out) >= limit_decisions:
            break
    return out


def player_deck(game: dict, player: int) -> list[int] | None:
    for step in game.get("steps", []):
        if player < len(step) and isinstance(step[player], dict):
            action = step[player].get("action")
            if isinstance(action, list) and len(action) == 60:
                return action
    return None


def replay_decision(decision_id: str) -> tuple[dict, list[int]] | None:
    name, step_s, agent_s = decision_id.rsplit(":", 2)
    path = ROOT / "data" / "external" / "replays" / name
    if not path.exists():
        return None
    game = json.loads(path.read_text(encoding="utf-8"))
    step = game.get("steps", [])[int(step_s)]
    agent_i = int(agent_s)
    rec = step[agent_i]
    obs = rec.get("observation") or {}
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", agent_i)
    deck = player_deck(game, me)
    return (obs, deck) if deck else None


def option_payload(obs: dict, deck: list[int]) -> dict | None:
    sel = obs.get("select") or {}
    cur = obs.get("current") or {}
    opts = sel.get("option") or []
    if (sel.get("maxCount") or 0) != 1 or len(opts) < 2 or not cur.get("players"):
        return None
    if M._forced_move(obs) is not None:
        return None
    me = cur.get("yourIndex", 0)
    try:
        root = FT.vectorize(FT.encode_state(obs))
    except Exception:
        return None
    try:
        deltas = S.option_deltas(obs, deck)
    except Exception:
        deltas = None
    keys = [BAD.opt_key(o, cur, me) if isinstance(o, dict) else None for o in opts]
    uniq = {}
    for k in keys:
        if k not in uniq:
            uniq[k] = len(uniq)
    dense, cids, eqs = [], [], []
    for j, o in enumerate(opts):
        if not isinstance(o, dict):
            continue
        af = BAD.option_features(o, cur, me)
        dd = deltas[j] if (deltas and j < len(deltas) and deltas[j]) else {}
        for k in BAD.DELTA_KEYS:
            af["d_" + k] = float(dd.get(k, 0.0))
        af["has_delta"] = 1 if dd else 0
        row = {"root": root, "eq": uniq[keys[j]], **af}
        dense.append(dense_vec(row))
        cids.append(int(row.get("card_id", -1)))
        eqs.append(int(row["eq"]))
    if len(dense) < 2 or len(set(eqs)) < 2:
        return None
    key_to_eq = {tuple(k): int(v) for k, v in uniq.items() if k is not None}
    return {"dense": dense, "cids": cids, "eq": eqs, "key_to_eq": key_to_eq}


def teacher_targets(obs: dict, deck: list[int], repeats: int, n_determ: int, time_budget: float,
                    seed0: int, accept_z: float, min_agreement: float) -> dict | None:
    payload = option_payload(obs, deck)
    if payload is None:
        return None
    per = []
    label_counts = Counter()
    for r in range(repeats):
        result = T.query(
            obs,
            deck,
            n_determ=n_determ,
            time_budget=time_budget,
            leaf_mode="hand",
            seed=seed0 + r,
            accept_z=accept_z,
        )
        if not result.get("applicable"):
            continue
        chosen = result.get("chosen_option")
        opts = result.get("options") or []
        if chosen is not None and chosen < len(opts):
            label_counts[tuple(opts[chosen]["semantic_action_key"])] += 1
        per.append(result)
    if not per:
        return None

    key_to_eq = payload["key_to_eq"]
    soft_acc = defaultdict(list)
    adv_acc = defaultdict(list)
    acceptable_acc = defaultdict(list)
    for result in per:
        soft = result.get("soft_policy_target") or {}
        class_values = []
        for cls in result.get("eq_classes") or []:
            if cls.get("mean_value") is not None:
                class_values.append(float(cls["mean_value"]))
        mean_v = sum(class_values) / len(class_values) if class_values else 0.0
        acceptable = set(result.get("acceptable_action_set") or [])
        if result.get("forced_action_flag") and result.get("forced_eq_key") is not None:
            fk = tuple(result["forced_eq_key"])
            if fk in key_to_eq:
                acceptable.add(key_to_eq[fk])
        for cls in result.get("eq_classes") or []:
            k = tuple(cls.get("key") or ())
            if k not in key_to_eq or cls.get("mean_value") is None:
                continue
            eq = key_to_eq[k]
            teq = cls["eq_class"]
            soft_acc[eq].append(float(soft.get(teq, soft.get(str(teq), 0.0)) or 0.0))
            adv_acc[eq].append(float(cls["mean_value"]) - mean_v)
            acceptable_acc[eq].append(1.0 if teq in acceptable or eq in acceptable else 0.0)

    if not soft_acc:
        return None
    soft = {k: sum(v) / len(v) for k, v in soft_acc.items()}
    z = sum(soft.values())
    if z <= 0:
        return None
    soft = {k: v / z for k, v in soft.items()}
    adv = {k: sum(v) / len(v) for k, v in adv_acc.items()}
    acceptable = {k: sum(v) / len(v) for k, v in acceptable_acc.items()}
    label_agreement = (label_counts.most_common(1)[0][1] / len(per)) if label_counts else 0.0
    stability = "stable" if label_agreement >= min_agreement else "unstable"
    return {
        **payload,
        "soft": soft,
        "adv": adv,
        "acceptable": acceptable,
        "stability": stability,
        "label_agreement": label_agreement,
        "applicable_repeats": len(per),
    }


def load_stable_replay_decisions(artifact: Path, limit: int, args) -> list[dict]:
    if not artifact.exists() or limit == 0:
        return []
    src = json.loads(artifact.read_text(encoding="utf-8"))
    out = []
    for i, d in enumerate(src.get("accepted_decisions") or []):
        rec = replay_decision(d["id"])
        if not rec:
            continue
        obs, deck = rec
        target = teacher_targets(
            obs, deck, args.teacher_repeats, args.n_determ, args.time_budget,
            args.seed + 100000 + i * args.teacher_repeats, args.accept_z, args.min_teacher_agreement,
        )
        if target is None:
            continue
        target["source"] = "stable_replay"
        target["weight"] = args.stable_replay_weight
        out.append(target)
        if limit and len(out) >= limit:
            break
    return out


def collect_recovery_decisions(args) -> tuple[list[dict], dict]:
    decisions, game_summary = OPS.run_traced_games("rank", "heuristic", args.recovery_games, args.progress)
    if args.max_recovery_decisions:
        decisions = decisions[:args.max_recovery_decisions]
    out = []
    skipped_forced = skipped_not_applicable = skipped_features = 0
    t0 = time.time()
    for i, d in enumerate(decisions):
        if M._forced_move(d["obs"]) is not None:
            skipped_forced += 1
            continue
        target = teacher_targets(
            d["obs"], list(M.DECK), args.teacher_repeats, args.n_determ, args.time_budget,
            args.seed + i * args.teacher_repeats, args.accept_z, args.min_teacher_agreement,
        )
        if target is None:
            # Distinguish feature/forced skips from teacher skips for the summary.
            if option_payload(d["obs"], list(M.DECK)) is None:
                skipped_features += 1
            else:
                skipped_not_applicable += 1
            continue
        if target["stability"] == "stable":
            target["weight"] = args.recovery_stable_weight
        else:
            target["weight"] = args.recovery_unstable_weight * max(0.25, target["label_agreement"])
        target["source"] = "recovery"
        target["game"] = d["game"]
        target["call"] = d["call"]
        target["obs_hash"] = d["obs_hash"]
        out.append(target)
        if args.progress and ((i + 1) % args.progress == 0 or i + 1 == len(decisions)):
            print(f"  labelled recovery {i + 1}/{len(decisions)} | kept={len(out)} | {time.time() - t0:.0f}s", flush=True)
    game_summary.update({
        "traced_decisions": len(decisions),
        "kept_recovery_decisions": len(out),
        "skipped_forced_floor": skipped_forced,
        "skipped_not_applicable": skipped_not_applicable,
        "skipped_features": skipped_features,
    })
    return out, game_summary


def load_model(path: Path):
    blob = json.loads(path.read_text(encoding="utf-8"))
    model = Ranker(len(blob["card_ids"]), int(blob["dense_dim"]), emb=int(blob["emb"]), use_emb=bool(blob["use_emb"]))
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in blob["state_dict"].items()}
    model.load_state_dict(state)
    return blob, model


def score_options(model, decision: dict, mean: torch.Tensor, std: torch.Tensor,
                  id2ix: dict[int, int], emb_dim: int) -> torch.Tensor:
    dense = torch.tensor(np.array(decision["dense"], dtype=np.float32), dtype=torch.float32)
    dn = (dense - mean) / std
    xs = []
    for cid, row in zip(decision["cids"], dn):
        ix = id2ix.get(int(cid))
        emb = model.emb.weight[ix] if ix is not None else torch.zeros(emb_dim)
        xs.append(torch.cat([emb, row], dim=0))
    x = torch.stack(xs, dim=0)
    return model.net(x).squeeze(-1)


def class_view(option_logits: torch.Tensor, eqs: list[int]) -> tuple[list[int], torch.Tensor]:
    class_ids = []
    for e in eqs:
        if e not in class_ids:
            class_ids.append(e)
    logits = []
    for e in class_ids:
        idx = [i for i, x in enumerate(eqs) if x == e]
        logits.append(torch.logsumexp(option_logits[idx], dim=0))
    return class_ids, torch.stack(logits)


def one_loss(model, decision, mean, std, id2ix, emb_dim, lam_rank, lam_accept):
    opt_logits = score_options(model, decision, mean, std, id2ix, emb_dim)
    class_ids, logits = class_view(opt_logits, decision["eq"])
    soft = torch.tensor([float(decision["soft"].get(e, 0.0)) for e in class_ids], dtype=torch.float32)
    if float(soft.sum()) <= 0:
        return None
    soft = soft / soft.sum()
    policy_loss = -(soft * F.log_softmax(logits, dim=0)).sum()

    acc_target = torch.tensor([float(decision["acceptable"].get(e, 0.0)) for e in class_ids], dtype=torch.float32)
    accept_loss = F.binary_cross_entropy_with_logits(logits, acc_target)

    adv = [float(decision["adv"].get(e, 0.0)) for e in class_ids]
    max_diff = max(1e-6, max(adv) - min(adv))
    pair_losses = []
    for i in range(len(class_ids)):
        for j in range(len(class_ids)):
            if adv[i] <= adv[j] + 1e-9:
                continue
            w = min(1.0, (adv[i] - adv[j]) / max_diff)
            pair_losses.append(w * F.softplus(-(logits[i] - logits[j])))
    rank_loss = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0)
    return float(decision["weight"]) * (policy_loss + lam_rank * rank_loss + lam_accept * accept_loss)


def eval_decisions(model, decisions, mean, std, id2ix, emb_dim) -> dict:
    by_source = defaultdict(list)
    rows = []
    with torch.no_grad():
        for d in decisions:
            logits = score_options(model, d, mean, std, id2ix, emb_dim)
            class_ids, class_logits = class_view(logits, d["eq"])
            pred = class_ids[int(torch.argmax(class_logits).item())]
            best = max(d["adv"], key=lambda k: d["adv"][k])
            pred_adv = float(d["adv"].get(pred, min(d["adv"].values())))
            regret = float(d["adv"][best] - pred_adv)
            rec = {
                "source": d["source"],
                "stability": d.get("stability", "unknown"),
                "acceptable": float(d["acceptable"].get(pred, 0.0)) >= 0.5,
                "hard_top1": pred == best,
                "regret": regret,
            }
            rows.append(rec)
            by_source[(d["source"], d.get("stability", "unknown"))].append(rec)

    def summ(xs):
        if not xs:
            return {"n": 0}
        regrets = sorted(x["regret"] for x in xs)
        p90 = regrets[int(0.9 * (len(regrets) - 1))]
        return {
            "n": len(xs),
            "acceptable_agreement": sum(1 for x in xs if x["acceptable"]) / len(xs),
            "hard_top1": sum(1 for x in xs if x["hard_top1"]) / len(xs),
            "mean_regret": sum(regrets) / len(regrets),
            "p90_regret": p90,
            "high_regret_count": sum(1 for r in regrets if r >= 1000.0),
        }

    out = {"overall": summ(rows), "by_source_stability": {}}
    for key, xs in by_source.items():
        out["by_source_stability"]["/".join(key)] = summ(xs)
    return out


def train(model, decisions, blob, args):
    mean = torch.tensor(blob["mean"], dtype=torch.float32)
    std = torch.tensor(blob["std"], dtype=torch.float32)
    id2ix = {int(c): i for i, c in enumerate(blob["card_ids"])}
    emb_dim = int(blob["emb"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    before = eval_decisions(model, decisions, mean, std, id2ix, emb_dim)
    order = list(range(len(decisions)))
    epoch_losses = []
    for ep in range(args.epochs):
        rng = np.random.default_rng(args.seed + ep)
        rng.shuffle(order)
        total = 0.0
        nb = 0
        opt.zero_grad()
        for n, i in enumerate(order, start=1):
            loss = one_loss(model, decisions[i], mean, std, id2ix, emb_dim, args.lam_rank, args.lam_accept)
            if loss is None:
                continue
            (loss / args.batch_decisions).backward()
            total += float(loss.detach())
            nb += 1
            if n % args.batch_decisions == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        epoch_losses.append(total / max(1, nb))
        print(f"  epoch {ep + 1}/{args.epochs} loss={epoch_losses[-1]:.4f}", flush=True)
    after = eval_decisions(model, decisions, mean, std, id2ix, emb_dim)
    return before, after, epoch_losses


def save_model(model, blob, out_path: Path, args, n_decisions: int) -> None:
    new_blob = dict(blob)
    new_blob["state_dict"] = {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}
    new_blob["target"] = "dagger_round1_soft_adv_acceptable"
    new_blob["trained"] = "base_action_adv_plus_rank_recovery"
    new_blob["dagger_round"] = 1
    new_blob["n_decisions"] = n_decisions
    new_blob["round1_config"] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lam_rank": args.lam_rank,
        "lam_accept": args.lam_accept,
        "teacher_repeats": args.teacher_repeats,
        "n_determ": args.n_determ,
        "time_budget": args.time_budget,
        "recovery_games": args.recovery_games,
        "max_recovery_decisions": args.max_recovery_decisions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(new_blob, separators=(",", ":")), encoding="utf-8")


def summarize_dataset(decisions: list[dict]) -> dict:
    counts = Counter((d["source"], d.get("stability", "unknown")) for d in decisions)
    return {
        "total_decisions": len(decisions),
        "by_source_stability": {"/".join(k): v for k, v in sorted(counts.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-data", default=None)
    ap.add_argument("--base-limit", type=int, default=1218)
    ap.add_argument("--base-weight", type=float, default=0.35)
    ap.add_argument("--base-acceptable-margin", type=float, default=100.0)
    ap.add_argument("--stable-replay-artifact", type=Path,
                    default=ROOT / "docs" / "workstreams" / "robust_learner_v2_b1_2_train.json")
    ap.add_argument("--stable-replay-limit", type=int, default=64)
    ap.add_argument("--stable-replay-weight", type=float, default=0.6)
    ap.add_argument("--recovery-games", type=int, default=30)
    ap.add_argument("--max-recovery-decisions", type=int, default=800)
    ap.add_argument("--recovery-stable-weight", type=float, default=1.0)
    ap.add_argument("--recovery-unstable-weight", type=float, default=0.35)
    ap.add_argument("--teacher-repeats", type=int, default=3)
    ap.add_argument("--min-teacher-agreement", type=float, default=1.0)
    ap.add_argument("--n-determ", type=int, default=3)
    ap.add_argument("--time-budget", type=float, default=4.0)
    ap.add_argument("--accept-z", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-decisions", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam-rank", type=float, default=0.25)
    ap.add_argument("--lam-accept", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--model-in", type=Path, default=ROOT / "agent" / "ranker_model.json")
    ap.add_argument("--model-out", type=Path, default=ROOT / "agent" / "ranker_model_dagger_round1.json")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "docs" / "workstreams" / "robust_learner_v2_dagger_round1_train.json")
    ap.add_argument("--progress", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_path = resolve_base_data(args.base_data)
    print(f"loading base distill rows from {base_path}", flush=True)
    decisions = load_base_distill(base_path, args.base_limit, args.base_weight, args.base_acceptable_margin)
    print(f"base distill decisions: {len(decisions)}", flush=True)

    stable = load_stable_replay_decisions(args.stable_replay_artifact, args.stable_replay_limit, args)
    decisions.extend(stable)
    print(f"stable replay decisions: {len(stable)}", flush=True)

    recovery, game_summary = collect_recovery_decisions(args)
    decisions.extend(recovery)
    print(f"recovery decisions: {len(recovery)}", flush=True)
    if not recovery:
        raise SystemExit("No recovery decisions were labelled; refusing to run an empty DAgger round.")

    blob, model = load_model(args.model_in)
    before, after, losses = train(model, decisions, blob, args)
    save_model(model, blob, args.model_out, args, len(decisions))

    report = {
        "audit_version": "branch_b_dagger_round1.0",
        "config": {
            "base_data": str(base_path),
            "model_in": str(args.model_in),
            "model_out": str(args.model_out),
            "base_limit": args.base_limit,
            "stable_replay_limit": args.stable_replay_limit,
            "recovery_games": args.recovery_games,
            "max_recovery_decisions": args.max_recovery_decisions,
            "teacher_repeats": args.teacher_repeats,
            "n_determ": args.n_determ,
            "time_budget": args.time_budget,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
        },
        "dataset": summarize_dataset(decisions),
        "recovery_game_summary": game_summary,
        "offline_eval_before": before,
        "offline_eval_after": after,
        "epoch_losses": losses,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved model -> {args.model_out.relative_to(ROOT)}", flush=True)
    print(f"saved report -> {args.report.relative_to(ROOT)}", flush=True)
    print(json.dumps({"before": before["overall"], "after": after["overall"], "dataset": report["dataset"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
