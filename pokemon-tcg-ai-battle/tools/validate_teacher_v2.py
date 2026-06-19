"""Branch A / A4 validation -- is Teacher V2's terminal-outcome a real stronger-than-hand signal, or noise?

For high-criticality decisions: high-N(32) hand advantage (Teacher V1) + TWO independent k=32 paired-playout
outcome runs. From one run's per-world results it reads the outcome argmax at k=4/8/16/32 prefixes
(convergence: does hand-vs-outcome disagreement persist or collapse as k grows?); the second run gives
outcome-argmax stability; binomial SE gives per-option confidence. Also writes the scaled label batch for B.

    python tools/validate_teacher_v2.py --n 12 --k 32 --out teacher_v2_labels_v2
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import teacher_api_v1 as T1               # noqa: E402
import teacher_api_v2 as T2               # noqa: E402
import audit_teacher_stability as A2      # noqa: E402

KS = [4, 8, 16, 32]


def _winrate(seq, k):
    s = seq[:k]
    return (sum(s) / len(s)) if s else None


def _argmax_opt(results, k):
    wr = [(_winrate(results[i], k), i) for i in range(len(results)) if results[i]]
    wr = [(w, i) for w, i in wr if w is not None]
    return max(wr, key=lambda x: x[0])[1] if wr else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="replays_20260618.json")
    ap.add_argument("--split", default="replays_20260618_split.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=120)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-determ", type=int, default=32)
    ap.add_argument("--crit-threshold", type=float, default=0.3)
    ap.add_argument("--out", default="teacher_v2_labels_v2")
    args = ap.parse_args()

    manifest = json.load(open(ROOT / "data" / "manifests" / args.snapshot, encoding="utf-8"))
    split = json.load(open(ROOT / "data" / "splits" / args.split, encoding="utf-8"))
    cands = A2.sample_decisions(manifest, split, args.candidates, verify=False)
    scored = sorted(cands, key=lambda d: -T2.criticality_score(d["obs"])["score"])
    print(f"[A4-val] {len(cands)} candidates; validating up to {args.n} most-critical at k={args.k}", flush=True)

    labels, t0 = [], time.time()
    disagree_at_k = {k: [0, 0] for k in KS}     # k -> [disagreements, decisions]
    stable_eq = [0, 0]
    ses = []
    for d in scored:
        if len(labels) >= args.n:
            break
        obs, deck = d["obs"], d["deck"]
        if T2.criticality_score(obs)["score"] < args.crit_threshold:
            continue
        v1 = T1.query(obs, deck, n_determ=args.n_determ, time_budget=8.0, leaf_mode="hand", seed=1234)
        if not v1.get("applicable"):
            continue
        eq_of = {o["index"]: o["eq_class"] for o in v1["options"]}
        hand_eq = v1.get("argmax_eq_class")
        rA = T2.outcome_playouts(obs, deck, args.k, time_budget=40.0)
        rB = T2.outcome_playouts(obs, deck, args.k, time_budget=40.0)
        if not rA or not rB:
            continue
        # convergence (run A prefixes)
        conv = {}
        for k in KS:
            am = _argmax_opt(rA, k)
            if am is None:
                continue
            disagree = int(eq_of.get(am) != hand_eq)
            conv[k] = {"outcome_argmax_opt": am, "outcome_argmax_eq": eq_of.get(am), "disagree_hand": disagree}
            disagree_at_k[k][0] += disagree
            disagree_at_k[k][1] += 1
        # stability at full k (eq-class level)
        aA, aB = _argmax_opt(rA, args.k), _argmax_opt(rB, args.k)
        st = int(aA is not None and aB is not None and eq_of.get(aA) == eq_of.get(aB))
        stable_eq[0] += st; stable_eq[1] += 1
        # per-option outcome winrate + binomial SE at full k
        opt_out = []
        for i in range(len(rA)):
            n = len(rA[i])
            p = (sum(rA[i]) / n) if n else None
            se = math.sqrt(p * (1 - p) / n) if (p is not None and n) else None
            if se is not None:
                ses.append(se)
            opt_out.append({"index": i, "outcome_winrate": round(p, 3) if p is not None else None,
                            "outcome_playouts": n, "outcome_se": round(se, 3) if se is not None else None})
        labels.append({
            "decision_id": f"{d['file']}:{d['step']}", "source": {"file": d["file"], "step": d["step"]},
            "criticality": T2.criticality_score(obs), "me": v1["me"],
            "hand_argmax_eq_class": hand_eq, "top_two_margin": v1["top_two_margin"],
            "soft_policy_target": v1["soft_policy_target"], "acceptable_action_set": v1["acceptable_action_set"],
            "options": [{**o, **next(x for x in opt_out if x["index"] == o["index"])} for o in v1["options"]],
            "outcome_convergence": conv, "outcome_stable_runs": st,
            "outcome_argmax_eq_full": eq_of.get(aA), "hand_outcome_agree_full": int(eq_of.get(aA) == hand_eq),
            "config": {"n_determ": args.n_determ, "k_outcome": args.k, "paired_world": True, "seed": 1234},
        })
        print(f"  [{len(labels)}/{args.n}] {d['file']}:{d['step']} crit={labels[-1]['criticality']['score']} "
              f"hand_eq={hand_eq} conv_disagree={[conv.get(k, {}).get('disagree_hand') for k in KS]} "
              f"stable={st} ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "data" / "manifests" / f"{args.out}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")
    print(f"\n[A4-val] wrote {len(labels)} validated Teacher V2 labels -> {out.relative_to(ROOT)}")
    print("=== CONVERGENCE: hand-vs-outcome disagreement rate by k (does it persist or collapse?) ===")
    for k in KS:
        dd, nn = disagree_at_k[k]
        print(f"  k={k:3}: disagreement {dd}/{nn} = {dd/nn:.2f}" if nn else f"  k={k}: n/a")
    print(f"outcome-argmax stability across two k={args.k} runs (eq-class): {stable_eq[0]}/{stable_eq[1]} "
          f"= {stable_eq[0]/max(1,stable_eq[1]):.2f}")
    print(f"mean per-option outcome SE at k={args.k}: {statistics.fmean(ses):.3f}" if ses else "no SE")


if __name__ == "__main__":
    main()
