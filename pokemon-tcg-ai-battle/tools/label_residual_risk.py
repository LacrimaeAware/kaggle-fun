"""Branch A -- residual/risk Teacher V2 labels (the narrowed target).

Per high-criticality decision + legal sibling: current search value (live N=8, paired prefix), stronger
value (N=32), delta_to_search (+ within-decision normalized), regret + high-regret flag, unacceptable flag,
variance/SE, and the terminal-outcome winrate as a separate auxiliary. Self-contained (obs + legal_options).

Prioritizes the most useful states: Model B's failure states (where old ranker/option-0 beat the model),
then high-criticality snapshot decisions. No live agent, no screen, no generic hand-only batch.

    python tools/label_residual_risk.py --n 50 \
        --b-request <abs path to teacher_v2_label_request_for_A.json> \
        --out teacher_v2_residual_risk_labels
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import teacher_api_v2 as T2               # noqa: E402
import audit_teacher_stability as A2      # noqa: E402

MAN = ROOT / "data" / "manifests"


def _hash(obs):
    return hashlib.sha1(json.dumps(obs, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _b_states(path):
    if not path or not Path(path).is_file():
        return []
    raw = json.load(open(path, encoding="utf-8"))
    entries = raw["requests"] if isinstance(raw, dict) and "requests" in raw else raw
    out = []
    for e in entries:
        obs = e.get("observation")
        if not obs or not e.get("deck"):
            continue
        src = e.get("source") or {}
        out.append({"file": src.get("file"), "step": src.get("step"), "player": src.get("player"),
                    "obs": obs, "deck": e["deck"], "decision_id": e.get("decision_id"),
                    "tag": "B_failure_state"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="replays_20260618.json")
    ap.add_argument("--split", default="replays_20260618_split.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--candidates", type=int, default=220)
    ap.add_argument("--n-strong", type=int, default=32)
    ap.add_argument("--k-outcome", type=int, default=16)
    ap.add_argument("--high-regret-thresh", type=float, default=5000.0)
    ap.add_argument("--b-request", default="")
    ap.add_argument("--out", default="teacher_v2_residual_risk_labels")
    args = ap.parse_args()

    decs = _b_states(args.b_request)
    n_b = len(decs)
    if len(decs) < args.n:
        manifest = json.load(open(MAN / args.snapshot, encoding="utf-8"))
        split = json.load(open(ROOT / "data" / "splits" / args.split, encoding="utf-8"))
        cands = A2.sample_decisions(manifest, split, args.candidates, verify=False)
        cands = sorted(cands, key=lambda d: -T2.criticality_score(d["obs"])["score"])
        for d in cands:
            if len(decs) >= args.n:
                break
            d["tag"] = "high_criticality"
            decs.append(d)
    print(f"[resid/risk] {n_b} B-failure states + {len(decs)-n_b} high-criticality = {len(decs)} decisions", flush=True)

    labels, t0 = [], time.time()
    for d in decs:
        ts = time.time()
        lab = T2.residual_risk_label(d["obs"], d["deck"], n_strong=args.n_strong, k_outcome=args.k_outcome,
                                     high_regret_thresh=args.high_regret_thresh, seed=1234)
        if not lab.get("applicable"):
            continue
        opts = lab["options"]
        all_done = int(all(o["completed_determinizations"] >= args.n_strong and o["outcome_playouts"] >= args.k_outcome
                           for o in opts))
        lab.update({
            "decision_id": d.get("decision_id") or f"{d.get('file')}:{d.get('step')}",
            "obs_hash": _hash(d["obs"]), "observation": d["obs"],
            "legal_options": (d["obs"].get("select") or {}).get("option") or [],
            "source": {"file": d.get("file"), "step": d.get("step"), "player": d.get("player")},
            "state_tag": d.get("tag"), "coverage": {"all_siblings_completed": all_done},
            "timing": {"label_time_s": round(time.time() - ts, 2)},
        })
        labels.append(lab)
        if len(labels) % 10 == 0:
            print(f"  labeled {len(labels)}/{len(decs)} ({time.time()-t0:.0f}s)", flush=True)

    out = MAN / f"{args.out}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")

    # ---- summary ----
    deltas = [o["delta_to_search"] for r in labels for o in r["options"]]
    hr = sum(o["high_regret_flag"] for r in labels for o in r["options"])
    unacc = sum(o["unacceptable_flag"] for r in labels for o in r["options"])
    n_opt = sum(len(r["options"]) for r in labels)
    v_se = [o["value_se"] for r in labels for o in r["options"]]
    o_se = [o["outcome_se"] for r in labels for o in r["options"] if o["outcome_se"] is not None]

    def _ho_disagree(r):
        ow = [(o["index"], o["outcome_winrate"]) for o in r["options"] if o["outcome_winrate"] is not None]
        if not ow:
            return None
        out_best = max(ow, key=lambda x: x[1])[0]
        return int(out_best != r["stronger_argmax_option"])
    hod = [x for x in (_ho_disagree(r) for r in labels) if x is not None]
    full = sum(r["coverage"]["all_siblings_completed"] for r in labels)

    def pct(xs, q):
        xs = sorted(xs); return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None
    summary = {
        "n_decisions": len(labels), "n_options": n_opt, "n_b_failure_states": n_b,
        "residual_delta": {"mean": round(statistics.fmean(deltas), 1) if deltas else None,
                           "stdev": round(statistics.pstdev(deltas), 1) if len(deltas) > 1 else None,
                           "p05": pct(deltas, 0.05), "p50": pct(deltas, 0.5), "p95": pct(deltas, 0.95),
                           "abs_mean": round(statistics.fmean(abs(x) for x in deltas), 1) if deltas else None},
        "high_regret_thresh": args.high_regret_thresh,
        "high_regret_options": hr, "unacceptable_options": unacc,
        "hand_outcome_argmax_disagreement": f"{sum(hod)}/{len(hod)}" if hod else "n/a",
        "mean_value_se": round(statistics.fmean(v_se), 2) if v_se else None,
        "mean_outcome_se": round(statistics.fmean(o_se), 3) if o_se else None,
        "all_siblings_completed": f"{full}/{len(labels)}",
        "cost_per_decision_s": round((time.time() - t0) / max(1, len(labels)), 1),
    }
    json.dump(summary, open(MAN / f"{args.out}_summary.json", "w", encoding="utf-8"), indent=1)
    print("\n=== RESIDUAL/RISK SUMMARY ===")
    print(json.dumps(summary, indent=1))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
