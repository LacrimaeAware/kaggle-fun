"""Branch A / A4 -- generate the Teacher V2 label artifact.

Samples decisions from the FROZEN snapshot, keeps the high-criticality ones, and runs teacher_api_v2.query_v2
on each (low-noise high-N hand advantage + terminal-outcome auxiliary). Writes the per-decision labels Model B
consumes, and reports how often the outcome signal disagrees with the hand argmax (does the stronger signal
add information?).

    python tools/query_teacher_v2.py --n 8 --out teacher_v2_labels_pilot     # small validation
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import teacher_api_v2 as T2              # noqa: E402
import audit_teacher_stability as A2     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="replays_20260618.json")
    ap.add_argument("--split", default="replays_20260618_split.json")
    ap.add_argument("--n", type=int, default=8, help="number of high-criticality decisions to label")
    ap.add_argument("--candidates", type=int, default=80, help="decision pool to filter for criticality")
    ap.add_argument("--n-determ", type=int, default=32)
    ap.add_argument("--k-outcome", type=int, default=4)
    ap.add_argument("--crit-threshold", type=float, default=0.3)
    ap.add_argument("--out", default="teacher_v2_labels_pilot")
    args = ap.parse_args()

    manifest = json.load(open(ROOT / "data" / "manifests" / args.snapshot, encoding="utf-8"))
    split = json.load(open(ROOT / "data" / "splits" / args.split, encoding="utf-8"))
    cands = A2.sample_decisions(manifest, split, args.candidates, verify=False)
    # rank candidates by criticality, take the most critical
    scored = sorted(cands, key=lambda d: -T2.criticality_score(d["obs"])["score"])
    print(f"[A4] {len(cands)} candidates; labeling up to {args.n} most-critical (threshold {args.crit_threshold})", flush=True)

    labels, t0, disagree, evald = [], time.time(), 0, 0
    for d in scored:
        if evald >= args.n:
            break
        lab = T2.query_v2(d["obs"], d["deck"], n_determ=args.n_determ, k_outcome=args.k_outcome,
                          crit_threshold=args.crit_threshold, seed=1234)
        if not lab.get("evaluated"):
            continue
        evald += 1
        lab["source"] = {"file": d["file"], "step": d["step"], "deck_n": len(d["deck"])}
        labels.append(lab)
        if lab.get("hand_outcome_agree") is False and lab.get("outcome_argmax_option") is not None:
            disagree += 1
        print(f"  [{evald}/{args.n}] crit={lab['criticality']['score']} margin={lab.get('top_two_margin')} "
              f"hand_top_eq={lab.get('hand_argmax_eq_class')} outcome_top_opt={lab.get('outcome_argmax_option')} "
              f"agree={lab.get('hand_outcome_agree')} ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "data" / "manifests" / f"{args.out}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")
    print(f"\n[A4] wrote {len(labels)} Teacher V2 labels -> {out.relative_to(ROOT)}")
    print(f"  outcome disagreed with hand argmax on {disagree}/{evald} labeled decisions "
          f"(higher = the outcome signal adds information beyond the hand leaf)")
    if labels:
        ex = labels[0]
        print("  sample label option fields:", list(ex["options"][0].keys()))


if __name__ == "__main__":
    main()
