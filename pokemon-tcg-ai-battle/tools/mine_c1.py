"""Branch A -- shardable miner for c1 states: high-criticality decisions where agent_search ITSELF selects a
high_regret option. This is B's #1-priority class and is intrinsically rare (top-165 screen found 0 beyond
the seed), so we fan out cheap hand-only screens across a large criticality-sorted candidate pool.

Deterministic sharding: sample_decisions iterates split['train'] in fixed order (no RNG), so we sample one
big pool, sort by criticality, and each shard screens a disjoint [start:start+count] slice. Hand-only
(k_outcome=0) -> ~0.5s/candidate. Writes only the c1 HIT identifiers (full labeling happens later).

    python tools/mine_c1.py --pool 3200 --start 0 --count 270 --out tools/_c1_shard0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import teacher_api_v2 as T2               # noqa: E402
import audit_teacher_stability as A2      # noqa: E402
import label_requested_states as LRS      # noqa: E402

MAN = ROOT / "data" / "manifests"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="replays_20260618.json")
    ap.add_argument("--split", default="replays_20260618_split.json")
    ap.add_argument("--pool", type=int, default=3200)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--high-regret-thresh", type=float, default=5000.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(MAN / args.snapshot, encoding="utf-8"))
    split = json.load(open(ROOT / "data" / "splits" / args.split, encoding="utf-8"))
    pool = A2.sample_decisions(manifest, split, args.pool, verify=False)
    pool = sorted(pool, key=lambda d: -T2.criticality_score(d["obs"])["score"])
    sl = pool[args.start:args.start + args.count]

    hits, screened, t0 = [], 0, time.time()
    for d in sl:
        lab = T2.residual_risk_label(d["obs"], d["deck"], n_strong=32, k_outcome=0,
                                     high_regret_thresh=args.high_regret_thresh, seed=1234)
        if not lab.get("applicable"):
            continue
        screened += 1
        sel = lab.get("search_selected_option")
        so = next((o for o in lab["options"] if o["index"] == sel), None)
        if so and so["high_regret_flag"] == 1:
            hits.append({"file": d["file"], "step": d["step"], "player": d.get("player"),
                         "obs_hash": LRS._hash(d["obs"]), "decision_id": f"{d['file']}:{d['step']}",
                         "sel_regret": so["regret"], "sel_unacc": so["unacceptable_flag"],
                         "search_selected_option": sel, "stronger_argmax_option": lab["stronger_argmax_option"],
                         "criticality": round(lab["criticality"]["score"], 3)})
    json.dump({"start": args.start, "count": args.count, "screened": screened, "hits": hits,
               "cost_s": round(time.time() - t0, 0)}, open(args.out, "w", encoding="utf-8"))
    print(f"shard start={args.start} screened={screened} c1_hits={len(hits)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
