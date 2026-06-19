"""Branch A -- targeted residual/risk ENRICHMENT for Model B's risk-only model (round 2).

B's risk model missed the case where agent_search ITSELF selects a high-regret option, and false-blocked a
safe search pick. So this densifies those classes. Two-phase to keep it cheap:
  Phase 1 (cheap, hand-only k_outcome=0): screen many high-criticality candidates, classify by B's criteria.
  Phase 2 (full, k_outcome=16 + outcome playouts): label the 2 seed states + the kept criterion matches.

Criteria (B):
  c1 search_selected_high_regret : the agent_search-selected option has high_regret_flag=1   [priority]
  c2 safe_search_false_positive  : search picks a non-high-regret option but some sibling is unacceptable
  c3 near_miss_boundary          : mixed high_regret among siblings, or large |delta_to_search|

    python tools/label_risk_round2.py --request <abs path> --target 50 --scan 240
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
import label_requested_states as LRS      # noqa: E402  (reuse _recover_from_source / _hash)

MAN = ROOT / "data" / "manifests"
BOUNDARY_DELTA = 2000.0


def classify(lab):
    opts = {o["index"]: o for o in lab["options"]}
    sel = lab.get("search_selected_option")
    so = opts.get(sel)
    tags = []
    if so and so["high_regret_flag"] == 1:
        tags.append("c1_search_selected_high_regret")
    if so and so["high_regret_flag"] == 0 and any(o["unacceptable_flag"] == 1 for o in lab["options"] if o["index"] != sel):
        tags.append("c2_safe_search_false_positive")
    hr = [o["high_regret_flag"] for o in lab["options"]]
    dmax = max((abs(o["delta_to_search"]) for o in lab["options"]), default=0)
    if (1 in hr and 0 in hr) or dmax > BOUNDARY_DELTA:
        tags.append("c3_near_miss_boundary")
    return tags


def full_label(obs, deck, k, hrt, seed=1234):
    return T2.residual_risk_label(obs, deck, n_strong=32, k_outcome=k, high_regret_thresh=hrt, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--scan", type=int, default=240)
    ap.add_argument("--k-outcome", type=int, default=16)
    ap.add_argument("--high-regret-thresh", type=float, default=5000.0)
    ap.add_argument("--c1-hits", default="", help="merged mine_c1 hits json (densifies the c1 class)")
    ap.add_argument("--out", default="teacher_v2_residual_risk_labels_round2")
    args = ap.parse_args()

    raw = json.load(open(args.request, encoding="utf-8"))
    seeds = raw.get("seed_examples", [])
    cache, labels, failed = {}, [], []

    def attach(lab, obs, deck_src, src, tag, did=None, timing_s=None):
        lab["decision_id"] = did or f"{src.get('file')}:{src.get('step')}"
        lab["obs_hash"] = LRS._hash(obs)
        lab["observation"] = obs
        lab["legal_options"] = (obs.get("select") or {}).get("option") or []
        lab["source"] = src
        lab["deck_source"] = deck_src
        lab["criterion_tags"] = tag
        lab["timing"] = {"label_time_s": round(timing_s, 2) if timing_s is not None else None,
                         "k_outcome": args.k_outcome, "n_strong": 32}
        lab["coverage"] = {"all_siblings_completed": int(all(
            o["completed_determinizations"] >= 32 and (o["outcome_playouts"] or 0) >= args.k_outcome for o in lab["options"]))}

    def timed_full(obs, deck):
        ts = time.time()
        return full_label(obs, deck, args.k_outcome, args.high_regret_thresh), round(time.time() - ts, 2)

    # Phase 2a: the 2 seed states (recover from source, full label, always include)
    t0 = time.time()
    for s in seeds:
        src = s.get("source") or {}
        obs, deck, _ = LRS._recover_from_source(src, s.get("obs_hash"), cache)
        if obs is None or not deck:
            failed.append({"seed": s.get("decision_id"), "reason": "obs/deck not recoverable"})
            continue
        lab, dt = timed_full(obs, deck)
        if not lab.get("applicable"):
            failed.append({"seed": s.get("decision_id"), "reason": "not applicable"})
            continue
        attach(lab, obs, "source", src, ["SEED"] + classify(lab), s.get("decision_id"), timing_s=dt)
        lab["seed_reason"] = s.get("why_requested")
        labels.append(lab)
        print(f"  SEED {lab['decision_id']} sel_opt={lab['search_selected_option']} tags={lab['criterion_tags']}", flush=True)

    # Phase 2a': mined c1 states (densify B's #1-priority class), recovered from source, full label
    seen = {lab["decision_id"] for lab in labels}
    if args.c1_hits and Path(args.c1_hits).is_file():
        raw_hits = json.load(open(args.c1_hits, encoding="utf-8"))
        c1_hits = raw_hits.get("hits", raw_hits) if isinstance(raw_hits, dict) else raw_hits
        # dedup, keep strongest-regret first
        uniq = {}
        for h in c1_hits:
            uniq.setdefault(h["decision_id"], h)
        ordered = sorted(uniq.values(), key=lambda h: -abs(h.get("sel_regret", 0)))
        for h in ordered:
            if h["decision_id"] in seen:
                continue
            obs, deck, _ = LRS._recover_from_source(h.get("source") or h, h.get("obs_hash"), cache)
            if obs is None or not deck:
                failed.append({"mined_c1": h["decision_id"], "reason": "obs/deck not recoverable"})
                continue
            lab, dt = timed_full(obs, deck)
            if not lab.get("applicable"):
                failed.append({"mined_c1": h["decision_id"], "reason": "not applicable"})
                continue
            attach(lab, obs, "source", {"file": h.get("file"), "step": h.get("step"), "player": h.get("player")},
                   ["MINED_c1"] + classify(lab), h["decision_id"], timing_s=dt)
            seen.add(h["decision_id"])
            labels.append(lab)
        print(f"  mined-c1 ingested: {sum(1 for r in labels if 'MINED_c1' in r['criterion_tags'])} "
              f"(of {len(uniq)} unique hits)", flush=True)

    # Phase 1: cheap hand-only screen to FIND criterion matches
    manifest = json.load(open(MAN / "replays_20260618.json", encoding="utf-8"))
    split = json.load(open(ROOT / "data" / "splits" / "replays_20260618_split.json", encoding="utf-8"))
    cands = A2.sample_decisions(manifest, split, args.scan, verify=False)
    cands = sorted(cands, key=lambda d: -T2.criticality_score(d["obs"])["score"])
    buckets = {"c1_search_selected_high_regret": [], "c2_safe_search_false_positive": [], "c3_near_miss_boundary": []}
    screened = 0
    for d in cands:
        if sum(len(v) for v in buckets.values()) >= args.target * 2:
            break
        hl = full_label(d["obs"], d["deck"], 0, args.high_regret_thresh)   # k_outcome=0 -> hand-only, cheap
        if not hl.get("applicable"):
            continue
        screened += 1
        for t in classify(hl):
            buckets[t].append(d)
        if screened % 40 == 0:
            print(f"  screened {screened} (c1={len(buckets['c1_search_selected_high_regret'])} "
                  f"c2={len(buckets['c2_safe_search_false_positive'])} c3={len(buckets['c3_near_miss_boundary'])}) "
                  f"{time.time()-t0:.0f}s", flush=True)

    # Phase 2b: full-label kept matches, priority c1 > c2 > c3, dedup, up to target
    for tag in ("c1_search_selected_high_regret", "c2_safe_search_false_positive", "c3_near_miss_boundary"):
        for d in buckets[tag]:
            if len(labels) >= args.target:
                break
            did = f"{d['file']}:{d['step']}"
            if did in seen:
                continue
            lab, dt = timed_full(d["obs"], d["deck"])
            if not lab.get("applicable"):
                continue
            attach(lab, d["obs"], "source", {"file": d["file"], "step": d["step"], "player": d.get("player")},
                   classify(lab), did, timing_s=dt)
            seen.add(did)
            labels.append(lab)
        if len(labels) >= args.target:
            break

    out = MAN / f"{args.out}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")

    # class balance / report
    sel_hr = sum(1 for r in labels if (next((o for o in r["options"] if o["index"] == r["search_selected_option"]), {}) or {}).get("high_regret_flag") == 1)
    hr_opts = sum(o["high_regret_flag"] for r in labels for o in r["options"])
    unacc = sum(o["unacceptable_flag"] for r in labels for o in r["options"])
    n_opt = sum(len(r["options"]) for r in labels)
    tagc = {}
    for r in labels:
        for t in r.get("criterion_tags", []):
            tagc[t] = tagc.get(t, 0) + 1
    mined_c1 = sum(1 for r in labels if "MINED_c1" in r.get("criterion_tags", []))
    summary = {"requested_count": raw.get("requested_state_count"), "labeled": len(labels), "seeds": len(seeds),
               "mined_c1": mined_c1, "failed": failed, "screened_candidates": screened, "n_options": n_opt,
               "search_selected_high_regret_states": sel_hr, "high_regret_options": hr_opts,
               "unacceptable_options": unacc, "criterion_tag_counts": tagc,
               "high_regret_option_rate": round(hr_opts / max(1, n_opt), 3),
               "cost_s": round(time.time() - t0, 0)}
    json.dump(summary, open(MAN / f"{args.out}_summary.json", "w", encoding="utf-8"), indent=1)
    print("\n=== ROUND-2 RISK ENRICHMENT SUMMARY ===")
    print(json.dumps(summary, indent=1))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
