"""Continuous Terrain V1 -- A4 repeated-label measurement (shardable).

For each selected root decision: REPEAT the teacher measurement to capture distributions, not a single hard
label. Live search (N=8) x live_reps and stronger teacher (N=32) x strong_reps; within each run siblings
share paired hidden worlds (T1._per_world_values). Records per-option value distributions + acceptable /
high-regret / unacceptable PROBABILITIES across repeats, and per-decision live selected-action distribution /
entropy / modal stability. Attaches the A5 semantic vector. Self-contained for Model B.

    python tools/label_terrain_v1.py --manifest data/manifests/_terrain_selection.json --start 0 --count 60 --out tools/_terrain_lab0.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import teacher_api_v1 as T1                # noqa: E402
import teacher_api_v2 as T2                # noqa: E402
import main as M                            # noqa: E402
import search as S                          # noqa: E402
import features as FT                       # noqa: E402
import state_action_schema_v2 as SCH        # noqa: E402
import action_semantics_v1 as AS            # noqa: E402
import label_requested_states as LRS        # noqa: E402

MAN = ROOT / "data" / "manifests"


def _means(pw):
    return [statistics.fmean(v) if v else None for v in pw]


def repeated_label(obs, deck, *, live_n=8, strong_n=32, live_reps=8, strong_reps=4,
                   hrt=5000.0, accept_z=1.0, hand_budget=8.0):
    """Repeated live + stronger measurements with paired worlds. Returns the per-option + per-decision
    distribution record, or None if not applicable."""
    if not SCH.is_single_pick_decision(obs):
        return None
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    opts = obs["select"]["option"]
    n = len(opts)
    try:
        fm = M._forced_move(obs)
        forced = fm[0] if fm else None
    except Exception:
        forced = None

    live_runs, live_sel = [], []
    for _ in range(live_reps):
        pw = T1._per_world_values(obs, deck, live_n, hand_budget, "hand")
        if not pw:
            continue
        mm = _means(pw)
        live_runs.append(mm)
        valued = [i for i, m in enumerate(mm) if m is not None]
        if valued:
            live_sel.append(forced if forced is not None else max(valued, key=lambda i: mm[i]))
    strong_runs, strong_compl = [], []
    for _ in range(strong_reps):
        pw = T1._per_world_values(obs, deck, strong_n, hand_budget, "hand")
        if not pw:
            continue
        strong_runs.append(_means(pw))
        strong_compl.append([len(v) for v in pw])
    if not live_runs or not strong_runs:
        return None

    def col(runs, i):
        return [r[i] for r in runs if i < len(r) and r[i] is not None]

    # per-strong-run HIGH-REGRET flag (regret vs that run's best > hrt). unacceptable is a DISTINCT
    # CI-overlap criterion (regret beyond noise) computed post-aggregation below -- NOT a copy of high_regret.
    hr_cnt = [0] * n
    for r in strong_runs:
        valued = [i for i, m in enumerate(r) if m is not None]
        if not valued:
            continue
        best = max(r[i] for i in valued)
        for i in valued:
            if best - r[i] > hrt:
                hr_cnt[i] += 1
    # acceptable: within hrt of the run-best (a softer band than high-regret)
    acc_band = [0] * n
    for r in strong_runs:
        valued = [i for i, m in enumerate(r) if m is not None]
        if not valued:
            continue
        best = max(r[i] for i in valued)
        for i in valued:
            if best - r[i] <= hrt:
                acc_band[i] += 1

    ns = len(strong_runs)
    nl = len(live_runs)
    # per-option aggregation
    valued_all = [i for i in range(n) if col(strong_runs, i)]
    strong_mean = {i: statistics.fmean(col(strong_runs, i)) for i in valued_all}
    best_i = max(valued_all, key=lambda i: strong_mean[i]) if valued_all else None
    best_strong = strong_mean[best_i] if best_i is not None else 0.0
    se_by_i = {i: (statistics.pvariance(col(strong_runs, i)) / max(1, ns)) ** 0.5 if len(col(strong_runs, i)) > 1 else 0.0
               for i in valued_all}
    best_se = se_by_i.get(best_i, 0.0)
    mean_delta = statistics.fmean(
        statistics.fmean(col(strong_runs, i)) - statistics.fmean(col(live_runs, i))
        for i in valued_all if col(live_runs, i)) if valued_all else 0.0

    options = []
    for i in valued_all:
        lc, scv = col(live_runs, i), col(strong_runs, i)
        ml = statistics.fmean(lc) if lc else None
        ms = statistics.fmean(scv)
        lvar = statistics.pvariance(lc) if len(lc) > 1 else 0.0
        svar = statistics.pvariance(scv) if len(scv) > 1 else 0.0
        regret = best_strong - ms
        eqs = SCH.equivalence_classes(opts, cur, me)
        deltas_i = None  # filled below in batch
        options.append({
            "index": i,
            "semantic_action_key": list(SCH.semantic_action_key(opts[i], cur, me)),
            "eq_class": eqs[i],
            "mean_live_value": round(ml, 2) if ml is not None else None,
            "live_value_variance": round(lvar, 2),
            "mean_stronger_value": round(ms, 2),
            "stronger_value_variance": round(svar, 2),
            "delta_to_search": round(ms - ml, 2) if ml is not None else None,
            "delta_to_search_norm": round((ms - ml) - mean_delta, 2) if ml is not None else None,
            "hand_norm_advantage": round(ms - best_strong, 2),
            "regret": round(regret, 2),
            "value_se": round((svar / max(1, ns)) ** 0.5, 3),
            "acceptable_prob": round(acc_band[i] / ns, 3),
            "high_regret_prob": round(hr_cnt[i] / ns, 3),
            # DISTINCT from high_regret: CI-overlap criterion -- regret beyond the paired noise band
            # (not statistically tied with the best). accept_z=1.0.
            "unacceptable_prob": float(int(regret > 1.0 * (best_se + se_by_i.get(i, 0.0)))),
            "completed_determinizations": int(statistics.fmean([c[i] for c in strong_compl if i < len(c)])) if strong_compl else strong_n,
        })

    # forward-model option deltas (one extra cheap pass) -> attach to semantics
    feats = FT.encode_state(obs)
    deltas = S.option_deltas(obs, deck) or [None] * n
    for o in options:
        i = o["index"]
        o["semantic_vector"] = AS.semantic_vector(obs, i, deltas[i] if i < len(deltas) else None, feats, cur, me)

    # per-decision distributions
    from collections import Counter
    seldist = Counter(live_sel)
    tot = sum(seldist.values()) or 1
    sel_dist = {str(k): round(v / tot, 3) for k, v in seldist.items()}
    ent = -sum((v / tot) * math.log(v / tot + 1e-12) for v in seldist.values()) if seldist else 0.0
    modal = max(seldist.values()) / tot if seldist else 0.0
    # stronger soft policy = softmax over hand_norm_advantage (temperature on hand-eval scale)
    adv = {o["index"]: o["hand_norm_advantage"] for o in options}
    T = 2000.0
    ex = {k: math.exp(max(-50, v / T)) for k, v in adv.items()}
    Z = sum(ex.values()) or 1.0
    soft = {str(k): round(v / Z, 3) for k, v in ex.items()}
    spread = (max(adv.values()) - min(adv.values())) if adv else 0.0
    crit = T2.criticality_score(obs)

    return {
        "me": me, "n_options": n, "forced_action_flag": forced is not None,
        "search_selected_option": forced if forced is not None else (live_sel[0] if live_sel else best_i),
        "stronger_argmax_option": best_i, "criticality": crit,
        "live_selected_distribution": sel_dist, "live_action_entropy": round(ent, 3),
        "modal_action_stability": round(modal, 3), "stronger_soft_policy": soft,
        "value_spread": round(spread, 2), "high_regret_thresh": hrt,
        "live_repeats": nl, "strong_repeats": ns, "options": options,
        "all_siblings_completed": int(all(o["completed_determinizations"] >= strong_n for o in options)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10 ** 9)
    ap.add_argument("--live-repeats", type=int, default=8)
    ap.add_argument("--strong-repeats", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sel = json.load(open(args.manifest, encoding="utf-8"))
    rows = sel["rows"] if isinstance(sel, dict) else sel
    rows = rows[args.start:args.start + args.count]
    cache, out, failed, t0 = {}, [], [], time.time()
    for r in rows:
        src = r.get("source") or {"file": r.get("group_id"), "step": r.get("step"), "player": r.get("player")}
        obs, deck, _ = LRS._recover_from_source(src, r.get("obs_hash"), cache)
        if obs is None or not deck:
            failed.append({"decision_id": r.get("decision_id"), "reason": "unrecoverable"})
            continue
        ts = time.time()
        lab = repeated_label(obs, deck, live_reps=args.live_repeats, strong_reps=args.strong_repeats)
        if not lab:
            failed.append({"decision_id": r.get("decision_id"), "reason": "not_applicable"})
            continue
        lab.update({
            "decision_id": r.get("decision_id"), "group_id": r.get("group_id"),
            "obs_hash": LRS._hash(obs), "observation": obs,
            "legal_options": (obs.get("select") or {}).get("option") or [],
            "source": {"file": src.get("file"), "step": src.get("step"), "player": src.get("player")},
            "terrain_class": r.get("terrain_class"), "ring": r.get("ring"), "anchor_id": r.get("anchor_id"),
            "eval_only": bool(r.get("eval_only")),
            "timing": {"label_time_s": round(time.time() - ts, 2),
                       "live_repeats": args.live_repeats, "strong_repeats": args.strong_repeats},
            "coverage": {"all_siblings_completed": lab["all_siblings_completed"]},
        })
        out.append(lab)
        if len(out) % 10 == 0:
            print(f"  labeled {len(out)}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"labeled": len(out), "failed": failed, "records": out, "cost_s": round(time.time() - t0, 0)},
              open(args.out, "w", encoding="utf-8"))
    print(f"shard start={args.start} labeled={len(out)} failed={len(failed)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
