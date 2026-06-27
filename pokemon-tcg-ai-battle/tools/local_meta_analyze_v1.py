"""Local meta analyzer V1 (Model B) -- the REUSABLE trustworthy analysis template.

Consumes any local_meta_harness_v1 / selector_v3_powered_ab run dir (stage_*_summary.json + changed_decisions +
game_summary) and produces the standard report that judges a treatment mode vs a baseline mode the RIGHT way:
  * PRIMARY = deployed+mirror combined (the cells that decide the ladder), each also reported separately;
  * SENTINELS reported individually with Holm multiple-comparison correction;
  * negative controls + weak field reported SEPARATELY and never allowed to decide promotion;
  * Wilson CIs, Fisher exact, two-proportion delta CI, minimum detectable effect, and an EARLY-STOPPING warning
    that shows the interim-look trajectory (the trap that turned a +15pp n=20 smoke into +2.6pp at N500);
  * trigger coverage, override-intensity-by-result (with the game-length-confound caveat), family matrix, examples.

Generic over (baseline_mode, treatment_mode), so it serves every future tactical candidate, not just the selector.

  PYTHONIOENCODING=utf-8 python tools/local_meta_analyze_v1.py --dir <run_dir> \
      --baseline off --treatment selector_v3_transplant \
      --primary deployed,mirror --sentinels denpa92,lucario,koraidon,abomasnow,alakazam --neg first,random
"""
from __future__ import annotations
import argparse
import collections
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

Z = 1.96


def wilson(k, n, z=Z):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def dci(vw, vn, ow, on, z=Z):
    if not vn or not on:
        return [None, None]
    pv, po = vw / vn, ow / on
    se = math.sqrt(pv * (1 - pv) / vn + po * (1 - po) / on)
    d = pv - po
    return [round(100 * (d - z * se), 1), round(100 * (d + z * se), 1)]


def mde_pp(n_arm, p0, z_a=1.96, z_b=0.84):
    if not n_arm:
        return None
    return round(100 * (z_a + z_b) * math.sqrt(2 * p0 * (1 - p0) / n_arm), 1)


def _mode_key(summary, value):
    for k, v in summary.get("modes", {}).items():
        if v == value or k == value:
            return k
    return None


def _cell(results, mk, opps):
    w = sum(results[mk][o]["win"] for o in opps if o in results[mk])
    l = sum(results[mk][o]["loss"] for o in opps if o in results[mk])
    return w, l


def _report_cell(results_pool, bk, tk, opps):
    ow, ol = _cell(results_pool, bk, opps)
    vw, vl = _cell(results_pool, tk, opps)
    if (ow + ol) == 0 or (vw + vl) == 0:
        return None
    op, vp = 100 * ow / (ow + ol), 100 * vw / (vw + vl)
    return {
        "n_per_arm": vw + vl, "baseline": f"{ow}-{ol}", "treatment": f"{vw}-{vl}",
        "baseline_pct": round(op, 1), "treatment_pct": round(vp, 1), "delta_pp": round(vp - op, 1),
        "fisher_p": round(fisher_exact([[vw, vl], [ow, ol]])[1], 4),
        "treatment_wilson_ci": wilson(vw, vw + vl), "delta_ci_pp": dci(vw, vw + vl, ow, ow + ol),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--baseline", default="off")
    ap.add_argument("--treatment", default="selector_v3_transplant")
    ap.add_argument("--primary", default="deployed,mirror")
    ap.add_argument("--sentinels", default="denpa92,alakazam,lucario,koraidon,abomasnow")
    ap.add_argument("--neg", default="first,random")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = Path(args.dir)
    primary = [x for x in args.primary.split(",") if x]
    sentinels = [x for x in args.sentinels.split(",") if x]
    neg = [x for x in args.neg.split(",") if x]

    stages = sorted({p.name.split("_")[1] for p in d.glob("stage_*_summary.json")})
    if not stages:
        raise SystemExit(f"no stage_*_summary.json in {d}")
    summaries = {s: json.load(open(d / f"stage_{s}_summary.json", encoding="utf-8")) for s in stages}
    rows, games = [], []
    for s in stages:
        tf, gf = d / f"stage_{s}_changed_decisions.jsonl", d / f"stage_{s}_game_summary.jsonl"
        if tf.exists():
            rows += [json.loads(l) for l in open(tf, encoding="utf-8")]
        if gf.exists():
            games += [json.loads(l) for l in open(gf, encoding="utf-8")]

    s0 = summaries[stages[0]]
    bk, tk = _mode_key(s0, args.baseline), _mode_key(s0, args.treatment)
    if not bk or not tk:
        raise SystemExit(f"baseline/treatment mode not found in summary modes {s0.get('modes')}")

    # pool all stages
    pooled = {bk: collections.defaultdict(lambda: {"win": 0, "loss": 0}),
              tk: collections.defaultdict(lambda: {"win": 0, "loss": 0})}
    present_opps = set()
    total_err = 0
    for s in stages:
        R = summaries[s]["results"]
        for mk in (bk, tk):
            for o, cell in R.get(mk, {}).items():
                pooled[mk][o]["win"] += cell["win"]
                pooled[mk][o]["loss"] += cell["loss"]
                total_err += cell.get("err", 0)
                present_opps.add(o)
    primary = [o for o in primary if o in present_opps]
    sentinels = [o for o in sentinels if o in present_opps]
    neg = [o for o in neg if o in present_opps]
    field = [o for o in present_opps if o not in primary]

    rep = {"run_dir": str(d), "stages_pooled": stages, "baseline_mode": args.baseline, "treatment_mode": args.treatment,
           "PRIMARY_combined_deployed_mirror": _report_cell(pooled, bk, tk, primary),
           "primary_cells_separate": {o: _report_cell(pooled, bk, tk, [o]) for o in primary},
           "sentinel_cells": {o: _report_cell(pooled, bk, tk, [o]) for o in sentinels},
           "negative_controls": {o: _report_cell(pooled, bk, tk, [o]) for o in neg},
           "weak_field_aggregate_secondary": _report_cell(pooled, bk, tk, field),
           "errors": total_err}

    # Holm correction across primary-combined + sentinels
    tests = [("primary_combined", rep["PRIMARY_combined_deployed_mirror"])]
    tests += [(f"sentinel:{o}", rep["sentinel_cells"][o]) for o in sentinels if rep["sentinel_cells"][o]]
    ps = sorted([(name, c["fisher_p"]) for name, c in tests if c], key=lambda x: x[1])
    m = len(ps)
    rep["holm_adjusted_p"] = {name: round(min(1.0, p * (m - i)), 4) for i, (name, p) in enumerate(ps)}

    # MDE on primary combined
    pc = rep["PRIMARY_combined_deployed_mirror"]
    if pc:
        ow, ol = _cell(pooled, bk, primary)
        rep["primary_MDE_pp_80pct"] = mde_pp(pc["n_per_arm"], ow / (ow + ol))
        rep["primary_significant"] = bool(pc["fisher_p"] < 0.05 and pc["delta_pp"] > 0)

    # early-stopping trajectory (cumulative interim looks on the primary combined)
    traj = []
    cum = []
    for s in stages:
        cum.append(s)
        cp = {bk: collections.defaultdict(lambda: {"win": 0, "loss": 0}),
              tk: collections.defaultdict(lambda: {"win": 0, "loss": 0})}
        for ss in cum:
            R = summaries[ss]["results"]
            for mk in (bk, tk):
                for o, cell in R.get(mk, {}).items():
                    cp[mk][o]["win"] += cell["win"]
                    cp[mk][o]["loss"] += cell["loss"]
        c = _report_cell(cp, bk, tk, primary)
        if c:
            traj.append({"through_stage": "+".join(cum), "n_per_arm": c["n_per_arm"],
                         "delta_pp": c["delta_pp"], "fisher_p": c["fisher_p"]})
    rep["early_stopping_trajectory"] = traj
    crossed = any(t["fisher_p"] < 0.05 for t in traj[:-1])
    final_sig = traj and traj[-1]["fisher_p"] < 0.05
    rep["early_stopping_warning"] = ("an interim look crossed p<0.05 but the full sample did not -- DO NOT trust the "
                                     "interim 'significant' look" if (crossed and not final_sig) else
                                     "no interim-look inflation detected" if traj else "single stage")

    # trigger coverage + override intensity by result (with length-confound caveat) + family matrix
    t_rows = [r for r in rows if r["mode"] == args.treatment]
    t_ov = [r for r in t_rows if not r.get("blocked_terminal") and r.get("selector_raw") != r.get("baseline_raw")]
    t_games = [g for g in games if g["mode"] == args.treatment]
    touched = {g["game_id"] for g in t_games if g.get("overrides", 0) > 0}
    by_res = collections.defaultdict(list)
    for g in t_games:
        by_res[g.get("result")].append(g.get("overrides", 0))
    fam = collections.Counter((r["baseline_family"], r["selector_family"]) for r in t_ov)
    rep["trigger_diagnostics"] = {
        "treatment_games": len(t_games), "games_with_any_trigger": len(touched),
        "fraction_triggered": round(len(touched) / max(1, len(t_games)), 3),
        "applied_overrides": len(t_ov),
        "terminal_overrides": sum(1 for r in t_ov if r.get("selector_family") in {"ATTACK", "END", "RETREAT"}),
        "mean_overrides_by_result": {k: round(sum(v) / len(v), 2) for k, v in by_res.items() if v},
        "intensity_caveat": "overrides vs result is within-arm/observational (confounded by game length); trust the randomized cell delta, not this.",
        "family_transition_top": {f"{a}->{b}": c for (a, b), c in sorted(fam.items(), key=lambda x: -x[1])[:10]},
    }
    rep["examples"] = [{"game_id": r["game_id"], "matchup": r["matchup"], "result": r.get("game_result"),
                        "from": r["baseline_family"], "to": r["selector_family"],
                        "key": r.get("transplant_lookup_key")} for r in t_ov[:15]]
    rep["READING"] = ("Judge on PRIMARY combined (deployed+mirror), Holm-corrected sentinels next; weak field + "
                      "negative controls are secondary and never decide promotion. A positive PRIMARY point "
                      "estimate that is not significant / below MDE is 'positive point estimate, underpowered / "
                      "not established, do not promote'. Local self-play does not predict the ladder.")

    outp = Path(args.out) if args.out else (d / "analysis_report.json")
    outp.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"PRIMARY": rep["PRIMARY_combined_deployed_mirror"], "MDE": rep.get("primary_MDE_pp_80pct"),
                      "significant": rep.get("primary_significant"), "early_stopping": rep["early_stopping_warning"],
                      "errors": total_err, "wrote": str(outp)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
