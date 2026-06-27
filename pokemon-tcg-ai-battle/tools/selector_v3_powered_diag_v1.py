"""Diagnostic for the powered/sequential V3 A/B. Pools stage A/B/C traces + summaries. Primary metric is the
deployed+mirror combined win-rate delta at full power (n=500/cell); interim looks are reported only to expose the
early-stopping bias, NOT as significance claims. Also: override-efficacy with a game-LENGTH confound check, family
transitions, top lookup keys by win/loss split, sentinel + field. Honest about the effect shrinking under power.

  PYTHONIOENCODING=utf-8 python tools/selector_v3_powered_diag_v1.py
"""
from __future__ import annotations
import collections
import html
import json
import math
import statistics
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

OUT = Path(__file__).resolve().parent.parent / "data" / "generated" / "starmie_selector_v3_powered_ab"
STAGES = ["A", "B", "C"]
PRIMARY = ["deployed", "mirror"]
TERMINAL = {"ATTACK", "END", "RETREAT"}


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def dci(vw, vn, ow, on, z=1.96):
    pv, po = vw / vn, ow / on
    se = math.sqrt(pv * (1 - pv) / vn + po * (1 - po) / on)
    d = pv - po
    return (round(100 * (d - z * se), 1), round(100 * (d + z * se), 1))


def main() -> int:
    summaries = {s: json.load(open(OUT / f"stage_{s}_summary.json", encoding="utf-8"))["results"] for s in STAGES}
    rows, games = [], []
    for s in STAGES:
        rows += [json.loads(l) for l in open(OUT / f"stage_{s}_changed_decisions.jsonl", encoding="utf-8")]
        games += [json.loads(l) for l in open(OUT / f"stage_{s}_game_summary.jsonl", encoding="utf-8")]

    def pool(mode, opp):
        return (sum(summaries[s][mode][opp]["win"] for s in STAGES),
                sum(summaries[s][mode][opp]["loss"] for s in STAGES))

    def cell(mode, opps):
        return (sum(pool(mode, o)[0] for o in opps), sum(pool(mode, o)[1] for o in opps))

    per_opp = {}
    for o in ["deployed", "mirror", "denpa92"]:
        ow, ol = pool("S0", o)
        vw, vl = pool("S3", o)
        per_opp[o] = {
            "n_per_arm": vw + vl, "off": f"{ow}-{ol}", "v3": f"{vw}-{vl}",
            "off_pct": round(100 * ow / (ow + ol), 1), "v3_pct": round(100 * vw / (vw + vl), 1),
            "delta_pp": round(100 * vw / (vw + vl) - 100 * ow / (ow + ol), 1),
            "fisher_p": round(fisher_exact([[vw, vl], [ow, ol]])[1], 3),
            "v3_wilson_ci": wilson(vw, vw + vl), "delta_ci_pp": dci(vw, vw + vl, ow, ow + ol),
        }
    ow, ol = cell("S0", PRIMARY)
    vw, vl = cell("S3", PRIMARY)
    combined = {
        "n_per_arm": vw + vl, "off": f"{ow}-{ol}", "v3": f"{vw}-{vl}",
        "off_pct": round(100 * ow / (ow + ol), 1), "v3_pct": round(100 * vw / (vw + vl), 1),
        "delta_pp": round(100 * vw / (vw + vl) - 100 * ow / (ow + ol), 1),
        "fisher_p": round(fisher_exact([[vw, vl], [ow, ol]])[1], 4),
        "delta_ci_pp": dci(vw, vw + vl, ow, ow + ol),
    }

    # interim looks -> early-stopping bias (combined primary delta at each cumulative sample)
    def look(stage_set):
        ow = sum(summaries[s]["S0"][o]["win"] for s in stage_set for o in PRIMARY)
        ol = sum(summaries[s]["S0"][o]["loss"] for s in stage_set for o in PRIMARY)
        vw = sum(summaries[s]["S3"][o]["win"] for s in stage_set for o in PRIMARY)
        vl = sum(summaries[s]["S3"][o]["loss"] for s in stage_set for o in PRIMARY)
        return {"n_per_arm": vw + vl, "delta_pp": round(100 * vw / (vw + vl) - 100 * ow / (ow + ol), 1),
                "fisher_p": round(fisher_exact([[vw, vl], [ow, ol]])[1], 3)}
    interim = {"A": look(["A"]), "A+B": look(["A", "B"]), "A+B+C(full)": look(["A", "B", "C"])}

    # safety + non-inert (pooled trace)
    v3_rows = [r for r in rows if r["mode"] == "selector_v3_transplant"]
    v3_ov = [r for r in v3_rows if not r.get("blocked_terminal") and r.get("selector_raw") != r.get("baseline_raw")]
    term = sum(1 for r in v3_ov if r.get("selector_family") in TERMINAL)
    table_hits = sum(1 for r in v3_ov if r.get("transplant_table_hit"))
    total_err = sum(summaries[s][m][o]["err"] for s in STAGES for m in ("S0", "S3")
                    for o in ("deployed", "mirror", "denpa92"))

    # override efficacy + LENGTH-CONFOUND check (overrides per game, and per decision-length proxy = max step)
    v3_games = [g for g in games if g["mode"] == "selector_v3_transplant"]
    steps_by_game = collections.defaultdict(int)
    for r in v3_rows:
        steps_by_game[r["game_id"]] = max(steps_by_game[r["game_id"]], int(r.get("step") or 0))
    win_c = [g["overrides"] for g in v3_games if g["result"] == "win"]
    loss_c = [g["overrides"] for g in v3_games if g["result"] == "loss"]
    win_len = [steps_by_game[g["game_id"]] for g in v3_games if g["result"] == "win"]
    loss_len = [steps_by_game[g["game_id"]] for g in v3_games if g["result"] == "loss"]
    win_rate = [g["overrides"] / steps_by_game[g["game_id"]] for g in v3_games
                if g["result"] == "win" and steps_by_game[g["game_id"]]]
    loss_rate = [g["overrides"] / steps_by_game[g["game_id"]] for g in v3_games
                 if g["result"] == "loss" and steps_by_game[g["game_id"]]]
    rate_mw_p = round(mannwhitneyu(loss_rate, win_rate, alternative="greater")[1], 3)
    efficacy = {
        "mean_overrides_win": round(statistics.mean(win_c), 2), "mean_overrides_loss": round(statistics.mean(loss_c), 2),
        "mean_len_proxy_win": round(statistics.mean(win_len), 1), "mean_len_proxy_loss": round(statistics.mean(loss_len), 1),
        "override_RATE_win": round(statistics.mean(win_rate), 3), "override_RATE_loss": round(statistics.mean(loss_rate), 3),
        "override_RATE_mannwhitney_p_loss_gt_win": rate_mw_p,
        "reading": ("COUNT differs (9.7 vs 10.1) mostly because losses run longer (len proxy 34.7 vs 35.6; "
                    "corr(overrides,length)~0.91). The RATE is slightly higher in losses (Mann-Whitney p~0.01), so a "
                    "small residual remains -- BUT any within-arm override-vs-result split is observational and "
                    "confounded (overrides and outcome are both downstream of board state). The clean CAUSAL estimate "
                    "is the randomized BETWEEN-ARM delta (+2.6pp ITT), not this within-arm slice. Trust the arm delta."),
    }
    # multiple-comparison correction across the 4 cells (deployed/mirror/denpa92/combined)
    raw_ps = sorted([("deployed", per_opp["deployed"]["fisher_p"]), ("mirror", per_opp["mirror"]["fisher_p"]),
                     ("denpa92", per_opp["denpa92"]["fisher_p"]), ("combined", combined["fisher_p"])], key=lambda x: x[1])
    m = len(raw_ps)
    holm = {name: round(min(1.0, p * (m - i)), 3) for i, (name, p) in enumerate(raw_ps)}
    # minimum detectable effect at this n (two-proportion, 80% power, alpha 0.05, around off ~47.5%)
    n_arm = combined["n_per_arm"]
    p0 = ow / (ow + ol)
    mde_pp = round(100 * (1.96 + 0.84) * math.sqrt(2 * p0 * (1 - p0) / n_arm), 1)

    # family transitions + top lookup keys by win/loss
    fam = collections.Counter((r["baseline_family"], r["selector_family"]) for r in v3_ov)
    key_wl = collections.defaultdict(lambda: [0, 0])
    for r in v3_ov:
        key_wl[r.get("transplant_lookup_key")][0 if r.get("game_result") == "win" else 1] += 1
    top_keys = sorted(key_wl.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:15]

    if combined["fisher_p"] < 0.05 and combined["delta_pp"] > 0:
        verdict = "A_V3_POWERED_POSITIVE"
    elif total_err or term:
        verdict = "D_V3_UNSAFE_OR_INVALID"
    elif combined["delta_pp"] <= -5:
        verdict = "C_V3_POWERED_REGRESSIVE"
    else:
        verdict = "B_V3_POWERED_NEUTRAL"

    report = {
        "DIAGNOSTIC_VERDICT": verdict,
        "PROMOTION_STATUS": "DO_NOT_SUBMIT / PARK_V3",
        "primary_metric_combined_deployed_mirror": combined,
        "per_opponent": per_opp,
        "all_opponent_incl_sentinel": {
            "off_pct": round(100 * (cell("S0", PRIMARY)[0] + pool("S0", "denpa92")[0]) /
                             (sum(cell("S0", PRIMARY)) + sum(pool("S0", "denpa92"))), 1),
            "v3_pct": round(100 * (cell("S3", PRIMARY)[0] + pool("S3", "denpa92")[0]) /
                            (sum(cell("S3", PRIMARY)) + sum(pool("S3", "denpa92"))), 1),
        },
        "interim_looks_early_stopping_bias": interim,
        "early_stopping_note": ("Effect shrank monotonically as power rose: +15.0pp (n=20) -> +11.7 (A) -> +7.5 "
                                "(A+B, p=0.040) -> +2.6 (full, p=0.263). The A+B look crossed p<0.05; the full sample "
                                "did not. Only the full pooled n=1000 is a valid significance claim. Stopping at the "
                                "first significant interim look would have been a false positive."),
        "safety": {"errors": total_err, "terminal_overrides": term, "applied_overrides": len(v3_ov),
                   "table_hit_overrides": table_hits, "total_changed_rows": len(v3_rows),
                   "blocked_or_veto_rows": len(v3_rows) - len(v3_ov),
                   "note": "applied_overrides counts source=selector picks; the extra changed rows are blocked-terminal/veto fallbacks (kept baseline)."},
        "multiple_comparison_holm_adjusted_p": holm,
        "minimum_detectable_effect_pp_80pct_power": mde_pp,
        "power_reading": (f"MDE at n={n_arm}/arm (80% power, alpha 0.05) ~= {mde_pp}pp; observed +{combined['delta_pp']}pp "
                          "is BELOW the MDE. This is 'underpowered to confirm a small edge' (absence of evidence), "
                          "NOT 'proven zero effect'. A real ~2-3pp self-play edge would routinely fail to reach "
                          "significance here -- but a sub-MDE self-play number cannot justify promotion, and local "
                          "self-play does not predict the ladder."),
        "override_efficacy_length_confound": efficacy,
        "family_transitions_top": {f"{a}->{b}": c for (a, b), c in sorted(fam.items(), key=lambda x: -x[1])[:12]},
        "top_lookup_keys_win_loss": {str(k): {"win": v[0], "loss": v[1], "win_pct": round(100 * v[0] / (v[0] + v[1]), 1)}
                                     for k, v in top_keys},
        "caveat": ("Local self-play does NOT predict the ladder. Primary metric is deployed+mirror combined; do not "
                   "promote on the sentinel/field cells. denpa92 +6.5pp is the only individually significant cell "
                   "(p=0.042) and does not survive multiple-comparison correction across 4 tests."),
    }
    (OUT / "diagnostic_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    css = "body{font:13px system-ui,sans-serif;margin:18px;background:#0f1117;color:#dde}.c{border:1px solid #2a2f3a;border-radius:7px;padding:7px 10px;margin:5px 0;background:#161a22}.tag{display:inline-block;padding:0 6px;border-radius:4px;background:#22303f;margin-right:4px;font-size:11px}.win{color:#7ee787}.loss{color:#ff7a7a}.big{font-size:15px}"
    keyrows = "".join(
        f"<div class='c'><span class='tag'>{html.escape(str(k))}</span>"
        f"<span class='tag win'>{v['win']}W</span><span class='tag loss'>{v['loss']}L</span>"
        f"<span class='tag'>{v['win_pct']}% win</span></div>"
        for k, v in report["top_lookup_keys_win_loss"].items())
    (OUT / "review_examples.html").write_text(
        f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"
        f"<h1>Powered V3 A/B ({verdict})</h1>"
        f"<p class='big'>PRIMARY deployed+mirror combined (n={combined['n_per_arm']}/arm): "
        f"off {combined['off_pct']}% -> V3 {combined['v3_pct']}% = <b>{combined['delta_pp']:+}pp</b>, "
        f"Fisher p={combined['fisher_p']}, CI [{combined['delta_ci_pp'][0]:+},{combined['delta_ci_pp'][1]:+}]pp</p>"
        f"<p>interim looks (early-stopping bias): A {interim['A']['delta_pp']:+}pp -> "
        f"A+B {interim['A+B']['delta_pp']:+}pp (p={interim['A+B']['fisher_p']}) -> "
        f"full {interim['A+B+C(full)']['delta_pp']:+}pp (p={interim['A+B+C(full)']['fisher_p']})</p>"
        f"<p>deployed {per_opp['deployed']['delta_pp']:+}pp (p={per_opp['deployed']['fisher_p']}) | "
        f"mirror {per_opp['mirror']['delta_pp']:+}pp (p={per_opp['mirror']['fisher_p']}) | "
        f"denpa92 {per_opp['denpa92']['delta_pp']:+}pp (p={per_opp['denpa92']['fisher_p']})</p>"
        f"<p>safety: {term} terminal overrides, {total_err} errors, {len(v3_ov)} overrides ({table_hits} table-hit)</p>"
        f"<p>override RATE win {efficacy['override_RATE_win']} vs loss {efficacy['override_RATE_loss']} "
        f"(len proxy win {efficacy['mean_len_proxy_win']} vs loss {efficacy['mean_len_proxy_loss']})</p>"
        f"<h2>Top transplant lookup keys by win/loss</h2>{keyrows}</body></html>", encoding="utf-8")

    print(json.dumps({"verdict": verdict, "promotion": report["PROMOTION_STATUS"], "combined": combined,
                      "interim": interim, "safety": report["safety"], "efficacy": efficacy}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
