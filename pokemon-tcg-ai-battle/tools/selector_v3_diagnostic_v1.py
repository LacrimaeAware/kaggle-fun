"""Diagnostic for the repaired V3 transplant smoke. Consumes live_smoke_summary.json + changed_decisions.jsonl.
Judged on deployed+mirror combined (primary); field aggregate secondary. Adds transplant-source / table-hit
breakdown. Verdict A-F + promotion. Honest about n=20 underpowering.

  PYTHONIOENCODING=utf-8 python tools/selector_v3_diagnostic_v1.py
"""
from __future__ import annotations
import collections
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_selector_v3_smoke_repaired"
V2 = ROOT / "data" / "generated" / "starmie_selector_v2_smoke" / "live_smoke_report.json"
TERMINAL = {"ATTACK", "END", "RETREAT"}
KEY = ["deployed", "mirror"]
V3 = "selector_v3_transplant"


def main() -> int:
    summary = json.load(open(OUT / "live_smoke_summary.json", encoding="utf-8"))
    rows = [json.loads(l) for l in open(OUT / "changed_decisions.jsonl", encoding="utf-8")]
    res, opps = summary["results"], summary["opponents"]
    modes = summary["modes"]  # {S0:off, S2:c3_family_limited, S3:selector_v3_transplant}
    mk = {v: k for k, v in modes.items()}  # value->key
    FIELD = [o for o in opps if o not in KEY]

    def comb(mode_key, group):
        w = sum(res[mode_key][o]["win"] for o in group)
        l = sum(res[mode_key][o]["loss"] for o in group)
        return round(100 * w / (w + l), 1) if (w + l) else None

    v3k = mk.get(V3)
    offk = mk.get("off")
    c3k = mk.get("c3_family_limited")
    v3_rows = [r for r in rows if r["mode"] == V3]
    v3_overrides = [r for r in v3_rows if not r.get("blocked_terminal") and r.get("selector_raw") != r.get("baseline_raw")]
    v3_terminal_overrides = [r for r in v3_overrides if r["selector_family"] in TERMINAL]
    v3_table_hits = sum(1 for r in v3_rows if r.get("transplant_table_hit"))
    src_breakdown = collections.Counter(r.get("transplant_support_source") for r in v3_rows)
    fam_to = collections.Counter(r["selector_family"] for r in v3_overrides)
    fam_matrix = collections.Counter((r["baseline_family"], r["selector_family"]) for r in v3_overrides)
    total_err = sum(res[m][o]["err"] for m in modes for o in opps)
    metrics_v3 = {k: v for k, v in summary.get("metrics", {}).items() if k.startswith(f"{v3k}:")}
    blocked_terminal = sum(v.get("blocked_terminal", 0) for v in metrics_v3.values())

    # first-changed outcomes for V3
    fc = [r for r in v3_rows if r.get("first_changed")]
    fc_out = collections.Counter(r.get("game_result") for r in fc)

    report = {
        "n_games_per_matchup": summary["games_per_matchup"], "total_errors": total_err,
        "v3_non_inert": {"overrides": len(v3_overrides), "table_hits": v3_table_hits,
                         "is_inert": len(v3_overrides) == 0},
        "win_rate_by_mode_opponent": {modes[k]: {o: res[k][o]["win_pct"] for o in opps} for k in modes},
        "key_matchups_deployed_mirror": {modes[k]: comb(k, KEY) for k in modes},
        "v3_minus_off_on_key_pp": round((comb(v3k, KEY) or 0) - (comb(offk, KEY) or 0), 1) if v3k and offk else None,
        "v3_minus_c3_on_key_pp": round((comb(v3k, KEY) or 0) - (comb(c3k, KEY) or 0), 1) if v3k and c3k else None,
        "field_aggregate": {modes[k]: comb(k, FIELD) for k in modes},
        "v3_safety": {"terminal_overrides": len(v3_terminal_overrides), "blocked_terminal_live": blocked_terminal,
                      "override_family_distribution": dict(fam_to)},
        "transplant_support_source_breakdown": dict(src_breakdown),
        "v3_family_transition_matrix": {f"{a}->{b}": c for (a, b), c in sorted(fam_matrix.items(), key=lambda x: -x[1])},
        "first_changed_outcomes_v3": {"n": len(fc), "result_dist": dict(fc_out)},
        "metrics_v3": {k: dict(v) for k, v in metrics_v3.items()},
    }

    # verdict on deployed+mirror combined
    key_delta = report["v3_minus_off_on_key_pp"] or 0
    field_deltas = [(comb(v3k, [o]) or 0) - (comb(offk, [o]) or 0) for o in FIELD]
    catastrophic_field = any(d <= -30 for d in field_deltas)
    if total_err > 0 or len(v3_terminal_overrides) > 0:
        verdict = "D_REPAIRED_V3_UNSAFE_OR_INVALID"
    elif len(v3_overrides) == 0:
        verdict = "E_V3_STILL_INERT"
    elif key_delta <= -25 or catastrophic_field:
        verdict = "C_REPAIRED_V3_SMOKE_REGRESSIVE"
    elif key_delta >= 10 and not catastrophic_field:
        verdict = "A_REPAIRED_V3_SMOKE_CLEAN_DIRECTIONAL"
    else:
        verdict = "B_REPAIRED_V3_SMOKE_NEUTRAL"
    report["DIAGNOSTIC_VERDICT"] = verdict
    report["PROMOTION_STATUS"] = "DO_NOT_SUBMIT / NEEDS_N500"
    report["caveat"] = ("Judged on deployed+mirror combined (primary), NOT field aggregate. n=20/matchup is "
                        "underpowered (~+/-20pp CIs; need ~+30pp/cell for significance); win-rate direction is "
                        "indicative only. Local self-play does NOT predict the ladder. V3 = C3 + transplant gating; "
                        "the live transplant lookup is enabled by Model B's compact_semantic_action_key bridge.")
    (OUT / "diagnostic_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # review html
    css = "body{font:13px system-ui,sans-serif;margin:18px;background:#0f1117;color:#dde}.c{border:1px solid #2a2f3a;border-radius:7px;padding:8px 11px;margin:6px 0;background:#161a22}.tag{display:inline-block;padding:0 6px;border-radius:4px;background:#22303f;margin-right:4px;font-size:11px}.win{color:#7ee787}.loss{color:#ff7a7a}"
    exs = [r for r in v3_overrides if r.get("transplant_table_hit")][:30]
    cards = "".join(
        f"<div class='c'><span class='tag {'win' if e.get('game_result')=='win' else 'loss'}'>{e.get('game_result')}</span>"
        f"<span class='tag'>{e['matchup']}</span><span class='tag'>{e['baseline_family']}->{e['selector_family']}</span>"
        f"<span class='tag'>key={html.escape(str(e.get('transplant_lookup_key')))}</span>"
        f"<span class='tag'>conf={e.get('transplant_confidence')}</span> step{e.get('step')}</div>" for e in exs)
    (OUT / "review_examples.html").write_text(
        f"<html><head><meta charset='utf-8'><style>{css}</style></head><body><h1>Repaired V3 smoke review ({verdict})</h1>"
        f"<p>key(deployed+mirror): off {report['key_matchups_deployed_mirror'].get('off')}% / "
        f"c3 {report['key_matchups_deployed_mirror'].get('c3_family_limited')}% / "
        f"v3 {report['key_matchups_deployed_mirror'].get('selector_v3_transplant')}% | v3-off {report['v3_minus_off_on_key_pp']}pp | "
        f"overrides {len(v3_overrides)} table_hits {v3_table_hits} terminal {len(v3_terminal_overrides)} errors {total_err}</p>"
        f"<h2>V3 table-supported overrides ({len(exs)})</h2>{cards}</body></html>", encoding="utf-8")

    print(json.dumps({"verdict": verdict, "promotion": report["PROMOTION_STATUS"],
                      "key_matchups": report["key_matchups_deployed_mirror"], "v3_minus_off_key": report["v3_minus_off_on_key_pp"],
                      "v3_minus_c3_key": report["v3_minus_c3_on_key_pp"], "non_inert": report["v3_non_inert"],
                      "safety": report["v3_safety"], "support_source": report["transplant_support_source_breakdown"],
                      "errors": total_err}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
