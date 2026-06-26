"""Diagnostic report for the conservative C3 second smoke (Section 7-8). Consumes live_smoke_summary.json +
changed_decisions.jsonl. Answers: did C3 eliminate ATTACK/END/RETREAT overrides; did ATTACH/SELECT/EVOLVE/PLAY
improvements remain; did the mirror regression disappear; did the field stay non-catastrophic. Computes the
DIAGNOSTIC verdict + promotion status. Honest about n=20 underpowering.

  PYTHONIOENCODING=utf-8 python tools/selector_v2_diagnostic_v1.py
"""
from __future__ import annotations
import collections
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_selector_v2_smoke"
V1 = ROOT / "data" / "generated" / "starmie_selector_live_smoke_v1" / "live_smoke_report.json"
TERMINAL = {"ATTACK", "END", "RETREAT"}


def main() -> int:
    summary = json.load(open(OUT / "live_smoke_summary.json", encoding="utf-8"))
    rows = [json.loads(l) for l in open(OUT / "changed_decisions.jsonl", encoding="utf-8")]
    res = summary["results"]
    opps = summary["opponents"]
    n = summary["games_per_matchup"]

    def winpct(mode, opp):
        return res[mode][opp]["win_pct"]

    # C3 safety: terminal override families must be empty
    c3 = [r for r in rows if r["mode"] == "c3_family_limited"]
    c3_terminal_overrides = [r for r in c3 if r["selector_family"] in TERMINAL]
    c3_fam_to = collections.Counter(r["selector_family"] for r in c3)
    c3_blocked_terminal = sum(1 for r in c3 if r.get("terminal_override_blocked"))
    total_err = sum(res[m][o]["err"] for m in summary["modes"] for o in opps)

    # mirror regression check vs off + vs V1 top3 (20%)
    v1 = json.load(open(V1, encoding="utf-8")) if V1.exists() else None
    v1_top3_mirror = v1["results"]["S2"]["deployed"]["win_pct"] if v1 else None  # V1 S2 was top3 vs 'deployed'

    # first-changed-decision outcomes (per mode)
    def first_changed_outcomes(mode):
        fc = [r for r in rows if r["mode"] == mode and r.get("first_changed")]
        out = collections.Counter(r.get("game_result") for r in fc)
        return {"n_games_with_change": len(fc), "result_dist": dict(out)}

    # family transition matrix (c3)
    fam_matrix = collections.Counter((r["baseline_family"], r["selector_family"]) for r in c3)

    report = {
        "n_games_per_matchup": n, "total_errors": total_err,
        "win_rate_by_mode_opponent": {m: {o: winpct(m, o) for o in opps} for m in summary["modes"]},
        "mirror": {
            "S0_off": winpct("S0", "mirror"), "S1_top1_gate": winpct("S1", "mirror"),
            "S2_c3_family_limited": winpct("S2", "mirror"),
            "S2_minus_off_pp": round((winpct("S2", "mirror") or 0) - (winpct("S0", "mirror") or 0), 1),
            "deployed_mirror_S2_minus_off_pp": round((winpct("S2", "deployed") or 0) - (winpct("S0", "deployed") or 0), 1),
            "v1_top3_deployed_mirror_winpct_for_reference": v1_top3_mirror,
        },
        "c3_safety": {
            "terminal_override_count": len(c3_terminal_overrides),
            "eliminated_terminal_overrides": len(c3_terminal_overrides) == 0,
            "blocked_terminal_in_live": c3_blocked_terminal,
            "override_family_distribution": dict(c3_fam_to),
            "nonterminal_improvements_retained": {k: c3_fam_to.get(k, 0) for k in ("ATTACH", "SELECT_CARD", "EVOLVE", "PLAY")},
        },
        "field_non_catastrophic": {o: {"S0": winpct("S0", o), "S2": winpct("S2", o),
                                       "delta_pp": round((winpct("S2", o) or 0) - (winpct("S0", o) or 0), 1)}
                                   for o in opps if o not in ("mirror", "deployed")},
        "first_changed_outcomes": {m: first_changed_outcomes(m) for m in summary["modes"]},
        "c3_family_transition_matrix": {f"{a}->{b}": c for (a, b), c in sorted(fam_matrix.items(), key=lambda x: -x[1])},
        "metrics": summary.get("metrics", {}),
    }

    # ---- verdict ----
    field_deltas = [v["delta_pp"] for v in report["field_non_catastrophic"].values()]
    mirror_delta = report["mirror"]["S2_minus_off_pp"]
    deployed_delta = report["mirror"]["deployed_mirror_S2_minus_off_pp"]
    catastrophic_field = any(d <= -30 for d in field_deltas)
    if total_err > 0 or len(c3_terminal_overrides) > 0:
        verdict = "D_C3_SELECTOR_UNSAFE_OR_INVALID"
    elif (mirror_delta <= -25 or deployed_delta <= -25) or catastrophic_field:
        verdict = "C_C3_SELECTOR_STILL_REGRESSIVE"
    elif (mirror_delta >= 5 or deployed_delta >= 5) and not catastrophic_field and c3_fam_to:
        verdict = "A_C3_SELECTOR_SMOKE_CLEAN_DIRECTIONAL"
    else:
        verdict = "B_C3_SELECTOR_SMOKE_NEUTRAL"
    # promotion: n=20 is underpowered (per the V1 diagnostic), so never submit/large-AB on this alone
    if verdict.startswith("D"):
        promotion = "NEEDS_SELECTOR_REPAIR"
    elif verdict.startswith("C"):
        promotion = "NEEDS_SELECTOR_REPAIR"
    elif verdict.startswith("A"):
        promotion = "NEEDS_N500"
    else:
        promotion = "NEEDS_N500"
    report["DIAGNOSTIC_VERDICT"] = verdict
    report["PROMOTION_STATUS"] = "DO_NOT_SUBMIT / " + promotion
    report["caveat"] = ("n=20/matchup is underpowered (per the V1 diagnostic: a 20-game mirror cell carries ~+/-20pp "
                        "CIs). The terminal-override elimination and 0-error safety are solid; win-rate direction is "
                        "indicative only. Local self-play does NOT predict the ladder.")
    (OUT / "diagnostic_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ---- review html ----
    cats = [
        ("C3 overrides retained (ATTACH/SELECT/EVOLVE/PLAY)", lambda r: r["mode"] == "c3_family_limited", 24),
        ("C3 changed-decision in MIRROR games", lambda r: r["mode"] == "c3_family_limited" and r["matchup"] == "mirror", 16),
        ("top1_gate changed decisions (reference)", lambda r: r["mode"] == "top1_gate", 12),
    ]
    css = ("body{font:13px/1.5 system-ui,sans-serif;margin:18px;background:#0f1117;color:#dde}"
           "h2{font-size:15px;border-bottom:1px solid #333;margin-top:22px}.c{border:1px solid #2a2f3a;border-radius:7px;"
           "padding:8px 11px;margin:6px 0;background:#161a22}.tag{display:inline-block;padding:0 6px;border-radius:4px;"
           "background:#22303f;margin-right:4px;font-size:11px}.win{color:#7ee787}.loss{color:#ff7a7a}.k{color:#9fc5ff}td{padding:1px 9px 1px 0}")
    parts = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
             f"<h1>Starmie C3 second smoke review ({verdict})</h1>",
             f"<p>n={n}/matchup, errors={total_err}, C3 terminal overrides={len(c3_terminal_overrides)} "
             f"(blocked live={c3_blocked_terminal}). Mirror: off {report['mirror']['S0_off']}% / top1 "
             f"{report['mirror']['S1_top1_gate']}% / c3 {report['mirror']['S2_c3_family_limited']}%.</p>"]
    for title, pred, lim in cats:
        exs = [r for r in rows if pred(r)][:lim]
        parts.append(f"<h2>{html.escape(title)} ({len(exs)})</h2>")
        for r in exs:
            wl = "win" if r.get("game_result") == "win" else ("loss" if r.get("game_result") == "loss" else "")
            t = r.get("tactical") or {}
            parts.append(
                "<div class='c'>"
                f"<span class='tag'>{r['matchup']}</span><span class='tag {wl}'>{r.get('game_result')}</span>"
                f"<span class='tag'>{r['baseline_family']}->{r['selector_family']}</span>"
                f"<span class='tag'>conf={round(r['confidence'],3) if isinstance(r.get('confidence'),(int,float)) else r.get('confidence')}</span>"
                f"<span class='tag'>{'blockedT' if r.get('terminal_override_blocked') else ''}</span>"
                f"<span class='tag'>KO={t.get('commitment.guaranteed_ko_available')}</span>"
                f"<span class='tag'>safedev={t.get('commitment.safe_development_available')}</span>"
                f" <span class='k'>{r['game_id']} step{r.get('step')}</span> raw {r['baseline_raw']}-&gt;{r['selector_raw']}"
                "</div>")
    parts.append("</body></html>")
    (OUT / "review_examples.html").write_text("\n".join(parts), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "promotion": report["PROMOTION_STATUS"], "mirror": report["mirror"],
                      "c3_safety": {k: report["c3_safety"][k] for k in ("terminal_override_count", "eliminated_terminal_overrides", "blocked_terminal_in_live", "override_family_distribution")},
                      "field": report["field_non_catastrophic"], "errors": total_err}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
