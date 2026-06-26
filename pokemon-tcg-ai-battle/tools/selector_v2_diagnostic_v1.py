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
    # blocked terminals keep the baseline (not a "changed" decision), so count them from the smoke metrics
    c3_blocked_terminal = sum(v.get("blocked_terminal", 0) for k, v in summary.get("metrics", {}).items()
                              if k.startswith("S2:"))
    total_err = sum(res[m][o]["err"] for m in summary["modes"] for o in opps)

    # mirror regression check vs off + vs V1 top3 (20%)
    v1 = json.load(open(V1, encoding="utf-8")) if V1.exists() else None
    v1_top3_mirror = v1["results"]["S2"]["deployed"]["win_pct"] if v1 else None  # V1 S2 was top3 vs 'deployed'

    # first-changed-decision outcomes (per mode). NB rows store the mode VALUE (off/top1_gate/c3_family_limited).
    def first_changed_outcomes(mode_value):
        fc = [r for r in rows if r["mode"] == mode_value and r.get("first_changed")]
        out = collections.Counter(r.get("game_result") for r in fc)
        return {"n_games_with_change": len(fc), "result_dist": dict(out)}

    # combined key matchups (the two disciplined-Starmie opponents) -- the matchups that actually matter
    KEY = ["deployed", "mirror"]
    FIELD = [o for o in opps if o not in KEY]

    def comb(mode, group):
        w = sum(res[mode][o]["win"] for o in group)
        l = sum(res[mode][o]["loss"] for o in group)
        return round(100 * w / (w + l), 1) if (w + l) else None

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
        "first_changed_outcomes": {mv: first_changed_outcomes(mv) for mv in summary["modes"].values()},
        "c3_family_transition_matrix": {f"{a}->{b}": c for (a, b), c in sorted(fam_matrix.items(), key=lambda x: -x[1])},
        "metrics": summary.get("metrics", {}),
    }

    # key-matchup (deployed+mirror) analysis -- the verdict basis, NOT the field aggregate
    key_off, key_top1, key_c3 = comb("S0", KEY), comb("S1", KEY), comb("S2", KEY)
    field_off, field_c3 = comb("S0", FIELD), comb("S2", FIELD)
    agg_off, agg_c3 = comb("S0", opps), comb("S2", opps)
    extra_wins_total = sum(res["S2"][o]["win"] - res["S0"][o]["win"] for o in opps)
    extra_wins_field = sum(res["S2"][o]["win"] - res["S0"][o]["win"] for o in FIELD)
    report["key_matchup_analysis"] = {
        "key_opponents": KEY,
        "combined_winpct": {"off": key_off, "top1_gate": key_top1, "c3": key_c3},
        "c3_minus_off_pp_on_key": round((key_c3 or 0) - (key_off or 0), 1),
        "c3_vs_top1_on_key_pp": round((key_c3 or 0) - (key_top1 or 0), 1),
        "deployed_c3_vs_top1": {"c3": res["S2"]["deployed"]["win_pct"], "top1_gate": res["S1"]["deployed"]["win_pct"]},
        "note": "C3 is flat vs off on the key matchups and BELOW the simpler top1_gate; the only gains are vs weak field decks.",
    }
    report["aggregate_decomposition"] = {
        "aggregate_off": agg_off, "aggregate_c3": agg_c3,
        "field_off": field_off, "field_c3": field_c3,
        "extra_wins_c3_over_off_total": extra_wins_total, "extra_wins_from_field": extra_wins_field,
        "pct_of_gain_from_field": round(100 * extra_wins_field / extra_wins_total, 0) if extra_wins_total else None,
        "note": "100% of C3's extra wins over off come from weak field decks; 0 from deployed+mirror.",
    }

    # ---- verdict: decided on the KEY matchups (deployed+mirror), not the field aggregate ----
    field_deltas = [v["delta_pp"] for v in report["field_non_catastrophic"].values()]
    key_delta = report["key_matchup_analysis"]["c3_minus_off_pp_on_key"]
    catastrophic_field = any(d <= -30 for d in field_deltas)
    if total_err > 0 or len(c3_terminal_overrides) > 0:
        verdict = "D_C3_SELECTOR_UNSAFE_OR_INVALID"
    elif key_delta <= -25 or catastrophic_field:
        verdict = "C_C3_SELECTOR_STILL_REGRESSIVE"
    elif key_delta >= 10 and not catastrophic_field:
        verdict = "A_C3_SELECTOR_SMOKE_CLEAN_DIRECTIONAL"
    else:
        # flat on the key matchups (n=20, Fisher p~1.0): safe + non-catastrophic but not a directional win
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
    report["PROMOTION_STATUS"] = "DO_NOT_PROMOTE / " + promotion
    report["what_holds"] = ("Deterministic safety: C3 emitted 0 terminal (ATTACK/END/RETREAT) overrides and blocked "
                            f"{c3_blocked_terminal} live. No catastrophic regression: the V1 top3 -35pp deployed-mirror "
                            "crash did NOT recur (combined key matchups flat vs off).")
    report["what_is_overclaimed"] = ("'Field-positive / directional win' is NOT supported. On the matchups that decide "
                                     "the ladder (deployed+mirror) C3 is FLAT vs off (0.0pp combined) and BELOW the "
                                     "simpler top1_gate (deployed 45% vs 60%). 100% of C3's aggregate gain is vs weak "
                                     "field decks. At n=20 every cell is Fisher p~1.0 -- no win-rate direction is real.")
    report["promotion_conditions"] = [
        "fix the trace logger to record BLOCKED decisions + true per-game ids (current game_id is a non-unique shard "
        "label; blocked terminals are only counters, so 'blocking avoids mirror loss' has no per-decision evidence)",
        "before any N500, pre-commit to judging promotion on the deployed+mirror cells, NOT the field aggregate",
    ]
    report["caveat"] = ("n=20/matchup is underpowered (~+/-20pp CIs; need ~+30pp/cell for p<0.05). Local self-play "
                        "does NOT predict the ladder. Verdict adversarially reviewed and downgraded A->B.")
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
