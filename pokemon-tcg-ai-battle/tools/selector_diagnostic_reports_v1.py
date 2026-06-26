"""STARMIE SELECTOR DIAGNOSTIC PACK V1 -- reports stage (sections 2 + 3).

Consumes changed_decision_classes.jsonl (mechanism, per decision x mode) and live_smoke_report.json (aggregate
win rate by mode x opponent -- the only outcome data the smoke saved). Produces the aggregate failure report and
the mirror-regression mechanism report. Honest about the two data gaps:
  (1) no per-game outcome linkage -> outcome attribution is at the mode x opponent level only;
  (2) the classification is on EXPERT replay states; the regression manifested on OUR-AGENT mirror states
      (distribution shift) -- so terminal-override rates here are a lower bound for the live regression.

  PYTHONIOENCODING=utf-8 python tools/selector_diagnostic_reports_v1.py
"""
from __future__ import annotations
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_selector_live_smoke_v1"
MODES = ("top1_gate", "top3_selector")
MODE_TO_SMOKE = {"top1_gate": "S1", "top3_selector": "S2"}


def _q(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    vals.sort()
    return {"n": len(vals), "mean": round(statistics.mean(vals), 4), "median": round(statistics.median(vals), 4),
            "p10": round(vals[max(0, int(0.1 * len(vals)) - 0)], 4), "p90": round(vals[min(len(vals) - 1, int(0.9 * len(vals)))], 4)}


def main() -> int:
    rows = [json.loads(l) for l in open(OUT / "changed_decision_classes.jsonl", encoding="utf-8")]
    smoke = json.load(open(OUT / "live_smoke_report.json", encoding="utf-8"))

    agg = {"data_provenance": {
        "decisions_source": "Starmie single-select decisions sampled from replay episodes (expert-pilot states)",
        "per_game_outcome_linkage": "UNAVAILABLE -- live smoke saved aggregate win/loss only; games not re-run",
        "outcome_evidence": "mode x opponent aggregate win rate from live_smoke_report.json",
        "distribution_shift_caveat": "classification is on expert-pilot states; the top3_selector regression "
                                     "manifested on OUR-AGENT mirror states, which differ. Terminal-override rates "
                                     "here are a LOWER BOUND for the live mirror regression.",
    }, "modes": {}}

    for mode in MODES:
        mr = [r for r in rows if r["mode"] == mode]
        changed = [r for r in mr if r["changed"]]
        term = [r for r in changed if r["terminal_override"]]
        prem = [r for r in changed if r["premature_terminal_override"]]
        veto_blocked = [r for r in mr if r.get("hard_veto_on_pick")]
        trans = collections.Counter(r["transition_class"] for r in changed)
        fam_matrix = collections.Counter((r["baseline_family"], r["selector_family"]) for r in changed)
        # win rate (this mode) by opponent + aggregate
        sm = smoke["results"][MODE_TO_SMOKE[mode]]
        off = smoke["results"]["S0"]
        win_by_opp = {opp: {"mode_win_pct": sm[opp]["win_pct"], "off_win_pct": off[opp]["win_pct"],
                            "delta_pp": round(sm[opp]["win_pct"] - off[opp]["win_pct"], 1)} for opp in smoke["opponents"]}
        agg["modes"][mode] = {
            "baseline_decisions": len(mr),
            "overrides": len(changed),
            "override_rate_pct": round(100 * len(changed) / max(1, len(mr)), 1),
            "terminal_overrides": len(term),
            "terminal_override_rate_of_overrides_pct": round(100 * len(term) / max(1, len(changed)), 1),
            "premature_terminal_overrides": len(prem),
            "premature_terminal_rate_of_overrides_pct": round(100 * len(prem) / max(1, len(changed)), 1),
            "develop_to_attack": trans.get("DEVELOP_TO_ATTACK", 0),
            "develop_to_end": trans.get("DEVELOP_TO_END", 0),
            "transition_matrix": dict(trans),
            "family_transition_matrix": {f"{a}->{b}": c for (a, b), c in sorted(fam_matrix.items(), key=lambda x: -x[1])},
            "hard_veto_blocked_overrides": len(veto_blocked),
            "confidence_terminal_overrides": {
                "proposer_prob": _q([r["proposer_prob_of_pick"] for r in term]),
                "selector_score_margin": _q([r["selector_score_margin"] for r in term]),
                "entropy": _q([r["entropy"] for r in term]),
            },
            "confidence_nonterminal_overrides": {
                "proposer_prob": _q([r["proposer_prob_of_pick"] for r in changed if not r["terminal_override"]]),
                "selector_score_margin": _q([r["selector_score_margin"] for r in changed if not r["terminal_override"]]),
            },
            "win_rate_by_opponent": win_by_opp,
            "aggregate_win_pct": round(100 * sum(sm[o]["win"] for o in smoke["opponents"])
                                       / max(1, sum(sm[o]["win"] + sm[o]["loss"] for o in smoke["opponents"])), 1),
        }
    # ---- KEY FINDING: rank x confidence of terminal overrides (the top1 vs top3 differentiator) ----
    def _rankconf(mode, subset):
        rs = [r for r in rows if r["mode"] == mode and r["changed"] and subset(r)]
        by_rank = collections.Counter(r["proposer_rank_of_pick"] for r in rs)
        return {"count": len(rs), "by_proposer_rank": dict(sorted(by_rank.items(), key=lambda x: (x[0] is None, x[0])))}
    t3 = [r for r in rows if r["mode"] == "top3_selector" and r["changed"]]
    term_rank1_prob = _q([r["proposer_prob_of_pick"] for r in t3 if r["terminal_override"] and r["proposer_rank_of_pick"] == 1])
    term_rank23_prob = _q([r["proposer_prob_of_pick"] for r in t3 if r["terminal_override"] and (r["proposer_rank_of_pick"] or 9) >= 2])
    agg["key_finding_rank_confidence"] = {
        "claim": "The top1_gate (neutral) vs top3_selector (regressive) difference is the rank-2/3 terminal "
                 "overrides that top3 admits and the rank-1 gate filters. These are LOW proposer-confidence.",
        "terminal_overrides": {"top1_gate": _rankconf("top1_gate", lambda r: r["terminal_override"]),
                               "top3_selector": _rankconf("top3_selector", lambda r: r["terminal_override"])},
        "premature_terminal_overrides": {"top1_gate": _rankconf("top1_gate", lambda r: r["premature_terminal_override"]),
                                        "top3_selector": _rankconf("top3_selector", lambda r: r["premature_terminal_override"])},
        "top3_extra_terminal_overrides_rank2plus": sum(1 for r in t3 if r["terminal_override"] and (r["proposer_rank_of_pick"] or 9) >= 2),
        "top3_extra_premature_terminal_rank2plus": sum(1 for r in t3 if r["premature_terminal_override"] and (r["proposer_rank_of_pick"] or 9) >= 2),
        "proposer_prob_terminal_rank1": term_rank1_prob,
        "proposer_prob_terminal_rank2plus": term_rank23_prob,
        "implication": "A conservative selector must gate terminal (ATTACK/END) overrides on high confidence "
                       "(proposer rank-1 / KO or gamewin available / no safe development remaining). Low-confidence "
                       "rank-2/3 proposer picks must never override development into a turn-ending action.",
    }
    (OUT / "failure_aggregate_report.json").write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")

    # ---- mirror regression mechanism report ----
    s2 = agg["modes"]["top3_selector"]
    s1 = agg["modes"]["top1_gate"]
    # context of terminal overrides (top3): tactical coordinates when the selector terminates
    t3_term = [r for r in rows if r["mode"] == "top3_selector" and r["terminal_override"]]
    ctx = {
        "safe_development_available_pct": round(100 * sum(1 for r in t3_term if r["safe_development_available"]) / max(1, len(t3_term)), 1),
        "no_ko_no_gamewin_pct": round(100 * sum(1 for r in t3_term if not r["guaranteed_ko_available"] and not r["game_winning_attack_available"]) / max(1, len(t3_term)), 1),
        "nonterminal_attack_available_pct": round(100 * sum(1 for r in t3_term if r["nonterminal_attack_available"]) / max(1, len(t3_term)), 1),
        "prize_diff_when_terminating": _q([r["prize_diff"] for r in t3_term]),
        "my_ready_main_attackers_when_terminating": _q([r["my_ready_main_attackers"] for r in t3_term]),
    }
    mirror = {
        "focus": "deployed Starmie mirror -- where top3_selector regressed (20% vs 55% off)",
        "outcome_from_smoke": {"S0_off_win_pct": smoke["results"]["S0"]["deployed"]["win_pct"],
                               "S1_top1_gate_win_pct": smoke["results"]["S1"]["deployed"]["win_pct"],
                               "S2_top3_selector_win_pct": smoke["results"]["S2"]["deployed"]["win_pct"],
                               "games_per_matchup": smoke["games_per_matchup"]},
        "per_game_first_changed_decision_analysis": "UNAVAILABLE -- requires instrumented game logs; smoke saved "
                                                    "aggregate win/loss only and games may not be re-run.",
        "mechanism_top3_vs_top1": {
            "top3_override_rate_pct": s2["override_rate_pct"], "top1_override_rate_pct": s1["override_rate_pct"],
            "top3_terminal_overrides": s2["terminal_overrides"], "top1_terminal_overrides": s1["terminal_overrides"],
            "top3_develop_to_attack": s2["develop_to_attack"], "top3_develop_to_end": s2["develop_to_end"],
            "top1_develop_to_attack": s1["develop_to_attack"], "top1_develop_to_end": s1["develop_to_end"],
        },
        "terminal_override_context_top3": ctx,
        "interpretation": ("top3_selector overrides more often and terminates the turn (ATTACK/END) far more than "
                           "top1_gate. On expert states most terminal overrides occur with no KO/gamewin available; "
                           "a share occur while safe development still remains (premature). Because the mirror "
                           "regression is on OUR-AGENT states (which develop more and reach different boards), the "
                           "live premature-termination rate is expected to exceed these expert-state rates -- "
                           "consistent with develop-before-attack: terminating early surrenders tempo in the mirror."),
        "field_vs_mirror_contrast": {
            "top3_delta_vs_off_pp": {opp: agg["modes"]["top3_selector"]["win_rate_by_opponent"][opp]["delta_pp"]
                                     for opp in smoke["opponents"]},
            "reading": ("top3_selector BEATS off on 3/5 field opponents (+10pp) but crashes the deployed mirror "
                        "(-35pp). The same terminal aggression that helps against weaker/differently-piloted decks "
                        "is punished by the disciplined mirror -- the classic 'selection helps weak policies, hurts "
                        "strong policies' failure: premature termination surrenders tempo exactly where the opponent "
                        "exploits it."),
        },
        "mode_difference_is_terminal": {
            "develop_to_attack": {"top1": s1["develop_to_attack"], "top3": s2["develop_to_attack"],
                                  "extra_in_top3": s2["develop_to_attack"] - s1["develop_to_attack"]},
            "develop_to_end": {"top1": s1["develop_to_end"], "top3": s2["develop_to_end"],
                               "extra_in_top3": s2["develop_to_end"] - s1["develop_to_end"]},
            "select_card_change": {"top1": agg["modes"]["top1_gate"]["transition_matrix"].get("SELECT_CARD_CHANGE", 0),
                                   "top3": agg["modes"]["top3_selector"]["transition_matrix"].get("SELECT_CARD_CHANGE", 0)},
            "reading": "The top1->top3 difference is concentrated in DEVELOP->ATTACK/END (terminal). Nonterminal "
                       "SELECT_CARD changes are ~equal across modes, so they are not the differentiator.",
        },
        "distribution_shift_caveat": agg["data_provenance"]["distribution_shift_caveat"],
    }
    (OUT / "mirror_regression_report.json").write_text(json.dumps(mirror, indent=2, default=str), encoding="utf-8")

    print("failure_aggregate_report.json + mirror_regression_report.json written")
    for mode in MODES:
        m = agg["modes"][mode]
        print(f"  {mode}: overrides {m['overrides']}/{m['baseline_decisions']} ({m['override_rate_pct']}%); "
              f"terminal {m['terminal_overrides']} (D->ATK {m['develop_to_attack']}, D->END {m['develop_to_end']}); "
              f"premature-terminal {m['premature_terminal_overrides']}; aggregate_win {m['aggregate_win_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
