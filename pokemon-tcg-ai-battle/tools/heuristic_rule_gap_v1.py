"""Heuristic rule reconstruction GAP audit V1 (Model B). READ-ONLY: no games, no feature impl, no gameplay change.

Re-classifies every feature/composite the V0 reconstruction did NOT recover cleanly into one of 7 categories,
verified against the actual V0 feature table (tools/heuristic_rule_recon_v0.py NUM_FEATS), the worktree helpers
(learned_selector_bridge.option_features, starmie_tactical_state, agent/attack_stats.json), and the rule agent
(submissions/sub_archaludon/main.py). Emits the gap reports.

  CAT 1 present_in_v0 | 2 in_raw_or_cardstats_not_extracted | 3 derivable_with_helpers_not_extracted
      | 4 missing_from_public_data | 5 trace_support_absent | 6 hidden_internal_crossturn | 7 model_framing
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "generated" / "heuristic_rule_reconstruction_gap_v1"
CAT = {1: "present_in_v0", 2: "in_raw_or_cardstats_not_extracted", 3: "derivable_with_helpers_not_extracted",
       4: "missing_from_public_data", 5: "trace_support_absent", 6: "hidden_internal_crossturn", 7: "model_framing"}

# Each item: (id, primary_cat, secondary_cats, v0_status, evidence, fix)
ITEMS = [
 ("attack_damage", 3, [2],
  "NOT in V0 NUM_FEATS (only energy_on_target extracted).",
  "option carries attackId (learned_selector_bridge.option_features 'attack_id' :249; agent reads opt.attackId). "
  "agent/attack_stats.json is keyed by attackId with 'd'=damage,'c'=cost. Raging-Hammer scaling = 80 + "
  "damage_on(active) where damage = maxHp-hp (derivable). best_attack_damage(obs,attack_id) at agent:285 does "
  "exactly this. -> DERIVABLE, not missing.",
  "extract attack_id per ATTACK option + attack_stats[attackId].d (+ damage_on(active) for Raging Hammer)."),
 ("attack_route_needs_promote", 3, [],
  "NOT extracted (no route/needs_promote feature).",
  "agent archaludon_ex_attack_route(obs) :358 is a COMPOSITE of public-derivable parts: bench/active energy "
  "units (TS.energy_units :76), retreat cost (TS._retreat_cost :173 / CARD_DB.retreatCost), and the retreated "
  "flag (obs.current.retreated -- ALREADY in V0). The composite was never built. -> DERIVABLE composite.",
  "build a needs_promote feature = (best bench attacker attack_ready) AND (active not attack-route) AND "
  "(not retreated) AND (active energy_units >= retreat_cost)."),
 ("retreat_route", 3, [1],
  "NOT extracted; RETREAT family F1 0.0.",
  "retreat_to_attack_ready_ex uses the SAME archaludon_ex_attack_route composite (derivable, above). "
  "retreat_dont_break_hp400_tank needs active.id + has_tool(active) + active.hp -- all public/derivable; "
  "'retreated' is already in V0. -> DERIVABLE composite.",
  "same route composite + active tool/hp reads."),
 ("opponent_last_attack", 6, [2],
  "NOT extracted; rule boss_save_vs_mega_brave_stuck unrecovered.",
  "agent _update_opp_attack_tracking :116-119 reads obs.logs but ACCUMULATES _opp_last_attack_id across turns "
  "into a module global. A single public obs only carries a short obs.logs window (partial, cat 2); the reliable "
  "cross-turn value is internal state. -> HIDDEN/CROSS-TURN (partial in-window signal exists).",
  "out of scope for single-obs features; would need a stateful per-game tracker (mirrors the agent's own)."),
 ("matchup_specific_rules", 5, [3],
  "Never triggered; ~12 Crustle/Hop/Starmie/Lucario rules UNIDENTIFIABLE.",
  "detect_matchup(obs) :396 classifies the opponent deck from opp board card ids -- DERIVABLE from public obs. "
  "But V0 traces used random/first opponents, so those archetypes never appeared and the rules never fired. "
  "-> TRACE-SUPPORT absent (the feature itself is derivable).",
  "re-run pi_R vs the real archetype opponents already wired in local_meta_v1 (lucario/koraidon/abomasnow)."),
 ("attach_graded_target_scoring", 7, [3, 1],
  "ATTACH family F1 0.0 despite target_card_id + energy_on_target present.",
  "PRIMARY cause is FRAMING: attach_target_score is a per-decision ARGMAX over graded scores (target id x energy "
  "x area x HP-ratio); V0 pooled all ATTACH options into a binary y_select classifier, which cannot represent "
  "'highest score within THIS decision'. SECONDARY: area (option 'target_zone'/'source_zone' :266,294, cat 2) and "
  "HP-ratio (target.hp/maxHp, derivable cat 3) were not extracted. target_card_id + energy_on_target ARE present "
  "(cat 1). -> mostly MODEL FRAMING, plus 2 derivable features.",
  "frame as within-decision ranking (group by decision_id, learning-to-rank or per-decision argmax) and add "
  "area + hp_ratio features."),
 ("select_card_source_target", 1, [7],
  "SELECT_CARD F1 0.68 -- SEMANTIC (not EXACT).",
  "source_card_id + target_card_id ARE in V0 NUM_FEATS and WERE recovered (the tree split on source_card_id). "
  "Not EXACT because (a) the multi-level priority is a within-decision ranking -> a flat tree approximates it, "
  "and (b) the model leaned partly on the raw_index tie-break shortcut. -> FEATURES PRESENT; imperfection is "
  "FRAMING, not a feature gap.",
  "per-family within-decision ranking would tighten SEMANTIC -> EXACT; no new feature needed."),
 ("global_selector_option_index_shortcut", 7, [],
  "Global selector F1 0.385 ~= option-zero null 0.389.",
  "Pooling heterogeneous decision types into one binary classifier makes option order (raw_index) the strongest "
  "signal -- a tie-break SHORTCUT, not a state rule. Reconstruction is only meaningful PER decision family "
  "(where the real positives -- EVOLVE metal_in_discard, SELECT_CARD card identity -- were found). -> FRAMING.",
  "always analyze per decision-family, never a pooled global selector."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    cat_counts = {}
    for rid, pc, sec, v0, ev, fix in ITEMS:
        table.append({"item": rid, "primary_category": pc, "primary_category_name": CAT[pc],
                      "secondary_categories": [CAT[c] for c in sec], "v0_status": v0, "evidence": ev, "fix": fix})
        cat_counts[CAT[pc]] = cat_counts.get(CAT[pc], 0) + 1
    (OUT / "rule_gap_table.json").write_text(json.dumps({"category_legend": CAT, "items": table}, indent=2), encoding="utf-8")

    summary = {
        "VERDICT": "E_MIXED",
        "headline": "No item is category 4 (truly missing from public data). The plurality is category 3 "
                    "(derivable-with-existing-helpers, not extracted: attack damage, attack/retreat route); the "
                    "other material causes are model FRAMING (category 7: ATTACH argmax + the pooled global "
                    "selector), trace SUPPORT (category 5: matchup rules never faced Crustle/Hop/Starmie/Lucario), "
                    "and exactly one HIDDEN cross-turn rule (category 6: opponent last-attack). SELECT_CARD "
                    "source/target were already present and recovered.",
        "primary_category_counts": cat_counts,
        "zero_truly_missing_public_data": True,
        "biggest_levers": ["extract derivable features: attack_damage (attackId+attack_stats), attack/retreat "
                           "route composite (energy_units+retreat_cost+retreated)",
                           "fix model FRAMING: per-decision-family within-decision ranking instead of a pooled "
                           "binary selector (fixes ATTACH F1=0 and the global option-index shortcut)",
                           "give matchup rules trace support: re-run pi_R vs lucario/koraidon/abomasnow (local_meta_v1)"],
        "only_genuine_dead_end": "opponent_last_attack (cross-turn module state; out of scope for single-obs features)",
        "correction_to_v0": "V0 over-labeled attack damage + attack/retreat route as MISSING_FEATURE; they are "
                            "DERIVABLE-not-extracted (category 3), not missing. ATTACH's F1=0 is mainly FRAMING "
                            "(category 7), not a feature gap.",
    }
    (OUT / "gap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# Feature missingness table (heuristic rule reconstruction gap V1)", "",
          "Category: 1 present_in_v0 | 2 in_raw/cardstats_not_extracted | 3 derivable_with_helpers_not_extracted "
          "| 4 missing_from_public_data | 5 trace_support_absent | 6 hidden_internal_crossturn | 7 model_framing",
          "", "| item | primary | secondary | why V0 missed / partial | fix |",
          "|---|---|---|---|---|"]
    for rid, pc, sec, v0, ev, fix in ITEMS:
        secs = ",".join(str(c) for c in sec) or "-"
        why = ev.replace("\n", " ").split(" -> ")[-1] if " -> " in ev else ev[:80]
        md.append(f"| {rid} | {pc} {CAT[pc]} | {secs} | {why} | {fix[:90]} |")
    md += ["", f"**Verdict: E_MIXED.** No category-4 (truly missing) gaps. Plurality = category 3 "
           "(derivable-not-extracted); material secondary = 7 (framing), 5 (trace support); one 6 (hidden)."]
    (OUT / "feature_missingness_table.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"VERDICT": summary["VERDICT"], "primary_category_counts": cat_counts,
                      "zero_truly_missing": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
