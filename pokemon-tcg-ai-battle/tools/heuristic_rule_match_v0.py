"""Natural Heuristic Rule Reconstruction -- PHASE 3: manifest + feature-contract + identifiability + rule-match.

Synthesizes the rule manifest (catalogued from submissions/sub_archaludon/main.py), the feature contract (which
rule features the public layer exposes), and the Phase-2 reconstruction results into the identifiability +
rule-match reports + review pack. This is the core diagnostic output: did the feature layer recover the rule
triggers from NATURAL traces?

  PYTHONIOENCODING=utf-8 python tools/heuristic_rule_match_v0.py
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "generated" / "heuristic_rule_reconstruction_v0"

# Condensed faithful manifest (from the full read of sub_archaludon/main.py; 57 distinct rules, ~100-110 clauses).
# feature_class: card_identity (recoverable from action metadata) | derived_state (needs public board/discard/energy)
#                | internal_hidden (module-global opp tracking) | matchup (needs opp-deck classification).
M = [
 ("setup_active_priority", "priority_order", "setup", "SELECT_CARD", "card_identity", True, "Cinderace>>Duraludon>>Relicanth as setup Active"),
 ("setup_no_mulligan", "positive_select", "setup", "meta", "card_identity", True, "never mulligan / go second"),
 ("activate_explosiveness_always", "positive_select", "ability", "ABILITY", "card_identity", True, "always YES on ACTIVATE"),
 ("play_pokemon_basic", "positive_select", "bench", "PLAY", "card_identity", True, "bench Duraludon/Relicanth"),
 ("play_full_metal_lab", "positive_select", "stadium", "PLAY", "derived_state", True, "play FML unless Active non-Metal"),
 ("ultra_ball_fuel_alloy", "positive_select", "item", "PLAY", "derived_state", True, "UB to discard a Metal when 0 in discard & >=1 in hand"),
 ("ultra_ball_search_line", "positive_select", "item", "PLAY", "derived_state", True, "UB to search line when safe_discard>=2 & need line"),
 ("night_stretcher_save_unless_urgent", "negative_veto", "item", "PLAY", "derived_state", True, "save NS unless key piece in discard & urgent"),
 ("ice_cream_skip_not_ex", "negative_veto", "item", "PLAY", "derived_state", True, "skip Ice Cream unless Active is Archaludon ex"),
 ("explorer_supporter_lock", "positive_select", "supporter", "PLAY", "derived_state", True, "Explorer 16000 unless supporterPlayed"),
 ("override_no_explorer_low_deck", "negative_veto", "supporter", "PLAY", "derived_state", True, "never Explorer at deckCount<=10"),
 ("boss_lethal_bench_when_active_ko", "positive_select", "supporter", "PLAY", "derived_state", True, "Boss bench for lethal KO (prize math)"),
 ("boss_pull_bench_value", "positive_select", "supporter", "PLAY", "derived_state", True, "Boss best KO-able bench target by prize+energy"),
 ("boss_save_vs_mega_brave_stuck", "negative_veto", "matchup_tech", "PLAY", "internal_hidden", False, "save Boss if opp last attack was Mega Brave (module-global tracking)"),
 ("evolve_active_metal_discard_alloy", "positive_select", "evolution", "EVOLVE", "derived_state", True, "evolve Active Duraludon: mc>=2 -> 28000; mc==1 -> 8000; else hold"),
 ("evolve_active_3energy_no_ex_yet", "positive_select", "evolution", "EVOLVE", "derived_state", True, "evolve 3-energy Active Duraludon if no ex in play"),
 ("evolve_bench_duraludon", "positive_select", "evolution", "EVOLVE", "derived_state", True, "bench evolve only with mc>=2"),
 ("attach_metal_target_scoring", "priority_order", "energy_attach", "ATTACH", "derived_state", True, "graded by target id x energy x area x HP; Cinderace@0, Dura/Arch@2 boosted; cap @3"),
 ("attach_heros_cape_archaludon", "positive_select", "energy_attach", "ATTACH", "card_identity", True, "Cape -> untooled Archaludon ex"),
 ("attach_already_attached_block", "negative_veto", "energy_attach", "ATTACH", "derived_state", True, "no 2nd attach if energyAttached"),
 ("retreat_to_attack_ready_ex", "positive_select", "retreat", "RETREAT", "derived_state", True, "retreat if attack route needs promote (composite: active/bench energy+retreatCost+retreated)"),
 ("retreat_dont_break_hp400_tank", "negative_veto", "retreat", "RETREAT", "derived_state", True, "don't retreat tooled Archaludon ex hp>200"),
 ("attack_score_by_damage", "priority_order", "attack", "ATTACK", "derived_state", True, "rank attacks by damage (attackId -> base dmg; Raging Hammer scales)"),
 ("tohand_generic_line_and_resources", "priority_order", "draw_search", "SELECT_CARD", "card_identity", True, "search/draw priority Archaludon>Duraludon>Metal>supporters; skip Cinderace"),
 ("tohand_explorer_metal_first_only", "priority_order", "draw_search", "SELECT_CARD", "card_identity", True, "take first Metal only (option-index dedup)"),
 ("discard_generic_priorities", "priority_order", "discard", "SELECT_CARD", "derived_state", True, "discard Metal-to-fuel>Cinderace>utility; keep Archaludon/Duraludon"),
 ("target_to_field_bench_pick", "priority_order", "draw_search", "SELECT_CARD", "card_identity", True, "to-field: Archaludon>Duraludon>Cinderace"),
 ("promote_own_pokemon_priority", "priority_order", "gust_promote_heal", "SELECT_CARD", "card_identity", True, "promote Cinderace>Archaludon>Duraludon"),
 ("target_heal_archaludon", "priority_order", "gust_promote_heal", "SELECT_CARD", "derived_state", True, "heal Archaludon (weighted by damage)"),
 ("gust_opponent_ko_or_drag", "priority_order", "gust_promote_heal", "SELECT_CARD", "derived_state", True, "gust KO-able target by prize+energy"),
 ("override_crustle_no_evolve_ex", "negative_veto", "matchup_tech", "EVOLVE", "matchup", "unknown", "vs Crustle: don't evolve to ex (10 Crustle rules total)"),
 ("boss_hop_pull_snorlax", "positive_select", "matchup_tech", "PLAY", "matchup", "unknown", "vs Hop: Boss the Snorlax"),
 ("opp_attack_log_tracking", "meta", "matchup_tech", "meta", "internal_hidden", False, "track opp last attack across turns via obs.logs (module-global)"),
]

# Phase-2 reconstruction outcomes -> rule-match class per family/rule (informed by the recovered trees).
RULE_MATCH = {
 "evolve_active_metal_discard_alloy": ("EXACT_FEATURE_MATCH", "recovered tree split metal_in_discard>1.5 -> select; matches the known mc>=2 evolve threshold exactly. F1 0.44 within EVOLVE."),
 "tohand_generic_line_and_resources": ("SEMANTIC_FEATURE_MATCH", "SELECT_CARD tree splits on source_card_id (card identity) -> recovers that card identity drives the pick (F1 0.68); the raw_index component is a partial shortcut."),
 "target_to_field_bench_pick": ("SEMANTIC_FEATURE_MATCH", "card-identity priority recoverable via source_card_id within SELECT_CARD."),
 "promote_own_pokemon_priority": ("SEMANTIC_FEATURE_MATCH", "card-identity own-promote priority recoverable via card id; partial support."),
 "setup_active_priority": ("UNIDENTIFIABLE", "setup decisions are few in natural traces; card-identity rule plausibly recoverable with support but n too small here."),
 "attach_metal_target_scoring": ("PARTIAL_MATCH", "required features present (target_card_id, energy_on_target, area) but the finely-graded multi-target priority is not recovered by a depth-2 tree (ATTACH F1 0.0). Needs a richer model or per-target framing, NOT a missing feature."),
 "retreat_to_attack_ready_ex": ("NO_MATCH", "depends on the archaludon_ex_attack_route COMPOSITE (active+bench energy vs retreatCost + retreated flag); that composite is not in the feature set -> MISSING_FEATURE. RETREAT F1 0.0."),
 "attack_score_by_damage": ("NO_MATCH", "attack damage / attackId is not in the feature set -> MISSING_FEATURE. ATTACK F1 0.11 (~option order only)."),
 "override_crustle_no_evolve_ex": ("UNIDENTIFIABLE", "opponents were random/first -> Crustle never appeared -> matchup never triggered (UNDERIDENTIFIED_NO_SUPPORT). Same for all ~12 Crustle/Hop matchup rules."),
 "boss_hop_pull_snorlax": ("UNIDENTIFIABLE", "no Hop opponent in traces -> no support."),
 "boss_save_vs_mega_brave_stuck": ("UNIDENTIFIABLE", "depends on module-global _opp_last_attack_id (cross-turn) -> HIDDEN_OR_INTERNAL, not reconstructible from a single public obs."),
 "opp_attack_log_tracking": ("UNIDENTIFIABLE", "internal cross-turn state via obs.logs replay; HIDDEN_OR_INTERNAL."),
}

# feature contract: which rule-needed feature classes the public extractor exposes.
FEATURE_CONTRACT = {
 "card_identity (source_card_id/target_card_id/family/option_index)": "AVAILABLE -- deck_policy_v3.option_card_id/option_target_entity + option.type; recovered SELECT_CARD/EVOLVE card splits.",
 "discard contents (metal_in_discard)": "AVAILABLE -- direct read of players[me].discard card ids; recovered the evolve trigger.",
 "deck/hand/prize counts": "AVAILABLE -- direct obs reads.",
 "turn context (turnActionCount/supporterPlayed/energyAttached/retreated)": "AVAILABLE -- obs.current.*; used in item/attach gates.",
 "target energy units / area (active vs bench) / HP-ratio": "AVAILABLE -- entity.energies/hp/maxHp + option.inPlayArea; present but the graded attach priority needs a richer model.",
 "attack damage / attackId / best_attack_damage": "MISSING from the diagnostic feature set (computable via attack_stats but not included here).",
 "attack-route composite (needs_retreat: bench attacker energy vs retreatCost)": "MISSING as a single feature -- a derived composite the rule computes; would need a dedicated extractor.",
 "matchup classification (opp deck = Crustle/Hop/Starmie/Lucario)": "AVAILABLE from opp board ids BUT UNSUPPORTED in these traces (opponents were random/first).",
 "opponent last-attack across turns (_opp_last_attack_id)": "HIDDEN/INTERNAL -- module-global from obs.logs replay; not in a single public obs.",
}


def cls_counts(d):
    out = {}
    for v in d.values():
        out[v[0]] = out.get(v[0], 0) + 1
    return out


def main():
    recon = json.load(open(OUT / "reconstruction_report.json", encoding="utf-8"))
    manifest = [{"rule_id": r[0], "rule_type": r[1], "decision_type": r[2], "action_family": r[3],
                 "feature_class": r[4], "runtime_observable": r[5], "condition": r[6]} for r in M]
    (OUT / "rule_manifest.json").write_text(json.dumps(
        {"pi_R": "submissions/sub_archaludon (Archaludon ex rule agent)", "n_distinct_rules_catalogued": len(M),
         "n_literal_clauses_estimate": "100-110", "rules": manifest,
         "note": "condensed from a full read; matchup tech (~12 Crustle/Hop rules) + 2 internal opp-tracking rules grouped."},
        indent=2), encoding="utf-8")
    (OUT / "feature_contract.json").write_text(json.dumps(FEATURE_CONTRACT, indent=2), encoding="utf-8")

    # identifiability per rule
    ident = {}
    for r in M:
        rid, fclass, obs_ok = r[0], r[4], r[5]
        if fclass == "internal_hidden":
            c = "HIDDEN_OR_INTERNAL"
        elif fclass == "matchup":
            c = "UNDERIDENTIFIED_NO_SUPPORT (opponents random/first -> matchup never triggered)"
        elif rid in ("attack_score_by_damage",):
            c = "MISSING_FEATURE (attack damage not in feature set)"
        elif rid in ("retreat_to_attack_ready_ex",):
            c = "MISSING_FEATURE (attack-route composite)"
        elif rid in ("evolve_active_metal_discard_alloy",):
            c = "IDENTIFIABLE (recovered)"
        elif fclass == "card_identity":
            c = "IDENTIFIABLE (card-identity; recovered where support present, else low-support)"
        else:
            c = "IDENTIFIABLE_PARTIAL (features present; graded priority under-recovered by shallow model)"
        ident[rid] = c
    (OUT / "identifiability_report.json").write_text(json.dumps(
        {"deviation_rate_from_option_zero": recon["deviation_rate_from_option_zero"],
         "global_selector_vs_option_zero_null": {"tree_f1": recon["selector_global"]["tree_f1"],
            "option_zero_null_f1": recon["selector_global"]["option_zero_null_f1"],
            "reading": "global selector is NOT better than the option-zero null and leans on raw_index -> a TIEBREAKER/option-order shortcut; reconstruction is only meaningful PER DECISION FAMILY."},
         "per_family_selector_f1": {k: v.get("tree_f1") for k, v in recon["per_family_selector"].items()},
         "per_rule_identifiability": ident}, indent=2), encoding="utf-8")

    # rule-match
    rm = {rid: {"match_class": v[0], "evidence": v[1]} for rid, v in RULE_MATCH.items()}
    (OUT / "rule_match_report.json").write_text(json.dumps(
        {"match_class_counts": cls_counts(RULE_MATCH), "per_rule": rm,
         "headline": "EXACT recovery of the core combo trigger (evolve when metal_in_discard>=2); SEMANTIC recovery "
                     "of card-identity SELECT_CARD priorities; PARTIAL on graded attach scoring (features present, "
                     "shallow model); MISSING_FEATURE for attack-damage + retreat-route composites; UNIDENTIFIABLE "
                     "for matchup tech (no Crustle/Hop opponents) and internal opp-tracking (hidden)."}, indent=2), encoding="utf-8")

    verdict = "B_HEURISTIC_RULE_RECONSTRUCTION_PARTIAL"
    closeout = {
        "task": "NATURAL HEURISTIC RULE RECONSTRUCTION LAB V0", "model": "B",
        "VERDICT": verdict,
        "pi_R": "submissions/sub_archaludon (Archaludon rule agent)",
        "traces": {"games": 20, "opponents": ["random", "first"], "decisions": recon["n_decisions"],
                   "option_rows": recon["n_option_rows"], "deviation_rate": recon["deviation_rate_from_option_zero"]},
        "what_recovered": ["EVOLVE: metal_in_discard>=2 trigger (EXACT)", "SELECT_CARD: card-identity priority (SEMANTIC, F1 0.68)"],
        "what_not": ["ATTACH graded target scoring (PARTIAL -- features present, shallow model)",
                     "ATTACK damage + RETREAT route composites (MISSING_FEATURE)",
                     "matchup tech (NO_SUPPORT: random/first opponents)", "opp last-attack tracking (HIDDEN/INTERNAL)"],
        "feature_layer_verdict": "NOT insufficient -- it exposed the key triggers (card identity, discard contents, "
                                 "turn context, energy/area). Gaps are (a) two derived COMPOSITES not yet extracted "
                                 "(attack damage, attack-route), and (b) natural-trace under-identification of matchup "
                                 "rules (need Crustle/Hop opponents) + genuinely hidden cross-turn state.",
    }
    (OUT / "closeout.json").write_text(json.dumps(closeout, indent=2), encoding="utf-8")
    print(json.dumps({"VERDICT": verdict, "match_counts": cls_counts(RULE_MATCH),
                      "recovered": closeout["what_recovered"], "per_family_f1": closeout["traces"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
