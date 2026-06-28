# Heuristic rule reconstruction GAP audit V1 (2026-06-27)

Model B. Re-classified every feature/composite the V0 reconstruction did not recover cleanly, verified against
the actual V0 feature table, the worktree helpers, and the rule agent. Read-only: no new games, no feature
implementation, no gameplay change. **Verdict: E_MIXED.**

## Headline
**No item is truly missing from public data (category 4 = 0).** The V0 handoff over-labeled several things
"MISSING_FEATURE"; on inspection they are mostly *derivable but not extracted*. The gaps split across four
distinct causes, which is why the verdict is MIXED:

| category | count | items |
|---|---|---|
| 3 derivable_with_helpers_not_extracted | 3 | attack damage, attack route/needs-promote, retreat route |
| 7 model_framing | 2 | ATTACH graded target scoring, global option-index shortcut |
| 5 trace_support_absent | 1 | matchup-specific rules |
| 6 hidden_internal_crossturn | 1 | opponent last attack |
| 1 present_in_v0 | 1 | SELECT_CARD source/target (already recovered) |
| 4 missing_from_public_data | 0 | — |

## Per-item verification (the 7 you asked about)
- **attack damage -> CAT 3 (derivable, not extracted).** The option carries `attackId`
  (`learned_selector_bridge.option_features` 'attack_id' :249), and `agent/attack_stats.json` is keyed by
  attackId with `d`=damage. The agent's own `best_attack_damage(obs,attack_id)` (:285) does exactly this lookup;
  Raging-Hammer scaling = 80 + damage_on(active) (damage = maxHp-hp, derivable). V0 simply didn't extract it.
- **attack route / needs-promote -> CAT 3 (derivable composite).** `archaludon_ex_attack_route` (agent:358) is a
  composite of public-derivable parts: bench/active energy units (`TS.energy_units` :76), retreat cost
  (`TS._retreat_cost` :173), and the `retreated` flag (already in V0). The composite was never built.
- **retreat route -> CAT 3.** Same composite; plus `retreat_dont_break_tank` needs only active.id + has_tool +
  hp (all public). RETREAT F1 0.0 because the composite wasn't extracted, not because data is missing.
- **opponent last attack -> CAT 6 (hidden/cross-turn).** `_update_opp_attack_tracking` (agent:116-119) reads
  `obs.logs` but accumulates `_opp_last_attack_id` across turns into a module global. A single obs carries only a
  short logs window (partial, cat 2); the reliable value is internal cross-turn state. The one genuine dead-end.
- **matchup-specific rules -> CAT 5 (trace support absent).** `detect_matchup(obs)` (agent:396) classifies the
  opponent deck from public opp-board ids -- derivable. They went unrecovered only because V0 ran vs random/first,
  so Crustle/Hop/Starmie/Lucario never appeared and the ~12 rules never fired.
- **ATTACH graded target scoring -> CAT 7 primary (+3).** F1 0.0 is mainly FRAMING: the rule is a per-decision
  ARGMAX over graded scores; V0 pooled all attach options into a binary classifier, which cannot represent
  "highest score within THIS decision." Secondary: area (`target_zone`/`source_zone` in option metadata,
  :266,294) and HP-ratio (target.hp/maxHp) weren't extracted. target_card_id + energy_on_target ARE present.
- **SELECT_CARD source/target -> CAT 1 (present) + 7.** Both ARE in V0 and WERE recovered (tree split on
  `source_card_id`, F1 0.68). Not EXACT only because the multi-level priority is a within-decision ranking a flat
  pooled tree approximates, and it leaned partly on the raw_index shortcut. No feature gap.

## Biggest levers (if/when this is acted on -- not now)
1. Extract two derivable feature groups: attack_damage (attackId + attack_stats) and the attack/retreat route
   composite (energy_units + retreat_cost + retreated). Closes the 3 category-3 items.
2. Fix model FRAMING: analyze per decision-family with within-decision ranking (learning-to-rank / per-decision
   argmax) instead of a pooled binary selector. Closes ATTACH (F1 0->meaningful) and removes the option-index
   shortcut; likely tightens SELECT_CARD SEMANTIC -> EXACT.
3. Give matchup rules trace support: re-run pi_R vs the archetype opponents already wired in local_meta_v1
   (lucario/koraidon/abomasnow), not random/first.

Only `opponent_last_attack` (category 6) is out of scope for single-obs features.

## Artifacts
`data/generated/heuristic_rule_reconstruction_gap_v1/`: gap_summary.json, rule_gap_table.json,
feature_missingness_table.md. Tool: `tools/heuristic_rule_gap_v1.py` (read-only). Corrects the V0 handoff's
MISSING_FEATURE labels (attack damage, routes are derivable; ATTACH is framing).
