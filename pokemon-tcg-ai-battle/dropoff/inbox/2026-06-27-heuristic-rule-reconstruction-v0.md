# Natural heuristic rule reconstruction lab V0 (2026-06-27)

Model B feature-forensics diagnostic. **Verdict: B_HEURISTIC_RULE_RECONSTRUCTION_PARTIAL.** Not a gameplay
change, not a claim pi_R is good, default behavior intact, 14/14 suites pass. The question was narrow: can the
public FEATURE LAYER recover the state/action conditions behind a KNOWN rule policy from NATURAL game traces?

## Setup
- pi_R = the Archaludon rule agent (`submissions/sub_archaludon/main.py`, ~100 scoring clauses, 57 distinct
  rules catalogued). The `.ipynb` itself isn't in the repo, but the built rule agent IS, so the section-0 gate
  is met (the prompt treats notebook/rule-agent as pi_R).
- Two-phase pipeline to avoid the sub_archaludon `cg` colliding with the worktree engine: Phase 1 runs pi_R in
  natural games and dumps raw obs + action; Phase 2 (worktree env) extracts features + fits interpretable models.
- Natural traces only (no artificial states): 20 games vs random + first, **725 decisions / 3869 option rows**,
  deviation-from-option-zero rate 0.57 (good support). Starmie baseline was NOT used as an opponent to keep
  pi_R's bundled cg isolated.

## What the feature layer RECOVERED (the positive result)
- **EVOLVE -- the core combo trigger, EXACT match.** A depth-2 tree on the EVOLVE options recovered
  `metal_in_discard > 1.5 -> select`, which is exactly pi_R's known rule "evolve Active Duraludon when >=2 Metal
  in discard (Assemble Alloy attaches 2)". The public feature `metal_in_discard` (read from the discard pile)
  exposes the trigger.
- **SELECT_CARD -- card-identity priorities, SEMANTIC match (F1 0.68).** The tree splits on `source_card_id`,
  recovering that card identity (take Archaludon/Duraludon line pieces, skip Cinderace) drives the search/draw
  pick. Public `option_card_id` exposes it.

## What it did NOT recover (and why -- the honest gaps)
- **ATTACH graded target scoring -- PARTIAL.** The required features (target_card_id, energy_on_target, area,
  HP) ARE present, but pi_R's finely-graded multi-target priority (Cinderace@0 / Dura-Arch@2, HP-ratio tweaks)
  is not captured by a depth-2 tree -> F1 0.0. This is a model-capacity / framing gap, NOT a missing feature.
- **ATTACK + RETREAT -- MISSING_FEATURE.** Attack selection ranks by attack DAMAGE (attackId -> base dmg), and
  retreat depends on an attack-route COMPOSITE (bench attacker energy vs retreatCost + the retreated flag).
  Neither was in the diagnostic feature set -> F1 ~0. Both are computable (attack_stats / a route extractor) but
  not exposed as single features here.
- **Matchup tech (~12 Crustle/Hop/Starmie/Lucario rules) -- UNDERIDENTIFIED_NO_SUPPORT.** Opponents were
  random/first, so those decks never appeared and the matchup rules never triggered. Not a feature failure;
  needs the right opponents (the local_meta_v1 roster has lucario/koraidon/abomasnow -- a natural follow-up).
- **Opp last-attack tracking (Mega-Brave Boss) -- HIDDEN_OR_INTERNAL.** Depends on pi_R's module-global
  `_opp_last_attack_id`, accumulated by replaying `obs.logs` across turns. Genuinely not in a single public obs.
- **Global selector ~ option-zero null (raw_index shortcut).** The pooled selector (F1 0.385) is no better than
  "always pick option 0" and leans on option order -- a tie-break SHORTCUT. Reconstruction is only meaningful
  PER DECISION FAMILY, which is how the positives above were found.

## Rule-match tally
EXACT 1, SEMANTIC 3, PARTIAL 1, NO_MATCH (missing-feature) 2, UNIDENTIFIABLE (no-support/hidden) 5.

## What this tells the project (the point of the lab)
The feature layer is **NOT insufficient** -- it exposed the conditions for the highest-value rules (the combo
trigger via discard contents, card-identity priorities). The gaps are specific and actionable:
1. add two derived COMPOSITES as features when needed: attack damage (attack_stats) and an attack-route /
   needs-promote composite;
2. to identify matchup rules, re-run vs the real archetype opponents (local_meta_v1 lucario/koraidon/abomasnow),
   not random/first;
3. some rules are inherently hidden (cross-turn opp tracking) -- accept as out-of-scope for single-obs features.

This validates the "atomic representation first" direction: a public x(s,a) with card identity + discard/turn/
energy context can recover a rule policy's key triggers, while exposing exactly which composites are missing.

## Artifacts
`data/generated/heuristic_rule_reconstruction_v0/`: rule_manifest.json (57 rules), feature_contract.json,
trace_manifest.json, natural_trace_raw.jsonl (725 decisions), reconstruction_report.json (per-family trees +
F1 + nulls), identifiability_report.json, rule_match_report.json, review_examples.{jsonl,html}, closeout.json.
Tools: heuristic_trace_gen_v0.py (Phase 1, runs pi_R), heuristic_rule_recon_v0.py (Phase 2, features+models),
heuristic_rule_match_v0.py (Phase 3, manifest+match). All read-only; no gameplay change.
