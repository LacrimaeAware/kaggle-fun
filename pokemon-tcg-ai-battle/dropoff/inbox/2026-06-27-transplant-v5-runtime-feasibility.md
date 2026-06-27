# Transplant V5 runtime-feature feasibility — FEASIBLE (2026-06-27)

Model B support pack for Model A's context-delta transplant V5 (T(s,a,delta)). **Verdict:
A_V5_RUNTIME_FEATURES_FEASIBLE.** Audit/data only: no A/B, no gameplay change, no heuristic/selector/table
edit, selector default off, 12/12 test suites pass. V3 stays parked (powered-neutral).

## The question and the answer
Can the live runtime compute the `(s, a, delta_a)` inputs a state-conditioned transplant needs, from public
CABT obs + legal options, WITHOUT a full turn rollout, opponent simulation, hidden cards, or the result?
**Yes.** Verified empirically on 129 golden-state fixtures / 200 example payloads, deterministic, 0 missing
context fields, ~3ms/fixture.

## What is computable live (mechanisms, all already in the repo)
- **CONTEXT** (static read): `turn_context_v0.extract_turn_context(obs)` +
  `learned_selector_bridge.tactical_state_features(obs)` + `state_features(obs, options)` cover turn/turnAction
  Count/supporter/energy/retreat/first-player, deck/hand/prize counts, ready-attackers, backup continuity,
  safe-development, deck-out pressure, board HP, prize diff. All direct or derivable.
- **ACTION** (static, per option): `learned_proposer_adapter.option_index_to_key` (compact key) +
  `learned_selector_bridge.option_features` (family, source/target card, target owner/zone, terminal flag);
  target_role via `starmie_tactical_state.semantic_role`.
- **DELTA-realized** (one engine apply/option via `search_v3.option_deltas`, deterministic): cards_drawn,
  deck_used, dmg_dealt (opp active), energy_attached, opp_ko/prizes_taken, board_dev, ends_turn, wins_now.
  Verified nonzero across fixtures (cards_drawn 697, energy_attached 354, prizes_taken 210, opp_ko 12).
- **DELTA-capability/threshold** (the only NEW code needed: a THIN adapter): recompute the EXISTING static
  tactical levels on the post-apply obs and diff pre/post -> ready_attacker_delta, ko_available_delta,
  attack_affordability/threshold-crossing, energy_shortfall_delta, line_completion_delta. Demonstrated working
  (e.g. PLAY:Ultra Ball -> nonterminal_attack_available delta = -1). Valid for non-terminal options; terminal
  options use the realized deltas.

Nothing is blocked, hidden, or needs a full rollout. Live `T(s,a,delta)` lookup is feasible: `s`/`a` static,
`delta_a` = one apply + thin pre/post recompute.

## Four gotchas Model A must handle (in the contract)
1. **damage**: `entity['damage']` is absent live; `learned_selector_bridge._damage` returns 0 (latent bug) ->
   the model's per-entity damage feature is dead on the live path. Use `max(0, maxHp-hp)`.
2. **key format**: compact (`proposer.semantic_key`) vs structured JSON (packer) differ -- the exact mismatch
   that made V3 inert. Pick ONE canonical key for memory-build AND runtime-lookup.
3. **board_hp_delta** via option_deltas is opp-active only; full board needs per-entity hp pre/post.
4. **energy units vs cards**: standardize affordability/shortfall on UNITS (Ignition=3), not card count.

## Cost
context+action static (~free); realized+capability deltas = one `cg.api.search_begin` + one `search_step` per
option (~3ms/fixture across all options on golden fixtures; well within the 0.6s/decision budget). Scope the
apply to candidate options if option counts are large.

## Artifacts
`data/generated/transplant_v5_runtime_support/`: feature_feasibility_report.json (per-axis classification +
verdict), immediate_delta_probe.json (empirical probe: determinism, realized + capability deltas by family,
cost), example_payloads.jsonl (200 live-shape `{decision_id, legal_option_index, context, action, delta,
missing_fields, runtime_safe}`), model_a_contract.md (exact field names B can provide live, offline-only,
gaps, cost, feasibility). Tools: `tools/transplant_v5_feasibility_probe_v1.py`. Tests:
`tests/test_transplant_v5_support_v1.py` (added to run_all; no-mutation, no-leakage, key-stable, terminal-flag,
missing-explicit, engine-backed ATTACH/PLAY delta).

## For Model A
The full field-name contract is in `model_a_contract.md`. The only build needed for runtime is the thin
capability-delta adapter (pre/post tactical recompute across one apply) plus the four gotcha fixes. Everything
else is already callable. Model B can, on request: build that adapter, emit a larger payload corpus, or run a
runtime cost benchmark.
