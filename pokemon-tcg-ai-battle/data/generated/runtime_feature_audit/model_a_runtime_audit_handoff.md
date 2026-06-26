# Model A Handoff: Starmie Runtime Observation Feature Audit V0

Model B, runtime-side cross-check for the Public Information Completeness Audit. **Audit only — no gameplay, no
model/heuristic changes.** Verdict: **B_RUNTIME_TURN_CONTEXT_GAPS_FOUND**.

## Headline (validates the "turn context is most urgent" hypothesis, with good news)
The live CABT obs is **information-rich** — every public fact I audited is observable. The one actionable runtime
gap is **turn context**: it is present in **every** observation but reaches **neither the live adapter nor the
model's features**. It is an EXTRACTION gap (cheap), not an availability gap, and it maps directly onto the
selector's develop-vs-attack failure (the selector cannot see how much of the turn is already spent).

## Turn context is ALL there, in `current.*` (100% presence, sampled n=501)
| feature | obs source | classification |
|---|---|---|
| who went first | `current.firstPlayer` | DIRECT |
| global turn number | `current.turn` | DIRECT |
| action index in turn | `current.turnActionCount` | DIRECT |
| supporter used this turn | `current.supporterPlayed` | DIRECT |
| attachment used this turn | `current.energyAttached` | DIRECT |
| retreat used this turn | `current.retreated` | DIRECT |
| stadium in play | `current.stadium`/`stadiumPlayed` | DIRECT |
| summoning sickness | `entity.appearThisTurn` | DIRECT |
| status conditions | `player.asleep/paralyzed/confused/burned/poisoned` | DIRECT |
| attack/end available, nonterminal count, player-turn count | scan options / derive | DERIVABLE |
| previous action families this turn | `obs.logs` is only a short recent list | NEEDS LOCAL MEMORY |
| ability-used flags | not in obs | UNSUPPORTED / LOCAL MEMORY |

13 of 16 are direct-or-derivable; **no history tracker is needed for the key signals.**

## Where the gap sits on your feature-path (A->H)
- **A raw observation**: HAS all turn-context fields (confirmed, 100%).
- **C Feature-V2 packer / D-E model input**: the model's 11 state features (from feature_vocab.json) are
  bias/option_count/our+opp hand/deck/prize/bench/attack_ready — **zero turn-context**. So the trained model is
  blind to it too: adding it **requires retraining**, not just adapter extraction.
- **G live adapter (learned_selector_bridge)**: consumes entity hp/maxHp/energies/id + counts + select only; it
  does **not** read `current.turn*`/status/tools/stadium/discard. So even a retrained model couldn't see turn
  context live until the adapter extracts it.
- **Net:** turn context needs BOTH a packer/training-feature addition (your lane) AND an adapter extraction (mine),
  in lockstep, to preserve parity. (Recall: missing tactical features previously broke adapter parity until restored.)

## Card mechanic observability (Section 3)
- `card_stats.json` provides printed facts at runtime: HP, weakness (`wk`), resistance (`rs`), type (`ty`), stage,
  retreat, prize, ex/mega, attacks (cost/dmg/cE). All available; the model just sees them opaquely via `card_id`.
- `card_effects.json` (581 cards) HAS action taxonomy: search/heal/gust/switch/discard/has_ability.
- **Gaps in the effect taxonomy** (the "card_id is opaque" cases you flagged): ignore-weakness/resistance/effects,
  damage-prevention, and conditional/scaling damage are **NOT** in card_effects. Nebula Beam's flat-210-ignore and
  Alakazam's 20x-hand are **hardcoded in deck_policy.attack_profile**, not clean features -> invisible to the model.
- Special-energy units (Ignition = 3) are pre-expanded by the engine into `entity.energies` (OBSERVABLE); but
  "Ignition only on evolution / discarded end of turn" is hardcoded, not a card_effect feature.

## Ranked runtime blind spots
1. **CRITICAL turn_context_features** — present every obs, extracted nowhere; drives develop-vs-attack. Needs retrain + adapter.
2. **HIGH status_conditions + summoning_sickness** — present, not extracted; affect attack/retreat legality + tempo.
3. **HIGH discard_zone_belief** — both discards fully visible; unused for prize/deck-copy belief.
4. **MEDIUM conditional_damage / ignore / prevention taxonomy** — hardcoded in deck_policy, not card_effects (your lane).
5. **LOW tools/stadium** — `entity.tools` (Hero's Cape +100 HP shifts KO threshold), `current.stadium`; not extracted.
6. **LOW full same-turn action history** — only `obs.logs` short list; needs a local tracker.

## Forbidden runtime field to exclude
`current.result` (game outcome, -1 mid-game) is present in the obs — must NEVER become a feature.

## Artifacts
`data/generated/runtime_feature_audit/`: live_observation_inventory_v0.json, turn_context_tracker_feasibility_v0.json,
card_mechanic_observability_v0.json, runtime_blind_spots_v0.json, model_a_runtime_audit_handoff.md.
Tool: `tools/runtime_feature_audit_v0.py`.

VERDICT=B_RUNTIME_TURN_CONTEXT_GAPS_FOUND
