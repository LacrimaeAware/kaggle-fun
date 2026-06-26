# Starmie Runtime Observation Feature Audit V0 (2026-06-26)

Model B runtime-side cross-check for the Public Information Completeness Audit. **Audit only; gameplay unchanged.**
Verdict: **B_RUNTIME_TURN_CONTEXT_GAPS_FOUND**.

## The one finding that matters
Turn context is **present in every live obs at 100%** (`current.turn`, `firstPlayer`, `turnActionCount`,
`supporterPlayed`, `energyAttached`, `retreated`, plus `entity.appearThisTurn`, `player.<status>`) — but it reaches
**neither the live adapter nor the model's 11 state features**. The user's hypothesis was right that turn context
is the urgent blind spot; the good news is it is an EXTRACTION gap, not an availability gap. 13 of 16 turn-context
features are direct `current.*` fields or trivially derivable — no history tracker needed for the key ones.

This maps directly onto the selector failure: the develop-vs-attack decision needs "how much of the turn is already
spent" (turnActionCount + supporterPlayed + energyAttached), and the selector was blind to all of it.

## Where the gap sits
- Raw obs: HAS it (100%).
- Model features (feature_vocab): the 11 state features are all board/prize counts -> ZERO turn-context. So the
  trained model is blind too; fixing requires RETRAINING (Model A), not just adapter extraction.
- Live adapter (learned_selector_bridge): reads entity hp/energies/id + counts + select only -> does not read
  current.turn*/status/tools/stadium/discard.
- => needs BOTH a packer/training-feature add (Model A) AND an adapter extraction (Model B), in lockstep for parity.

## Card mechanics (secondary)
card_stats provides printed facts at runtime (HP/wk/rs/type/stage/retreat/attacks); card_effects has
search/heal/gust/switch/discard tags but LACKS ignore/prevent/conditional-damage. Nebula flat-ignore + Alakazam
20x-hand are hardcoded in deck_policy, not clean features -> opaque to the model (your "card_id is too opaque" case).

## Ranked blind spots
1 CRITICAL turn-context; 2 HIGH status+summoning-sickness; 3 HIGH discard-belief; 4 MEDIUM conditional/ignore/prevent
taxonomy; 5 LOW tools(Cape +100 HP KO threshold)/stadium; 6 LOW full action-history.

Forbidden runtime field present: `current.result` (outcome) -- must never be a feature.

## Recommended next step (when greenlit)
Turn-context is the highest-leverage, cheapest add. It needs a coordinated change: Model A adds the features to the
packer + retrains the proposer/selector; Model B extends learned_selector_bridge to extract `current.*` into
state_features + tactical, re-running the parity gate to keep the adapter bit-exact. Do them together or parity breaks.

## Artifacts
`data/generated/runtime_feature_audit/`: live_observation_inventory_v0.json, turn_context_tracker_feasibility_v0.json,
card_mechanic_observability_v0.json, runtime_blind_spots_v0.json, model_a_runtime_audit_handoff.md.
Tool: `tools/runtime_feature_audit_v0.py`.
