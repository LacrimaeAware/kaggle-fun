# Teacher V2 Integration Plan for Contextual Ranker

Branch: `exp/robust-learner-v2`

Base commit before this preparation: `a5137ba`

Status: implemented loader/alignment preparation. No Teacher V2 training yet, no arena rerun, no live promotion.

## Current State

Live baseline remains `agent_search`.

Current Branch B live candidate remains `agent_search_ctx`, consuming `agent/contextual_ranker_v1.json`. It is implemented and calibrated, but not accepted:

- calibrated `search_ctx` vs `search`: 10-10 over 20 games;
- calibrated offline replay test: full mean regret 122.3 versus old ranker 86.8;
- calibrated on-policy: acceptable 0.924, hard top-1 0.643, mean regret 6307.7, high-regret count 2.

Evidence type: offline, on-policy, and arena. Status: implemented/trained/evaluated, inconclusive-to-negative for promotion.

## Loader

Implemented:

- `tools/align_teacher_v2_labels.py`

Artifacts:

- `docs/workstreams/teacher_v2_alignment_selftest.json`
- `docs/workstreams/teacher_v2_alignment_check.json`

The self-test verifies decision/option alignment on a synthetic positive fixture:

- teacher decisions: 1;
- matched decisions: 1;
- alignment ready for training: true.

The real alignment check against `docs/workstreams/contextual_action_ranker_v1_dataset.json` found no Teacher V2 artifact in this worktree yet:

- teacher decisions: 0;
- matched decisions: 0;
- alignment ready for training: false.

This is expected. It is not a failure of the loader and it is not training evidence.

## Alignment Contract

Teacher V2 records should align by at least one of:

- `decision_id`;
- `root_decision_id`;
- `id`;
- `obs_hash`;
- replay identity such as `game_file` plus `step` or `call`;
- recovery identity such as `source`, `game`, and `call`.

Each decision should expose legal sibling actions under one of:

- `legal_siblings`;
- `options`;
- `actions`;
- `siblings`.

Each option must include:

- `option_index`;
- `semantic_action_key`;
- `eq_class`;
- `hand_mean_value`;
- `hand_value_variance`;
- `hand_norm_advantage`.

Auxiliary option fields:

- `outcome_winrate`;
- `outcome_playouts`;
- `outcome_variance` or `outcome_confidence`.

Decision-level fields:

- hand soft policy;
- acceptable set;
- criticality score;
- determinization/coverage/timing metadata;
- seed or pairing metadata.

The loader checks option index range, semantic key equality, and eq-class match/remappability. It will not treat a record as training-ready if siblings cannot be aligned.

## Target Usage

Primary target, once Teacher V2 is validated:

```text
hand_norm_advantage
  weighted by criticality_score
  weighted by inverse hand_value_variance / stability
  grouped by semantic sibling-action class
```

Auxiliary targets:

- Teacher V2 soft policy;
- acceptable set;
- high-regret recovery flag;
- outcome_winrate only if Model A validates higher-k reliability.

Do not make k=4 outcome win rate a primary target. If Model A reports higher-k disagreement is persistent and confidence improves, outcome can become an auxiliary ranking/regression target with conservative weight. If disagreement collapses or remains noisy, outcome should stay out of the main objective.

## Objective Revision

The next revised training run should be narrow and tail-focused:

1. Keep the contextual architecture and grouped sibling-action objective.
2. Add criticality-weighted pairwise/listwise loss.
3. Add high-regret tail penalty:
   - upweight examples where selected/predicted class has high Teacher V2 regret;
   - penalize unacceptable predictions on high-criticality states more strongly than ordinary near-ties.
4. Use confidence/stability weighting:
   - down-weight high-variance hand labels;
   - down-weight unstable or incomplete coverage;
   - keep outcome labels auxiliary unless validated.
5. Revisit option-delta calibration:
   - the calibrated diagnostic showed zeroing deltas improved aggregate regret;
   - keep deltas as trainable inputs, but consider lower interaction capacity, stronger regularization, or delta-section dropout.
6. Reduce overreaction to sparse effects/deltas:
   - keep `min_std=0.05` and `clip_z=8.0`;
   - retain normalized-feature ablation parity between training and live inference.

No old-objective DAgger Round 3. No global state value proxy. No hard argmax-only imitation objective.

## Required Diagnostics

Every Teacher V2 retrain must report:

- full contextual;
- no decoded effects;
- no card embedding;
- no option deltas;
- old ranker baseline;
- option-0 baseline.

Report slices:

- high-criticality decisions;
- high-regret tail;
- recovery states;
- stable versus unstable labels;
- mixed strategic decisions;
- held-out player/deck where available.

Metrics:

- within-decision top-1;
- top-k;
- pairwise accuracy;
- MRR/NDCG;
- acceptable-set agreement;
- mean regret;
- p90/p95 regret;
- high-regret count.

## When To Retrain

Retrain only after at least one condition is true:

- Model A provides a Teacher V2 batch that passes `align_teacher_v2_labels.py`;
- Model A validates higher-k outcome labels well enough for auxiliary use;
- a specific objective/weighting fix is approved that targets the current high-regret tail without needing Teacher V2 outcome labels.

Do not run another `search_ctx` vs `search` arena screen unless the model changes through one of those routes.

## Model A Request

Most useful Teacher V2 label batch:

- high-criticality decisions;
- high-regret recovery states;
- stable hand soft policies and advantages;
- hand variance and completed determinization counts;
- higher-k outcome win rates only with confidence/variance;
- common-seed or paired counterfactual metadata where possible;
- candidate coverage and actual search time;
- whether all sibling options completed;
- whether deeper/opponent-sensitive search changed the preferred action;
- semantic action keys and option indices for every sibling.

Model B can consume hand advantage immediately once aligned. Outcome supervision should wait for Model A's higher-k validation recommendation.

## Recommendation

Wait for Teacher V2 labels. Branch B is prepared to ingest them, but no reliable Teacher V2 batch is present in this worktree yet. Keep `agent_search` as live baseline and keep `agent_search_ctx` unpromoted.
