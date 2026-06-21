# Contextual Action Ranker V1 Calibration

Branch: `exp/robust-learner-v2`

Status: calibration implemented and evaluated. No promotion, no main merge, no old-objective DAgger Round 3.

## Artifacts

- Pre-calibration diagnostics: `docs/workstreams/contextual_action_ranker_v1_precalibration_diagnostics.json`
- Calibrated train/eval: `docs/workstreams/contextual_action_ranker_v1_calibrated_train_eval.json`
- Calibrated diagnostics: `docs/workstreams/contextual_action_ranker_v1_calibrated_diagnostics.json`
- Calibrated arena: `docs/workstreams/contextual_action_ranker_v1_calibrated_arena_search_ctx_vs_search_20g.json`
- Calibrated on-policy: `docs/workstreams/contextual_action_ranker_v1_calibrated_on_policy_search_ctx_4g.json`
- Live model consumed by `agent_search_ctx`: `agent/contextual_ranker_v1.json`

## Trainable vs Fixed

Decoded effects are trainable inputs. They enter as dense effect features and state x effect interaction features, then flow through the learned MLP. They are not final fixed bonuses.

Tactic Miner features are not consumed by V1. The closest hand-engineered feature is `target_engine_role`, a target/entity scalar built from decoded card effects with fixed preprocessing coefficients. It is an input to the learned scorer, not a final score, but it should be treated as a calibration risk.

Option deltas are trainable inputs. They are immediate one-step action consequences from `search.option_deltas`, not full-turn rollout leaves and not fixed final bonuses.

State x effect interactions are hand-constructed products, then trainable inputs. The model can assign different value to the same decoded effect in different root contexts.

Manual fixed weights in the final scorer: none. Manual preprocessing exists: feature scaling constants, `target_engine_role`, and the production forced-move floor for lethal/go-first before search. Search remains final authority; the contextual model orders candidates and breaks exact search-value ties.

## Calibration Fix

Found issue: the original model used `std + 1e-6` normalization. The 160-decision dataset is sparse, so 78 dense features had effectively zero standard deviation. Four features varied outside that train support, producing normalized values up to millions, especially in root/effect-interaction sections.

Fix:

- dense-feature std floor: `min_std=0.05`;
- normalized input clipping: `clip_z=8.0`;
- corrected ablation handling so ablations zero normalized features, matching live inference.

Effect:

| Diagnostic | Before | After |
|---|---:|---:|
| min std | 0.000001 | 0.050 |
| std <= 1e-5 | 78 | 0 |
| tiny-std features that vary | 4 | 0 |
| test full mean score margin | 9208.2 | 0.29 |
| test acceptable agreement | 0.563 | 0.625 |

The numeric instability is fixed. This did not make the live agent a promotion candidate.

## Offline Components

Held-out replay test, `n=16`:

| Model | Top-1 | Top-2 | Pairwise | MRR | NDCG | Acceptable | Mean Regret | P95 | High-Regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full calibrated | 0.375 | 0.438 | 0.643 | 0.546 | 0.930 | 0.625 | 122.3 | 726.6 | 0 |
| no decoded effects | 0.313 | 0.563 | 0.651 | 0.550 | 0.932 | 0.688 | 90.3 | 285.0 | 0 |
| no card embedding | 0.313 | 0.375 | 0.593 | 0.499 | 0.917 | 0.563 | 136.7 | 726.6 | 0 |
| no option deltas | 0.375 | 0.438 | 0.624 | 0.542 | 0.925 | 0.625 | 124.8 | 726.6 | 0 |
| old ranker baseline | 0.438 | 0.688 | 0.847 | 0.638 | 0.937 | 0.875 | 86.8 | 390.6 | 0 |
| option-0 baseline | 0.375 | 0.563 | 0.491 | 0.545 | 0.902 | 0.625 | 108.7 | 358.8 | 0 |

Interpretation: the old ranker remains better aligned to this tiny replay test slice. Do not claim offline replay improvement.

## Slice Diagnostics

Calibrated full model versus baselines:

| Slice | Model | Top-1 | Acceptable | Mean Regret | P95 | High-Regret |
|---|---|---:|---:|---:|---:|---:|
| all, n=160 | full | 0.513 | 0.919 | 10531.0 | 991.3 | 8 |
| all, n=160 | old ranker | 0.344 | 0.875 | 29607.4 | 170212.9 | 23 |
| recovery, n=80 | full | 0.588 | 0.925 | 20919.3 | 10653.1 | 8 |
| recovery, n=80 | old ranker | 0.325 | 0.800 | 59047.1 | 505426.6 | 19 |
| stable labels, n=91 | full | 0.604 | 0.912 | 123.5 | 640.0 | 3 |
| stable labels, n=91 | old ranker | 0.418 | 0.846 | 27207.8 | 2422.0 | 10 |
| unstable labels, n=69 | full | 0.391 | 0.928 | 24256.8 | 100310.2 | 5 |
| high-criticality, n=79 | full | 0.342 | 0.873 | 21304.2 | 18833.2 | 8 |
| high-criticality, n=79 | old ranker | 0.228 | 0.861 | 59883.8 | 511678.1 | 23 |
| high-regret recovery, n=18 | full | 0.389 | 0.722 | 83660.7 | 536399.5 | 7 |
| high-regret recovery, n=18 | old ranker | 0.000 | 0.611 | 132614.4 | 642815.1 | 13 |

The contextual model helps on the mixed recovery-heavy dataset it was trained for, especially relative to the old ranker tail. But the high-regret recovery tail remains large and not promotion-safe.

## Feature Contribution

Full-model sensitivity on all 160 decisions:

| Perturbation | Top-1 | Acceptable | Mean Regret | P95 | High-Regret |
|---|---:|---:|---:|---:|---:|
| full calibrated | 0.513 | 0.919 | 10531.0 | 991.3 | 8 |
| zero decoded effects + interactions | 0.381 | 0.831 | 27005.6 | 2367.6 | 17 |
| zero card embedding | 0.438 | 0.894 | 14437.4 | 1881.0 | 10 |
| zero target/entity | 0.450 | 0.875 | 11118.4 | 2238.9 | 14 |
| zero option deltas | 0.538 | 0.913 | 4302.9 | 828.2 | 7 |

Decoded effects, interactions, embeddings, and target/entity features are actively consumed by the trained scorer. Option deltas are trainable inputs, but this diagnostic says their current calibration is not clearly helpful; zeroing them improved aggregate regret. Treat deltas as inconclusive, not failed.

## Confirmation

Calibrated `agent_search_ctx` vs `agent_search`, same deck, same wall-clock policy:

| A | B | Result | Win Rate | Errors |
|---|---|---:|---:|---:|
| search_ctx | search | 10-10 | 0.500 | 0 |

Small calibrated on-policy diagnostic for `search_ctx` vs heuristic, first 120 teacher-labelled visited decisions:

| Metric | Value |
|---|---:|
| trace | 3-1 |
| teacher-applicable | 119 |
| acceptable agreement | 0.924 |
| hard top-1 agreement | 0.643 |
| mean regret | 6307.7 |
| p90 regret | 66.5 |
| p95 regret | 120.5 |
| high-regret count | 2 |

The low p90/p95 with high mean regret means a small number of extreme outliers dominate the tail. This fails the promotion safety bar.

## Teacher V2 Request

Model B needs Teacher V2 labels that emphasize:

- high-criticality states and high-regret recovery states;
- stable soft policies, not just hard argmax;
- deeper-search advantages/regret with common-seed or paired counterfactual values where possible;
- per-action uncertainty/stability and action spread;
- trigger metadata explaining why selective computation was used;
- determinization count, candidate coverage, actual search time, and whether deeper/opponent-sensitive search changed the action;
- legal sibling metadata aligned to semantic action keys and option-delta rows.

Most useful format: one record per root decision containing legal sibling actions, semantic keys, Teacher V2 soft distribution, class advantages/regrets, acceptable set, criticality score, uncertainty/stability fields, and enough metadata to join back to the contextual feature rows.

## Recommendation

Pause promotion. The calibrated model is a real contextual scorer and it fixes a numeric bug, but confirmation washed against `agent_search`, and the on-policy high-regret tail is still unacceptable.

Next step: wait for Teacher V2 high-criticality labels or run a narrow objective/weighting revision focused on high-regret tail suppression and option-delta calibration. Do not run old-ranker DAgger Round 3, do not merge to main, and keep `agent_search` as the live baseline.
