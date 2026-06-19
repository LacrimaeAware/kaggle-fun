# Continuous Terrain Representation V1

Status: round-2 smoke test only.

No live agent was modified and no arena screen was run.

## Dataset

- Input: `data\manifests\teacher_v2_residual_risk_labels_round2.jsonl`
- Decisions/options/games: 60 / 588 / 13
- Eval-only decisions: 2
- Failures: 0
- Split: train games ['80251230.json', '80251834.json', '80252178.json', '80252701.json', '80253882.json', '80253888.json', '80270516.json', '80277480.json', '80280539.json']; val games ['80253044.json', '80275931.json']; test games ['80271583.json', '80279946.json']; eval-only ['80251230.json', '80252701.json']

## Architecture

- Learned card-id embedding dimension: 32.
- Effect vector MLP, dynamic entity MLP, DeepSets zone pooling, 128-d state encoder.
- Action encoder combines action type, acting card embedding, target entity, decoded effects, option deltas, and action scalars.
- Semantic latent is separated from the search-metadata branch; R3 uses semantic only, R4 adds metadata.
- Homoscedastic task weights are learned for ranking, risk, acceptability, instability, residual, and contrastive losses.

## Trainable Embedding Check

- `R3_semantic` card embedding grad norm, last epoch: 0.011268
- `R4_semantic_plus_search` card embedding grad norm, last epoch: 0.007920
- `R4_no_card_embedding` card embedding grad norm, last epoch: 0.000000
- `R4_no_decoded_effects` card embedding grad norm, last epoch: 0.006930
- `R4_no_target_entity` card embedding grad norm, last epoch: 0.008878
- `R4_no_option_deltas` card embedding grad norm, last epoch: 0.011374
- `R4_no_contrastive` card embedding grad norm, last epoch: 0.001068
- `R4_no_ranking` card embedding grad norm, last epoch: 0.008344

## Learned Task Weights

| variant | ranking | high_regret | unacceptable | acceptable | instability | residual | contrastive |
|---|---:|---:|---:|---:|---:|---:|---:|
| `R3_semantic` | 0.995 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 0.995 |
| `R4_semantic_plus_search` | 0.995 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 0.995 |
| `R4_no_card_embedding` | 0.995 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 0.995 |
| `R4_no_decoded_effects` | 0.995 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 0.995 |
| `R4_no_target_entity` | 0.995 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 0.995 |
| `R4_no_option_deltas` | 0.995 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 0.995 |
| `R4_no_contrastive` | 0.995 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 |
| `R4_no_ranking` | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 0.995 |

## Predictive Metrics

High-regret AP/AUROC on held-out test games:

| representation | AP | AUROC | recall@FPR10 |
|---|---:|---:|---:|
| `R0_root_engineered` | 0.742 | 0.768 | 0.136 |
| `R1_search_metadata_only` | 0.849 | 0.830 | 0.227 |
| `R2_current_engineered_contextual` | 0.770 | 0.636 | 0.273 |
| `R3_semantic` | 0.784 | 0.776 | 0.273 |
| `R4_no_card_embedding` | 0.448 | 0.256 | 0.000 |
| `R4_no_contrastive` | 0.859 | 0.838 | 0.636 |
| `R4_no_decoded_effects` | 0.777 | 0.557 | 0.500 |
| `R4_no_option_deltas` | 0.668 | 0.486 | 0.136 |
| `R4_no_ranking` | 0.555 | 0.526 | 0.000 |
| `R4_no_target_entity` | 0.502 | 0.213 | 0.091 |
| `R4_semantic_plus_search` | 0.805 | 0.653 | 0.500 |

## Signal Radius

k=10 held-out-game high-regret enrichment:

| representation | bg rate | neighbor rate | enrich | queries |
|---|---:|---:|---:|---:|
| `R0_root_engineered` | 0.597 | 0.624 | 1.046 | 22 |
| `R1_search_metadata_only` | 0.597 | 0.608 | 1.019 | 22 |
| `R2_current_engineered_contextual` | 0.597 | 0.640 | 1.073 | 22 |
| `R3_semantic` | 0.597 | 0.632 | 1.059 | 22 |
| `R4_semantic_plus_search` | 0.597 | 0.616 | 1.032 | 22 |
| `R4_no_card_embedding` | 0.597 | 0.672 | 1.126 | 22 |
| `R4_no_decoded_effects` | 0.597 | 0.624 | 1.046 | 22 |
| `R4_no_target_entity` | 0.597 | 0.632 | 1.059 | 22 |
| `R4_no_option_deltas` | 0.597 | 0.640 | 1.073 | 22 |
| `R4_no_contrastive` | 0.597 | 0.640 | 1.073 | 22 |
| `R4_no_ranking` | 0.597 | 0.592 | 0.992 | 22 |

## Limitations

- Round-2 smoke-test data is not the expanded continuous terrain dataset.
- Only the final A artifact may be used for the true representation verdict.
- The old round-2 c1 class remains sparse and game-clustered.
- Smoke mode reconstructs some features from replays; final A data is expected to be self-contained.

## Verdict

**D. CURRENT DATA UNDERPOWERED**

Smoke-test only: final verdict requires Model A continuous_terrain_v1.jsonl. The round-2 artifact is too small and c1-clustered for the required gate.

One next experiment: Run this same pipeline on Model A's expanded continuous_terrain_v1.jsonl once available.
