# Continuous Terrain Representation V1

## Clean Rerun Warning

The original R1/R4 comparison in this report is confounded. A follow-up audit found N=32-derived metadata leakage through `value_spread`/`value_se`, schema-mismatch dead fields, teacher `policy_prob` leakage in semantic action scalars, and duplicated high-regret/unacceptable labels in the pre-patch dataset. Use `docs/workstreams/CONTINUOUS_TERRAIN_REPRESENTATION_V1_CLEAN_RERUN.md` for the corrected gate.

Status: final terrain experiment.

No live agent was modified and no arena screen was run.

## Post-Run Audit Note

See `docs/workstreams/SEARCH_METADATA_DOMINANCE_AUDIT_V1.md` before using the R1/R4 tables below. The follow-up audit found that strict live N=8 metadata still has strong warning signal, but the original R1 was not a strict live-only whitelist, the semantic/R4 path leaked teacher `policy_prob` through action scalars, and the high-regret/unacceptable labels are identical in this dataset. Treat the prior representation result as diagnostic only, not a deployable search-guidance gate.

## Dataset

- Input: `data\manifests\continuous_terrain_v1.jsonl`
- Decisions/options/games: 702 / 5743 / 160
- Eval-only decisions: 2
- Failures: 0
- Split: train games ['80271105.json', '80272080.json', '80272390.json', '80274139.json', '80274923.json', '80276040.json', '80276549.json', '80277498.json', '80279270.json', '80280866.json', '80281374.json', '80285820.json', '80287441.json', '80289754.json', '80289755.json', '80291836.json', '80292689.json', '80293984.json', '80294020.json', '80294486.json', '80294505.json', '80294824.json', '80295141.json', '80296137.json', '80297627.json', '80298133.json', '80300148.json', '80300282.json', '80300572.json', '80300783.json', '80300890.json', '80300895.json', '80300896.json', '80301388.json', '80301390.json', '80301415.json', '80301891.json', '80302213.json', '80302553.json', '80303361.json', '80304505.json', '80307669.json', '80308887.json', '80308995.json', '80309332.json', '80310187.json', '80311046.json', '80312386.json', '80312719.json', '80313600.json', '80313721.json', '80313741.json', '80313747.json', '80314056.json', '80314382.json', '80315229.json', '80315231.json', '80315742.json', '80316081.json', '80316429.json', '80318252.json', '80318946.json', '80319449.json', '80326672.json', '80328795.json', '80329668.json', '80332993.json', '80333529.json', '80333932.json', '80333944.json', '80334476.json', '80334484.json', '80334826.json', '80334847.json', '80335177.json', '80335200.json', '80336030.json', '80336047.json', '80338044.json', '80338409.json', '80339177.json', '80339202.json', '80339224.json', '80339996.json', '80342910.json', '80344206.json', '80345100.json', '80345988.json', '80347254.json', '80347656.json', '80348267.json', '80348808.json', '80349900.json', '80350454.json', '80352279.json', '80352296.json', '80354992.json', '80359144.json', '80359693.json', '80361280.json', '80361284.json', '80363659.json', '80363955.json', '80365405.json', '80365421.json', '80365448.json', '80366705.json', '80368054.json', '80369855.json', '80373811.json', '80374166.json']; val games ['80273641.json', '80280046.json', '80296615.json', '80298599.json', '80301898.json', '80301899.json', '80310709.json', '80312706.json', '80318283.json', '80331975.json', '80332353.json', '80333948.json', '80336589.json', '80337121.json', '80338035.json', '80338763.json', '80338809.json', '80340004.json', '80345969.json', '80352722.json', '80353921.json', '80354482.json', '80355511.json', '80370387.json']; test games ['80277165.json', '80300466.json', '80308998.json', '80309852.json', '80311027.json', '80313727.json', '80314899.json', '80318256.json', '80322198.json', '80324949.json', '80335175.json', '80335697.json', '80337112.json', '80340754.json', '80342153.json', '80345084.json', '80345620.json', '80345947.json', '80348795.json', '80350956.json', '80361846.json', '80367521.json', '80371307.json']; eval-only ['80251230.json', '80252701.json']
- Terrain-authoritative records: 702
- Semantic-vector option rows: 5743
- Repeated live/stronger records: 702 / 702
- Missing required record fields: `{}`
- Missing required option fields: `{}`
- Split violations: decisions `[]`, groups `[]`

## Architecture

- Learned card-id embedding dimension: 32.
- Effect vector MLP, dynamic entity MLP, DeepSets zone pooling, 128-d state encoder.
- Action encoder combines action type, acting card embedding, target entity, decoded effects, option deltas, and action scalars.
- Semantic latent is separated from the search-metadata branch; R3 uses semantic only, R4 adds metadata.
- Homoscedastic task weights are learned for ranking, risk, acceptability, instability, residual, and contrastive losses.

## Trainable Embedding Check

- `R3_semantic` card embedding grad norm, last epoch: 0.001493
- `R4_semantic_plus_search` card embedding grad norm, last epoch: 0.003440
- `R4_no_card_embedding` card embedding grad norm, last epoch: 0.000000
- `R4_no_decoded_effects` card embedding grad norm, last epoch: 0.003716
- `R4_no_target_entity` card embedding grad norm, last epoch: 0.001943
- `R4_no_option_deltas` card embedding grad norm, last epoch: 0.003717
- `R4_no_contrastive` card embedding grad norm, last epoch: 0.000712
- `R4_no_ranking` card embedding grad norm, last epoch: 0.003596

## Learned Task Weights

| variant | ranking | high_regret | unacceptable | acceptable | instability | residual | contrastive |
|---|---:|---:|---:|---:|---:|---:|---:|
| `R3_semantic` | 0.990 | 1.010 | 1.010 | 1.010 | 1.010 | 1.010 | 0.990 |
| `R4_semantic_plus_search` | 0.999 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 0.999 |
| `R4_no_card_embedding` | 0.998 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 0.998 |
| `R4_no_decoded_effects` | 0.997 | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | 0.997 |
| `R4_no_target_entity` | 0.995 | 1.005 | 1.005 | 1.005 | 1.005 | 1.005 | 0.995 |
| `R4_no_option_deltas` | 0.999 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 0.999 |
| `R4_no_contrastive` | 0.999 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 |
| `R4_no_ranking` | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 0.999 |

## Predictive Metrics

High-regret AP/AUROC on held-out test games:

| representation | AP | AUROC | recall@FPR5 | recall@FPR10 |
|---|---:|---:|---:|---:|
| `R0_root_engineered` | 0.135 | 0.767 | 0.242 | 0.364 |
| `R1_search_metadata_only` | 0.747 | 0.990 | 1.000 | 1.000 |
| `R2_current_engineered_contextual` | 0.162 | 0.730 | 0.273 | 0.333 |
| `R3_semantic` | 0.222 | 0.757 | 0.212 | 0.303 |
| `R4_no_card_embedding` | 0.281 | 0.711 | 0.545 | 0.667 |
| `R4_no_contrastive` | 0.029 | 0.260 | 0.000 | 0.000 |
| `R4_no_decoded_effects` | 0.037 | 0.422 | 0.000 | 0.000 |
| `R4_no_option_deltas` | 0.024 | 0.112 | 0.000 | 0.000 |
| `R4_no_ranking` | 0.225 | 0.448 | 0.303 | 0.303 |
| `R4_no_target_entity` | 0.211 | 0.682 | 0.545 | 0.576 |
| `R4_semantic_plus_search` | 0.252 | 0.373 | 0.333 | 0.333 |

Unacceptable AP/AUROC on held-out test games:

| representation | AP | AUROC | recall@FPR5 | recall@FPR10 |
|---|---:|---:|---:|---:|
| `R0_root_engineered` | 0.135 | 0.767 | 0.242 | 0.364 |
| `R1_search_metadata_only` | 0.747 | 0.990 | 1.000 | 1.000 |
| `R2_current_engineered_contextual` | 0.162 | 0.730 | 0.273 | 0.333 |
| `R3_semantic` | 0.042 | 0.476 | 0.000 | 0.000 |
| `R4_no_card_embedding` | 0.040 | 0.476 | 0.000 | 0.000 |
| `R4_no_contrastive` | 0.446 | 0.738 | 0.667 | 0.667 |
| `R4_no_decoded_effects` | 0.069 | 0.560 | 0.030 | 0.152 |
| `R4_no_option_deltas` | 0.107 | 0.722 | 0.152 | 0.242 |
| `R4_no_ranking` | 0.048 | 0.512 | 0.000 | 0.000 |
| `R4_no_target_entity` | 0.048 | 0.475 | 0.000 | 0.152 |
| `R4_semantic_plus_search` | 0.296 | 0.703 | 0.606 | 0.667 |

## Signal Radius

k=10 held-out-game high-regret enrichment:

| representation | bg rate | neighbor rate | enrich | queries |
|---|---:|---:|---:|---:|
| `R0_root_engineered` | 0.035 | 0.145 | 4.112 | 33 |
| `R1_search_metadata_only` | 0.035 | 0.536 | 15.163 | 33 |
| `R2_current_engineered_contextual` | 0.035 | 0.079 | 2.227 | 33 |
| `R3_semantic` | 0.035 | 0.118 | 3.341 | 33 |
| `R4_semantic_plus_search` | 0.035 | 0.403 | 11.393 | 33 |
| `R4_no_card_embedding` | 0.035 | 0.418 | 11.822 | 33 |
| `R4_no_decoded_effects` | 0.035 | 0.376 | 10.622 | 33 |
| `R4_no_target_entity` | 0.035 | 0.403 | 11.393 | 33 |
| `R4_no_option_deltas` | 0.035 | 0.421 | 11.907 | 33 |
| `R4_no_contrastive` | 0.035 | 0.415 | 11.736 | 33 |
| `R4_no_ranking` | 0.035 | 0.379 | 10.708 | 33 |

## Ablations

| ablation | high-regret AP | high-regret k10 enrich | read |
|---|---:|---:|---|
| `R4_no_card_embedding` | 0.281 | 11.822 | little change |
| `R4_no_decoded_effects` | 0.037 | 10.622 | component appears helpful |
| `R4_no_target_entity` | 0.211 | 11.393 | component appears helpful |
| `R4_no_option_deltas` | 0.024 | 11.907 | component appears helpful |
| `R4_no_contrastive` | 0.029 | 11.736 | component appears helpful |
| `R4_no_ranking` | 0.225 | 10.708 | little change |

## Retrieval And Failure Examples

- Cross-game R3 semantic neighbor examples: 10
- Semantic succeeds / search metadata fails examples: 12
- Search metadata succeeds / semantic fails examples: 12
- R3 semantic high-regret failure cases: 12

`semantic_succeeds_search_fails` sample:
- 80348795.json:6#3 group=80348795.json label=False R3=0.479 R1=0.895
- 80345947.json:148#0 group=80345947.json label=False R3=0.465 R1=0.811
- 80345947.json:148#1 group=80345947.json label=False R3=0.462 R1=0.938

`search_succeeds_semantic_fails` sample:
- 80348795.json:6#0 group=80348795.json label=True R3=0.478 R1=0.712
- 80348795.json:6#1 group=80348795.json label=True R3=0.475 R1=0.896
- 80348795.json:6#2 group=80348795.json label=True R3=0.487 R1=0.898

`semantic_failure_cases` sample:
- 80348795.json:6#0 group=80348795.json label=True R3=0.478 R1=0.712
- 80348795.json:6#1 group=80348795.json label=True R3=0.475 R1=0.896
- 80348795.json:6#2 group=80348795.json label=True R3=0.487 R1=0.898

## Limitations

- No live agent was built or screened; this is an offline representation gate.
- The A terrain artifact has no terminal-outcome targets; this run uses hand/repeated-search terrain labels.
- c1 remains rare relative to c2/boundary/safe terrain even after expansion, so c1-specific metrics are still high variance.
- This run uses one fixed training seed; three-seed replication was not run in this bounded pass.

## Verdict

**E. CURRENT SEMANTIC REPRESENTATION NOT VALIDATED**


One next experiment: Do not build a live search-guidance candidate; diagnose why search metadata dominates before another representation run.
