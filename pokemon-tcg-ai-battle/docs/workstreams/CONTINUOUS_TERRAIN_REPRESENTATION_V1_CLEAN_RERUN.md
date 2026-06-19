# Continuous Terrain Representation V1 Clean Rerun

Status: clean offline rerun after Search Metadata Dominance Audit.

Final verdict: **C. LIVE METADATA IS THE USEFUL SIGNAL**

Strict live metadata remains the strongest corrected signal; semantic features add little or hurt.

No arena screen was run. `agent_search` was not modified. Main was not merged.

## Dataset Verification

- decision_count: `702`
- option_count: `5743`
- group_count: `160`
- high_regret_count: `599`
- high_regret_rate: `0.104300888037611`
- unacceptable_count: `3348`
- unacceptable_rate: `0.5829705728713216`
- high_equals_unacceptable_rows: `2796`
- high_equals_unacceptable_rate: `0.4868535608566951`
- eval_only_seed_count: `2`
- high_regret and unacceptable distinct: `True`
- missing record fields: `{}`
- missing option fields: `{}`
- duplicate decision ids: `[]`
- duplicate decision/option ids: `[]`

## Allowed And Forbidden Inputs

### R1_live_metadata_only

Allowed:
- mean_live_value
- live_value_variance
- live_selected_distribution
- live_action_entropy
- modal_action_stability
- live margin/spread computed from mean_live_value only
- criticality
- option index
- search_selected_option
- n_options

Forbidden:
- acceptable_prob
- delta_to_search
- delta_to_search_norm
- hand_norm_advantage
- high_regret_prob
- mean_stronger_value
- policy_prob
- regret
- stronger_soft_policy
- stronger_value_variance
- unacceptable_prob
- value_se
- value_spread

### R3_semantic_only

Allowed:
- observation/root state features
- legal option/action descriptor
- semantic_action_key
- eq_class
- learned card-id embedding
- decoded card effect vector
- target/entity features
- state-effect interactions from root and effect features
- one-step option deltas from semantic_vector
- card metadata from semantic_vector

Forbidden:
- acceptable_prob
- delta_to_search
- delta_to_search_norm
- hand_norm_advantage
- high_regret_prob
- live_action_entropy
- live_selected_distribution
- mean_stronger_value
- modal_action_stability
- policy_prob
- regret
- stronger_soft_policy
- stronger_value_variance
- unacceptable_prob
- value_se
- value_spread

### R4_semantic_plus_live_metadata

Allowed:
- all R3 semantic inputs
- all strict R1 live metadata inputs

Forbidden:
- acceptable_prob
- delta_to_search
- delta_to_search_norm
- hand_norm_advantage
- high_regret_prob
- mean_stronger_value
- policy_prob
- regret
- stronger_soft_policy
- stronger_value_variance
- unacceptable_prob
- value_se
- value_spread

## Leakage And Schema Assertions

- PASS: r1_margin_uses_mean_live_value - `R1 is constructed directly from mean_live_value, not current_search_value.`
- PASS: r1_variance_uses_live_value_variance - `R1 is constructed directly from live_value_variance, not value_variance.`
- PASS: forbidden_feature_sources_absent - `{'R1_sources': ['mean_live_value', 'live_value_variance', 'live_selected_distribution', 'live_action_entropy', 'modal_action_stability', 'live margin/spread computed from mean_live_value only', 'criticality', 'option index', 'search_selected_option', 'n_options'], 'R3_sources': ['observation/root state features', 'legal option/action descriptor', 'semantic_action_key', 'eq_class', 'learned card-id embedding', 'decoded card effect vector', 'target/entity features', 'state-effect interactions from root and effect features', 'one-step option deltas from semantic_vector', 'card metadata from semantic_vector']}`
- PASS: eval_only_excluded_from_train_val - `[]`
- PASS: group_split_no_game_crossing - `[]`
- PASS: R0_root_engineered_all_zero_columns_explicitly_allowed - `{'count': 3, 'reason': 'legacy contextual vector contains sparse unused action/effect slots'}`
- PASS: R0_root_engineered_no_near_perfect_target_column - `[]`
- PASS: R1_live_metadata_only_no_all_zero_columns - `[]`
- PASS: R1_live_metadata_only_no_near_perfect_target_column - `[]`
- PASS: R2_current_engineered_contextual_all_zero_columns_explicitly_allowed - `{'count': 18, 'reason': 'legacy contextual vector contains sparse unused action/effect slots'}`
- PASS: R2_current_engineered_contextual_no_near_perfect_target_column - `[]`
- PASS: R3_R4_global_features_all_zero_columns_explicitly_allowed - `{'count': 3, 'columns': [28, 32, 33], 'reason': 'semantic/card/action feature taxonomies include sparse slots that may be unused by the fixed deck or batch'}`
- PASS: R3_R4_global_features_no_near_perfect_target_column - `[]`
- PASS: R3_R4_action_effects_all_zero_columns_explicitly_allowed - `{'count': 1, 'columns': [13], 'reason': 'semantic/card/action feature taxonomies include sparse slots that may be unused by the fixed deck or batch'}`
- PASS: R3_R4_action_effects_no_near_perfect_target_column - `[]`
- PASS: R3_R4_target_effects_all_zero_columns_explicitly_allowed - `{'count': 1, 'columns': [13], 'reason': 'semantic/card/action feature taxonomies include sparse slots that may be unused by the fixed deck or batch'}`
- PASS: R3_R4_target_effects_no_near_perfect_target_column - `[]`
- PASS: R3_R4_target_dynamic_all_zero_columns_explicitly_allowed - `{'count': 0, 'columns': [], 'reason': 'semantic/card/action feature taxonomies include sparse slots that may be unused by the fixed deck or batch'}`
- PASS: R3_R4_target_dynamic_no_near_perfect_target_column - `[]`
- PASS: R3_R4_option_deltas_all_zero_columns_explicitly_allowed - `{'count': 1, 'columns': [1], 'reason': 'semantic/card/action feature taxonomies include sparse slots that may be unused by the fixed deck or batch'}`
- PASS: R3_R4_option_deltas_no_near_perfect_target_column - `[]`
- PASS: R3_R4_action_scalars_all_zero_columns_explicitly_allowed - `{'count': 2, 'columns': [29, 38], 'reason': 'semantic/card/action feature taxonomies include sparse slots that may be unused by the fixed deck or batch'}`
- PASS: R3_R4_action_scalars_no_near_perfect_target_column - `[]`
- PASS: R4_metadata_branch_no_all_zero_columns - `[]`
- PASS: R3_R4_metadata_no_near_perfect_target_column - `[]`

## Corrected High-Regret Metrics

| representation | AP | AUROC | recall@FPR5 | recall@FPR10 | ECE | positives/n |
|---|---:|---:|---:|---:|---:|---:|
| `R0_root_engineered` | 0.135 | 0.767 | 0.242 | 0.364 | 0.275 | 33/763 |
| `R1_live_metadata_only` | 0.985 | 0.999 | 1.000 | 1.000 | 0.036 | 33/763 |
| `R2_current_engineered_contextual` | 0.162 | 0.730 | 0.273 | 0.333 | 0.241 | 33/763 |
| `R3_semantic_only` | 0.328 | 0.825 | 0.515 | 0.636 | 0.415 | 33/763 |
| `R4_semantic_plus_live_metadata` | 0.650 | 0.800 | 0.667 | 0.667 | 0.421 | 33/763 |

## Corrected Unacceptable Metrics

| representation | AP | AUROC | recall@FPR5 | recall@FPR10 | ECE | positives/n |
|---|---:|---:|---:|---:|---:|---:|
| `R0_root_engineered` | 0.788 | 0.745 | 0.137 | 0.191 | 0.155 | 467/763 |
| `R1_live_metadata_only` | 0.978 | 0.973 | 0.857 | 0.940 | 0.102 | 467/763 |
| `R2_current_engineered_contextual` | 0.808 | 0.733 | 0.171 | 0.373 | 0.157 | 467/763 |
| `R3_semantic_only` | 0.581 | 0.438 | 0.039 | 0.086 | 0.140 | 467/763 |
| `R4_semantic_plus_live_metadata` | 0.635 | 0.597 | 0.006 | 0.026 | 0.100 | 467/763 |

## Other Target Metrics

### selected_high_regret

| representation | AP | AUROC | recall@FPR10 | positives/n |
|---|---:|---:|---:|---:|
| `R0_root_engineered` | 0.051 | 0.568 | 0.000 | 2/90 |
| `R1_live_metadata_only` | 0.333 | 0.966 | 1.000 | 2/90 |
| `R2_current_engineered_contextual` | 0.021 | 0.233 | 0.000 | 2/90 |
| `R3_semantic_only` | 0.151 | 0.778 | 0.500 | 2/90 |
| `R4_semantic_plus_live_metadata` | 0.292 | 0.955 | 1.000 | 2/90 |

### instability

| representation | AP | AUROC | recall@FPR10 | positives/n |
|---|---:|---:|---:|---:|
| `R0_root_engineered` | 0.476 | 0.764 | 0.354 | 195/763 |
| `R1_live_metadata_only` | 1.000 | 1.000 | 1.000 | 195/763 |
| `R2_current_engineered_contextual` | 0.482 | 0.764 | 0.287 | 195/763 |
| `R3_semantic_only` | 0.190 | 0.355 | 0.036 | 195/763 |
| `R4_semantic_plus_live_metadata` | 0.286 | 0.619 | 0.041 | 195/763 |

### acceptable

| representation | AP | AUROC | recall@FPR10 | positives/n |
|---|---:|---:|---:|---:|
| `R0_root_engineered` | 0.992 | 0.849 | 0.436 | 734/763 |
| `R1_live_metadata_only` | 1.000 | 0.998 | 0.996 | 734/763 |
| `R2_current_engineered_contextual` | 0.988 | 0.788 | 0.417 | 734/763 |
| `R3_semantic_only` | 0.986 | 0.734 | 0.492 | 734/763 |
| `R4_semantic_plus_live_metadata` | 0.923 | 0.257 | 0.008 | 734/763 |


## Signal Radius

| target | representation | k | neighbor rate | background | enrichment | recall coverage |
|---|---|---:|---:|---:|---:|---:|
| `high_regret` | `R1_live_metadata_only` | 10 | 0.479 | 0.035 | 13.535 | 0.667 |
| `high_regret` | `R1_live_metadata_only` | 25 | 0.326 | 0.035 | 9.217 | 1.000 |
| `high_regret` | `R2_current_engineered_contextual` | 10 | 0.079 | 0.035 | 2.227 | 0.424 |
| `high_regret` | `R2_current_engineered_contextual` | 25 | 0.084 | 0.035 | 2.364 | 0.727 |
| `high_regret` | `R3_semantic_only` | 10 | 0.061 | 0.035 | 1.713 | 0.364 |
| `high_regret` | `R3_semantic_only` | 25 | 0.062 | 0.035 | 1.748 | 0.485 |
| `high_regret` | `R4_semantic_plus_live_metadata` | 10 | 0.473 | 0.035 | 13.364 | 0.667 |
| `high_regret` | `R4_semantic_plus_live_metadata` | 25 | 0.222 | 0.035 | 6.271 | 0.667 |
| `unacceptable` | `R1_live_metadata_only` | 10 | 0.828 | 0.570 | 1.455 | 0.497 |
| `unacceptable` | `R1_live_metadata_only` | 25 | 0.813 | 0.570 | 1.428 | 0.604 |
| `unacceptable` | `R2_current_engineered_contextual` | 10 | 0.608 | 0.570 | 1.068 | 0.497 |
| `unacceptable` | `R2_current_engineered_contextual` | 25 | 0.612 | 0.570 | 1.074 | 0.632 |
| `unacceptable` | `R3_semantic_only` | 10 | 0.585 | 0.570 | 1.028 | 0.548 |
| `unacceptable` | `R3_semantic_only` | 25 | 0.572 | 0.570 | 1.004 | 0.730 |
| `unacceptable` | `R4_semantic_plus_live_metadata` | 10 | 0.877 | 0.570 | 1.540 | 0.546 |
| `unacceptable` | `R4_semantic_plus_live_metadata` | 25 | 0.836 | 0.570 | 1.468 | 0.679 |
| `selected_high_regret` | `R1_live_metadata_only` | 10 | 0.100 | 0.011 | 8.850 | 1.000 |
| `selected_high_regret` | `R1_live_metadata_only` | 25 | 0.040 | 0.011 | 3.540 | 1.000 |
| `selected_high_regret` | `R2_current_engineered_contextual` | 10 | 0.000 | 0.011 | 0.000 | 0.000 |
| `selected_high_regret` | `R2_current_engineered_contextual` | 25 | 0.000 | 0.011 | 0.000 | 0.000 |
| `selected_high_regret` | `R3_semantic_only` | 10 | 0.100 | 0.011 | 8.850 | 1.000 |
| `selected_high_regret` | `R3_semantic_only` | 25 | 0.040 | 0.011 | 3.540 | 1.000 |
| `selected_high_regret` | `R4_semantic_plus_live_metadata` | 10 | 0.100 | 0.011 | 8.850 | 1.000 |
| `selected_high_regret` | `R4_semantic_plus_live_metadata` | 25 | 0.040 | 0.011 | 3.540 | 1.000 |
| `c1` | `R1_live_metadata_only` | 10 | 0.100 | 0.011 | 8.850 | 1.000 |
| `c1` | `R1_live_metadata_only` | 25 | 0.040 | 0.011 | 3.540 | 1.000 |
| `c1` | `R2_current_engineered_contextual` | 10 | 0.000 | 0.011 | 0.000 | 0.000 |
| `c1` | `R2_current_engineered_contextual` | 25 | 0.000 | 0.011 | 0.000 | 0.000 |
| `c1` | `R3_semantic_only` | 10 | 0.100 | 0.011 | 8.850 | 1.000 |
| `c1` | `R3_semantic_only` | 25 | 0.040 | 0.011 | 3.540 | 1.000 |
| `c1` | `R4_semantic_plus_live_metadata` | 10 | 0.100 | 0.011 | 8.850 | 1.000 |
| `c1` | `R4_semantic_plus_live_metadata` | 25 | 0.040 | 0.011 | 3.540 | 1.000 |
| `c2` | `R1_live_metadata_only` | 10 | 0.791 | 0.606 | 1.307 | 0.862 |
| `c2` | `R1_live_metadata_only` | 25 | 0.697 | 0.606 | 1.151 | 0.948 |
| `c2` | `R2_current_engineered_contextual` | 10 | 0.810 | 0.606 | 1.338 | 0.638 |
| `c2` | `R2_current_engineered_contextual` | 25 | 0.683 | 0.606 | 1.128 | 0.724 |
| `c2` | `R3_semantic_only` | 10 | 0.678 | 0.606 | 1.119 | 0.810 |
| `c2` | `R3_semantic_only` | 25 | 0.648 | 0.606 | 1.069 | 0.914 |
| `c2` | `R4_semantic_plus_live_metadata` | 10 | 0.724 | 0.606 | 1.196 | 0.914 |
| `c2` | `R4_semantic_plus_live_metadata` | 25 | 0.655 | 0.606 | 1.082 | 0.948 |
| `c3` | `R1_live_metadata_only` | 10 | 0.508 | 0.261 | 1.948 | 0.800 |
| `c3` | `R1_live_metadata_only` | 25 | 0.301 | 0.261 | 1.154 | 0.880 |
| `c3` | `R2_current_engineered_contextual` | 10 | 0.360 | 0.261 | 1.381 | 0.680 |
| `c3` | `R2_current_engineered_contextual` | 25 | 0.355 | 0.261 | 1.362 | 0.720 |
| `c3` | `R3_semantic_only` | 10 | 0.404 | 0.261 | 1.549 | 0.880 |
| `c3` | `R3_semantic_only` | 25 | 0.379 | 0.261 | 1.454 | 1.000 |
| `c3` | `R4_semantic_plus_live_metadata` | 10 | 0.480 | 0.261 | 1.841 | 0.880 |
| `c3` | `R4_semantic_plus_live_metadata` | 25 | 0.254 | 0.261 | 0.976 | 0.880 |

## Training

- `R3_semantic_only`: best_epoch=1, card_embedding_grad_norm_last_epoch=0.002
- `R4_semantic_plus_live_metadata`: best_epoch=2, card_embedding_grad_norm_last_epoch=0.002

## Final Decision

**C. LIVE METADATA IS THE USEFUL SIGNAL**

Strict live metadata remains the strongest corrected signal; semantic features add little or hurt.

One recommended next action: Prefer selective-compute/instability trigger analysis over another semantic representation run.
