# Signal Radius Audit V1

Status: bounded diagnostic on round-2 Teacher V2 residual/risk labels. No live agent change, no arena screen, and no production risk-policy retrain.

## Dataset And Class Summary

- Canonical labels: `data\manifests\teacher_v2_residual_risk_labels_round2.jsonl`
- Alias labels: `data\manifests\teacher_v2_risk_labels_for_B_request.jsonl`
- Canonical and alias byte-identical: `True`
- Records/options loaded: 60 decisions / 588 options
- Feature rows generated: 588
- Unique games (`group_id`): 13
- c1 reproduced: 9
- c1 candidate but not reproduced: 7
- c2 safe-search false-positive states: 49
- c3 near-miss/boundary states: 21
- High-regret options: 127
- Unacceptable options: 404
- Eval-only seed decisions: 2 (excluded from fitting and feature-selection steps)
- Duplicate decisions: 0; duplicate option identities: 0
- Missing fields: `{}`

Class rates by game are in the JSON report. The important caveat is that reproduced c1 positives are game-clustered, so group-held-out estimates are high-variance.

| group_id | decisions | options | high-regret rate | unacceptable rate | c1 repr | c2 | c3 | eval seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 80251230.json | 12 | 85 | 0.188 | 0.247 | 1 | 11 | 6 | 1 |
| 80251834.json | 2 | 14 | 0.000 | 0.857 | 0 | 2 | 0 | 0 |
| 80252178.json | 1 | 11 | 0.000 | 0.909 | 0 | 1 | 0 | 0 |
| 80252701.json | 8 | 82 | 0.000 | 0.780 | 0 | 8 | 0 | 1 |
| 80253044.json | 9 | 119 | 0.000 | 0.748 | 0 | 9 | 0 | 0 |
| 80253882.json | 6 | 75 | 0.000 | 0.707 | 0 | 6 | 0 | 0 |
| 80253888.json | 6 | 54 | 0.000 | 0.759 | 0 | 6 | 0 | 0 |
| 80270516.json | 6 | 74 | 0.878 | 0.838 | 3 | 3 | 6 | 0 |
| 80271583.json | 1 | 5 | 0.600 | 0.600 | 1 | 0 | 1 | 0 |
| 80275931.json | 1 | 9 | 0.000 | 0.778 | 0 | 1 | 0 | 0 |
| 80277480.json | 1 | 21 | 0.952 | 0.952 | 1 | 0 | 1 | 0 |
| 80279946.json | 6 | 33 | 0.576 | 0.545 | 3 | 1 | 6 | 0 |
| 80280539.json | 1 | 6 | 0.667 | 0.667 | 0 | 1 | 1 | 0 |

## Feature Packs

- `A_root`: root-state vector plus public turn/history only
- `B_root_action`: A + action descriptor, semantic_action_key numeric encoding, eq_class/option count
- `C_plus_card_identity`: B + categorical card identity one-hot proxy for trainable embedding
- `D_plus_decoded_effects`: C + decoded card-effect features
- `E_plus_target_entity`: D + target/entity properties
- `F_plus_state_effect_interactions`: E + state x effect interaction features
- `G_plus_option_deltas`: F + immediate one-step option deltas
- `H_plus_search_uncertainty`: G + search value rank/margin/spread, value variance/SE, determinization, coverage, criticality metadata
- `baseline_criticality_only`: criticality fields only
- `baseline_search_variance_only`: search/teacher uncertainty fields only, without semantic features
- `class_frequency`: fold-local class prevalence baseline

Continuous features were robust-normalized with median/IQR from non-eval fitting rows. Card identity is a categorical one-hot proxy for the trainable card embedding used by the contextual model.

## Neighborhood Enrichment

Table shows k=10 standardized-Euclidean neighbor enrichment with same-game neighbors forbidden and eval-only rows excluded from neighbor pools.

| target | best pack | bg rate | neighbor rate | enrich | coverage | queries |
|---|---:|---:|---:|---:|---:|---:|
| high_regret_flag | `A_root` | 0.152 | 0.344 | 2.262 | 0.150 | 127 |
| unacceptable_flag | `H_plus_search_uncertainty` | 0.687 | 0.791 | 1.152 | 0.686 | 404 |
| selected_option_high_regret_flag | `H_plus_search_uncertainty` | 0.110 | 0.200 | 1.813 | 0.750 | 9 |
| c1_reproduced_this_label | `H_plus_search_uncertainty` | 0.110 | 0.200 | 1.813 | 0.750 | 9 |
| c2_safe_search_false_positive | `H_plus_search_uncertainty` | 0.807 | 0.969 | 1.201 | 0.708 | 49 |
| c3_near_miss_boundary | `H_plus_search_uncertainty` | 0.293 | 0.305 | 1.041 | 0.750 | 21 |

## Predictive Probes

Grouped leave-one-game-out linear probes; eval-only seeds are final-eval only. Full per-pack metrics are in the JSON.

| target | best probe pack | AP | AUROC | recall@FPR10 | valid folds | full-pack AP | search-meta AP | root AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_regret_flag | `baseline_criticality_only` | 0.676 | 0.709 | 0.708 | 13 | 0.462 | 0.304 | 0.252 |
| unacceptable_flag | `baseline_search_variance_only` | 0.926 | 0.859 | 0.533 | 13 | 0.827 | 0.926 | 0.725 |
| selected_option_high_regret_flag | `baseline_criticality_only` | 0.560 | 0.802 | 0.750 | 13 | 0.287 | 0.405 | 0.126 |
| c1_reproduced_this_label | `baseline_criticality_only` | 0.575 | 0.810 | 0.750 | 13 | 0.290 | 0.407 | 0.143 |
| c2_safe_search_false_positive | `baseline_search_variance_only` | 0.983 | 0.921 | 0.896 | 13 | 0.840 | 0.983 | 0.819 |
| c3_near_miss_boundary | `baseline_search_variance_only` | 0.980 | 0.983 | 0.950 | 13 | 0.794 | 0.980 | 0.421 |

## Feature Contribution Conclusions

| target | root AP | full AP | search-meta AP | contribution read |
|---|---:|---:|---:|---|
| high_regret_flag | 0.252 | 0.462 | 0.304 | signal visible after feature enrichment |
| unacceptable_flag | 0.725 | 0.827 | 0.926 | signal dominated by search metadata or no gain beyond it |
| selected_option_high_regret_flag | 0.126 | 0.287 | 0.405 | signal dominated by search metadata or no gain beyond it |
| c1_reproduced_this_label | 0.143 | 0.290 | 0.407 | signal dominated by search metadata or no gain beyond it |
| c2_safe_search_false_positive | 0.819 | 0.840 | 0.983 | signal dominated by search metadata or no gain beyond it |
| c3_near_miss_boundary | 0.421 | 0.794 | 0.980 | signal dominated by search metadata or no gain beyond it |

Component-level read:

| component | mean AP delta | mean k10 enrich delta | read |
|---|---:|---:|---|
| card_embedding_or_identity | 0.009 | -0.244 | little consistent incremental signal |
| decoded_effects | 0.010 | -0.118 | little consistent incremental signal |
| target_entity_features | 0.037 | 0.181 | little consistent incremental signal |
| state_effect_interactions | 0.006 | -0.249 | little consistent incremental signal |
| immediate_option_deltas | 0.001 | -0.010 | little consistent incremental signal |
| search_uncertainty_metadata | 0.104 | 0.802 | adds probe signal on average |

The explicit contribution answer is: card identity, decoded effects, target/entity features, interactions, and option deltas do not yet add reliable c1/search-failure signal; search/criticality metadata dominates most state-level labels; option-level high-regret has some broader structure but the c1 class remains too clustered for a reliable policy conclusion.

## Limitations

- c1 reproduced positives are clustered across fewer than six games, so c1 generalization is fragile.
- full features improve over root-only for option-level high_regret.
- c1 probe signal is not clearly beyond search metadata.
- Raw regret magnitude was not used as a primary target; it is noisy under the round-2 artifact notes.
- c1 positives are rare and clustered; a failed c1 probe does not prove there is no underlying structure.
- Linear probes are diagnostic only. They are not a deployable model and were not written to `agent/`.

## Verdict

**D. CURRENT ARTIFACTS INCONCLUSIVE**

Before another risk-policy retrain, mine disjoint replay shards until there are at least 25 reproduced search-selected-high-regret c1 decisions across at least 12 group_id games, plus matched c2/c3/background states from the same high-criticality band, then rerun this audit with the current two eval-only seeds still held out.

Model A remains idle unless that next experiment is explicitly authorized and asks for independent c1 labels from disjoint games.
