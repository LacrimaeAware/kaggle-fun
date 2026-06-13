# EXP54 - Model Class Matrix

Date: 2026-06-12

Purpose: summarize every expert-viewer model overall and within each geometry class. This answers
which candidate is broadly best, and which candidates are only class-specific.

Harness: `experiments/exp54_model_class_matrix.py`

Ignored output bundle: `results/exp54_model_class_matrix/`

Key outputs:

- `per_image_model_scores.csv`
- `model_class_matrix.csv`
- `class_winners.csv`

## Overall Ranking

| model | overall | PA | FL | MT |
|---|---:|---:|---:|---:|
| `median_weight_blend_best` | **0.143337** | 0.148942 | 0.210820 | 0.070248 |
| `story_weight_grid_best` | 0.143535 | 0.149963 | **0.210393** | 0.070248 |
| `weighted_story_fl` | 0.149199 | 0.149963 | 0.227385 | 0.070248 |
| `story_weight_same_story` | 0.149285 | 0.157714 | 0.219893 | 0.070248 |
| `story_stack` | 0.150726 | **0.139936** | 0.241995 | 0.070248 |
| `best_local_all_features` | 0.153201 | 0.144130 | 0.245225 | 0.070248 |
| `robust_triangle_anchor` | 0.170276 | 0.149772 | 0.278101 | 0.082955 |
| `our_pipeline_true_scale` | 0.251092 | 0.149772 | 0.519459 | 0.084046 |

## Class Overall Winners

| class | n | winner | score |
|---|---:|---|---:|
| all | 35 | `median_weight_blend_best` | 0.143337 |
| boundaries_not_parallel | 10 | `median_weight_blend_best` | 0.122198 |
| expert_FL_below_projected_median | 22 | `story_weight_grid_best` | 0.166345 |
| high_PA_sensitivity | 8 | `weighted_story_fl` | 0.150880 |
| low_support_any | 33 | `median_weight_blend_best` | 0.145187 |
| lower_side_angle_changes_strongly | 11 | `story_weight_grid_best` | 0.153069 |
| multi_band_risk | 9 | `mt_vertical_center` | 0.103650 |
| severe_low_support | 18 | `median_weight_blend_best` | 0.160713 |
| sparse_fragments | 3 | `story_weight_grid_best` | 0.058876 |
| strong_lower_curve_any_direction | 10 | `best_local_all_features` | 0.132954 |
| strong_upper_curve_any_direction | 12 | `story_weight_grid_best` | 0.115772 |
| two_band_simple | 14 | `story_weight_grid_best` | 0.178607 |
| upper_middle_deep_sag | 4 | `pa_conflict_gate` | 0.205170 |
| upper_middle_shallow_arch | 12 | `story_weight_grid_best` | 0.115772 |
| upper_side_angle_changes_strongly | 9 | `story_weight_grid_best` | 0.121641 |

## Interpretation

`median_weight_blend_best` is the best average benchmark model, but not the best story for every
class. `story_weight_grid_best` is nearly tied overall and wins many curved/sparse classes. The older
`story_stack` has the best PA term but loses too much FL. This supports a class-aware stack, but the
class gates need target-label validation before we trust them publicly.
