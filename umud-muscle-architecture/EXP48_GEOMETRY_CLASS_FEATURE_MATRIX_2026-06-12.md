# EXP48 - Geometry Class Feature Matrix

Date: 2026-06-12

Purpose: make the "story" layer explicit. Instead of asking whether a feature is globally good, classify benchmark images by coarse geometry and ask which feature helps which class.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

## Classes

The harness creates coarse diagnostic classes from `results/benchmark_error_taxonomy.csv`.

| class | count |
|---|---:|
| `low_support_any` | 33 |
| `expert_FL_below_projected_median` | 22 |
| `severe_low_support` | 18 |
| `two_band_simple` | 14 |
| `strong_upper_curve_any_direction` | 12 |
| `upper_middle_shallow_arch` | 12 |
| `lower_side_angle_changes_strongly` | 11 |
| `strong_lower_curve_any_direction` | 10 |
| `boundaries_not_parallel` | 10 |
| `multi_band_risk` | 9 |
| `upper_side_angle_changes_strongly` | 9 |
| `high_PA_sensitivity` | 8 |
| `upper_middle_deep_sag` | 4 |
| `sparse_fragments` | 3 |

## Story Candidate Scores

| variant | overall | PA | FL | MT | read |
|---|---:|---:|---:|---:|---|
| `story_FL_scan_on_low_support_PA_per_band_on_multi_band_else_conflict_all_MT_vertical_all` | **0.151** | **0.140** | **0.242** | 0.070 | best story stack |
| `best_local_all_features` | 0.153 | 0.144 | 0.245 | 0.070 | prior local stack |
| `story_FL_scan_on_upper_curve_or_low_support_PA_conflict_all_MT_vertical_all` | 0.153 | 0.144 | 0.245 | 0.070 | same as all-feature stack because low support covers 33/35 |
| `story_FL_scan_on_curved_or_nonparallel_PA_conflict_on_PA_story_MT_vertical_all` | 0.155 | 0.148 | 0.248 | 0.070 | more selective, slightly worse |
| `robust_triangle_anchor` | 0.170 | 0.150 | 0.278 | 0.083 | baseline |

## Class-Specific Reads

- Per-band PA helps most in `multi_band_risk`, `upper_side_angle_changes_strongly`, and `boundaries_not_parallel`.
- Conflict-gated PA helps `upper_middle_deep_sag`, `severe_low_support`, and `two_band_simple`.
- Vertical-center MT helps the curved-boundary classes most, especially `strong_upper_curve_any_direction`, `upper_middle_shallow_arch`, and `upper_side_angle_changes_strongly`.
- Strict scan-region FL helps `boundaries_not_parallel`, `strong_lower_curve_any_direction`, and `expert_FL_below_projected_median`.

## Interpretation

This supports the user's "selling a coherent story" hypothesis:

- When the image looks like a multi-band or strong-side-angle case, per-band PA is a better story than one global PA aggregation.
- When the image looks like a simpler two-band case, conflict-gated PA is better than per-band PA.
- When boundary shape is strongly curved or nonparallel, vertical-center MT and scan-region FL are more coherent with the visual story than the original measurement.

Caveat: this is still only 35 benchmark images. The classes are diagnostic, not final production gates. Fine-grained class gating can overfit quickly. The useful next step is to make the viewer show these classes and inspect whether the selected story matches the actual geometry by eye.

## Files

- Harness: `experiments/exp48_geometry_class_feature_matrix.py`
- Ignored output bundle: `results/exp48_geometry_class_feature_matrix/`
- Key outputs:
  - `geometry_classes.csv`
  - `class_feature_scores.csv`
  - `story_candidate_summary.csv`
  - per-candidate CSVs

