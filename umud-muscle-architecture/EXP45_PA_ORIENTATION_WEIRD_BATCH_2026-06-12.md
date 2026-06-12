# EXP45 - PA Orientation Weird Batch

Date: 2026-06-12

Purpose: test orthogonal ways to estimate the internal-strand angle while keeping FL and MT fixed to the robust-triangle anchor.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

## Summary

| variant | overall | PA | read |
|---|---:|---:|---|
| `robust_triangle_anchor` | **0.170** | **0.150** | baseline |
| `PA_cv2_fitLine_component_orientation_area_median` | 0.170 | 0.150 | identical to PCA baseline |
| `PA_visible_length_weighted_median_of_fragment_orientations` | 0.173 | 0.159 | worse |
| `PA_endpoint_extreme_axis_component_orientation_area_median` | 0.175 | 0.163 | worse |
| `PA_middle_60_percent_circular_mean_of_fragment_orientations` | 0.175 | 0.164 | worse |
| `PA_quadratic_xy_orientation_field_area_median` | 0.177 | 0.171 | worse |
| `PA_RANSAC_component_orientation_area_median` | 0.178 | 0.173 | worse |
| `PA_area_times_visible_length_weighted_median_of_fragment_orientations` | 0.179 | 0.176 | worse |
| `PA_linear_xy_orientation_field_area_median` | 0.179 | 0.177 | worse |
| `PA_area_weighted_circular_mean_of_fragment_orientations` | 0.181 | 0.181 | worse |

## Read

- The existing component PCA plus area-weighted median is hard to beat.
- Circular averaging, visible-length weighting, endpoint-axis orientation, RANSAC orientation, and x/y orientation fields all worsened PA.
- This points away from changing the basic line fitter or global aggregation. The useful PA path is more likely local conflict detection or structured family assignment, not a new universal angle estimator.

## Files

- Harness: `experiments/exp45_pa_orientation_weird_batch.py`
- Ignored output bundle: `results/exp45_pa_orientation_weird_batch/`
- Key outputs: `summary.csv`, per-variant CSVs, `geometry_bundle.json`

