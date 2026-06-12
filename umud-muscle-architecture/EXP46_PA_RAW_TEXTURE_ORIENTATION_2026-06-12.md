# EXP46 - PA Raw Texture Orientation

Date: 2026-06-12

Purpose: test whether grayscale texture orientation around each predicted fragment can improve PA.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

## Summary

| variant | overall | PA | read |
|---|---:|---:|---|
| `robust_triangle_anchor` | **0.170** | **0.150** | baseline |
| `PA_texture_far_disagreement_replace_area_median` | 0.170 | 0.150 | no meaningful movement |
| `PA_25_percent_blend_PCA_toward_raw_texture_area_median` | 0.197 | 0.229 | rejected |
| `PA_texture_close_refinement_only_area_median` | 0.236 | 0.348 | rejected |
| `PA_50_percent_blend_PCA_toward_raw_texture_area_median` | 0.240 | 0.358 | rejected |
| `PA_raw_texture_orientation_support_weighted_median` | 0.328 | 0.623 | rejected |
| `PA_raw_texture_orientation_area_median` | 0.333 | 0.637 | rejected |
| `PA_raw_texture_orientation_texture_weighted_median` | 0.334 | 0.642 | rejected |

## Read

- Raw grayscale texture orientation is not aligned with the scored PA convention on this benchmark.
- Blending toward raw texture is also bad.
- Texture may still be useful as a QA/disagreement signal, but not as a direct PA estimator.

## Files

- Harness: `experiments/exp46_pa_raw_texture_orientation.py`
- Ignored output bundle: `results/exp46_pa_raw_texture_orientation/`
- Key outputs: `summary.csv`, per-variant CSVs, `geometry_bundle.json`

