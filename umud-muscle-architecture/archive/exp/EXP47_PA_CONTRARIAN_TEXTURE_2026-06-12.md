# EXP47 - PA Contrarian Texture

Date: 2026-06-12

Purpose: test the negative-information hypothesis from exp46. If raw grayscale texture points the wrong way, does moving slightly away from it improve PA?

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

## Summary

| variant | overall | PA | signed PA | read |
|---|---:|---:|---:|---|
| `robust_triangle_anchor` | **0.170** | **0.150** | -0.32 deg | baseline |
| `PA_move_10pct_away_from_raw_texture_orientation_area_median` | 0.173 | 0.157 | -0.01 deg | fixes mean bias, worsens MAE |
| `PA_move_25pct_away_from_raw_texture_orientation_area_median` | 0.182 | 0.184 | +0.54 deg | rejected |
| `PA_move_50pct_away_from_raw_texture_orientation_area_median` | 0.212 | 0.276 | +1.32 deg | rejected |
| `PA_move_100pct_away_from_raw_texture_orientation_area_median` | 0.285 | 0.494 | +2.86 deg | rejected |

## Read

- Moving away from raw texture can remove signed bias, but the absolute error gets worse.
- This confirms raw texture is not a useful PA correction direction in the current form.
- Keep raw texture as a possible diagnostic/uncertainty feature only, not as an angle estimator.

## Files

- Harness: `experiments/exp47_pa_contrarian_texture.py`
- Ignored output bundle: `results/exp47_pa_contrarian_texture/`
- Key outputs: `summary.csv`, per-variant CSVs, `geometry_bundle.json`

