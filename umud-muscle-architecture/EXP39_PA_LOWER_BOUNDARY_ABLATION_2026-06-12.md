# EXP39 - PA Lower-Boundary / Local Conflict Ablation

Date: 2026-06-12

Purpose: test whether the remaining PA error is helped by using lower-boundary shape or by locally correcting obvious fragment-angle conflicts. This is an expert-benchmark harness only, not a submission generator.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

Important isolation rule: this experiment changes `pa_deg` only. `fl_mm` and `mt_mm` are intentionally left as the robust-triangle values so PA signal is not confounded with FL/MT geometry.

## Summary

| variant | overall | PA | FL | MT | signed PA |
|---|---:|---:|---:|---:|---:|
| `pa_conflict_gated_7deg` | 0.168 | 0.144 | 0.278 | 0.083 | -0.33 deg |
| `pa_conflict_gated_4deg` | 0.169 | 0.145 | 0.278 | 0.083 | -0.31 deg |
| `robust_anchor` | 0.170 | 0.150 | 0.278 | 0.083 | -0.32 deg |
| `pa_local_smooth_25` | 0.172 | 0.154 | 0.278 | 0.083 | -0.34 deg |
| `pa_local_smooth_50` | 0.173 | 0.159 | 0.278 | 0.083 | -0.36 deg |
| `pa_lower_quartile_polyline_tangent` | 0.183 | 0.187 | 0.278 | 0.083 | -0.16 deg |
| `pa_lower_smooth_tangent` | 0.191 | 0.213 | 0.278 | 0.083 | -0.29 deg |

## Over / Under Matrix For PA

Groups are defined by the robust-triangle PA error sign.

| variant | group | n | PA MAE | delta vs robust | signed bias |
|---|---:|---:|---:|---:|---:|
| `robust_anchor` | all | 35 | 0.899 deg | +0.000 | -0.32 deg |
| `robust_anchor` | base_over | 15 | 0.675 deg | +0.000 | +0.67 deg |
| `robust_anchor` | base_under | 20 | 1.067 deg | +0.000 | -1.07 deg |
| `pa_lower_smooth_tangent` | all | 35 | 1.279 deg | +0.380 | -0.29 deg |
| `pa_lower_smooth_tangent` | base_over | 15 | 0.921 deg | +0.246 | +0.81 deg |
| `pa_lower_smooth_tangent` | base_under | 20 | 1.548 deg | +0.481 | -1.11 deg |
| `pa_lower_quartile_polyline_tangent` | all | 35 | 1.123 deg | +0.225 | -0.16 deg |
| `pa_lower_quartile_polyline_tangent` | base_over | 15 | 1.154 deg | +0.479 | +1.01 deg |
| `pa_lower_quartile_polyline_tangent` | base_under | 20 | 1.100 deg | +0.034 | -1.04 deg |
| `pa_local_smooth_25` | all | 35 | 0.923 deg | +0.025 | -0.34 deg |
| `pa_local_smooth_25` | base_over | 15 | 0.686 deg | +0.011 | +0.69 deg |
| `pa_local_smooth_25` | base_under | 20 | 1.102 deg | +0.035 | -1.10 deg |
| `pa_local_smooth_50` | all | 35 | 0.956 deg | +0.058 | -0.36 deg |
| `pa_local_smooth_50` | base_over | 15 | 0.701 deg | +0.027 | +0.70 deg |
| `pa_local_smooth_50` | base_under | 20 | 1.148 deg | +0.081 | -1.15 deg |
| `pa_conflict_gated_4deg` | all | 35 | 0.867 deg | -0.032 | -0.31 deg |
| `pa_conflict_gated_4deg` | base_over | 15 | 0.624 deg | -0.051 | +0.62 deg |
| `pa_conflict_gated_4deg` | base_under | 20 | 1.049 deg | -0.017 | -1.01 deg |
| `pa_conflict_gated_7deg` | all | 35 | 0.865 deg | -0.034 | -0.33 deg |
| `pa_conflict_gated_7deg` | base_over | 15 | 0.626 deg | -0.049 | +0.62 deg |
| `pa_conflict_gated_7deg` | base_under | 20 | 1.044 deg | -0.022 | -1.04 deg |

## Interpretation

- Lower-boundary tangent PA is not supported. Both smooth and quartile-polyline lower-boundary tangent variants worsen PA, especially on the under-predicted PA group.
- Plain local PA smoothing is also not supported. It nudges angles toward local medians but worsens the benchmark.
- The only positive signal is conflict-gated local correction: replace a fragment angle with the local median only when it is clearly different from neighbors. This matches the user's "don't arbitrarily smooth everything; only fix obvious local conflicts" intuition.
- The gain is small: PA normalized score improves from 0.150 to 0.144 and overall from 0.170 to 0.168. This is not a standalone submission lever. It is a possible add-on after the larger geometry candidates are decided.

## Files

- Harness: `experiments/exp39_pa_lower_boundary_ablation.py`
- Ignored output bundle: `results/exp39_pa_lower_boundary_ablation/`
- Key outputs: `summary.csv`, `matrix.csv`, per-variant CSVs, `geometry_bundle.json`

## Next

Do not submit this alone. If robust-triangle or curve-blend candidates survive public testing, wire `pa_conflict_gated_7deg` behind an explicit opt-in flag and inspect its overlay before stacking it into a candidate CSV.
