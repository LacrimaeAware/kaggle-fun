# EXP43 - PA/MT Geometry Conventions

Date: 2026-06-12

Purpose: test geometric measurement conventions for PA and MT without changing FL.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

## Summary

| variant | overall | PA | FL | MT | read |
|---|---:|---:|---:|---:|---|
| `MT_only_vertical_center_gap_keep_PA_FL` | **0.166** | 0.150 | 0.278 | **0.070** | useful MT convention |
| `robust_triangle_anchor` | 0.170 | 0.150 | 0.278 | 0.083 | baseline |
| `MT_only_perpendicular_mean_across_boundary_width_keep_PA_FL` | 0.177 | 0.150 | 0.278 | 0.103 | rejected |
| `MT_only_vertical_three_positions_gap_keep_PA_FL` | 0.182 | 0.150 | 0.278 | 0.118 | rejected |
| `PA_only_relative_to_average_upper_and_lower_boundary_direction_keep_FL_MT` | 0.231 | 0.333 | 0.278 | 0.083 | rejected |
| `PA_only_relative_to_local_upper_boundary_tangent_keep_FL_MT` | 0.316 | 0.586 | 0.278 | 0.083 | rejected |
| `PA_only_smaller_angle_to_upper_or_lower_boundary_keep_FL_MT` | 0.323 | 0.609 | 0.278 | 0.083 | rejected |

## Read

- The benchmark strongly prefers PA relative to the lower/deep boundary convention. Upper-boundary tangent and average-boundary PA conventions are decisively wrong locally.
- MT improves when measured as the vertical center gap instead of perpendicular center gap: MT term `0.083 -> 0.070`.
- Three-position MT and mean-across-width MT are worse under robust-triangle geometry. The useful MT convention is specifically center vertical gap, not broad averaging.

## Files

- Harness: `experiments/exp43_pa_mt_geometry_conventions.py`
- Ignored output bundle: `results/exp43_pa_mt_geometry_conventions/`
- Key outputs: `summary.csv`, per-variant CSVs, `geometry_bundle.json`
