# EXP40 - Untested Feature Benchmark

Date: 2026-06-12

Purpose: start testing the untested feature-database rows against the current robust-triangle expert-benchmark anchor.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

This is a benchmark harness only, not a submission generator.

## Summary

| variant | overall | PA | FL | MT | signed PA | signed FL | signed MT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strict_scan_region_linear_support_weighted_FL_only` | **0.159** | 0.150 | **0.245** | 0.083 | -0.32 deg | +0.49 mm | -0.09 mm |
| `lower_edge_quartile_median_polyline_FL_only` | 0.167 | 0.150 | 0.268 | 0.083 | -0.32 deg | +1.33 mm | -0.09 mm |
| `robust_triangle_anchor` | 0.170 | 0.150 | 0.278 | 0.083 | -0.32 deg | +2.39 mm | -0.09 mm |
| `lower_edge_quartile_median_polyline_FL_MT` | 0.176 | 0.150 | 0.268 | 0.110 | -0.32 deg | +1.33 mm | -0.21 mm |
| `lower_edge_quartile_median_polyline_MT_only` | 0.179 | 0.150 | 0.278 | 0.110 | -0.32 deg | +2.39 mm | -0.21 mm |
| `strict_scan_region_and_visible_support_weighted_FL_only` | 0.198 | 0.150 | 0.361 | 0.083 | -0.32 deg | -1.01 mm | -0.09 mm |
| `strict_scan_region_and_visible_support_weighted_PA_FL` | 0.216 | 0.205 | 0.361 | 0.083 | +0.18 deg | -1.01 mm | -0.09 mm |

## Read

- The real on-screen/off-screen idea has signal, but only in a gentle form.
- `strict_scan_region_linear_support_weighted_FL_only` improves the robust anchor from `0.170 -> 0.159`, with FL `0.278 -> 0.245`. It leaves PA and MT unchanged by design.
- The improvement is not symmetric: on rows where robust triangle already overshot FL, FL MAE improves by `-0.723 mm`; on rows where robust triangle undershot FL, it worsens by `+0.555 mm`.
- The harsher squared visible+scan weighting is rejected. It overcorrects downward and worsens FL to `0.361`.
- Weighting PA with the same support rule is rejected. PA worsens from `0.150 -> 0.205`.
- The lower-edge quartile median polyline is mixed: it helps FL a little if used for lower intersections (`0.278 -> 0.268`) but worsens MT (`0.083 -> 0.110`). MT-only lower-boundary replacement is rejected on the current benchmark.

## Matrix Highlights

Negative delta is better.

| variant | metric | group | n | delta MAE | signed bias |
|---|---|---:|---:|---:|---:|
| `strict_scan_region_linear_support_weighted_FL_only` | FL | all | 35 | -0.395 mm | +0.49 mm |
| `strict_scan_region_linear_support_weighted_FL_only` | FL | base_over | 26 | -0.723 mm | +1.47 mm |
| `strict_scan_region_linear_support_weighted_FL_only` | FL | base_under | 9 | +0.555 mm | -2.33 mm |
| `lower_edge_quartile_median_polyline_FL_only` | FL | all | 35 | -0.117 mm | +1.33 mm |
| `lower_edge_quartile_median_polyline_FL_only` | FL | base_over | 26 | -0.649 mm | +2.92 mm |
| `lower_edge_quartile_median_polyline_FL_only` | FL | base_under | 9 | +1.421 mm | -3.26 mm |
| `lower_edge_quartile_median_polyline_MT_only` | MT | all | 35 | +0.081 mm | -0.21 mm |
| `lower_edge_quartile_median_polyline_MT_only` | MT | base_over | 14 | -0.009 mm | +0.10 mm |
| `lower_edge_quartile_median_polyline_MT_only` | MT | base_under | 21 | +0.141 mm | -0.42 mm |

## Files

- Harness: `experiments/exp40_untested_feature_benchmark.py`
- Ignored output bundle: `results/exp40_untested_feature_benchmark/`
- Key outputs: `summary.csv`, `matrix.csv`, per-variant CSVs, `geometry_bundle.json`

## Next

This creates one new serious local feature: `strict_scan_region_linear_support_weighted_FL_only`.

Do not submit it alone yet. Next step is to stack it with robust triangle and inspect the viewer geometry, then decide if it deserves a public slot. Per-band separation remains untested in this harness because the existing per-band prototype imports the full model stack and needs a clean benchmark-only extraction.
