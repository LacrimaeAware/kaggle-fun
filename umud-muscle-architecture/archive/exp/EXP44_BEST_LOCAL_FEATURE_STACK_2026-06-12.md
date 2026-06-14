# EXP44 - Best Local Feature Stack

Date: 2026-06-12

Purpose: combine isolated locally useful benchmark features into a clean research anchor. This does not discover new features and does not generate a submission.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

## Summary

| variant | overall | PA | FL | MT |
|---|---:|---:|---:|---:|
| `FL_scan_region_linear_plus_PA_conflict_gate_plus_MT_vertical_center` | **0.153** | **0.144** | **0.245** | **0.070** |
| `FL_scan_region_linear_plus_PA_per_band_avg_plus_MT_vertical_center` | 0.154 | 0.146 | 0.245 | 0.070 |
| `PA_conflict_gate_plus_MT_vertical_center_keep_FL_baseline` | 0.164 | 0.144 | 0.278 | 0.070 |
| `PA_per_band_avg_plus_MT_vertical_center_keep_FL_baseline` | 0.165 | 0.146 | 0.278 | 0.070 |
| `robust_triangle_anchor` | 0.170 | 0.150 | 0.278 | 0.083 |

## Components

- FL: strict scan-region linear support weighting from exp40.
- PA: conflict-gated local angle replacement from exp39, or per-band fragment-count PA from exp42.
- MT: vertical center gap from exp43.

## Read

- Best local composition improves robust-triangle benchmark `0.170 -> 0.153`.
- This is a useful research anchor, not a public-transfer claim. Each component still needs viewer inspection and production wiring before any submission.
- PA/MT together can improve `0.170 -> 0.164` even with FL frozen, so the PA/MT work is not imaginary. It is just smaller than the FL/scale levers.

## Files

- Harness: `experiments/exp44_best_local_feature_stack.py`
- Ignored output bundle: `results/exp44_best_local_feature_stack/`
- Key output: `summary.csv`
