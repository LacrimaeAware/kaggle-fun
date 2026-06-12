# EXP42 - Per-Band PA/MT Isolation

Date: 2026-06-12

Purpose: understand the useful part of exp41. Exp41 showed per-band fragment-count averaging improved PA and MT but worsened FL. This experiment freezes FL at the robust-triangle baseline and swaps only PA and/or MT.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

## Summary

| variant | overall | PA | FL | MT | read |
|---|---:|---:|---:|---:|---|
| `fragment_count_average_all_detected_bands_pa_mt_keep_FL_baseline` | **0.169** | **0.146** | 0.278 | **0.082** | tiny real PA/MT gain |
| `fragment_count_average_all_detected_bands_pa_only_keep_FL_baseline` | 0.169 | 0.146 | 0.278 | 0.083 | PA gain only |
| `fragment_count_average_all_detected_bands_mt_only_keep_FL_baseline` | 0.170 | 0.150 | 0.278 | 0.082 | tiny MT gain only |
| `robust_triangle_anchor` | 0.170 | 0.150 | 0.278 | 0.083 | baseline |
| `simple_average_all_detected_bands_pa_mt_keep_FL_baseline` | 0.177 | 0.165 | 0.278 | 0.087 | rejected |

## Read

- The per-band PA/MT signal is real but small.
- PA improvement comes from reducing PA overestimates; PA underestimates are essentially unchanged.
- MT improvement is tiny and mostly helps existing MT underestimates.
- Simple averaging across bands is worse. Fragment-count weighting is the only useful per-band aggregation in this test.
- This supports the user's intuition that exp41's PA/MT movement is worth understanding, but it is not a large enough lever by itself.

## Files

- Harness: `experiments/exp42_per_band_pa_mt_isolation.py`
- Ignored output bundle: `results/exp42_per_band_pa_mt_isolation/`
- Key outputs: `summary.csv`, `matrix.csv`, per-variant CSVs, `geometry_bundle.json`
