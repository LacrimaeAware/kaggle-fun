# EXP41 - Per-Band Benchmark

Date: 2026-06-12

Purpose: extract the older per-gap/per-band viewer idea into a benchmark-only harness that does not import the model stack.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv`

Terminology: this document uses **per-band** instead of per-gap to avoid confusion with local fragment gaps.

## Summary

| variant | overall | PA | FL | MT | signed PA | signed FL | signed MT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `robust_triangle_anchor` | **0.170** | 0.150 | 0.278 | 0.083 | -0.32 deg | +2.39 mm | -0.09 mm |
| `per_band_largest_fragment_count_gap_only` | 0.171 | 0.150 | 0.279 | 0.083 | -0.32 deg | +2.41 mm | -0.09 mm |
| `per_band_fragment_count_weighted_average` | 0.174 | 0.146 | 0.293 | 0.082 | -0.34 deg | +2.57 mm | -0.09 mm |

## Multi-Band Subset

The benchmark has 12/35 images with more than two detected bands:

`im_06_arch`, `im_07_arch`, `im_09_arch`, `im_19_arch`, `im_20_arch`, `im_22_arch`, `im_23_arch`, `im_24_arch`, `im_26_arch`, `im_28_arch`, `im_31_arch`, `im_32_arch`.

| variant | subset | overall | PA | FL | MT |
|---|---|---:|---:|---:|---:|
| robust triangle | multi-band | **0.137** | 0.126 | 0.205 | 0.079 |
| per-band weighted average | multi-band | 0.147 | **0.115** | 0.249 | **0.076** |
| per-band largest gap | multi-band | 0.139 | 0.128 | 0.209 | 0.079 |
| robust triangle | two-band | 0.188 | 0.162 | 0.316 | 0.085 |
| per-band weighted average | two-band | 0.188 | 0.162 | 0.316 | 0.085 |
| per-band largest gap | two-band | 0.188 | 0.162 | 0.316 | 0.085 |

## Read

- Per-band separation is locally not a broad win on the 35-image expert benchmark.
- Fragment-count weighted averaging improves PA and MT slightly on the 12 multi-band images, but worsens FL enough to lose overall.
- Largest-fragment-count gap only is almost neutral.
- This does not fully kill per-band for the Kaggle test set, because the public failure of facing-FL may still be a test-set multi-band routing issue. It does mean the naive per-band aggregation tested here should not be promoted.

## Files

- Harness: `experiments/exp41_per_band_benchmark.py`
- Ignored output bundle: `results/exp41_per_band_benchmark/`
- Key outputs: `summary.csv`, per-variant CSVs, `geometry_bundle.json`

## Next

Do not use naive per-band averaging. If per-band returns, it should be as a routing/filtering mechanism for known multi-band failure families, not as a global average over every detected band.
