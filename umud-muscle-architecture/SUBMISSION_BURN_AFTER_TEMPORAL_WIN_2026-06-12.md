# Follow-up Burn Pack After Temporal Win - 2026-06-12

`submission_burn_04_temporal_smooth_092.csv` scored **0.60961**, improving the previous 0.61918
baseline. Do not keep uploading the original unstacked burn order blindly. Treat temporal smoothing as
the new working baseline and test remaining axes on top of it.

## Submit Next

| order | file | axis | note |
|---:|---|---|---|
| 1 | `results/submission_burn_06_temporal_subpixel_scale.csv` | temporal + subpixel scale precision | Low-risk, tiny movement: FL mean 0.094mm, MT mean 0.024mm vs temporal. |
| 2 | `results/submission_burn_07_temporal_shape_neighbor_scale.csv` | temporal + clean shape-neighbor fallback scale | Changes only 10 fallback rows; avoids the old global FL-recenter ripple. |
| 3 | `results/submission_burn_08_temporal_img00275_ocr_scale.csv` | temporal + isolated IMG_00275 OCR scale fix | One verified tick-vs-printed-ruler anomaly; one-row high-leverage probe. |
| 4 | `results/submission_burn_09_temporal_fl_min_extrap_top3.csv` | temporal + top-3 minimal-extrapolation FL | Higher-risk core FL test; submit after the localized scale/sequence probes. |

Optional fifth/tomorrow:

`results/submission_burn_10_temporal_fl_visibility_weighted.csv`

This tests the weaker broad FL-combiner alternative. It moves 307 FL rows and was worse than top-3 on
the rough local labels, so it should not displace the four above unless you want another high-risk FL
probe.

## How These Were Built

Reproducible builder:

```powershell
python umud-muscle-architecture\experiments\exp32_temporal_stack_burn_pack.py
```

Stacking policy:

- One-row/fallback scale probes are applied as deltas **after** temporal smoothing so corrections do
  not smear into neighboring frames.
- Broad FL-combiner probes are temporally smoothed directly, because that is the natural combined
  pipeline.

Summary CSV:

`results/submission_burn_pack_after_temporal_win_summary.csv`

## Interpretation

- If #1 improves, subpixel scale precision is a tiny but real additive cleanup.
- If #2 improves, the remaining fallback-row scale issue matters and should be wired cleanly.
- If #3 improves, wire the isolated IMG_00275 OCR-scale correction.
- If #4 improves, FL aggregation is still alive; pursue support-aware/minimal-extrapolation methods
  with temporal smoothing retained.
- If #1-#3 fail but #4 improves, the wall is FL measurement, not scale.
- If all stacked probes fail, keep temporal smoothing as the current best and move the next work toward
  better segmentation/orientation modeling.
