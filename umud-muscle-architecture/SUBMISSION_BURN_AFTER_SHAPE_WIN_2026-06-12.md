# Follow-up Burn Pack After Shape-Neighbor Win - 2026-06-12

`submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv` scored **0.58910**, a large
improvement over burn #6 at 0.60936. Treat burn #11 as the current working baseline.

## Submit Next

| order | file | axis | note |
|---:|---|---|---|
| 1 | `results/submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv` | current best + isolated IMG_00275 OCR scale fix | One verified tick-vs-printed-ruler anomaly. This replaces older `12`, which lacked the shape-neighbor gain. |
| 2 | `results/submission_burn_14_temporal_subpixel_shape_fl_min_extrap_top3.csv` | current best + top-3 minimal-extrapolation FL | Higher-risk core FL aggregation probe; moves 307 FL rows. |

## Generation

```powershell
python umud-muscle-architecture\experiments\exp34_after_shape_win_stack.py
```

Summary CSV:

`results/submission_burn_pack_after_shape_win_summary.csv`

## Interpretation

- If #13 improves, wire the isolated IMG_00275 OCR scale correction on top of the current best.
- If #14 improves, FL aggregation is still a live lever; keep temporal+subpixel+shape as the baseline
  and develop support-aware/minimal-extrapolation FL.
- If #13 fails and #14 fails, current best is burn #11 and the next work is likely model/segmentation
  quality rather than scale cleanup.
