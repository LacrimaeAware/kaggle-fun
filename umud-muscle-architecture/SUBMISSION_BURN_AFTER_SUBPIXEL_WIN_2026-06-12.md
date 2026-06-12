# Follow-up Burn Pack After Subpixel Win - 2026-06-12

`submission_burn_06_temporal_subpixel_scale.csv` scored **0.60936**, improving the temporal-only
0.60961 result. Treat burn #6 as the current working baseline.

Update: `submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv` then improved to **0.58910**,
so the next order is superseded by `SUBMISSION_BURN_AFTER_SHAPE_WIN_2026-06-12.md`.

## Submit Next

| order | file | axis | note |
|---:|---|---|---|
| 1 | `results/submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv` | temporal + subpixel + clean shape-neighbor fallback scale | Changes only 10 fallback rows relative to burn #6. This replaces the older `07` file. |
| 2 | `results/submission_burn_12_temporal_subpixel_img00275_ocr_scale.csv` | temporal + subpixel + isolated IMG_00275 OCR scale fix | One verified tick-vs-printed-ruler anomaly relative to burn #6. This replaces the older `08` file. |

After those two, if slots remain:

`results/submission_burn_09_temporal_fl_min_extrap_top3.csv`

That is still the best broad FL-combiner probe. It is higher-risk because it moves 307 FL rows.

## Generation

```powershell
python umud-muscle-architecture\experiments\exp33_after_subpixel_win_stack.py
```

Summary CSV:

`results/submission_burn_pack_after_subpixel_win_summary.csv`

## Interpretation

- If #11 improves, the fallback scale rows add on top of temporal+subpixel and should be wired cleanly.
- If #12 improves, wire the isolated IMG_00275 OCR correction.
- If both fail, keep burn #6 as the current best and only use remaining slots on high-information FL
  probes, not more scale polishing.
