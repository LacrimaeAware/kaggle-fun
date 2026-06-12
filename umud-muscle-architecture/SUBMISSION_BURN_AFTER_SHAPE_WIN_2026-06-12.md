# Follow-up Burn Pack After Shape-Neighbor Win - 2026-06-12

`submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv` scored **0.58910**, a large
improvement over burn #6 at 0.60936. Treat burn #11 as the current working baseline.

Update: `submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv` also scored **0.58910**.
The isolated IMG_00275 OCR correction is public-score neutral on top of burn #11.

Update: `submission_burn_14_temporal_subpixel_shape_fl_min_extrap_top3.csv` scored **0.62994**.
The broad top-3/minimal-extrapolation FL combiner is rejected on the real board.

## Current Selection

Best public score is **0.58910**, tied by:

- `results/submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv`
- `results/submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv`

Prefer selecting `13` over `11` for final/private if choosing only one, because it keeps the same public
score while adding the structurally justified one-row OCR scale correction. Selecting both is also
reasonable if the competition allows up to 3 final submissions.

## Generation

```powershell
python umud-muscle-architecture\experiments\exp34_after_shape_win_stack.py
```

Summary CSV:

`results/submission_burn_pack_after_shape_win_summary.csv`

## Interpretation

- #13 was neutral publicly: keep it as a reasonable final/private candidate because it is structurally
  justified and does not cost public score.
- #14 failed: do not wire top-3 minimal-extrapolation FL into production.
- The current best path is temporal smoothing + subpixel scale precision + clean shape-neighbor fallback
  scale. The next serious work is model/segmentation quality or a better scale fallback audit, not broad
  FL aggregation.
