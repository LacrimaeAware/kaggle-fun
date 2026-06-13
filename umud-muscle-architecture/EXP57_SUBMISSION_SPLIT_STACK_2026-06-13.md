# EXP57 - Submission Split Stack

Date: 2026-06-13

Purpose: create concrete leaderboard CSVs that approximate the EXP56 split using already-generated
production deltas. This was necessary because the best EXP55/EXP56 route is benchmark-only and cannot
yet be produced for all 309 test rows without wiring reducers/class gates into `local_infer.py`.

Harness: `experiments/exp57_submission_split_stack.py`

Ignored output summary: `results/submission_burn_16_17_split_stack_summary.csv`

## Current Anchor

Public anchor:

`results/submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv`

Known public score: `0.58910`.

## Generated CSVs

| file | role | movement vs public anchor | public score | read |
|---|---|---:|---:|---|
| `results/submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv` | core production geometry | 0.073669 normalized movement | 0.60102 | rejected vs `0.58910`; robust triangle local win did not transfer enough |
| `results/submission_burn_16_core_plus_visibility_weighted_fl_proxy.csv` | FL-support proxy split | 0.182325 normalized movement | 0.64511 | rejected hard; broad visibility-weighted FL proxy moved too much |
| `results/submission_burn_17_core_plus_vertical_mt_proxy.csv` | MT proxy split | 0.077956 normalized movement | 0.60720 | rejected vs `0.58910`; less bad than #16 but not useful |

## Exact Movements

| candidate | PA changed | FL changed | FL mean abs | FL p95 abs | MT changed | MT mean abs | note |
|---|---:|---:|---:|---:|---:|---:|---|
| burn 15 core robust | 0 | 307 | 2.304 mm | 9.394 mm | 294 | 0.087 mm | safest geometry probe |
| burn 16 core + visibility FL | 0 | 307 | 6.215 mm | 17.531 mm | 294 | 0.087 mm | high movement; submit only as an FL-axis burn |
| burn 17 core + vertical MT | 0 | 307 | 2.304 mm | 9.394 mm | 295 | 0.126 mm | MT-axis burn; prior public MT signal was poor |

## Recommendation

Submitted scores:

1. `submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv`: `0.60102`
2. `submission_burn_16_core_plus_visibility_weighted_fl_proxy.csv`: `0.64511`
3. `submission_burn_17_core_plus_vertical_mt_proxy.csv`: `0.60720`

The actual best benchmark route is not one of these files yet. The next engineering step is to wire
the EXP50/EXP53 weighted reducers and EXP55 class gates into the 309-image production path.
