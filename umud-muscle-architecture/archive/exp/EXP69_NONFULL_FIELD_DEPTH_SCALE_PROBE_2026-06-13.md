# EXP69 Non-Full-Height Field-Depth Scale Probe

## Purpose

Fix the exact failure mode found after burn #22.

Burn #22 regressed to `0.66197` because most changed rows used the whole image
height as the scale span:

```text
90 / 114 changed rows: 50 mm depth and field_h_px = 800 -> 160 px/cm
```

EXP69 keeps only field-depth proposals where the detected field height is not
basically the whole image height.

## Script

```text
experiments/exp69_nonfull_field_depth_scale_probe.py
```

## Outputs

Public-best baseline plus corrected scale gate:

```text
results/submission_burn_24_field_depth_nonfull_scale_probe.csv
```

Best actual benchmark-derived 309-row candidate plus corrected scale gate:

```text
results/submission_burn_25_robust_triangle_nonfull_scale_probe.csv
```

Summary:

```text
results/submission_burn_24_25_nonfull_field_depth_scale_summary.csv
```

## Important Naming Clarification

The best *local benchmark route* is the EXP55/EXP56 route around `0.131`, but it
is not production-wired as a real 309-row submission CSV.

The best actual 309-row benchmark-derived CSV available right now is robust
triangle, burn #15:

```text
results/submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv
public LB: 0.60102
```

Therefore burn #25 is the literal "best available benchmark-derived CSV plus
the corrected scale gate," not the unwired EXP55/EXP56 benchmark route.

## Selection Rule

EXP69 starts from EXP67's algorithmic depth + field-height proposals, then keeps
only rows where:

```text
field_h_px / image_height < 0.98
```

This removes all full-canvas span detections.

## Rows Changed

Rows changed per candidate: `9`

| image_id | old tier | old scale | depth | field h | field fraction | new scale |
|---|---|---:|---:|---:|---:|---:|
| `IMG_00031.tif` | tick-only | 152.3 | 40.0 | 540 | 0.675 | 135.000 |
| `IMG_00198.tif` | mean | n/a | 30.0 | 478 | 0.598 | 159.333 |
| `IMG_00199.tif` | mean | n/a | 30.0 | 478 | 0.598 | 159.333 |
| `IMG_00200.tif` | mean | n/a | 30.0 | 478 | 0.598 | 159.333 |
| `IMG_00251.tif` | mean | n/a | 30.0 | 478 | 0.598 | 159.333 |
| `IMG_00268.png` | verified | 134.2 | 45.0 | 545 | 0.681 | 121.111 |
| `IMG_00278.png` | verified | 151.1 | 40.0 | 528 | 0.660 | 132.000 |
| `IMG_00288.png` | text-confirmed | 120.0 | 50.0 | 522 | 0.653 | 104.400 |
| `IMG_00307.png` | verified | 153.8 | 50.0 | 682 | 0.853 | 136.400 |

## Movement

Burn #24 versus public-best burn #13:

| target | changed rows | mean signed delta | min | max |
|---|---:|---:|---:|---:|
| FL | 9 | +9.268 mm | -2.724 | +16.925 |
| MT | 9 | +0.624 mm | -5.434 | +3.681 |

Burn #25 versus robust-triangle burn #15:

| target | changed rows | mean signed delta | min | max |
|---|---:|---:|---:|---:|
| FL | 9 | +8.556 mm | -3.478 | +15.783 |
| MT | 9 | +0.642 mm | -5.354 | +3.673 |

## Recommendation

Burn #24 is the better public-improvement probe because it starts from the
current public best.

Burn #25 is the literal benchmark-derived diagnostic requested by the user.
Submit it only if the goal is to test the robust-triangle interaction, not if
the goal is the highest expected leaderboard score.
