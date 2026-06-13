# EXP66 Robust Triangle Scale Candidates

## Purpose

Retest the robust-triangle geometry stack under the newly repaired 3 cm scale
rows.

The old robust-triangle public result was:

```text
results/submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv
public LB: 0.60102
```

That result is still a real public regression versus `0.58910`, but it did not
include the EXP64/EXP65 3 cm scale-span repair. So the correct next question is
not "resubmit burn 15"; it is "does robust triangle still regress after the
known 3 cm scale rows are repaired?"

## Script

```text
experiments/exp66_robust_triangle_with_scale_candidates.py
```

The script starts from burn 15 and recomputes changed FL/MT values from:

```text
results/calibration_debug_robust_triangle_only.csv
```

This avoids applying an approximate delta from the public-best geometry. PA is
unchanged; only FL and MT are rescaled on the named rows.

## Candidate Files

### Burn 20: robust triangle plus 3-row 3 cm scale repair

```text
results/submission_burn_20_robust_triangle_plus_3cm_scale_198_200.csv
```

Rows changed from burn 15:

| image_id | FL delta | MT delta |
|---|---:|---:|
| `IMG_00198.tif` | `+14.594 mm` | `-0.295 mm` |
| `IMG_00199.tif` | `+7.754 mm` | `-2.176 mm` |
| `IMG_00200.tif` | `+15.783 mm` | `-0.323 mm` |

This is the cleaner robust-triangle retest.

### Burn 21: robust triangle plus 4-row 3 cm scale repair

```text
results/submission_burn_21_robust_triangle_plus_3cm_scale_198_200_251.csv
```

Rows changed from burn 15:

| image_id | FL delta | MT delta |
|---|---:|---:|
| `IMG_00198.tif` | `+14.594 mm` | `-0.295 mm` |
| `IMG_00199.tif` | `+7.754 mm` | `-2.176 mm` |
| `IMG_00200.tif` | `+15.783 mm` | `-0.323 mm` |
| `IMG_00251.tif` | `-3.478 mm` | `-5.354 mm` |

This is higher risk because `IMG_00251.tif` moves MT by more than one MT
tolerance.

## Validation

Both CSVs were checked after generation:

- shape: `309 x 4`
- columns: `image_id`, `pa_deg`, `fl_mm`, `mt_mm`
- duplicate IDs: `0`
- missing values: `0`

## Recommendation

If using two public submissions now:

1. Submit burn 18 first as the cleanest scale-only probe:
   `results/submission_burn_18_oracle_scale_198_200_direct.csv`
2. Submit burn 20 as the repaired-scale robust-triangle retest:
   `results/submission_burn_20_robust_triangle_plus_3cm_scale_198_200.csv`

Do not submit burn 21 before burn 20 unless specifically isolating
`IMG_00251.tif`; it compounds the robust geometry retest with a risky MT move.
