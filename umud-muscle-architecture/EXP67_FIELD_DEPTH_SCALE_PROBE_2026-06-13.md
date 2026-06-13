# EXP67 Field-Depth Scale Probe

## Purpose

Make one deliberately broad but guarded scale-only public probe.

This tests the user's scale hypothesis directly:

```text
Are many current test-set FL/MT values wrong because the old scale router used
~134 px/cm where displayed depth plus scan-field height implies ~160 px/cm?
```

It does not change PA, boundaries, fragment aggregation, temporal smoothing, or
segmentation.

## Candidate

Script:

```text
experiments/exp67_field_depth_scale_probe.py
```

Output:

```text
results/submission_burn_22_field_depth_guarded_scale_probe.csv
```

Summary:

```text
results/submission_burn_22_field_depth_guarded_scale_probe_summary.csv
```

Baseline:

```text
results/submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv
public LB: 0.58910
```

## Inputs

- `results/exp64_text_scale_ocr/depth_ocr_summary.csv`
  - algorithmic displayed-depth guess
  - full review only used previously to verify these guesses, not as this
    script's predictor
- `results/scale_partition.csv`
  - old scale/router state
- `experiments/exp61_oracle_scale_patch.py::detect_field_rect`
  - heuristic scan-field rectangle detector
- `results/calibration_measurement_debug.csv`
  - pixel FL/MT only for rows that previously had no trusted old scale

## Selection Rules

A row is changed when:

- displayed depth is between `25` and `90` mm;
- the field rectangle is detected;
- proposed scale is between `80` and `180` px/cm;
- old tier is not `flag`;
- if an old scale exists, old-vs-new disagreement is between `8%` and `35%`;
- if no old scale exists, the row is allowed if the new scale is plausible.

This intentionally rejects:

- impossible/garbage OCR depths such as `18 mm`;
- the flagged `IMG_00275.png` tick/ruler conflict;
- extreme field-height disagreements like `IMG_00013.tif`.

## Resulting Movement

Rows changed: `114`

By old tier:

| old tier | rows |
|---|---:|
| text-confirmed | 91 |
| mean | 14 |
| tick-only | 6 |
| verified | 3 |

Submission-level movement versus burn #13:

| target | changed rows | mean abs delta | max abs delta | mean signed delta |
|---|---:|---:|---:|---:|
| PA | 0 | 0.000 | 0.000 | 0.000 |
| FL | 114 | 4.674 mm | 19.801 mm | -4.116 mm |
| MT | 114 | 1.301 mm | 5.434 mm | -1.211 mm |

Most changed rows move downward because the proposed field-depth scale is
usually higher than the old scale (`~134 -> 160 px/cm`).

## Interpretation

This is not a final production scale solver. It is a leaderboard probe for a
single question:

```text
Does broad field-depth scale correction help more than it hurts?
```

If burn #22 improves, scale span detection is a major remaining lever and the
next step is to replace the heuristic rectangle with a learned scan-field/ruler
span detector.

If burn #22 regresses, the old tick/family scale is probably closer on many of
these rows, or the field rectangle detector is still overcounting UI height.
