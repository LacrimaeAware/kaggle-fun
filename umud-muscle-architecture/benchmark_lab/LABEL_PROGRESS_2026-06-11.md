# Label Progress - 2026-06-11

Target manifest:

`results/human_benchmark/target_seed_manifest.csv`

Saved labels:

`results/human_benchmark/target_labels/`

Snapshot archive:

`results/human_benchmark/target_labels_snapshot_2026-06-11.zip`

Score table:

`results/human_benchmark/target_scores.csv`

Human-mask vs 0.619 submission comparison:

`results/human_benchmark/target_human_vs_submission.csv`

## Status

- 19 of 24 target manifest rows have saved label folders.
- 19 of 24 rows have both `apo` and `fasc` mask pixels.
- 19 of 24 rows are measurable by `score_labels.py` using the light cv2/numpy geometry path.
- 57 PNG layer files and 19 metadata JSON files are present in `target_labels/`.
- A first rough comparison against `results/submission_local.csv` was generated for all 19 measurable
  rows.

## Labeled Rows

| image_id | apo_pixels | fasc_pixels | measured_fascicles |
|---|---:|---:|---:|
| IMG_00275 | 5132 | 7336 | 7 |
| IMG_00086 | 8658 | 13120 | 7 |
| IMG_00186 | 8791 | 29046 | 9 |
| IMG_00033 | 8774 | 25020 | 12 |
| IMG_00259 | 4354 | 7233 | 4 |
| IMG_00305 | 5082 | 14397 | 11 |
| IMG_00059 | 7992 | 16287 | 5 |
| IMG_00048 | 4536 | 5144 | 4 |
| IMG_00152 | 8473 | 12436 | 4 |
| IMG_00235 | 9130 | 20209 | 8 |
| IMG_00123 | 15896 | 19871 | 8 |
| IMG_00280 | 5731 | 9337 | 7 |
| IMG_00266 | 4335 | 8494 | 6 |
| IMG_00108 | 7901 | 18306 | 7 |
| IMG_00178 | 8486 | 14005 | 6 |
| IMG_00220 | 8691 | 28641 | 11 |
| IMG_00014 | 6319 | 15855 | 10 |
| IMG_00278 | 4340 | 9435 | 6 |
| IMG_00054 | 21091 | 16088 | 9 |

## Unlabeled Manifest Rows

- IMG_00105
- IMG_00044
- IMG_00006
- IMG_00064
- IMG_00055

## Quality Notes

The user described these first labels as useful but rough: early rows used the curve tool more
intentionally, later rows were faster and more straight-line oriented due fatigue. Treat this pack as a
first diagnostic benchmark, not final ground truth. It is already valuable for checking whether the
pipeline agrees with a human-visible structure on target images, but any high-stakes method decision
should inspect overlays and possibly refine a smaller subset.

## First Comparison to the 0.619 Submission

Using `calibration_measurement_debug.csv` to convert human-mask pixels to millimeters, the rough
human-mask geometry differs from the 0.619 submission by:

| target | mean absolute difference | median absolute difference | mean tolerance units |
|---|---:|---:|---:|
| PA | 3.85 deg | 2.87 deg | 0.64 |
| FL | 7.87 mm | 5.45 mm | 0.66 |
| MT | 1.13 mm | 0.76 mm | 0.38 |

Largest normalized disagreements:

| image_id | PA delta | FL delta | MT delta |
|---|---:|---:|---:|
| IMG_00123 | +3.03 deg | +32.44 mm | +9.46 mm |
| IMG_00275 | +9.94 deg | -10.63 mm | -0.77 mm |
| IMG_00278 | +8.57 deg | -8.27 mm | -0.88 mm |
| IMG_00259 | +7.96 deg | -5.45 mm | -0.69 mm |
| IMG_00266 | +7.87 deg | -5.32 mm | -0.76 mm |
| IMG_00178 | +4.05 deg | -12.75 mm | -0.08 mm |
| IMG_00059 | +2.61 deg | -10.85 mm | -1.13 mm |
| IMG_00186 | +2.75 deg | +12.18 mm | -0.40 mm |

Interpretation: this is a triage list. It shows where the human labels and shipped submission disagree
most, but the labels were produced quickly and should not yet be treated as final truth. `IMG_00123`
is the first row to inspect because it disagrees on both FL and MT.

Important labeling convention confirmed during the session:

- For `apo`, drawing the gap-facing boundary line is acceptable.
- Multiple strokes on one boundary are acceptable.
- Avoid connecting the upper and lower `apo` boundaries into one blob.
- For `fasc`, draw visible slanted fragments only; do not extrapolate into missing/off-screen regions.

## Next Use

1. Build an overlay/report comparing production predictions to these 19 human masks.
2. Compare the safe 0.619 baseline geometry to human-mask geometry on the same rows.
3. Use disagreements to choose the next small label/refinement batch, preferably only a few images at
   a time with more deliberate review.
