# UMUD Current State - 2026-06-13

This is the current resume point. Older files such as `MASTER_REVIEW.md`, `handoff_brief.md`, and
the dated `EXP*.md` notes are still useful as history, but this file wins when they disagree.

## Straight Answer

Current best public score:

`0.58910`

Best public files:

- `results/submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv`
- `results/submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv`

Burn #13 tied #11 while adding the isolated `IMG_00275` OCR scale correction, so use #13 when one
file must be selected and #11/#13 when multiple final candidates are allowed.

## Latest Public Lessons

| burn | file | public score | read |
|---:|---|---:|---|
| 11 | `submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv` | **0.58910** | current best; temporal + subpixel + clean shape-neighbor scale fallback |
| 13 | `submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv` | **0.58910** | neutral tie; keep as a final-candidate variant |
| 15 | `submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv` | 0.60102 | robust-triangle proxy rejected publicly |
| 16 | `submission_burn_16_core_plus_visibility_weighted_fl_proxy.csv` | 0.64511 | visibility/support FL proxy rejected hard |
| 17 | `submission_burn_17_core_plus_vertical_mt_proxy.csv` | 0.60720 | vertical-MT proxy rejected publicly |
| 22 | `submission_burn_22_field_depth_guarded_scale_probe.csv` | 0.66197 | broad field-depth scale override rejected hard |
| 28 | `submission_burn_28_local_benchmark_proxy_plus_missing_scale.csv` | 0.65917 | stacked local-benchmark proxy rejected hard |

Burn #28 was not a clean "burn #15 plus scale" retest. Outside the four missing-scale rows, its FL
equals burn #16's visibility-weighted FL proxy and its MT equals burn #17's vertical-MT proxy.

## Current Verdict By Dimension

Scale:

- Narrow/gated scale routing is real and was the largest recent public win.
- Broad field-depth scale overrides are dangerous. Knowing displayed depth is not enough; the
  remaining problem is a trusted pixel span.
- EXP64's text/depth OCR is useful: direct OCR finds displayed depth on 237/309 and fused rules cover
  309/309 versus the human review. That solves depth reading, not px/mm by itself.

Geometry:

- The 35-image expert benchmark is useful for debugging conventions, but it has repeatedly
  over-predicted public transfer.
- Robust triangle, top-3 FL, support/visibility FL, and vertical MT all looked plausible locally and
  worsened publicly as currently wired.
- Do not stack rejected geometry proxy deltas again without production-wiring the exact candidate and
  validating on target labels or a stronger benchmark.

Segmentation:

- This is the active direction. The current plan is to spend GPU time on better supervised masks
  rather than more post-processing patches.
- The no-edit first run is `kaggle_seg59_02_highres_512_unet_auto.ipynb`.
- The unattended run is `kaggle_seg59_sleep_matrix_auto.ipynb`.

## What We Were Working On Last

The active work is EXP59 segmentation retraining:

1. Run `kaggle_seg59_sleep_matrix_auto.ipynb` on Kaggle with the competition input attached, GPU on,
   and Internet on.
2. Let it produce `umud_seg59_sleep_matrix_outputs.zip`.
3. Inspect `seg59_sleep_matrix_status.json`, `seg59_sleep_matrix_summary.csv`, each
   `submission_*.csv`, each `calibration_measurement_debug_*.csv`, and `seg59_logs/<run_id>.log`.
4. Submit only a candidate whose output distribution and scale/debug counts look sane.
5. Record every public score immediately in `EXPERIMENT_LOG.md`, `FEATURE_DATABASE.md`, and
   `FEATURE_DATABASE.csv`.

## Next Agenda

Immediate:

1. Wait for the Kaggle sleep matrix or run the single `seg59_02` notebook if time is limited.
2. Compare each segmentation candidate's PA/FL/MT distributions against burn #13.
3. If a segmentation output is sane, submit it as a clean model-quality probe.

After that:

1. If segmentation improves, keep exploring model quality: U-Net++, stronger augmentation, longer
   512 training, and possibly ensembles.
2. If segmentation does not improve, build explicit labels/tooling for scale assets or field spans
   before attempting another scale correction.
3. Keep class-aware geometry work as research only until it is production-wired and target-validated.

## Do Not Repeat

- Do not submit broad field-depth scale overrides from depth text plus field rectangle alone.
- Do not treat local expert-benchmark FL wins as public evidence without a transfer check.
- Do not describe burn #28 as burn #15 plus scale.
- Do not promote support/visibility FL or vertical MT proxies based only on local benchmark wins.

## Public/Privacy Notes

The repo is public-facing. Keep `data/`, `results/`, target human labels, OCR token caches, trained
weights, and generated review artifacts out of git unless deliberately sanitized.
