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
- EXP78 is the compact current scale synthesis. The user manually reviewed all 309 test-image
  displayed depths; after repairs, the algorithm-only depth guesser matched the full review 309/309
  without reading the notes file as predictor input. The standing boundary is unchanged: depth is
  solved, broad px/cm scale is not.

Geometry:

- The 35-image expert benchmark is useful for debugging conventions, but it has repeatedly
  over-predicted public transfer.
- Robust triangle, top-3 FL, support/visibility FL, and vertical MT all looked plausible locally and
  worsened publicly as currently wired.
- Do not stack rejected geometry proxy deltas again without production-wiring the exact candidate and
  validating on target labels or a stronger benchmark.

Segmentation:

- This is the active direction, but EXP73/EXP74 now supersede the "just run EXP72" advice.
- EXP75 adds the missing external-method layer: published muscle-ultrasound pipelines support a
  hybrid approach of learned boundaries plus classical fascicle-line extraction, dominant-orientation
  clustering, and non-crossing line cleanup. It also argues for masked in-domain ultrasound
  pretraining and Kaggle-grade folds/OOF/TTA/threshold diagnostics.
- EXP76 is still useful as a controlled diagnostic matrix, but it is no longer the first notebook to
  run when the user asks for the strongest candidate.
- EXP77 is the current recommended overnight notebook: it runs the strongest implemented
  segmentation candidate first, then serious alternates if wall time remains.
- The no-edit first run is `kaggle_seg59_02_highres_512_unet_auto.ipynb`.
- The unattended run is `kaggle_seg59_sleep_matrix_auto.ipynb`.
- The current recommended tonight run is `kaggle_seg77_best_effort_heavy_auto.ipynb`.
- `seg59_02_highres_512_unet` is the current segmentation control: apo best Dice `0.7945`, fasc best
  Dice `0.2925`.
- `seg72_01_soft5_tversky_640_unetpp` underperformed the control in the partial log: apo best Dice
  `0.7873`, fasc best Dice `0.2594` by epoch 20. Treat EXP72 as under audit, not as the current
  recommended overnight run.
- The deeper audit found that EXP72 did not faithfully implement clDice/boundary/skeleton-recall style
  thin-structure training; it used soft/dilated targets plus hard post-hoc skeleton decoding and
  changed too many knobs at once.
- The next segmentation follow-up after an EXP77 checkpoint should include inference-only
  recall-heavy variants: lower fascicle thresholds, lower minimum component area, and threshold versus
  skeleton-dilate postprocess. This tests the user's "guess more, then let geometry filter" idea
  without spending another full training run.

## What We Were Working On Last

The active work is segmentation retraining, now split into two tiers:

1. EXP59 (`kaggle_seg59_sleep_matrix_auto.ipynb`) is the conservative GPU matrix: higher resolution,
   architecture, loss, and augmentation variations while keeping the same binary mask target.
2. EXP72 (`kaggle_seg72_thin_structure_heavy_auto.ipynb`) is the first heavy thin-structure attempt:
   soft/dilated fascicle targets, validation threshold sweep, skeleton-style decoding, and debug mask
   exports. Its first run underperformed and the matrix is too confounded; hold it unless gathering
   artifacts.
3. EXP73 (`EXP73_SEGMENTATION_METHOD_AUDIT_2026-06-13.md`) is now the active segmentation decision
   note. It recommends stopping/bundling `seg72_01`, holding remaining EXP72 runs, and adding
   instrumentation before more training.
4. EXP74 (`EXP74_CONTROLLED_SEGMENTATION_ABLATION_PLAN_2026-06-13.md`) is the next notebook spec:
   baseline settings plus one change at a time, decoder sweeps, probability/debug outputs, component
   counts, and downstream geometry summaries.
5. EXP75 (`EXP75_EXTERNAL_ULTRASOUND_AND_KAGGLE_METHOD_REVIEW_2026-06-14.md`) is the external-method
   correction. It says the next non-GPU local experiment should be a classical fascicle-line extractor:
   CLAHE/ridge filtering, skeletonized line segments, dominant-orientation clustering, extension to
   boundaries, and non-crossing cleanup.
6. EXP76 (`EXP76_TONIGHT_NOTEBOOK_AUDIT_2026-06-14.md`) is the controlled diagnostic Kaggle plan. It
   generates `kaggle_seg76_controlled_diagnostics_auto.ipynb`, which tests binary control, soft
   target, dilated target, Tversky loss, U-Net++ architecture, and a final soft+Tversky combo.
7. EXP77 (`EXP77_BEST_EFFORT_SEGMENTATION_NOTEBOOK_2026-06-14.md`) is the current recommended
   overnight Kaggle plan. It generates `kaggle_seg77_best_effort_heavy_auto.ipynb`, with
   `seg77_01_best_unetpp640_dilate_soft5_cldice` as the main best-effort candidate.
8. EXP78 (`EXP78_SCALE_REVIEW_AND_RECALL_SEGMENTATION_STATE_2026-06-14.md`) is the compact synthesis
   of the full 309-row scale-depth review and the recall-heavy segmentation follow-up idea.
9. Inspect each notebook's status JSON, summary CSV, run logs, submissions, calibration debug CSVs,
   and any `pred_debug_*` masks before submitting.
10. Submit only a candidate whose output distribution and scale/debug counts look sane.
11. Record every public score immediately in `EXPERIMENT_LOG.md`, `FEATURE_DATABASE.md`, and
   `FEATURE_DATABASE.csv`.

## Next Agenda

Immediate:

1. Run `kaggle_seg77_best_effort_heavy_auto.ipynb` on Kaggle GPU with Internet on and the UMUD
   competition input attached.
2. Download `umud_seg77_best_effort_outputs.zip` and inspect `seg77_best_effort_summary.csv`, run
   logs, submission CSVs, calibration debug CSVs, and `pred_debug_*` masks.
3. If EXP77 produces usable weights, generate recall-heavy inference-only variants before concluding
   the segmentation branch failed.
4. Bundle/download any partial EXP72 outputs and logs if they still exist.
5. Build the EXP75 classical fascicle-line extractor harness as a local, inspectable experiment. This
   is the fastest way to test whether ultrasound texture contains a recoverable geometry signal that
   our fascicle masks miss.
6. Run EXP76 only if we specifically want controlled one-axis diagnostics after the EXP77 best-effort
   run.
7. Add/build EXP74 instrumentation for threshold-only vs skeleton decoding, probability/debug maps,
   component counts, accepted fragment counts, and downstream geometry distributions.
8. Build the controlled thin-line ablation notebook from the EXP74 plan: baseline settings plus one
   change at a time, preferably with target-specific training/reuse so apo and fasc are not confounded.

After that:

1. If EXP77 or later controlled segmentation improves, then scale to longer/heavier variants.
2. If EXP75's classical extractor produces sane line candidates, compare it against current masks,
   human scratch labels, and the expert benchmark before deciding whether to use it as a production
   feature or pseudo-label source.
3. If segmentation does not improve, build explicit labels/tooling for scale assets or field spans
   before attempting another scale correction.
4. Keep class-aware geometry work as research only until it is production-wired and target-validated.

## Do Not Repeat

- Do not submit broad field-depth scale overrides from depth text plus field rectangle alone.
- Do not confuse displayed depth with px/cm scale. The 309-row depth review solved depth, not the
  trusted-span problem.
- Do not treat local expert-benchmark FL wins as public evidence without a transfer check.
- Do not describe burn #28 as burn #15 plus scale.
- Do not promote support/visibility FL or vertical MT proxies based only on local benchmark wins.
- Do not call a notebook "best effort" unless the strongest candidate runs first and the remaining
  runs are serious alternates, not deliberately weak experiments.

## Public/Privacy Notes

The repo is public-facing. Keep `data/`, `results/`, target human labels, OCR token caches, trained
weights, and generated review artifacts out of git unless deliberately sanitized.
