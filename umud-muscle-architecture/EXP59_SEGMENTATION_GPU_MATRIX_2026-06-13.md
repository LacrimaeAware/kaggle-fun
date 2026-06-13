# EXP59 - Segmentation GPU Matrix

Date: 2026-06-13

Purpose: pivot from broad geometry post-processing to better segmentation weights. Recent public
burns show geometry proxies regressed despite benchmark wins; the next credible lever is mask
quality, especially the sparse fragment masks.

Harness: `experiments/exp59_segmentation_gpu_matrix.py`

Code support added:

- `segment_then_measure.py`
  - `UMUD_IMG_SIZE`
  - `UMUD_TTA_EXTRA_SIZE`
  - `UMUD_MODEL_ARCH`
  - `UMUD_MODEL_ENCODER`
  - `UMUD_MODEL_ENCODER_WEIGHTS`
  - `UMUD_LOSS_MODE`
  - `UMUD_AUG_LEVEL`
  - `UMUD_APO_MASK_THRESHOLD`
  - `UMUD_FASC_MASK_THRESHOLD`
  - `UMUD_BATCH_SIZE`
  - `UMUD_WEIGHTS_TAG`
- `local_infer.py` now loads tagged/alternate-architecture weights through the shared model builder.
- `experiments/score_weights.py` now scores tagged/alternate-architecture weights on the expert
  benchmark.
- `kaggle_segment_notebook.ipynb` version gate updated to `2026-06-13.02` and now includes
  selectable EXP59 run presets.
- `kaggle_seg59_02_highres_512_unet_auto.ipynb` is the no-edit Kaggle notebook for the first
  serious run. Import it, add the competition input, set GPU + Internet, and Run All; it bundles
  the submission/debug CSVs and tagged weights into one downloadable zip.
- `kaggle_seg59_sleep_matrix_auto.ipynb` is the sleep-run notebook. It fails fast if the competition
  input is not attached, then runs the serious segmentation candidates sequentially and copies every
  submission/debug CSV to a run-specific filename before zipping all outputs.

## Ordered GPU Runs

| priority | run id | submit? | purpose |
|---:|---|---|---|
| 1 | `seg59_01_repro_384_unet` | only if unexpectedly different | control: old setup through new configurable path |
| 2 | `seg59_02_highres_512_unet` | yes if sanity checks pass | conservative high-resolution test |
| 3 | `seg59_03_highres_512_unetplusplus` | yes if masks look sane | stronger decoder/skip fusion |
| 4 | `seg59_04_highres_focal` | only if support improves without PA drift | sparse-structure loss test |
| 5 | `seg59_05_train_clahe` | exploratory | train-time contrast normalization |
| 6 | `seg59_06_highres_512_unet_strong_aug` | maybe | high-res U-Net with stronger geometric/intensity augmentation |

Exact commands are written to `results/exp59_segmentation_gpu_matrix.csv`. The easiest Kaggle path
for the first serious run is `kaggle_seg59_02_highres_512_unet_auto.ipynb`, which requires no preset
editing. Use `kaggle_segment_notebook.ipynb` only when deliberately switching to another EXP59 run.
The command strings are mainly for RunPod/local shell use.

For an unattended run, use `kaggle_seg59_sleep_matrix_auto.ipynb`. It currently runs:

1. `seg59_02_highres_512_unet`
2. `seg59_03_highres_512_unetplusplus`
3. `seg59_06_highres_512_unet_strong_aug`

This is intentionally segmentation-first. The supervised labels available here are the two mask
families. Text/ruler/tick/field scale logic remains deterministic diagnostics in this run; every
candidate saves a run-specific `calibration_measurement_debug_*.csv` so scale failures remain
visible instead of being hidden behind the new masks.

## Why This Direction

The public leaderboard has now rejected:

- robust triangle stack (`0.60102`)
- visibility-weighted FL proxy (`0.64511`)
- vertical-MT proxy (`0.60720`)
- top-3 FL (`0.62994`)

The only recent public gains were temporal smoothing and scale routing. Geometry work is not dead,
but it should no longer be the main submission source until masks or target labels improve.

## Submission Rule

For each GPU run:

1. Save `submission_segmentation.csv`, `calibration_measurement_debug.csv`, `seg_apo_<tag>.pt`, and
   `seg_fasc_<tag>.pt`.
2. Run `experiments/score_weights.py` with the same env vars and downloaded weights.
3. Check mask sanity: fragment count, geometry success count, scale coverage, and output distribution.
4. Submit only high-resolution/architecture runs that do not show obvious PA/FL/MT distribution drift.

The first serious public candidate is `seg59_02_highres_512_unet`, not another geometry stack.
