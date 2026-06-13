# EXP72 Thin-Structure Segmentation

Created after the EXP59 sleep matrix proved too conservative for the actual problem. EXP59 mostly
varied architecture, resolution, augmentation, and loss while keeping the same thin binary target.
EXP72 changes the target/decoding formulation itself.

## Why EXP59 Was Not Enough

Facts from repeated runs:

- The aponeurosis task usually validates around `0.79-0.80` Dice.
- The fascicle/thin-line task repeatedly validates far lower, often around `0.25-0.35`.
- More epochs on the same binary target have not obviously broken that pattern.
- Pixel Dice is harsh for thin lines: a near miss can score like a miss.
- Leaderboard burns have mostly rejected geometry/scale patches, so the next real lever is mask
  quality or scale extraction, with mask quality now prioritized.

The mistake in EXP59 was treating "heavy overnight" mostly as longer/more varied training settings.
The better next attempt is to change the representation used for the thin target.

## Code Changes

`segment_then_measure.py` now supports:

- `UMUD_FASC_TARGET_MODE`
  - `binary`
  - `dilate3`, `dilate5`, etc.
  - `soft5`, `soft7`, etc.
  - `dilate_soft5`, etc.
- `UMUD_AUTO_THRESHOLD=1`
  - Sweeps validation thresholds from `UMUD_THRESHOLD_SWEEP`.
  - Uses the best validation threshold for inference inside the run.
- `UMUD_FASC_POSTPROCESS`
  - `threshold`
  - `skeleton`
  - `skeleton_dilate`
- `UMUD_AUG_LEVEL=heavy`
  - Stronger but version-stable augmentation for longer GPU runs.
- `UMUD_SAVE_PRED_DEBUG`
  - Saves predicted apo/fasc masks for the first N test images per run.

Pipeline version is now `2026-06-13.03`.

## Notebook

Use:

`kaggle_seg72_thin_structure_heavy_auto.ipynb`

This is the heavy notebook. It writes:

- `seg72_logs/<run_id>.log`
- `seg72_thin_structure_status.json`
- `seg72_thin_structure_summary.csv`
- `submission_<run_id>.csv`
- `calibration_measurement_debug_<run_id>.csv`
- `pred_debug_<weights_tag>/*.png`
- `umud_seg72_thin_structure_outputs.zip`

## Run Matrix

1. `seg72_01_soft5_tversky_640_unetpp`
   - Soft Gaussian target, Tversky-style loss, auto thresholding, skeleton-dilate decoding.
   - Primary serious candidate.

2. `seg72_02_dilate_soft5_tversky_640_unetpp`
   - Thicker soft target. Tests whether near-miss gradients help more than pure soft centerline.

3. `seg72_03_soft7_focal_768_unet`
   - Higher effective resolution with a wider soft target. Tests whether resize is destroying thin detail.

4. `seg72_04_recall_dilate3_tversky_640_unetpp`
   - Recall-biased variant with a lower component area gate. Intended to reveal missed fragments.

## Expected Interpretation

If EXP72 still produces the same public wall, then the failure is probably not solved by standard
binary/soft mask training alone. The next step would be a more explicit geometry target: orientation
field, line-center heatmap plus endpoint/support prediction, or a separate field/ruler/text model.

If EXP72 improves validation thin-line Dice but not public score, inspect `pred_debug_*` and
`calibration_measurement_debug_*` before submitting more variants. The postprocessing may be finding
more fragments but feeding the measurement step worse components.
