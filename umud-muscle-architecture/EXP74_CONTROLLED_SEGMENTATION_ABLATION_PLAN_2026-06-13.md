# EXP74 Controlled Segmentation Ablation Plan

Date: 2026-06-13

Status: plan, not yet executed.

Purpose: define the next GPU notebook after EXP72 underperformed. EXP74 should answer why the
thin-line model is failing before we spend another long run.

## Core Principle

No more "change everything and hope." EXP74 changes one axis at a time and logs the downstream
measurement consequences.

The goal is not just higher validation Dice. The goal is better masks for the measurement pipeline:
stable boundary bands, enough usable thin fragments, sane PA/FL/MT distributions, and no public-score
style broad proxy drift.

## Control Settings

Use the EXP59 control as the anchor:

- `UMUD_IMG_SIZE=512`
- `UMUD_MODEL_ARCH=unet`
- `UMUD_MODEL_ENCODER=resnet34`
- `UMUD_MODEL_ENCODER_WEIGHTS=imagenet`
- `UMUD_LOSS_MODE=dice_bce`
- `UMUD_AUG_LEVEL=light`
- `UMUD_BATCH_SIZE=6` when memory allows
- `UMUD_FASC_TARGET_MODE=binary`
- `UMUD_FASC_POSTPROCESS=threshold`
- `UMUD_FASC_MIN_AREA=40`
- `UMUD_FASC_MIN_ANG=6`
- `UMUD_TTA=1`
- `UMUD_TOP_BOUNDARY_MODE=line`
- `UMUD_MT_MODE=perp_center`

The first EXP74 notebook should either load/reuse the known-good apo weights or make apo/fasc configs
separate. The fasc experiment should not be allowed to damage apo while solving a fasc problem.

## Required Instrumentation

Each run must write:

- `seg74_summary.csv`
- `seg74_decoder_sweep_<run_id>.csv`
- `seg74_geometry_summary_<run_id>.csv`
- `submission_<run_id>.csv`
- `calibration_measurement_debug_<run_id>.csv`
- `run_config_<run_id>.json`
- `pred_debug_<run_id>/` hard masks
- optional `prob_debug_<run_id>/` compressed probability maps or fixed-image PNG heatmaps

The decoder sweep should report, at minimum:

- target: apo/fasc,
- threshold,
- decoder: threshold/skeleton/skeleton_dilate,
- validation Dice,
- predicted area fraction,
- connected component count,
- for fasc: accepted fragment count after measurement filters.

The geometry summary should report:

- geometry success count,
- PA mean/median/std,
- FL mean/median/std,
- MT mean/median/std,
- count clipped at PA/FL/MT bounds,
- expert-benchmark score when local data/weights are available.

## Run Matrix

| run id | train change | decoder during scoring | purpose |
|---|---|---|---|
| `seg74_00_score_existing_seg59_02` | no training if weights are available | threshold/skeleton/skeleton_dilate sweep | measure whether decoder choice alone explains anything |
| `seg74_01_control_binary_auto_thr` | EXP59 control plus threshold sweep | threshold | clean control under the new instrumentation |
| `seg74_02_soft5_no_skeleton` | only `UMUD_FASC_TARGET_MODE=soft5` | threshold | isolate soft target without hard skeleton confound |
| `seg74_03_dilate3_no_skeleton` | only `UMUD_FASC_TARGET_MODE=dilate3` | threshold | isolate thicker target support |
| `seg74_04_dilate_soft5_no_skeleton` | only `UMUD_FASC_TARGET_MODE=dilate_soft5` | threshold | test near-miss gradients without decoder damage |
| `seg74_05_decoder_ablation_best_target` | reuse best checkpoint from 01-04 | threshold/skeleton/skeleton_dilate | test skeletonization as decoder, not training |

Do not put U-Net++, 640/768 sizes, heavy augmentation, focal/Tversky, and skeleton decoding into the
same first run again. Those can be second-stage tests after one axis shows a clean gain.

## True Thin-Structure Losses To Add After Instrumentation

Add these only after the decoder/probability diagnostics exist:

1. Soft-clDice or clDice-inspired loss.
   - Reason: directly trains centerline/topology overlap instead of skeletonizing only after threshold.
   - Source: https://arxiv.org/abs/2003.07311

2. Boundary or distance-transform loss.
   - Reason: the positive class is tiny; contour/distance information can stabilize imbalanced
     segmentation.
   - Source: https://proceedings.mlr.press/v102/kervadec19a.html

3. Skeleton Recall style loss.
   - Reason: uses skeleton/tubed ground truth with probability-map recall and lower overhead than full
     differentiable skeleton methods.
   - Source: https://arxiv.org/abs/2404.03010

## Submit Rule

Do not submit merely because a run finished.

A candidate becomes submission-worthy only if at least one of these is true:

- fasc validation Dice beats `0.2925` without apo degradation and output distributions remain sane;
- Dice is similar, but component/fragment/geometry diagnostics clearly improve versus EXP59;
- expert-benchmark score improves using the same production measurement path and no rejected geometry
  proxy is quietly stacked in.

If none of those happen, bundle the outputs and treat EXP74 as a negative result.

## What Success Would Mean

If a controlled target/decoder change improves both fasc diagnostics and downstream measurements, then
scale it:

1. longer training,
2. stronger architecture,
3. higher resolution,
4. true topology/boundary loss,
5. public candidate.

If controlled segmentation still does not move, the next durable work is not more blind geometry. It is
either stronger labels/validation on the target distribution or a better independent scale-span detector.
