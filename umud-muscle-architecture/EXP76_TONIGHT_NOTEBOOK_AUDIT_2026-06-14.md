# EXP76 Tonight Notebook Audit

Date: 2026-06-14

Status: notebook generated.

Notebook: `kaggle_seg76_controlled_diagnostics_auto.ipynb`

## Why This Notebook

The external audit says the next overnight run should not be another all-knobs-at-once heavy model.
The current code can run controlled supervised segmentation variants, threshold sweeps, target-shape
variants, postprocessing variants, TTA, temporal smoothing, and debug mask export. It cannot yet run
masked self-supervised pretraining, true clDice, skeleton-recall loss, OOF folds, or the classical
CLAHE/ridge/skeleton/line-extraction harness without new implementation work.

Therefore EXP76 is a conservative high-information notebook:

- keep the public-positive measurement side fixed where possible;
- avoid rejected geometry proxies such as robust triangle, broad visibility-FL, vertical-MT proxy, and
  broad field-depth scale;
- change one segmentation axis at a time;
- save logs, configs, submissions, calibration debug CSVs, debug masks, and a summary zip.

## Source Audit: Evidence For

1. DL_Track shows a learned-segmentation approach can detect aponeuroses and multiple fascicle
   fragments from B-mode ultrasound, then measure PA/FL/MT from those detections.
   Source: https://arxiv.org/abs/2009.04790

2. A muscle-ultrasound extraction pipeline with Attention U-Net uses CLAHE, ridge-like filtering,
   skeletonized line segments, dominant-orientation clustering, line extension, and non-crossing
   cleanup. This supports a future classical line-extractor harness.
   Source: https://www.mdpi.com/1424-8220/22/14/5230

3. Kaggle segmentation practice commonly uses threshold tuning, TTA, folds, probability handling,
   class-aware sampling, and careful postprocessing rather than a single naked training run.
   Source: https://dev.to/jakubczakon/image-segmentation-tips-and-tricks-from-39-kaggle-competitions-l97

4. Masked ultrasound pretraining has reported large gains under low-label ultrasound conditions, so
   it is a real future candidate.
   Source: https://www.nature.com/articles/s41598-025-11688-2

## Source Audit: Antagonistic Evidence

1. Ultrasound augmentation is not "more is always better." A broad review found useful augmentation
   effects, but also task sensitivity and declines after adding too many transformations.
   Source: https://arxiv.org/html/2501.13193v1

2. UltraAugment reports that standard cropping, elastic deformation, and additive noise can degrade
   ultrasound segmentation when they produce implausible ultrasound geometry or noise.
   Source: https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/papers/Ramakers_UltraAugment_Fan-shape_and_Artifact-based_Data_Augmentation_for_2D_Ultrasound_Images_CVPRW_2024_paper.pdf

3. Pseudo-labeling can damage medical segmentation if a single model's foreground errors are treated
   as truth. A better approach compares multiple confidence maps and keeps high-confidence labels.
   Source: https://arxiv.org/abs/2304.07519

4. Ultrasound architecture measurement has real validity limits. A systematic review says reliability
   is generally good, but validity evidence is limited and depends on conditions such as relaxed,
   stationary large muscles and probe alignment.
   Source: https://journals.physiology.org/doi/full/10.1152/japplphysiol.01430.2011

## Tonight Run Matrix

All runs use current public-safe measurement settings:

- scale router on;
- TTA on;
- temporal smoothing on;
- fragment FL on;
- FL recenter on;
- top boundary `line`;
- MT mode `perp_center`;
- no robust-triangle / broad visibility-FL / vertical-MT proxy / broad scale override.

Runs:

| run id | one-axis question |
|---|---|
| `seg76_00_control_binary_thr` | 512 U-Net control with auto threshold and debug masks |
| `seg76_01_soft5_thr` | does a soft fascicle target help without skeleton postprocess? |
| `seg76_02_dilate3_thr` | does a slightly thicker fascicle target help usable fragments? |
| `seg76_03_tversky_binary_thr` | does Tversky help sparse fascicle recall without target-shape changes? |
| `seg76_04_unetpp_binary_thr` | does U-Net++ help at the same target/loss? |
| `seg76_05_soft5_tversky_thr` | optional combo if there is wall time: soft target plus Tversky |

Default `MAX_RUNS` is 6 with a wall-time guard. If Kaggle time is scarce, set `MAX_RUNS = 3` inside
the run cell.

## How To Read Results

Do not submit a CSV only because it exists. Prefer candidates where:

- fasc validation Dice improves over `seg76_00`;
- prediction distributions are sane;
- calibration counts are not unexpectedly damaged;
- accepted fragment count does not collapse;
- debug masks visibly improve thin fragments without flooding noise.

If all variants are flat or worse, that is information: the next implementation should be EXP75's
classical line extractor or masked pretraining code, not another longer supervised run.
