# EXP77 Best-Effort Segmentation Notebook

Date: 2026-06-14

Status: generated; recommended overnight Kaggle run.

Notebook: `kaggle_seg77_best_effort_heavy_auto.ipynb`

## Straight Answer

EXP77 is the current answer to "make the best segmentation candidate first." EXP76 remains useful as
a controlled diagnostic notebook, but EXP77 is the stronger overnight run because the main candidate
uses the best currently implemented segmentation ideas immediately instead of treating them only as
separate one-axis probes.

The first run is the main bet:

`seg77_01_best_unetpp640_dilate_soft5_cldice`

If only one run finishes, inspect and consider the CSV from that run first.

## What It Actually Implements

The main candidate uses:

- U-Net++ with a ResNet34 ImageNet encoder;
- 640 px training resolution with 768 px extra TTA;
- soft/dilated fascicle targets;
- Tversky plus a CLDice-style topology loss;
- validation threshold sweep;
- public-positive measurement settings: scale router, TTA, temporal smoothing, fragment FL, FL
  recenter, original line boundary, and center-perpendicular MT;
- debug mask export, run configs, logs, status JSON, summary CSV, calibration debug CSVs, weights,
  and final output zip.

The remaining runs are serious alternates, not intentionally weak controls:

| run id | purpose |
|---|---|
| `seg77_01_best_unetpp640_dilate_soft5_cldice` | main best-effort candidate |
| `seg77_02_unetpp640_soft5_bce_tversky` | same family without topology loss, stronger augmentation |
| `seg77_03_unet768_soft5_cldice` | higher-resolution U-Net alternate |
| `seg77_04_unetpp640_clahe_dilate_soft5_cldice` | CLAHE input alternate |

## Why These Choices

Evidence for this direction:

- DL_Track supports learned boundary/fragment segmentation followed by explicit geometry
  measurement. Source: https://arxiv.org/abs/2009.04790
- A muscle-ultrasound extraction pipeline supports CLAHE, skeletonized/line-like fascicle extraction,
  dominant-orientation cleanup, and non-crossing logic. Source:
  https://www.mdpi.com/1424-8220/22/14/5230
- Kaggle segmentation practice supports threshold sweeps, TTA, architecture/loss variations, and
  saved diagnostics rather than one blind CSV. Source:
  https://dev.to/jakubczakon/image-segmentation-tips-and-tricks-from-39-kaggle-competitions-l97

Antagonistic evidence:

- Ultrasound augmentation can hurt when it creates implausible image geometry or artifacts. Source:
  https://arxiv.org/html/2501.13193v1
- Pseudo-labeling can damage medical segmentation if one model's full foreground output is treated as
  truth. Source: https://arxiv.org/abs/2304.07519
- EXP72 already showed that simply making a run "heavier" can underperform if the formulation is
  confounded or not actually topology-aware.

## Honest Limits

EXP77 is not the theoretical maximum. It does not yet implement:

- masked in-domain ultrasound pretraining;
- K-fold/OOF segmentation training;
- neural scale/text/ruler/field detectors;
- a full classical ridge/skeleton/line-extractor harness;
- model-mask ensembling inside one final submission.

Those require more code. EXP77 is the best current overnight notebook using implemented pipeline
capabilities plus the newly added CLDice-style loss.

## Run Instructions

1. Import `kaggle_seg77_best_effort_heavy_auto.ipynb` into Kaggle.
2. Attach the UMUD competition input.
3. Turn GPU and Internet on.
4. Run all cells.
5. Download `umud_seg77_best_effort_outputs.zip`.

If the notebook is interrupted, rerun it. It writes per-run status and skips completed runs.

## How To Decide What To Submit

Prefer a candidate only if:

- the run completed and produced `submission_<run_id>.csv`;
- `seg77_best_effort_summary.csv` shows non-collapsed outputs;
- `pred_debug_*` masks visibly improve thin fragments without flooding noise;
- calibration/debug counts are not unexpectedly damaged.

If the first run completes cleanly, it is the intended first candidate:

`submission_seg77_01_best_unetpp640_dilate_soft5_cldice.csv`

If all EXP77 outputs look flat or worse, the next project step is not another longer copy of EXP77.
The next step should be either masked pretraining implementation or the classical ultrasound
line-extraction harness from EXP75.
