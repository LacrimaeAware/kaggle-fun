# UMUD Muscle Architecture

Current public best: `0.58910`.

Start here:

- `docs/CURRENT_STATE_2026-06-13.md` - current verdict and next agenda.
- `docs/DOC_INDEX.md` - map of which docs are current versus historical.
- `EXPERIMENT_LOG.md` - chronological record of public submissions and experiment status.
- `FEATURE_DATABASE.md` / `FEATURE_DATABASE.csv` - feature ledger with benchmark and public deltas.

## Current Work

The active direction is measurement-quality work in two tracks. The latest scale/geometry proxy
submissions regressed publicly, so the next useful work is not stacking more downstream patches. It is:

1. a best-effort heavy segmentation candidate run;
2. controlled segmentation diagnostics and one-axis GPU ablations;
3. a local classical fascicle-line extractor based on external muscle-ultrasound pipelines.

Kaggle notebooks:

- `kaggle_seg59_02_highres_512_unet_auto.ipynb` - no-edit first serious high-resolution run.
- `kaggle_seg59_sleep_matrix_auto.ipynb` - unattended matrix run for several segmentation candidates;
  writes per-run logs/status/summary files and can skip completed runs when rerun.
- `kaggle_seg72_thin_structure_heavy_auto.ipynb` - heavy overnight thin-structure run; changes the
  fascicle target/decoding formulation with soft/dilated targets, threshold sweep, and skeleton-style
  postprocessing. Hold it for now unless deliberately collecting artifacts from the confounded run.
- `kaggle_seg76_controlled_diagnostics_auto.ipynb` - controlled diagnostic matrix; useful if we want
  one-axis evidence after the best-effort run.
- `kaggle_seg77_best_effort_heavy_auto.ipynb` - current recommended overnight run; strongest
  implemented segmentation candidate first, then serious alternates if wall time remains.

Current segmentation docs:

- `EXP73_SEGMENTATION_METHOD_AUDIT_2026-06-13.md` - deeper audit of EXP72 and the current pipeline.
- `EXP74_CONTROLLED_SEGMENTATION_ABLATION_PLAN_2026-06-13.md` - next notebook design: instrumentation
  and one-axis ablations before another long GPU run.
- `EXP75_EXTERNAL_ULTRASOUND_AND_KAGGLE_METHOD_REVIEW_2026-06-14.md` - external research synthesis:
  muscle-ultrasound line extraction, Kaggle segmentation practice, masked pretraining, pseudo-label
  discipline, and scale auxiliary modeling.
- `EXP76_TONIGHT_NOTEBOOK_AUDIT_2026-06-14.md` - controlled diagnostic notebook rationale; secondary
  after EXP77 if the goal is immediate best candidate first.
- `EXP77_BEST_EFFORT_SEGMENTATION_NOTEBOOK_2026-06-14.md` - rationale for the current best-effort
  overnight notebook.

Current recommended Kaggle notebook:

- `kaggle_seg77_best_effort_heavy_auto.ipynb` - best-effort heavy segmentation run. Main candidate:
  `seg77_01_best_unetpp640_dilate_soft5_cldice`.

Kaggle setup: import the notebook, attach the UMUD competition input, set GPU + Internet on, and run
all cells.

## Current Public Submission Read

- Current best: `submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv` / `submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv`, both `0.58910`.
- Rejected: robust triangle (#15), visibility-weighted FL (#16), vertical MT (#17), broad field-depth scale (#22), and local-benchmark proxy stack (#28).

## Repo Hygiene

Do not commit `data/`, `results/`, trained weights, target human labels, OCR token caches, or generated
review artifacts unless they have been deliberately sanitized.
