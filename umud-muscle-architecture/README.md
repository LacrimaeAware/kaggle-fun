# UMUD Muscle Architecture

Current public best: `0.58910`.

Start here:

- `docs/CURRENT_STATE_2026-06-13.md` - current verdict and next agenda.
- `docs/DOC_INDEX.md` - map of which docs are current versus historical.
- `EXPERIMENT_LOG.md` - chronological record of public submissions and experiment status.
- `FEATURE_DATABASE.md` / `FEATURE_DATABASE.csv` - feature ledger with benchmark and public deltas.

## Current Work

The active direction is EXP59 segmentation retraining. The latest scale/geometry proxy submissions
regressed publicly, so the next useful work is improving the supervised mask models rather than
stacking more downstream geometry patches.

Kaggle notebooks:

- `kaggle_seg59_02_highres_512_unet_auto.ipynb` - no-edit first serious high-resolution run.
- `kaggle_seg59_sleep_matrix_auto.ipynb` - unattended matrix run for several segmentation candidates;
  writes per-run logs/status/summary files and can skip completed runs when rerun.
- `kaggle_seg72_thin_structure_heavy_auto.ipynb` - heavy overnight thin-structure run; changes the
  fascicle target/decoding formulation with soft/dilated targets, threshold sweep, and skeleton-style
  postprocessing.

Kaggle setup: import the notebook, attach the UMUD competition input, set GPU + Internet on, and run
all cells.

## Current Public Submission Read

- Current best: `submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv` / `submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv`, both `0.58910`.
- Rejected: robust triangle (#15), visibility-weighted FL (#16), vertical MT (#17), broad field-depth scale (#22), and local-benchmark proxy stack (#28).

## Repo Hygiene

Do not commit `data/`, `results/`, trained weights, target human labels, OCR token caches, or generated
review artifacts unless they have been deliberately sanitized.
