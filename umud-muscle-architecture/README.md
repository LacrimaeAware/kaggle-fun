# UMUD Muscle Architecture

Current public best: **`0.52570`** (global FL scale x1.05 on the PA+2.5 base, `results/submission_fl_x105.csv`; 2026-06-14).

Start here (these five are the living docs; everything else is in `archive/`):

- `docs/CURRENT_STATE.md` - **canonical single source of truth** (undated). Read this first.
- `VERIFIED_FACTS.md` - code/LB-grounded facts only; nothing unproven.
- `FINDINGS_REGISTRY.md` - every idea/feature/experiment by concept, each tagged LIVE / FACT / UNTESTED / REJECTED / FALSE / PAST. The merged replacement for the old front-door and EXP journals.
- `EXPERIMENT_LOG.md` - chronological public-submission changelog.
- `competition_reference.md` / `FEATURE_DATABASE.md` - host rules and the feature ledger.

> Reset (2026-06-14): the repo was consolidated. ~25 narrative docs and 41 dated EXP journals moved to
> `archive/` (history preserved); their findings are folded into `FINDINGS_REGISTRY.md` with corrected
> status. Several old claims were falsified - notably "the FL recenter is a no-op / masks an overshoot":
> it is an active ~19% FL shrink, and the leaderboard wants FL longer (FL x1.05 -> 0.52570).

## Current Work

The live lever is **FL global scale** (FL x1.05 -> 0.52570, still climbing; bracket x1.10/1.15/1.20/1.25,
then bake the optimum into `segment_then_measure.py`). PA is tapped at ~+2.4. The next non-LB work is a
classical Frangi+Radon fascicle-orientation extractor (no GPU). The GPU segmentation pivot is on hold:
its premise ("FL is mask-limited") was falsified by the FL scale win. See `docs/CURRENT_STATE.md` for the
full plan. The segmentation notebooks below remain available but EXP77 was never run.

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
- `EXP78_SCALE_REVIEW_AND_RECALL_SEGMENTATION_STATE_2026-06-14.md` - compact synthesis of the full
  309-image depth review, failed scale submissions, and the recall-heavy segmentation follow-up.

Current recommended Kaggle notebook:

- `kaggle_seg77_best_effort_heavy_auto.ipynb` - best-effort heavy segmentation run. Main candidate:
  `seg77_01_best_unetpp640_dilate_soft5_cldice`.

Kaggle setup: import the notebook, attach the UMUD competition input, set GPU + Internet on, and run
all cells.

## Current Public Submission Read

- Current best: `results/submission_fl_x105.csv` at `0.52570` (FL x1.05 on the PA+2.5 base). The
  burn_11/burn_13 files (`0.58910`) and the PA-shift files (`0.55075`/`0.55033`) are superseded.
- Rejected: robust triangle (#15), visibility-weighted FL (#16), vertical MT (#17), broad field-depth scale (#22), and local-benchmark proxy stack (#28).
- Scale status: displayed depth is audited on all 309 test images and algorithmically recovered
  309/309 after EXP63/EXP64 repairs; the unsolved part is trusted `px/cm` span detection.

## Repo Hygiene

Do not commit `data/`, `results/`, trained weights, target human labels, OCR token caches, or generated
review artifacts unless they have been deliberately sanitized.
