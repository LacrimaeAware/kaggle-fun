# UMUD Muscle Architecture

Current public best: **`0.46041`** (aponeurosis band fix + FL x1.05, `results/submission_bandfix_flx105.csv`).
Reproducible one-run pipeline (`local_infer.py`, median FL) = `0.47473`; the ~0.014 gap is old CSV residue.

Start here:

- `docs/HANDOFF.md` - **full self-contained state**, written to be read cold. Read this first.
- `docs/CURRENT_STATE.md` - terse canonical decision-driver (undated).
- `VERIFIED_FACTS.md` - code/LB-grounded facts only; nothing unproven.
- `FINDINGS_REGISTRY.md` - every idea/feature/experiment by concept, each tagged LIVE / FACT / UNTESTED / REJECTED / FALSE / PAST. The merged replacement for the old front-door and EXP journals.
- `EXPERIMENT_LOG.md` - chronological public-submission changelog.
- `competition_reference.md` / `FEATURE_DATABASE.md` - host rules and the feature ledger.

> Reset (2026-06-14): the repo was consolidated. ~25 narrative docs and 41 dated EXP journals moved to
> `archive/` (history preserved); their findings are folded into `FINDINGS_REGISTRY.md` with corrected
> status. Several old claims were falsified - notably "the FL recenter is a no-op / masks an overshoot":
> it is an active ~19% FL shrink, and the leaderboard wants FL longer (FL x1.05 -> 0.52570).

## Current Work

**Methodology reset (2026-06-15).** The leaderboard-multiplier loop is retired: global PA/FL shifts only
move a column mean, and the 35-image benchmark does not predict the LB (min_extrap_top3 scored 0.39 on
the benchmark and regressed the LB to 0.49983). The plan is to build a real validation loop instead:
an error decomposition (run `measure()` on expert train masks vs predicted masks for the per-term
segmentation cost), a test-distribution gate from the user's correction-UI hand-labels, and GroupKFold
by subject/device for any model change. See `docs/HANDOFF.md` (full) and `docs/CURRENT_STATE.md` (terse).
The strategic direction (route-by-class vs new segmentation target vs rebuild measurement on apo +
orientation objects) is an open question for the user. The segmentation notebooks below remain available
but EXP77 was never run.

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

Current segmentation docs (all archived under `archive/exp/`; keep them chronological rather than editing them in place):

- `archive/exp/EXP73_SEGMENTATION_METHOD_AUDIT_2026-06-13.md` - deeper audit of EXP72 and the current pipeline.
- `archive/exp/EXP74_CONTROLLED_SEGMENTATION_ABLATION_PLAN_2026-06-13.md` - next notebook design: instrumentation
  and one-axis ablations before another long GPU run.
- `archive/exp/EXP75_EXTERNAL_ULTRASOUND_AND_KAGGLE_METHOD_REVIEW_2026-06-14.md` - external research synthesis:
  muscle-ultrasound line extraction, Kaggle segmentation practice, masked pretraining, pseudo-label
  discipline, and scale auxiliary modeling.
- `archive/exp/EXP76_TONIGHT_NOTEBOOK_AUDIT_2026-06-14.md` - controlled diagnostic notebook rationale; secondary
  after EXP77 if the goal is immediate best candidate first.
- `archive/exp/EXP77_BEST_EFFORT_SEGMENTATION_NOTEBOOK_2026-06-14.md` - rationale for the current best-effort
  overnight notebook.
- `archive/exp/EXP78_SCALE_REVIEW_AND_RECALL_SEGMENTATION_STATE_2026-06-14.md` - compact synthesis of the full
  309-image depth review, failed scale submissions, and the recall-heavy segmentation follow-up.

Current recommended Kaggle notebook:

- `kaggle_seg77_best_effort_heavy_auto.ipynb` - best-effort heavy segmentation run. Main candidate:
  `seg77_01_best_unetpp640_dilate_soft5_cldice`.

Kaggle setup: import the notebook, attach the UMUD competition input, set GPU + Internet on, and run
all cells.

## Current Public Submission Read

- Current best: `results/submission_bandfix_flx105.csv` at `0.46041` (band fix + FL x1.05). The
  family_b scale fix (~0.488), FL x1.05 (0.52570), PA-shift (0.55x), and burn_11/13 (0.58910) are
  superseded steps on the ladder.
- Rejected: min_extrap_top3 FL (0.49983, and earlier #14 0.62994), robust triangle (#15 0.60102),
  visibility-weighted FL (#16 0.64511), vertical MT (#17 0.60720), broad field-depth scale (#22 0.66197),
  local-benchmark proxy stack (#28 0.65917).
- Scale status: displayed depth is audited on all 309 test images and algorithmically recovered
  309/309 after EXP63/EXP64 repairs; the unsolved part is trusted `px/cm` span detection.

## Repo Hygiene

Do not commit `data/`, `results/`, trained weights, target human labels, OCR token caches, or generated
review artifacts unless they have been deliberately sanitized.
