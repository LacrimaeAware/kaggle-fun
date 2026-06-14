# UMUD Documentation Index

Use this map instead of opening random root-level notes.

## Current Orientation

- `docs/CURRENT_STATE_2026-06-13.md` - freshest operational state and next agenda.
- `README.md` - short project front door.
- `EXPERIMENT_LOG.md` - chronological public submissions and experiment status.
- `FEATURE_DATABASE.md` / `FEATURE_DATABASE.csv` - idea ledger with benchmark/public deltas.

## Active Operations

- `EXP59_SEGMENTATION_GPU_MATRIX_2026-06-13.md` - current GPU segmentation plan.
- `EXP72_THIN_STRUCTURE_SEGMENTATION_2026-06-13.md` - heavy thin-structure follow-up; now held after
  `seg72_01` underperformed and the method audit found the matrix too confounded.
- `EXP73_SEGMENTATION_METHOD_AUDIT_2026-06-13.md` - methodology audit after EXP72 underperformed;
  current segmentation decision document.
- `EXP74_CONTROLLED_SEGMENTATION_ABLATION_PLAN_2026-06-13.md` - next notebook design: instrumentation
  plus one-axis thin-line ablations before another long GPU run.
- `EXP75_EXTERNAL_ULTRASOUND_AND_KAGGLE_METHOD_REVIEW_2026-06-14.md` - missing external-method review:
  muscle-ultrasound extraction pipelines, Kaggle segmentation practice, pseudo-label discipline,
  masked pretraining, and the next classical line-extractor harness.
- `EXP76_TONIGHT_NOTEBOOK_AUDIT_2026-06-14.md` - audit and rationale for the secondary controlled
  diagnostic Kaggle notebook.
- `EXP77_BEST_EFFORT_SEGMENTATION_NOTEBOOK_2026-06-14.md` - current recommended overnight notebook:
  best implemented segmentation candidate first, then serious alternates if wall time remains.
- `EXP78_SCALE_REVIEW_AND_RECALL_SEGMENTATION_STATE_2026-06-14.md` - current synthesis of the full
  309-row depth review, scale boundary, EXP72 downloaded partial run, and recall-heavy segmentation
  follow-up.
- `kaggle_seg59_02_highres_512_unet_auto.ipynb` - no-edit single serious run.
- `kaggle_seg59_sleep_matrix_auto.ipynb` - unattended multi-run segmentation matrix.
- `kaggle_seg72_thin_structure_heavy_auto.ipynb` - no-edit heavy EXP72 Kaggle notebook; hold unless
  deliberately reproducing the rejected/confounded run for artifacts.
- `kaggle_seg76_controlled_diagnostics_auto.ipynb` - secondary controlled diagnostic run with logs,
  status, summaries, submissions, weights, and debug masks.
- `kaggle_seg77_best_effort_heavy_auto.ipynb` - current recommended overnight run. Main candidate:
  `seg77_01_best_unetpp640_dilate_soft5_cldice`.
- `experiments/README.md` - older experiment narrative; useful history but not current state.

## Chronological Experiment Notes

The dated files `EXP38_...` through `EXP78_...` are append-only experiment journals. They should keep
their original conclusions, then be superseded by later dated notes rather than rewritten into a
single story.

## Submission Notes

The `SUBMISSION_*.md` files are historical submission planning notes. They are not necessarily
current recommendations. Always check `EXPERIMENT_LOG.md` and `docs/CURRENT_STATE_2026-06-13.md`
before submitting anything named in those files.

## Collaboration / Audit Notes

- `CLAUDE_AUDIT.md`, `codex_review.md`, and similar files are model-specific review artifacts.
- `benchmark_lab/` contains viewers, label tooling, and domain-neutral prompts.
- `benchmark_lab/README.md` is the entrypoint for local review servers.

## Historical / Superseded Front Doors

- `MASTER_REVIEW.md` - valuable long-form synthesis, but parts are superseded after burn #15-#28.
- `handoff_brief.md` - older collaborator brief; now points to the current-state doc.
- `writeup.md` - public-style early writeup; kept as historical narrative.

## Maintenance Rule

When a public score lands, update these in one commit:

1. `EXPERIMENT_LOG.md`
2. `FEATURE_DATABASE.md`
3. `FEATURE_DATABASE.csv`
4. the relevant `EXP*.md` note
5. `docs/CURRENT_STATE_2026-06-13.md` if the project direction changes
