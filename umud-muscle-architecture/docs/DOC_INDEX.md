# UMUD Documentation Index

Use this map instead of opening random root-level notes.

## Current Orientation

- `docs/CURRENT_STATE_2026-06-13.md` - freshest operational state and next agenda.
- `README.md` - short project front door.
- `EXPERIMENT_LOG.md` - chronological public submissions and experiment status.
- `FEATURE_DATABASE.md` / `FEATURE_DATABASE.csv` - idea ledger with benchmark/public deltas.

## Active Operations

- `EXP59_SEGMENTATION_GPU_MATRIX_2026-06-13.md` - current GPU segmentation plan.
- `EXP72_THIN_STRUCTURE_SEGMENTATION_2026-06-13.md` - heavy thin-structure follow-up; preferred
  overnight run when trying to break the low fascicle-mask Dice wall, but see EXP73 before rerunning.
- `EXP73_SEGMENTATION_METHOD_AUDIT_2026-06-13.md` - methodology audit after EXP72 underperformed;
  current segmentation decision document.
- `kaggle_seg59_02_highres_512_unet_auto.ipynb` - no-edit single serious run.
- `kaggle_seg59_sleep_matrix_auto.ipynb` - unattended multi-run segmentation matrix.
- `kaggle_seg72_thin_structure_heavy_auto.ipynb` - no-edit heavy EXP72 Kaggle notebook.
- `experiments/README.md` - older experiment narrative; useful history but not current state.

## Chronological Experiment Notes

The dated files `EXP38_...` through `EXP71_...` are append-only experiment journals. They should keep
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
