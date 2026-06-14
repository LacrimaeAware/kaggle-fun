# UMUD Documentation Freshness Audit - 2026-06-13

## Scope

Audited the public-facing repository docs and UMUD root notes after public burns #22 and #28.

## Fresh Current Sources

- `docs/CURRENT_STATE_2026-06-13.md`
- `docs/DOC_INDEX.md`
- `EXPERIMENT_LOG.md`
- `FEATURE_DATABASE.md`
- `FEATURE_DATABASE.csv`
- `EXP59_SEGMENTATION_GPU_MATRIX_2026-06-13.md`
- `EXP71_LOCAL_BENCHMARK_PROXY_SCALE_STACK_2026-06-13.md`
- `EXP73_SEGMENTATION_METHOD_AUDIT_2026-06-13.md`
- `EXP74_CONTROLLED_SEGMENTATION_ABLATION_PLAN_2026-06-13.md`

## Fixed During This Pass

- Added a current-state doc that reflects the true current best (`0.58910`) and rejects #22/#28.
- Added a documentation index that separates current orientation from chronological journals.
- Updated the root README so UMUD is marked active and points to the current-state doc.
- Added a UMUD README front door.
- Marked `MASTER_REVIEW.md`, `handoff_brief.md`, and `writeup.md` as historical/superseded.
- Updated `FEATURE_DATABASE.md` and `FEATURE_DATABASE.csv` so burn #28 is public-tested/rejected.
- Re-audited the segmentation docs after EXP72 underperformed: EXP72 is now held, EXP73 is the deeper
  code/notebook/literature audit, and EXP74 is the next controlled-ablation plan.

## Known Stale Or Historical Docs

These are kept for chronology and should not be read as current recommendations:

- `MASTER_REVIEW.md`
- `handoff_brief.md`
- `writeup.md`
- `strategy_brief.md`
- older `SUBMISSION_*.md`
- older `EXP*.md` notes whose candidates have since been public-tested

## Organization Decision

I did not bulk-move the many root-level `EXP*.md` and `SUBMISSION_*.md` files into subfolders during
this pass. They are heavily cross-referenced by scripts, docs, and prior review notes. Moving them
would create link churn without improving the immediate resume state. The safer structure is:

- current docs in `docs/`;
- chronological experiment journals kept at root with dated filenames;
- generated/private artifacts kept in ignored `results/` and `data/`.

## Privacy Check

Ran a tracked-file scan for obvious secrets, private keys, API-key terms, hardcoded Windows user
paths, and committed data/results artifacts. Findings:

- no committed `data/` or `results/` artifacts found;
- no obvious secrets or private keys found;
- no hardcoded local username paths found in tracked public docs/code;
- benign hits remain for words like "secret" and "token" in explanatory text/code.

Keep OCR token caches, target human notes, trained weights, and generated review outputs ignored.

## Recurring Automation

Created Codex automation:

`kaggle-fun-weekly-docs-freshness-audit`

Schedule: weekly Monday morning. Scope: whole `kaggle-fun` repository, with special attention to
UMUD current-state docs, experiment logs, feature database rows, notebook paths, and public-facing
privacy checks.

## Next Freshness Tasks

1. After the Kaggle sleep matrix finishes, add the resulting public scores to the experiment log and
   feature database.
2. Build the EXP74 controlled segmentation notebook before another long GPU run.
3. If a segmentation candidate improves, promote it from "next direction" to "accepted branch" and
   write the exact submission file path.
4. If segmentation fails, write a new dated diagnostic note before changing strategy again.
