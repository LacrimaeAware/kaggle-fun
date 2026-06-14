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
- `EXP75_EXTERNAL_ULTRASOUND_AND_KAGGLE_METHOD_REVIEW_2026-06-14.md`
- `EXP76_TONIGHT_NOTEBOOK_AUDIT_2026-06-14.md`
- `EXP77_BEST_EFFORT_SEGMENTATION_NOTEBOOK_2026-06-14.md`
- `EXP78_SCALE_REVIEW_AND_RECALL_SEGMENTATION_STATE_2026-06-14.md`
- `kaggle_seg76_controlled_diagnostics_auto.ipynb`
- `kaggle_seg77_best_effort_heavy_auto.ipynb`

## Fixed During This Pass

- Added a current-state doc that reflects the true current best (`0.58910`) and rejects #22/#28.
- Added a documentation index that separates current orientation from chronological journals.
- Updated the root README so UMUD is marked active and points to the current-state doc.
- Added a UMUD README front door.
- Marked `MASTER_REVIEW.md`, `handoff_brief.md`, and `writeup.md` as historical/superseded.
- Updated `FEATURE_DATABASE.md` and `FEATURE_DATABASE.csv` so burn #28 is public-tested/rejected.
- Re-audited the segmentation docs after EXP72 underperformed: EXP72 is now held, EXP73 is the deeper
  code/notebook/literature audit, and EXP74 is the next controlled-ablation plan.
- Added EXP75 after a broader external-method review covering muscle-ultrasound measurement
  pipelines, Kaggle segmentation practice, masked pretraining, pseudo-labeling, and scale auxiliary
  models.
- Added EXP76 after the antagonistic source review: controlled segmentation matrix, not another
  all-knobs heavy run.
- Added EXP77 after the user clarified the immediate goal: strongest implemented segmentation
  candidate first. EXP76 is now secondary diagnostics; EXP77 is the recommended overnight notebook.
- Added EXP78 to consolidate the full 309-image manual depth review, the algorithm-only 309/309
  depth audit, the failed broad scale probes, the EXP72 downloaded-partial audit, and the
  recall-heavy segmentation follow-up idea.

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

1. After the EXP77 Kaggle matrix finishes, add the resulting public scores and diagnostics to the
   experiment log and feature database.
2. If EXP77 produces weights, generate recall-heavy inference-only variants before another full
   training run.
3. If needed, run EXP76 afterward for controlled one-axis diagnostics.
4. Build the EXP75 classical fascicle-line extractor harness so raw ultrasound texture is tested
   independently from the neural fascicle masks.
5. Build the EXP74 controlled segmentation notebook before another long GPU run.
6. If a segmentation candidate improves, promote it from "next direction" to "accepted branch" and
   write the exact submission file path.
7. If segmentation fails, write a new dated diagnostic note before changing strategy again.
