# Repo hygiene audit - 2026-06-29

Scope: weekly documentation freshness and tracked-file hygiene pass.

## Verified this pass

- `umud-muscle-architecture/docs/CURRENT_STATE.md`, `docs/DOC_INDEX.md`, `docs/HANDOFF.md`,
  `README.md`, `FEATURE_DATABASE.md`, and `EXPERIMENT_LOG.md` still agree on the current verified
  UMUD state:
  - best public score `0.46041` (`results/submission_bandfix_flx105.csv`)
  - reproducible one-run pipeline `0.47473` (`submission_reproduced.csv`)
  - `min_extrap_top3` refuted at `0.49983`
  - methodology-reset plan remains the active direction
- No newer tracked UMUD EXP or submission notes were found after the 2026-06-14/15 state already
  reflected in the living docs, so the front-door UMUD documents did not need content changes.
- Dated UMUD journals under `umud-muscle-architecture/archive/` were left chronological and
  untouched.
- `umud-muscle-architecture/FEATURE_DATABASE.csv` remains parseable and still contains 72 rows
  through `F072`.
- The referenced UMUD Kaggle notebooks still exist in-repo:
  `kaggle_seg59_02_highres_512_unet_auto.ipynb`,
  `kaggle_seg59_sleep_matrix_auto.ipynb`,
  `kaggle_seg72_thin_structure_heavy_auto.ipynb`,
  `kaggle_seg76_controlled_diagnostics_auto.ipynb`,
  `kaggle_seg77_best_effort_heavy_auto.ipynb`.
- No obvious tracked secrets, Kaggle tokens, OCR token-cache outputs, committed UMUD `data/` /
  `results/`, or committed target human-label artifacts were found in the tracked tree.

## Hygiene findings still open

- `pokemon-tcg-ai-battle/` still ships tracked `.pt` model artifacts under `agent/`. These may be
  intentional release assets, but they remain committed trained weights and should stay a deliberate
  choice.
- `pokemon-tcg-ai-battle/` still contains many hardcoded local `C:/Users/EcceNihilum/...` paths in
  public docs, helper tools, and some tests. The public-facing examples previously noted in
  `docs/HANDOFF.md` and `docs/OVERVIEW.md` are still present, and the broader tool/test surface is
  larger than last week's note summarized.
- The worktree also contains unrelated untracked inbox notes under `pokemon-tcg-ai-battle/dropoff/`.
  They were not modified by this audit.

## Next actions

1. Keep UMUD living docs unchanged until a new verified score, notebook run, or test-distribution
   validation result exists.
2. Normalize Pokemon docs/tools away from hardcoded local paths before broader sharing.
3. Decide whether the tracked Pokemon `.pt` files are intended distributable artifacts or should move
   behind documented fetch/build steps.
