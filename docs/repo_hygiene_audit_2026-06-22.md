# Repo hygiene audit - 2026-06-22

Scope: documentation freshness pass plus a lightweight tracked-file hygiene scan.

## Verified this pass

- No obvious committed secrets, Kaggle API tokens, or OCR token-cache outputs were found in tracked files.
- UMUD private artifacts remain out of git where expected: no tracked `results/`, correction-UI label
  outputs, or `.venv_ocr` cache products were found via `git ls-files`.
- The active UMUD orientation docs were refreshed to the verified `0.46041` / methodology-reset state.
- Dated UMUD experiment journals were left chronological; the live state now points back to
  `umud-muscle-architecture/docs/CURRENT_STATE.md` and `umud-muscle-architecture/docs/HANDOFF.md`.

## Hygiene findings still open

- `pokemon-tcg-ai-battle/` still contains tracked binary model artifacts under `agent/*.pt`. They may be
  intentional release assets, but they are still committed trained weights and should stay a deliberate choice.
- `pokemon-tcg-ai-battle/` still contains hardcoded local `C:/Users/EcceNihilum/...` paths in some public
  docs and helper scripts (for example `docs/HANDOFF.md`, `docs/OVERVIEW.md`, and several `tools/*.py`
  utilities). Those should be normalized to repo-relative or configurable paths before broader sharing.

## Next actions

1. Decide whether the tracked Pokemon `.pt` files are intended distributable artifacts or should move to ignored storage plus documented fetch/build steps.
2. Replace hardcoded Windows-user paths in Pokemon docs/tools with repo-relative paths or CLI arguments.
3. Keep future UMUD state changes in `CURRENT_STATE.md`, `HANDOFF.md`, `EXPERIMENT_LOG.md`, and the
   feature ledger instead of reviving new front-door status docs.
