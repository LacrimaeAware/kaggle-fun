# tools/ map

91 Python scripts plus per-run scratch. This file classifies them so a new contributor can
tell a stable entry point from a one-off probe. Source: the 2026-06-21 project audit
(`dropoff/inbox/2026-06-21-project-audit-and-refactor-plan.md`), verified against the files.

Run anything from the project root with the repo venv:

    PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe tools/<script>.py

(64 of 91 scripts carry their own `sys.path.insert(... "agent")` bootstrap because `agent/`
is not yet an installable package. A future `pyproject.toml` removes that boilerplate.)

## Two A/B harness lineages (know which you are in)

There are two unconnected harnesses that run cabt games. Pick the first for new work.

- `agent/cabt_arena.py` -- the DESIGNED harness: `AGENTS` registry, `winner_of()`,
  seat-swapped `run()`. Used by `run_ab.py` (the clean CLI), `deck_off_run.py`,
  `deck_robust.py`, `hoard_ab.py`, `panel_run.py`, `screen_tactics.py`, `search_sprint.py`,
  `measure_on_policy_shift.py`, `freeze_baseline_v2.py`.
- `tools/ab_candidate_v1.py` -- a 154-line script that reimplements the harness and has
  become a de-facto shared library `import`ed by 10 others: `ab_ablate_v1`, `ab_compute_v1`,
  `ab_control_v1`, `ab_depth_v1`, `ab_factorial_v1`, `ab_heuristic_search_v2`,
  `ab_v3_ablate_v1`, `ab_vs_first_v1`, `decision_hist_v1`, `trace_game_v1`.

Known duplication to consolidate (audit O2/O3): `def wilson` appears in 8 files and is NOT
identical (6 variants; `screen_tactics.py` and `freeze_baseline_v2.py` round their output, the
others return raw floats, so numbers can disagree). `winner_of` / seat-swap loop / `pilot_deck`
are duplicated 4-10x; the DENPA92 deck literal is hardcoded in 15 files. Reproducibility gap:
the cabt engine exposes no seed and the harnesses do not pair arms on common random numbers,
so small-n win-rate deltas are not reproducible (audit M6). Consolidation is deferred until the
concurrent v3 work settles, to avoid colliding with active edits.

## Stable entry points (keep, documented CLIs)

- `fetch_data.py` -- downloads the reference card DB + competitive decklists into `data/`.
- `build_card_stats.py`, `build_card_features.py`, `build_card_effects.py`, `build_stats.py`,
  `build_decks.py`, `build_replay_db.py` -- single-purpose generators that read official data
  and write the JSON the agent bundles. Re-run when the source data changes.
- `build_golden_fixtures.py` -- regenerates `tests/golden_state_action_fixtures/fixtures.json`
  (the frozen states the test suite and the new fixed-state heuristic tests run on).
- `build_viewer.py` -- builds `viewer.html`.
- `run_ab.py` -- the canonical A/B CLI (thin wrapper over `cabt_arena`, live progress +
  Wilson CIs). Use this for new A/B runs.
- `build_submission.sh` -- bundles the shipping agent. Copies only `search.py`, `eval.py`,
  `value_model.py`, `features.py` + card JSON + `cg/`; the shipped agent is `agent_search`.

## Experiment / workstream scripts (probe -> result, archive when recorded)

These were written to answer one question. Once the answer is a registry result
(`registry/results.jsonl`) or a committed `docs/workstreams/*` summary, the script is an
archive candidate. Verify the canonical one per family before deleting; git history preserves
the rest.

- A/B probes (`ab_*`, 9): `ab_candidate_v1` (the shared base, keep), `ab_ablate_v1`,
  `ab_compute_v1`, `ab_control_v1`, `ab_depth_v1`, `ab_factorial_v1`, `ab_heuristic_search_v2`,
  `ab_v3_ablate_v1`, `ab_vs_first_v1`. Successive search/heuristic probes; most answers live in
  `docs/workstreams/*results.json`.
- Continuous-terrain pipeline (8 scripts, one workstream): `mine_terrain_v1` -> `build_terrain_v1`
  -> `label_terrain_v1` -> `refeaturize_terrain_v1` -> `train_continuous_terrain_v1` ->
  `evaluate_continuous_terrain_v1` -> `finalize_terrain_v1`, plus `validate_action_semantics_v1`.
  NOTE: `train_continuous_terrain_v1_clean_rerun.py` supersedes `train_continuous_terrain_v1.py`
  (it re-runs the gate after the Search Metadata Dominance Audit); confirm which backs the
  committed terrain results before removing either.
- Teacher / ranker training (`train_*`): `train_value`, `train_action_ranker`,
  `train_contextual_action_ranker`, `train_residual_risk_contextual`,
  `train_risk_only_contextual`, `train_dagger_round1`. None of these models ship today (the
  submission is pure search); they back the learned-policy research line.
- Labeling (`label_*`, 5) and mining (`mine_c1`, `tactic_miner`) -- dataset construction for the
  ranker work.
- Audits (`audit_*`, 7) and diagnostics (`diag_*`, 3): one-shot methodology/representation checks
  (e.g. `audit_search_metadata_dominance`, `audit_representation_ceiling`,
  `audit_teacher_stability`). Their conclusions belong in the registry; the script is the record
  of how it was measured.
- Ship verification (`verify_*`, 3): `verify_submission` (validity), `verify_heuristics_v1`,
  `verify_ship_v1`. NOTE (audit M7): `verify_ship_v1.py` asserts N_DETERM==32 and budget>=1.0,
  which production (N=8 / 0.6s) does NOT satisfy; it tests presence of a code change, not its
  win-rate, and currently fails against the live source by design.

## Scratch (gitignored, regenerable; see ../.gitignore)

`_candv1/`, `_v3_control_*/`, `tools/_v3ab_*.json`, `tools/_ab_candidate_v1.json` are per-run
build packages and result dumps. `tools/_v3_src/` is the opposite: it is TRACKED staged source
that `ab_v3_ablate_v1.py` reads from; do not delete or ignore it.
