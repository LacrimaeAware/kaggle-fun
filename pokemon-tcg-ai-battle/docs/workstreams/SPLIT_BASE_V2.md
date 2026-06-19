# SPLIT_BASE_V2 — the frozen shared preflight

**Status:** built and tested on `main`. All golden acceptance tests pass (8/8). This commit is the
single base both research branches fork from. Do not start Branch A's A2 or any Branch B training until
this is reviewed.

This is the shared, frozen contract from `docs/workstreams/BRANCH_PLAN.md` (v2) steps P0–P5. It is
deliberately minimal: a frozen baseline, an immutable replay snapshot + splits, the semantic state/action
schema, Teacher API V1, and golden fixtures + tests. No research, no training, no planner redesign.

## Scope contract (authorized)

> Implement SPLIT_BASE_V2 only. Do NOT begin Branch A experiments, run teacher-strength sweeps, change
> the production agent or deck, train a model, edit Branch B files, update broad hypothesis conclusions,
> or call action ranking "refuted" / covariate shift "proven". Use the existing engine/search code; do
> not redesign the planner. >=100 golden decisions covering all major action types; all tests pass; then
> commit and stop (do not start A2).

Authorized deviation (by the user): build directly on `main` rather than `exp/split-base-v2`, because the
worktrees were removed and everything was consolidated onto `main`. The stray `exp/split-base-v2` and
`codex/robust-learner-v2-preflight` refs are left untouched.

## What did NOT change

The production agent is untouched: `agent/main.py`, `agent/search.py`, `agent/features.py`,
`agent/eval.py`, `agent/cabt_arena.py` are unmodified. Only NEW files were added, plus one scoped
`.gitignore` exception so the manifests/splits are tracked while raw replays stay ignored.

## P0 — frozen production baseline → `data/manifests/baseline_v2_20260618.json`

`agent_search` on the DENPA92 deck (signature `27ed2ff887c1488c`), `N_DETERM=8`, per-decision budget
`0.6s`, `DEPTH_CAP=80`, continuation `aggro`, leaf `hand`, 1-ply (`opp_k=0`, `opp_prior=None`), with the
`_forced_move` floor (lethal/KO ≥8000 else go-first) and the heuristic fallback. Base commit `c7b98b2`.

Results: anchored (with provenance) — `search vs first_agent` 0.585 (n=800), `search vs heuristic` 0.833
(n=60, DENPA92). Fresh preflight measurement at this config — `search vs first` **0.850** [0.64, 0.948]
(17-3, 0 errors, n=20, 6.49 s/game). The fresh number matches the DENPA92/N=8 vs-heuristic anchor; the
older 0.585 vs-first anchor predates this deck/config. Regenerate / definitively reproduce (A1):
`python tools/freeze_baseline_v2.py --stamp <id> --measure 400 --also-heuristic`.

## P1 — immutable replay snapshot + splits → `data/manifests/replays_20260618.json`, `data/splits/replays_20260618_split.json`

1289 games included / 10 skipped of 1299 (`corpus_sha256` recorded; per-file sha256 + player/deck/result
+ skip reasons in the manifest). Chronological split by `info.EpisodeId` (a proxy — replays carry no
timestamp) at the GAME level, so candidate rows from one decision never cross train/test: 902 / 193 / 194,
with 123 held-out players and 23 held-out decks for generalization tests. Regenerate:
`python tools/snapshot_replays_v2.py --stamp <id>`. The raw replays stay gitignored; only the manifest +
splits are committed.

## P2 — semantic schema → `agent/state_action_schema_v2.py`

The single source of truth for card/entity identity, the semantic equivalence key, the action descriptor,
the structural state (orderless zones as multisets), the canonical L1 encoding, and deck identity. Ported
from the fixed logic in `tools/build_action_dataset.py` and **verified byte-for-byte against it across
31,146 decisions / 198,423 options (0 mismatches)**, with 100% PLAY card-id resolution.

## P4 — Teacher API V1 → `agent/teacher_api_v1.py`

Wraps the existing search (no planner redesign). Per decision returns: per-option `semantic_action_key`,
`mean_value`, `value_variance`, `completed_determinizations`, `normalized_advantage`; and per decision
`top_two_margin`, `soft_policy_target`, `acceptable_action_set` (statistically-indistinguishable-from-best),
`forced_action_flag`, `chosen_option`, and a teacher config hash. Computed over semantic equivalence
classes, never raw index. Mirrors `search._search` (the deployed agent), confirmed exact on
deterministic-rollout decisions.

**Reproducibility finding (important for A2 and for "more determinizations ≠ stronger"):** the seed
controls the determinization DRAW (Python RNG) but NOT the native engine's rollout RNG — coin flips
(`manual_coin=False`, matching the agent) and shuffle effects resolve inside `cg.dll`, and `sim.py` exposes
no seed hook. Measured: **~93% of strategic decisions have engine rollout RNG**, so same-seed queries are
NOT bit-identical; each value is a Monte Carlo estimate and `value_variance` reports the combined noise.
A2 must average enough worlds and treat cross-seed disagreement as conflating both sources.

## P3 — golden fixtures + tests → `tests/golden_state_action_fixtures/fixtures.json`, `tests/test_split_base_v2.py`

130 real decisions covering all 8 major action types; selected for feature diversity so 45/47 features vary
(the 2 documented constants `my_asleep`/`my_paralyzed` are corpus-absent at single-pick decisions, not dead
wiring). Run: `PYTHONIOENCODING=utf-8 python tests/test_split_base_v2.py`. The eight gates (all PASS):

1. trainer encoding == live encoding == frozen encoding (byte-identical);
2. every PLAY option resolves a card identity;
3. equivalent options collapse, distinct keys never share a class;
4. distinct PLAY cards never collapse (the original bug);
5. option permutations transform the keys and preserve the partition;
6. no unexpected dead/constant feature;
7. teacher queries do not mutate the root state;
8. ≥100 fixtures covering all major action types.

## Environment notes (this machine)

- Engine: the forward model is the `cg/api.py` wrapper under
  `data/external/official/sample_submission/cg/` (gitignored), discovered by `search._api()`. The installed
  `kaggle_environments` package ships only the native `cg` (no `api.py`).
- Data: 1299 replays + the engine wrapper live in the main checkout (gitignored). No `kaggle.json`, so
  replay fetching is not possible here without auth.

## File ownership (frozen after this commit)

- **Shared, auditor-gated only:** `agent/state_action_schema_v2.py`, `agent/teacher_api_v1.py`,
  `tests/golden_state_action_fixtures/`, `data/manifests/`, `data/splits/`, this doc. Changes require
  auditor approval and a cherry-pick into both branches.
- **Branch A owns:** `agent/teacher_api_v2.py`, `agent/search_live_v2.py`, `tools/query_teacher_v2.py`,
  `tools/audit_teacher_stability.py`, `tools/evaluate_teacher_regret.py`, `docs/workstreams/PLANNER_TEACHER_V2.md`.
- **Branch B owns:** `agent/student_v2.py`, `agent/student_prior_v2.py`, `agent/entity_encoder_v2.py`,
  `tools/audit_representation_ceiling.py`, `tools/measure_on_policy_shift.py`, `tools/dagger_collect.py`,
  `tools/dart_recovery_states.py`, `tools/train_student_v2.py`, `docs/workstreams/ROBUST_LEARNER_V2.md`.

Neither branch edits `agent/main.py` or submission packaging.

## Next step (NOT done here)

Both branches fork from this `main` commit. Branch B (`codex/robust-learner-v2-preflight`) must pause,
review this preflight, and start from this exact SHA — it must not independently edit the shared schema.
Branch A starts the A2 teacher-stability audit; Branch B starts representation-ceiling and on-policy-shift
diagnostics on the frozen corpus with Teacher V1.

## Corrections held to

No broad conclusions were written. Covariate shift is NOT asserted proven; action ranking / H024 is NOT
called refuted beyond the specific standalone+hybrid configs already tested; the registry was not edited;
no old 2-ply / global-continuation / shallow-belief experiment was rerun; A2 was not started.
