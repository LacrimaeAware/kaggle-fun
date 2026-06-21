# Project audit + organization / refactor plan (2026-06-21)

Method: a 6-reader file-level audit of `pokemon-tcg-ai-battle/` (decision pipeline,
tools sprawl, methodology + results, docs, tests/hygiene, registry machinery), then a
20-claim adversarial verification pass that re-checked each load-bearing claim against
the files. 26 agents total. Every finding below carries a file reference. Verdicts are
marked `verified` (re-checked against source), `partial` (true with a caveat the
verification found), or `refuted` (the first-pass claim was wrong; corrected here).

Scope note: this is one branch (`exp/planner-teacher-v2`) of the working tree at audit
time, with uncommitted workstream JSON present.

---

## 0. One-paragraph summary

The methodology *machinery* (the registry, the prior 2026-06-18 audits, the existing
R001-R009 result notes) is good and honest. The recent work has stopped flowing through
it, and in doing so reintroduced the exact UMUD failure mode `AGENTS.md` was written to
prevent. The headline claim of the latest commit ("verified Powerful-Hand KO fix wins
0.75 (15-5) over baseline") is an n=20 over-read that the repo's own larger-n data
contradicts, it is not in the registry, and the contradicting file was left untracked
while the favorable one was committed. Separately, the codebase has sprawled (91 tool
scripts, 9 agent entry points, ~7.3 MB of leaked scratch copies) and there is no way to
test a single heuristic without running full games, which is the direct blocker to the
"easier to test features/heuristics" goal. The good news: the seam for a clean,
testable, pluggable heuristic interface already exists in the code and just needs to be
promoted out of a dead module.

---

## 1. Major problems: methodology (highest priority)

**M1. The "verified 0.75 (15-5)" headline is an over-read contradicted by the repo's own
data. [verified]**
- Commit `9a8daf1` message: "verified Powerful-Hand KO fix wins 0.75 (15-5) over baseline."
- Same intervention (`phfix`), three measurements in the tree:
  - `docs/workstreams/heuristic_search_v2_results.json:128` -> 0.75, 15-5, n=20 (committed).
  - `docs/workstreams/v3_ablation_results.json:51` -> 0.388, 31-49, n=80 (untracked).
  - working-tree diff of the v2 file -> `phfix_s32` (the actual ship candidate: PH-fix +
    N=32) = 0.45, 9-11, n=20.
- The n=80 Wilson [0.288, 0.497] sits entirely below 0.5 and does not overlap the n=20
  [0.531, 0.888]. Caveat the verification added: the two runs are not apples-to-apples
  (different code path `agent_search_phfix` vs `forced_move_ko_phaware`, different n), so
  the honest read is not "0.75 vs 0.388 on the same thing" but "PH-aware auto-KO is at or
  below parity on this deck, measured noisily, and the favorable n=20 cell was the one
  committed." The word "verified" is not supportable.
- Action: retract the 0.75 framing; treat PH-fix as at-or-below parity pending a seeded,
  paired, n>=200 rerun at a realistic budget against a named baseline (production
  `agent_search`).

**M2. The registry has gone write-only and stale; the anti-rehash contract is unenforced.
[verified]**
- `registry/results.jsonl` ends at R009 (2026-06-18); `experiments.jsonl` at E015;
  `hypotheses.jsonl` at H024. ~13 commits of experiments since then dumped loose JSON
  into `docs/workstreams/*results.json` and `tools/_*.json` and never entered the ledger.
- No agent or experiment-running code imports `registry.py` (project-wide grep matches
  only `registry/seed.py:11`; the lone reference in `agent/` is the comment
  `agent/main.py:18` "See registry H023"). The anti-rehash search gate
  (`registry.py search`) has no hook/CI/pre-commit binding it; compliance is voluntary.
- BELIEFS.md/GRAVEYARD.md DO regenerate byte-identical from the JSONL (verified by render
  + diff), and the JSONL passes referential integrity. So the design is sound; the
  failure is non-enforcement and non-use, exactly the "enactment, not knowledge" failure
  `AGENTS.md:1-13` names.
- Action: backfill PH-fix / N=32 / v3-ablation as H/E/R rows with n + baseline + verdict;
  add a `registry.py check` integrity subcommand; either wire the harnesses to append
  result rows automatically, or downgrade the AGENTS.md "source of truth that wins"
  language to "hand-curated index" so the contract matches reality.

**M3. Factorial "main effects" and "interaction" are computed from n=2 per cell.
[verified]**
- `docs/workstreams/factorial_v1_results.json`: every cell n=2; `effects.interaction =
  -0.5` etc., computed at `tools/ab_factorial_v1.py:146-157` as point-estimate net
  win-rate subtractions. One game flips any "effect" by 0.5. This is the win-rate
  aggregation fallacy `AGENTS.md:41-53` forbids, and the harness's own docstring concedes
  a +/-0.20 swing is noise. Mark this file a smoke artifact, not a result.

**M4. Search is evaluated under a self-imposed 0.6s budget; real budget is ~600s/game.
[verified]**
- `agent/search.py:35` `DEFAULT_BUDGET = 0.6`, `:36 N_DETERM = 8`. Every N=8 A/B caps
  search at 0.6s/decision, ~3 orders of magnitude below deployment; N=32 arms raise it
  only to 15s. Conclusions like "N=32 did not help" are drawn under compute starvation
  and cannot rule out that more search helps at the real budget. State the budget in
  every result headline; re-run the depth/sampling questions at a deployment-representative
  per-decision budget before concluding.

**M5. The deployed "search" is opp_k=0 1-ply rollout, not 2-ply adversarial search.
[verified]**
- `agent/main.py:236` `agent_search -> _agent_search(obs,"hand")` with default `opp_k=0`
  (`main.py:181`); `search.py:249` then uses `_simulate` (1-ply, plays my turn + one
  default-policy opponent reply, evaluates start of my next turn). The `opp_k>0` minimax
  branch (`_simulate2`, exposed as `agent_search2`) exists but is not deployed. The
  belief-conditioning null (R008) is the expected consequence of an opponent-blind leaf,
  not evidence belief is useless. Stop describing the live agent as multi-ply; if you test
  depth, actually enable `opp_k>0` and measure it.

**M6. A/B runs are unseeded and unpaired. [verified]**
- The cabt engine has zero seed plumbing (`kaggle_environments/envs/cabt/cabt.py`: 0
  occurrences of "seed"). `tools/ab_candidate_v1.py:93` `make("cabt")` takes no seed and
  is the shared `run()` behind the v2/factorial/v3 harnesses. Arms see different games AND
  different determinizations, so a 0.05-0.10 edge drowns in independent Wilson CIs and no
  run is replayable. The engine RNG is genuinely unseedable; the correctable gap is
  pairing (common random numbers between arms) + pinning the controllable knobs, plus
  writing a `{n, seat_rule, git_sha, engine_seedable:false}` block into each result so a
  run is at least re-describable.

**M7. `verify_ship_v1.py` asserts a deployment that production does not contain.
[verified]**
- `tools/verify_ship_v1.py:59,62` check `N_DETERM == 32` and `DEFAULT_BUDGET >= 1.0`;
  production is N=8 / 0.6 and `agent/main._attack_value` has no PH-awareness (PH logic
  lives only in the non-shipping `deck_policy_v2.py`). So the "ship" was never deployed.
  The tool tests presence of a code change, not its win-rate effect; do not cite it as
  evidence the agent improved. Record in CURRENT.md that production is still N=8/0.6s/no-PH.

Prior audits (the six 2026-06-18 dropoff/inbox docs) already diagnosed the deeper version
of M1-M2-M5 ("objective slippage," the control-plane bug, the opponent-blind leaf). They
were accurate and remain mostly open. This audit's new contribution is showing the gap
widened post-06-18, with the specific phfix/registry evidence above.

---

## 2. Major problems: architecture + testability (drives the refactor)

**A1. No single source of truth for "the agent." [verified]**
- `agent/main.py` defines 9 `agent_*` entry points (agent, agent_search [shipped],
  agent_search_ctx, agent_search2, agent_search_v, agent_combine, agent_rank,
  agent_rank_hybrid, agent_eff). `tools/build_submission.sh:16-20` bundles only
  `search.py, eval.py, value_model.py, features.py` + card JSON + `cg/`. Everything else
  (ranker, contextual_ranker, deck_policy_v2, search_live_v2, teacher_api_v1/v2,
  state_action_schema_v2, action_semantics_v1, tactics_ontology) is dead relative to the
  submission, reachable only from tools/tests. Nothing but a docstring sentence declares
  which variant is canonical, and the build script encodes a different contract.

**A2. Heuristics are hard-coded inline in 5+ places with duplicated decoding. [verified]**
- Forced/KO rule: `main._attack_value` (`main.py:82-101`) + `main._forced_move`
  (`main.py:154-178`). Leaf weights: `eval.py:22-32` (W_PRIZE/W_HP/W_BODY/W_ENERGY/
  W_POWERFUL_HAND, a shared magic-number scale). Rollout policy: `search._rollout_pick`.
  Signals: `features.encode_state`. The attack-value math exists in 3 copies
  (`main.py:82`, `deck_policy_v2.py:149`, `tactics_ontology.py:51`) and the option->card-id
  join in 4 (`main._opt_card_id`, `ranker._card_id`, contextual_ranker, and the
  golden-tested `state_action_schema_v2.card_identity`). Adding one heuristic touches many
  files and risks the copies drifting.

**A3. The forced-KO rule has already diverged into three thresholds. [verified]**
- `main._forced_move` forces any KO at `>=8000` (`main.py:172`); `deck_policy_v2.
  forced_move_m0` forces only a game-winning attack at `>=90000` (`deck_policy_v2.py:188`);
  `deck_policy_v2.forced_move_ko_phaware` forces any KO at `>=8000` with PH-aware damage
  (`deck_policy_v2.py:210`). Three behaviorally different policies for one decision,
  selected by a toggle in the A/B harness. The 0.75-cited variant lives in the
  non-shipping module, so the measured agent is not the shipped agent.

**A4. A single heuristic / leaf eval cannot be tested on a fixed state. [verified, blocker]**
- Every scoring path bails when the engine is absent: `search._search`
  (`search.py:184`), `option_deltas` (`:400`), `option_evals` (`:533`),
  `deck_policy_v2.compare_selections` (`:324`) all return None / default with no `cg`. The
  one test (`tests/test_split_base_v2.py`, 8 functions) asserts encoding/schema parity and
  never a decision or a leaf value. So "does this heuristic pick the KO here?" cannot be
  answered without full games (slow + high-variance) -- the exact loop the user wants to
  avoid. Note `eval.evaluate` and `main._forced_move` ARE pure given an obs dict; they are
  just not wired to any fixture with an expected answer.

**A5. The never-throw / never-timeout competition rule is untested. [verified, blocker]**
- No test runs `agent_search` through a game; the legality/timeout invariant (a crash or
  timeout forfeits on Kaggle) has zero coverage. `ab_candidate_v1.run()` already catches
  per-game exceptions and counts errors, so the building block exists; it is just not in
  any acceptance test.

Good raw material already present (reusable assets):
- `eval.evaluate` (`agent/eval.py:66`) -- a pure, engine-free state->score; the natural
  pluggable leaf-eval seam, unit-testable as-is.
- `deck_policy_v2.compare_selections` (`agent/deck_policy_v2.py:315`) -- a working
  fail-closed "heuristics PROPOSE, search VALIDATES, default wins ties" contract; the right
  boundary to generalize, currently quarantined in a non-shipping module.
- `state_action_schema_v2.card_identity` -- the single golden-tested option->card-id join
  that should replace the 4 private copies.
- `tests/golden_state_action_fixtures` + `build_golden_fixtures.py` -- a frozen-observation
  corpus ready to carry `expected_action` labels.

---

## 3. Major problems: organization + hygiene

**O1. tools/ sprawl: 91 scripts + 27 scratch files. [verified]**
- Prefix counts: build_ 11, ab_ 9, train_ 8, audit_ 7, label_ 5, deck* 4, diag_ 3,
  verify_ 3, misc 23. 32 scripts carry _v1/_v2 suffixes; the terrain workstream is 8 loose
  scripts forming one pipeline; the ab_* cluster is 9 successive probes. No tools/README,
  no marker for which versioned script is canonical.

**O2. Two unconnected A/B harness lineages; a scratch script became shared infra. [verified]**
- `agent/cabt_arena.py` is the designed harness (winner_of, seat-swapped run). Separately,
  `tools/ab_candidate_v1.py` (154 lines, a runnable script that dumps to
  `_ab_candidate_v1.json`) reimplements that harness and is `import`ed by 10 scripts
  (ab_ablate/compute/control/depth/factorial/heuristic_search_v2/v3_ablate/vs_first,
  decision_hist, trace_game). The disjoint set (run_ab, freeze_baseline, search_sprint,
  deck_off, hoard, panel) uses cabt_arena. The branch's most recent results all run through
  the scratch lineage, so any bug there propagates uncentralized.

**O3. Copy-pasted primitives with real drift. [partial -- corrected]**
- `def wilson` appears in 8 files, but the first-pass "identical" claim is **refuted**: a
  body diff finds 6 variants, and two of them (`screen_tactics.py`, `freeze_baseline_v2.py`)
  round their output while the other six return raw floats -- so even the numeric contract
  differs. `winner_of`/seat-swap loop/`pilot_deck` are duplicated 4-10x; the DENPA92 deck
  literal is hardcoded in 15 files; `sys.path.insert` boilerplate is in 64 of 91 scripts
  (agent/ is not an installable package). The drift is the danger: fix one wilson and the
  others silently keep the old behavior.

**O4. Working-tree scratch and committed artifacts. [partial -- corrected]**
- 19 scratch result JSONs (`_deckoff_res_*`, `_panel_*`, `_hoard_denpa92`,
  `_deckoff_slate`) are git-tracked under tools/. ~7.3 MB of untracked scratch dirs sit in
  the tree, none gitignored. Correction to the first pass: they are NOT all "full agent
  copies leaked by one cleanup." The 6 `_v3_control_*` dirs ARE leaked `_build_pkg` outputs
  whose `ab_v3_ablate_v1.py:101-103` `finally` cleanup did not fire; `_candv1` is built by
  `ab_candidate_v1.py` which has no cleanup; `tools/_v3_src` is hand-staged source (the
  build input, not an output). Each leaked dir is an ~11-file import-rewritten subset, not
  the full 47-entry agent. `submission/submission.tar.gz` is a committed binary build
  artifact (sibling `submissions/` is correctly gitignored). The only `.gitignore` is at
  the monorepo root, not project-local.

**O5. docs/workstreams/ is 45 MB of mostly data, in a docs tree. [verified]**
- 54 json/jsonl vs 34 md; largest are multi-MB training datasets (up to 8.2 MB). New
  result JSONs keep landing here untracked. This belongs under `data/` (gitignored,
  regenerated), leaving only .md + small summaries in docs.

**O6. .env holds a live-format Kaggle token at rest. [verified, low risk]**
- `pokemon-tcg-ai-battle/.env` has a `KGAT_` token. It is gitignored (`.gitignore:140`)
  and never committed (verified across all history), so no leak. Residual risk is local:
  move it to `~/.kaggle/kaggle.json` (the location the gitignore comments already intend);
  rotate if it has ever been pasted into a shared log.

---

## 4. Major problems: docs

**D1. Layered conflicting authority. [verified]**
- Four docs each claim to be the plan of record pointing different directions:
  `STRATEGY.md` -> LEARNING_PLAN/RESEARCH; `LEARNING_PLAN.md` -> the dropoff CONSENSUS memo;
  `RESEARCH.md`/`PLAN.md` -> CURRENT.md + ACTION_RANKER_PLAN.md; `CURRENT/HANDOFF/OVERVIEW`
  -> `workstreams/BRANCH_PLAN.md` (the actually-current one). OVERVIEW resolves it correctly,
  but only if you land on OVERVIEW. README's "How a session runs" (`README.md:50-51`) and
  `AGENTS.md:10-12` both still route a fresh session to `STRATEGY.md`, which self-marks
  "SUPERSEDED IN PART," and never name BRANCH_PLAN.

**D2. README pointers all resolve. [verified -- corrected my first assumption]**
- All README/OVERVIEW-referenced docs exist (OVERVIEW, PLAN, LANDSCAPE, COMPETITION,
  STRATEGY, conventions, CURRENT, HANDOFF, SUBMISSIONS, ACTION_RANKER_PLAN, BRANCH_PLAN).
  No dangling pointer. The doc problem is authority/staleness, not missing files.

**D3. Minor drifts. [partial / verified]**
- `ACTION_RANKER_PLAN.md:3` (not :2) cites `dropoff/inbox/pokemon_tcg_agent_roadblock_
  diagnosis`, which does not exist; the real file is
  `dropoff/inbox/2026-06-18-roadblock-diagnosis.md`. `OVERVIEW.md:348` says
  `ROBUST_LEARNER_V2.md` "lives on the other branch" but it is present and tracked on this
  branch.

---

## 5. Organization plan (phased, lowest-risk first)

**Phase 0 -- hygiene (mechanical, no behavior change).**
1. Add a project-local `.gitignore`: `_candv1/`, `_v3_control_*/`, `_v3_*/`,
   `tools/_v3_src/`, `tools/_*.json`, `tools/_*.log`, `submission/*.tar.gz`. `git rm
   --cached` the 19 scratch JSONs and `submission.tar.gz`; delete the leaked scratch dirs.
   Route experiment outputs to one gitignored `tools/_runs/`.
2. Move the multi-MB datasets out of `docs/workstreams/` into `data/` (gitignored).
3. Move the Kaggle token to `~/.kaggle/kaggle.json`.
Risk: preserve any gitignored load-bearing state before any worktree/branch deletion (the
data/ junction + registry ground-truth lesson).

**Phase 1 -- shared experiment library (mechanical).**
4. Extract `wilson` / `winner_of` / seat-swapped `run` / `pilot_deck` / deck constants into
   one module (promote into `cabt_arena` or a new `tools/_lib.py`); repoint the 10 ab_*
   dependents and delete the 8 wilson copies. Pick one harness lineage.
5. Add per-game arm pairing (common random numbers) + pin controllable knobs + write a run
   metadata block into every result JSON. Document that the cabt engine is unseedable.
6. Add `pyproject.toml` / editable install so `agent/` imports without `sys.path.insert` in
   64 files.
7. Pick the canonical script per versioned family; archive the rest under `tools/archive/`
   (git history suffices for deletes). Add `tools/README.md` classifying each script as
   stable-CLI / pipeline-step / archived.

**Phase 2 -- agent consolidation + the heuristic interface (the refactor, section 6).**

**Phase 3 -- methodology re-grounding.**
8. Backfill post-06-18 results into the registry; retract the "verified 0.75" framing; add
   the `registry.py check` integrity subcommand.
9. Reconcile the doc authority chain: one authoritative pointer (BRANCH_PLAN), banner the
   stale docs, fix the README/AGENTS session path. Fix D3.
10. Decide the real question: is the binding constraint the compute budget (0.6s vs 600s)
    rather than the heuristic? Re-run the key depth/sampling A/Bs at a deployment-
    representative budget, seeded+paired, n>=200, before more heuristic tuning.

---

## 6. Refactor proposal: a testable, pluggable heuristic / signal pipeline

Goal restated: a clear "order / signalling sequence" where signals are computed once,
heuristics are registered objects that fire in a defined order, each is individually
testable on a fixed state in milliseconds, and a new heuristic is one object in one file.

Current order (the real one, verified): `agent_search` -> `_agent_search` ->
`_forced_move` (inline floor) -> `search.best_option` -> `_search` (N=8 determinizations,
0.6s) -> `_simulate` (1-ply rollout) -> `eval.evaluate_obs` leaf. Heuristics are scattered
inline across that path.

Proposed structure (builds on assets already in the tree):
- `agent/signals.py`: extends `features.encode_state` into named, cached-per-decision
  signals. A `Signal` is `(name, compute(obs) -> value)`. One place to add a state signal.
- `agent/heuristics.py`: a `Heuristic` protocol with `name`, `applies(obs, signals)`, and
  either `propose(obs, signals) -> Optional[move]` (a forced/floor move) or
  `score(option, obs, signals) -> float` (a leaf/option weight). One ordered `REGISTRY`
  list defines the sequence.
- Integration rule (generalize `deck_policy_v2.compare_selections`): a heuristic PROPOSES,
  the forward model VALIDATES that the proposal beats the safe default by a margin, and the
  default wins ties. This fail-closed contract is the safety property that keeps the agent
  always-legal and never-worse, and it already exists in code -- it just needs to come out
  of the dead module and become the single integration path.
- `eval.evaluate` stays the pure leaf eval and becomes one registered scorer.
- Promote `state_action_schema_v2.card_identity` to the single option->card-id join;
  delete the 3 private copies of the attack-value math behind one shared function.
- Collapse the 9 `agent_*` variants: keep the shipping pipeline in `agent/`, move research
  variants + v2 modules into a `research/` package, and add an `agent/manifest` declaring
  the canonical shipping agent that `build_submission.sh` reads.

Testability layer (the precondition, do this first within Phase 2):
- Extend the golden fixtures with `expected_action` labels (and a few expected leaf
  orderings). Add unit tests that call each heuristic / `eval.evaluate` / `_forced_move`
  directly on frozen obs dicts -- millisecond tests, zero engine.
- Add one smoke test that runs `agent_search` over the fixtures and a few `make("cabt")`
  games asserting: no exception, a legal index returned, per-decision time under budget.
  This is the single highest-value test (covers the untested never-throw/never-timeout rule).

Net effect: adding or ablating a heuristic becomes "write one object + one fixture test,"
and the measured agent and the shipped agent are the same object (fixing A3/M7).

---

## 7. What I did not verify / open questions

- Whether `phfix` actually beats production at n>=200, seeded+paired, realistic budget. The
  repo has 0.75 (n=20) and 0.388 (n=80) and cannot say.
- Whether enabling `opp_k>0` (the existing `_simulate2` 2-ply branch) beats the deployed
  opp_k=0 agent.
- Whether the post-06-18 registry staleness is deliberate batching for later backfill or
  unnoticed drift.
- Whether `submission/submission.tar.gz` currently matches `submission/main.py` + deck.csv.
- The 38k-LOC figure in the prompt counts docs/JSON; measured Python is ~30,333 LOC
  (5,232 agent + 25,101 tools).
