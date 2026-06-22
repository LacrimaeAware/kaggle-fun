# MASTER HANDOFF: pokemon-tcg-ai-battle audit + refactor (2026-06-21)

Read-first, single self-contained doc. Everything worth knowing before the whole-project
refactor is here. Companion detail (same findings, longer) is in
`2026-06-21-project-audit-and-refactor-plan.md`. Snapshot of the pre-refactor repo is git tag
`audit-2026-06-21` on origin/main.

How this was produced: a 6-reader file-level audit (decision pipeline, tools sprawl,
methodology, docs, tests/hygiene, registry) + a 20-claim adversarial verification pass against
the actual files. Provenance is marked: `verified` = re-checked against source.

---

## 1. TL;DR

- The project's bones are good (honest registry, careful prior audits, a working search agent
  at ~700 ELO). The problems are sprawl and a control plane that the recent sprints stopped
  using, not bad core engineering.
- Shipped agent = `agent_search` (1-ply forward search + 5-term hand eval, DENPA92 deck).
  Only 4 modules ship: `search.py`, `eval.py`, `value_model.py`, `features.py`.
- The single biggest correctness lever is NOT a new heuristic. It is that whole-game A/B is too
  noisy to tune heuristics (selfbase = 0.567 at n=120, where identical code should be 0.50), so
  past "wins" (PH-fix 0.75/n=20, N=32 0.625/n=40) are noise and already retracted. Tune
  per-decision with the paired-world instrument, not whole-game win rate.
- The refactor goal (easier to test/add heuristics, a clear order/signalling sequence) is
  unblocked by one move: pull the heuristic interface out of the dead module it already lives in
  and give it fixed-state tests. The test layer is already built (Section 5).

---

## 2. The decision pipeline as it exists today (what you are refactoring)

Real order for the shipped agent (verified):

    agent_search                 (main.py:234)
      -> _agent_search(opp_k=0)  (main.py:181)
      -> _forced_move            (main.py:154)  inline heuristic floor: lethal KO / go-first
      -> search.best_option
      -> search._search          (N_DETERM=8, DEFAULT_BUDGET=0.6s)
      -> search._simulate        (1-ply: my turn + one default-policy opponent reply)
      -> eval.evaluate_obs       (linear 5-term leaf eval)
    main.agent                   (main.py:141) = crash/timeout fallback (never raises)

Heuristics/signals are hard-coded inline in 5+ places on a shared magic-number scale:
`main._attack_value` (82-101), `main._forced_move` (154-178), `eval` W_* constants (eval.py:22-32),
`search._rollout_pick` continuation policy, `features.encode_state` (signals). The same
attack-value math is re-implemented in 3 files; the option->card-id join in 4.

---

## 3. Major problems (condensed, with evidence)

### Methodology (mostly already known + retracted; one new)
- M1 [verified, KNOWN]: the "verified 0.75 (15-5) PH-fix win" was n=20 noise; the same
  intervention is 0.388 (n=80) elsewhere in the repo. Already retracted (selfbase=0.567/n=120).
- M2 [verified, NEW and OPEN]: the registry has gone write-only. `registry/results.jsonl` is
  frozen at R009 (2026-06-18) while ~13 commits of phfix/N32/v3 experiments dump loose JSON into
  `docs/workstreams/*.json`. No agent/tool code reads `registry.py`; the anti-rehash search gate
  has no hook/CI. The machinery is sound (BELIEFS/GRAVEYARD regenerate clean); it is unused.
- M3-M6 [verified, KNOWN]: factorial "effects" from n=2 cells (aggregation fallacy); search
  judged at a self-imposed 0.6s vs the real ~600s/game; deployed agent is opp_k=0 1-ply rollout,
  not 2-ply; A/Bs are unseeded and unpaired (engine has no seed; harnesses do not pair arms).
- M7 [verified]: `verify_ship_v1.py` asserts N_DETERM==32 / budget>=1.0, which production
  (N=8 / 0.6s) does not satisfy; it tests presence of a code change, not its win-rate.

### Architecture / testability (drives the refactor)
- A1 [verified]: no single source of truth. `main.py` has 9 `agent_*` variants; only 4 modules
  ship; `ranker`, `contextual_ranker`, `deck_policy_v2`, `search_live_v2`, `teacher_api_v1/v2`,
  `state_action_schema_v2`, `action_semantics_v1`, `tactics_ontology` are dead vs the submission.
- A2/A3 [verified]: heuristics inline in 5+ places; the forced-KO rule has diverged into THREE
  thresholds (`>=8000` any-KO in main; `>=90000` lethal-only and `>=8000` PH-aware in
  deck_policy_v2). The agent you measure is not always the agent you ship.
- A4 [verified, BLOCKER]: you cannot score one heuristic on a fixed state without the cg engine
  (every scoring path bails when `_api()` is None), and the never-throw/never-timeout rule had
  zero tests. (Both fixed by Section 5.)

### Organization / hygiene
- O1 [verified]: 91 tool scripts + scratch; 32 carry _v1/_v2 suffixes; the terrain workstream is
  8 loose scripts; the ab_* cluster is 9 probes.
- O2 [verified]: two A/B harness lineages. `agent/cabt_arena.py` is the designed one; a 154-line
  scratch script `tools/ab_candidate_v1.py` became a de-facto library imported by 10 others.
- O3 [partial]: `def wilson` is in 8 files but NOT identical (6 variants; 2 round output, so
  numbers can disagree). `winner_of`/seat-swap/`pilot_deck` duplicated 4-10x; deck literal in 15
  files; `sys.path.insert` in 64/91 (agent/ is not a package).

### Docs
- D1 [verified]: four docs each claim to be the authoritative plan, pointing different ways.
  `OVERVIEW.md` resolves it (BRANCH_PLAN is current) but README/AGENTS still route a fresh session
  to the superseded STRATEGY.md. (2 minor factual drifts already fixed this session.)

---

## 4. Refactor design: the pluggable, testable heuristic/signal sequence

Goal: signals computed once, heuristics as registered objects firing in a defined order, each
testable on a fixed state in ms, a new heuristic = one object in one file.

Proposed structure (most of the raw material already exists, just quarantined):
- `signals.py`: extend `features.encode_state` into named, per-decision-cached signals. One place
  to add a state signal.
- `heuristics.py`: a `Heuristic` protocol -- `name`, `applies(obs, signals)`, and either
  `propose(obs, signals) -> Optional[move]` (a forced/floor move) or
  `score(option, obs, signals) -> float` (a leaf/option weight). One ordered `REGISTRY` list IS
  the order/signalling sequence.
- Integration rule (generalize `deck_policy_v2.compare_selections`, agent/deck_policy_v2.py:315):
  a heuristic PROPOSES, the forward model VALIDATES it beats the safe default by a margin, the
  default wins ties. This fail-closed contract is the safety property that keeps the agent
  always-legal and never-worse. It already exists; lift it out of the dead module.
- `eval.evaluate` stays the pure leaf eval and becomes one registered scorer.
- Promote `state_action_schema_v2.card_identity` to the single option->card-id join; delete the 3
  copies of the attack-value math behind one shared function.
- Collapse the 9 `agent_*` variants: keep the shipping pipeline in `agent/`, move research
  variants + v2/v3 modules into a `research/` package, add a manifest declaring the canonical
  shipping agent that `build_submission.sh` reads.

Reusable assets to KEEP/promote:
- `eval.evaluate` (eval.py:66): pure engine-free state->score, the leaf-eval seam.
- `deck_policy_v2.compare_selections`: the fail-closed propose/validate contract.
- `state_action_schema_v2.card_identity`: the golden-tested card-id join.
- `tools/cabt_arena.py` + `tools/run_ab.py`: the designed A/B harness; consolidate the ab_*
  duplication onto it.
- `registry/registry.py`: clean JSONL-canon + generated-views pattern; add a `check` subcommand.

Per-decision tuning instrument (use instead of whole-game A/B):
- `tools/ko_sequencing_state_test_v1.py`: find states where heuristic H applies, score both
  choices on the SAME paired hidden worlds via the engine, average the paired leaf diff. Noise
  cancels. This is the only method with resolution for sub-percent piloting fixes.

---

## 5. The test layer (already built; survives the refactor)

`tests/test_heuristics_fixed_state.py` (11 tests) + `tests/run_all.py`. Run:

    PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe tests/run_all.py

Covers, with no game and (mostly) no engine, in milliseconds:
- `eval.evaluate`: determinism, terminal WIN/LOSS/DRAW, prize dominance (= W_PRIZE under equal
  boards), body gradient.
- `_forced_move` reproduces the golden `forced_option` label on all 130 fixtures (regression lock
  on the forced-move heuristic + `_attack_value`).
- `_attack_value` KO/lethal thresholds (>=8000 KO, >=90000 game-winning, <8000 non-KO).
- `agent` and `agent_search`: legal + never raises + per-decision under budget on all 130
  fixtures (the competition's never-throw/never-timeout rule, previously untested).

To test a new heuristic after the refactor: add a synthetic state or golden fixture, add one
assertion calling your function directly. This file is structure-agnostic as long as the entry
points still exist. Side note: `eval`'s docstring claim that prizes dominate "regardless of board
fluff" is not strictly true (board-HP at W_HP=1.0 can reach ~2000 vs W_PRIZE=1000); left as-is.

---

## 6. What changed this session, and where everything is preserved

Additive only, no live-agent behavior change. Commits on origin/main (tag `audit-2026-06-21`):
- Phase 0 hygiene: project-local `.gitignore`; untracked 19 scratch JSONs + `submission.tar.gz`
  (index-only, files kept on disk). Note: `tools/_v3_src/` is TRACKED source, left alone.
- Fixed-state test layer (Section 5).
- `tools/README` script catalog + 2 doc-drift fixes (ACTION_RANKER_PLAN citation, OVERVIEW
  ROBUST_LEARNER_V2 location).

DEFERRED on purpose (do NOT do blind): the destructive consolidation (collapse the 9 variants,
dedup the 10 ab_* harnesses). Parallel sessions were editing the same `agent/`+`tools/` files
mid-audit (commit 5671b0b added deck_policy_v3.py / search_v3.py). Land it coordinated or on a
branch.

Preserved in 3 places: `Downloads/pokemon-refactor-docs/`, git history on origin/main, and the
`audit-2026-06-21` tag.

## 7. First moves after the refactor (recommended order)

1. Backfill the post-06-18 results into the registry with n + baseline + verdict; add
   `registry.py check`. (Fixes M2, the one open methodology problem.)
2. Build the `signals.py` + `heuristics.py` registry + fail-closed contract (Section 4), porting
   the existing `compare_selections` logic. Wire the fixed-state tests to each heuristic.
3. Keep tuning per-decision with `ko_sequencing_state_test_v1.py`, never whole-game A/B at small n.
4. Reconcile the doc front door: make OVERVIEW.md the single entry point; banner the stale plans.
