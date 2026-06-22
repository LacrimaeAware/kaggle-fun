# Refactor brief — everything to carry forward (2026-06-21)

Read this BEFORE refactoring the project from scratch, so the rebuild starts from the evidence and does not
repeat what has already been tried and concluded. Pointers to the deeper docs are at the end. Self-contained
on purpose.

## 1. What the agent is, and what reliably works
`agent_search` = determinized Monte-Carlo / Perfect-Information-Monte-Carlo (PIMC) search over the
competition's own forward model (cabt `cg.api`). At each single-pick decision: sample `N_DETERM` hidden
worlds, run a short forward rollout in each, score the leaf with a small hand evaluator, average, pick best.
A heuristic floor forces the unambiguous moves (lethal KO, go-first); it never crashes and always returns a
legal move.
- Config that works: DENPA92 Dudunsparce/Alakazam deck, `N_DETERM=8`, ~0.18s/decision under a 0.6s cap.
- Evidence: beats the heuristic ~0.83; public LB climbed 617 -> 640 -> 697 as the search and deck improved.
- THE FORWARD-MODEL SEARCH IS THE STRONG PART. Keep it as the core of any rebuild.

## 2. Full audit — what was tried and the conclusion (do not re-run these blind)
SEARCH side, helped:
- `N_DETERM` 4->8: real improvement (~0.675 head-to-head). N=16/32 directional, limited by budget/saturation.
- Deck swap to DENPA92: moved the LB up.
SEARCH side, no effect or worse:
- 2-ply min-leaf search: WORSE than 1-ply (over-pessimistic worst-case opponent).
- Continuation/rollout policy (develop-first vs attack-first): no reliable effect (ties both ways).
- Opponent-belief determinization: no effect at 1-ply (the leaf is near-blind to opp hidden cards).
- Copying the #1 deck (Mega Lucario): 0.30 under our search (deck value is policy-coupled).
LEARNING side, ALL washed or lost to search at equal budget:
- Learned leaf value (search_v): parity. combine/blend: worse, LB 422.
- Distilled action ranker: offline-faithful, fails in play (covariate shift).
- Full contextual action ranker (card embeddings + decoded effects + option deltas + target/entity +
  state x effect interactions + short history, grouped sibling targets): 10-10 wash as a tie-breaker.
- DAgger rounds 1-2: round 1 cut on-policy regret but hurt offline fidelity; round 2 failed its gate.
- Teacher V2 (criticality-gated high-N advantage + outcome auxiliary) contextual retrain: worse than option-0.
- Risk-only model: real detection signal but worsened selected-action safety; not integrated.
- Tactic miner / hard floors: washes.
DIAGNOSED BLOCKER: the teacher/eval labels are NOISY and BIASED because they come from determinized
stochastic rollouts. ~half of strategic decisions are near-tie/unstable; rare catastrophic states are only
~53% reproducible. The instability is mostly the engine ROLLOUT RNG, not the hidden-world determinization.

## 3. Research verdict (adversarially-verified literature; full digest linked below)
- Reliable levers, in order of empirical strength: (1) a CONSISTENT real-time search on an approximate value
  (poker's decisive lever: Libratus's raw blueprint LOST -8 mbb/g, adding nested subgame search WON +63
  mbb/g vs the same opponent), (2) a good learned leaf evaluation -- but in imperfect-info it is only SOUND
  over BELIEF STATES (ReBeL), not action/observation history, (3) raw compute -- lawful but only logarithmic
  in search and it saturates.
- Poker: CFR regret math (limit hold'em) -> search (no-limit) -> regret + neural value/policy + search over
  belief states (ReBeL). Chess: classical search + handcrafted eval until ~2017, now search + learned (NNUE)
  evaluation together. In both, SEARCH done right was decisive and learning is the leaf eval, not a replacement.
- "Just simulate more" has a CEILING: determinization saturates (~20 worlds in Dou Di Zhu) and PIMC carries
  STRUCTURAL BIAS (strategy fusion + non-locality) that no rollout count removes -- ISMCTS exploitability
  stays flat with more time while a consistent (CFR/OOS) method's falls.
- WHY OUR LEARNING WASHED: distilling a determinized search into a history-keyed neural value is the
  AlphaZero port, which is PROVABLY UNSOUND in imperfect-information games (an action's value can depend on
  the probability it is played, so a history-keyed value has no unique target). The wash is the EXPECTED
  signature of training on biased/high-variance determinized labels -- not an effort or capacity failure. A
  from-scratch rebuild of the SAME approach will hit the SAME wall.
- Highest ceiling but gated: self-play Deep Monte-Carlo (DouZero-style) is the strongest "true AI" path with
  domain proof, but it is gated on SIMULATOR THROUGHPUT, our binding constraint.

## 4. Recommended direction for the refactor
First do the cheap DECISIVE diagnostic, before any learning:
- Measure leaf correlation / bias / disambiguation on the game tree (the three a-priori predictors of
  whether PIMC works); test whether a consistent search's error/exploitability FALLS with budget while the
  determinized one PLATEAUS; check whether `agent_search` is past determinization saturation or genuinely
  under-simulating.
Then pick the evidence-backed road by the result:
- VARIANCE-limited (cheap, keeps search authoritative): common random numbers / paired evaluation -- compare
  sibling options on the SAME sampled rollouts so the noise cancels -- more determinizations to saturation,
  and a lower-variance leaf value. Heuristics only as fail-closed proposals / pruning priors that the search
  validates; never let a heuristic or a learned head take control from search.
- BIAS-limited: move to a CONSISTENT / belief-state search (CFR / ReBeL-style: a value over public belief
  states + regret-minimizing look-ahead). Run a small feasibility spike for the ~0.6s/move budget on a single
  fixed deck (card-game belief states may be far smaller / more structured than no-limit poker).
STOP doing: more learned heads trained on determinized-search labels; the naive AlphaZero port.

## 5. Operational facts + landmines to preserve through the refactor
- Engine: the cabt forward model is bundled in the installed `kaggle_environments` package (`envs/cabt/cg`)
  AND in `data/external/official/sample_submission/cg`. `search._api()` finds it. Reentrancy-safe.
- Budget: 0.6s/decision cap; the agent must never crash or time out and must always return a legal move.
- Data: ~1820 replays in `data/external/replays/` (gitignored); parsed DB in `data/replay_db/` (gitignored).
  Pull via `tools/fetch_episodes.py` (auth via `kaggle.json`; the `.env` has a BOM that breaks shell sourcing).
- Submission: `PYTHON=<venv> bash tools/build_submission.sh search`; `submissions/` is gitignored; the human
  uploads. Submission log: `docs/SUBMISSIONS.md`.
- Recurring BUG CLASS (kill it in the rebuild with a frozen schema + golden fixtures): option card-identity
  depends on option TYPE (PLAY has no `area` field, its index is a hand index); canonical equivalence must
  not collapse distinct options; comparing two search configs under a time cap confounds quality with how
  many determinizations finished (fix N and raise the cap to isolate quality); a "meta/opponent" deck must
  actually differ from ours or the test is a no-op; agents fall back to the heuristic on any exception, so
  verify the thing under test is actually exercised; a learned target with no labels silently trains nothing.
- Teacher primitives that already exist (do not rebuild): `search.option_evals` (per-option leaf value
  averaged over determinizations), `search.option_deltas` (one-step consequence), `best_option` / `_search`.

## 6. Where the deeper detail lives (all on main)
- Verified, cited research: `dropoff/inbox/2026-06-19-deep-research-beyond-heuristics.md`
- Way-forward (source of section 4): `dropoff/outbox/2026-06-19-research-grounded-way-forward.md`
- Other models' branch synthesis: `docs/workstreams/RESEARCH_SYNTHESIS_2026-06-19.md`
- Branch plan: `docs/workstreams/BRANCH_PLAN.md`
- Prior handoff/state: `docs/HANDOFF.md`, `docs/CURRENT.md`; submissions: `docs/SUBMISSIONS.md`

## One-line summary
Keep the forward-model search; stop bolting learned heads onto its biased/high-variance determinized labels;
run the cheap variance-vs-bias diagnostic; then either de-noise the leaf value (paired evaluation / common
random numbers) or move to a consistent / belief-state search -- those are the two paths with real evidence.
