# Game model research (raw landing notes)

> SUPERSEDED 2026-06-17 where it says "no in-match forward model". A forward model IS available
> (cg/api.py search_begin/search_step; registry H001 SUPPORTED) and in-match search runs. The
> game-as-formal-problem framing below is still accurate; ignore any "no forward model / tree
> search blocked" statements. Current state in docs/RESEARCH.md and docs/LEARNING_PLAN.md.

Curated version in `../../docs/STRATEGY.md` section 2. This is the raw research plus what
was read directly from the engine.

## The game as a formal AI problem

- Imperfect information. Each player has a 60-card deck (hidden order), an Active and up to
  five Bench (each with HP, damage, attached Energy, status), a hand visible only to its
  owner (opponent sees a count), a discard, a Lost Zone, six hidden prizes, a shared
  Stadium. Genuine information sets: a perfect-information solver would cheat. Principled
  tools are ISMCTS, CFR, or determinization with belief sampling.
- Stochastic. Coin flips, prize reveals, draws, shuffle/search effects: chance handled by
  expectation, not by chasing one lucky resolution.
- Action space (verified by running the engine). The harness offers `obs["select"] =
  {"option": [...], "maxCount": k}`; you return up to k indices. Options are typed dicts,
  e.g. `{"type": 8, "inPlayArea": 4, "inPlayIndex": 0}` (attach to an in-play Pokemon),
  `{"type": 3, "area": 2, "index": 1, "playerIndex": 0}` (select a card/target),
  `{"type": 14}` (looks like end-turn/pass), `{"type": 0, "number": n}` (a numeric/coin
  choice). Per-decision branching is small (about 2 to 15); the depth is that one turn
  chains many micro-decisions. A match was 156 engine steps in one observed game.
- `obs["current"]` (a Struct) carries: `players`, `turn`, `turnActionCount`, `yourIndex`,
  `result`, `energyAttached`, `retreated`, `stadium`, `stadiumPlayed`, `supporterPlayed`,
  `looking`, `firstPlayer`. These are the features a board-aware evaluation would read.
- Reward (verified, `cabt.json`): Lost -1 / Won +1 / Draw 0.

## Tooling landscape (honest completeness)

The decision is to reuse the organizer engine, not build one. Only the organizer ruleset
predicts the ladder; any other sim is the wrong surface.

| Engine | Lang / license | Completeness | Use |
| --- | --- | --- | --- |
| cabt (organizer) | Python, in `kaggle_environments` | the scoring engine; runs locally | use directly |
| tcgone-engine-contrib | Groovy/Java, Apache 2.0 | mature full PTCG, server-shaped | rules reference only |
| deckgym-core | Rust, AGPL-3.0 | ~82% of cards, TCG Pocket (simpler) | architecture model; wrong ruleset |
| ryuu-play | TypeScript, MIT | moderate, from scratch | licensed rules reference |
| ptcg-sim | JS, MIT | manual playmat, no rules engine | not programmatic |
| poke-env | Python, MIT | the video game on Showdown, not the TCG | design reference |

Cloning ~2,000 cards is a multi-month effort and would still mismatch cabt. cabt already
gives a `kaggle_environments` env, the legal-options interface, the imperfect-info
observation, an event log, and a default deck.

## Forward-model finding (registry H1, refuted)

Read `cg/sim.py` and `cg/game.py`: the native lib binds only `BattleStart`,
`BattleFinish`, `GetBattleData`, `Select`, `VisualizeData`. One global mutable
`battle_ptr`, no clone/snapshot, no `SearchBegin`/`SearchStep`. The agent gets only `obs`.
So no in-match cloneable forward model for tree search. Offline self-play (`env.run`, or a
`battle_start`/`battle_select` loop) does work, which keeps offline eval/policy training on
the table.

## Sources

GitHub `Kaggle/kaggle-environments` `envs/cabt/` (read directly), the
`matsuoinstitute.github.io/cabt` docs, and the engine code in the installed package. See
`../sources.md`.
