# Landscape and forks (the "is this worth it and where do we go" doc)

> CONSOLIDATED 2026-06-19: see [docs/OVERVIEW.md](OVERVIEW.md) for the single-document project summary. Retained for the problem characterization + quant-transfer framing; slated for slimming.

Written 2026-06-16 to support one decision: how seriously to take this, and which direction.
Grounded in the engine as actually read and run, not in generalities. Provenance tiers as
in `COMPETITION.md`: verified / claim / unconfirmed.

> CORRECTION 2026-06-17: the official Data tab reversed the central constraint below. A
> forward-model SEARCH API does exist (`cg/api.py` `search_begin` / `search_step`, with
> determinized hidden-info inputs), the full card pool is known (`EN_Card_Data.csv`, 1267
> cards), and deck-building is confirmed. So in-match tree search / ISMCTS is the intended
> design, not blocked. Section 2's "no in-match forward model" was measured on the stripped
> installed package; ignore it. See `COMPETITION.md` "Update 2026-06-17" and registry H1.

## 1. What the problem actually is (verified by running the engine)

A turn-based, two-player, imperfect-information, stochastic card game. Concretely, from a
real game dump:

- The observation is RICH. Each in-play Pokemon exposes `id`, `hp`, `maxHp`, `energies`,
  `energyCards`, `tools`, `preEvolution` (the evolution stack), and status flags
  (`poisoned`/`burned`/`asleep`/`paralyzed`/`confused`). Each player exposes `active`,
  `bench` (max 5), `deckCount`, `discard`, `prize` (6 slots, hidden as `None`), `handCount`,
  and `hand` (your own cards listed; the opponent's `hand` is `None`). Top level: `turn`,
  `turnActionCount`, `yourIndex`, `firstPlayer`, `supporterPlayed`, `stadiumPlayed`,
  `energyAttached`, `retreated`, `stadium`. So a board-aware evaluation has everything it
  needs: prize race, HP, energy, evolution stage, status, hand and deck-out clocks.
- Hidden information: opponent hand and both deck orders and the face-down prizes. Genuine
  information sets.
- Win conditions (one confirmed directly, others standard PTCG): opponent has no Pokemon in
  play (seen: a player knocked out with an empty bench lost immediately), take all 6 prizes,
  or opponent decks out. Reward Lost -1 / Won +1 / Draw 0.
- Games are short to medium and always terminate. Random vs random over 150 games: median
  57 engine steps, p90 162, max 263, ZERO draws. The Magic-style infinite-loop worry does
  not apply: PTCG has no instant-speed stack, and the engine caps episodes at 10000 steps
  plus a time budget. First-player win rate was 0.453 (n=150), i.e. no clear first-player
  advantage in this pool, possibly a slight second-player edge (within noise).
- Action vocabulary (partly decoded from the option stream). Each decision offers a `select`
  with typed option dicts and a `maxCount` (almost always 1, occasionally more):
  - type 13 `{"attackId": N}`  = declare an attack
  - type 14 `{}`              = end turn / pass
  - type  8 `{inPlayArea,inPlayIndex,...}` = target an in-play Pokemon (attach/apply)
  - type  3 `{area,index,playerIndex}` = select a card (hand/board area)
  - type  7 `{index}`          = pick from an indexed list
  - type 1 / 2 `{}`            = menu-level choices (not yet pinned)
  The full legend is one short instrumentation task away.

## 2. The binding constraints (what makes this hard, specifically)

> SUPERSEDED 2026-06-17 (see top banner): item 1 below is WRONG. The forward model IS available
> (cg/api.py search_begin/search_step; registry H001 SUPPORTED) and in-match search is built and
> is our strongest agent. Read items 2+ for still-valid constraints; ignore item 1.

1. [INCORRECT, kept as history] NO in-match forward model. The engine drives a single global
   native battle; the agent gets only an observation, with no clone/step/search API (read in
   `cg/sim.py`). So we cannot do "try an action, evaluate the resulting state" in-match. This
   rules out clean MCTS/ISMCTS this version (registry H1, refuted). It is THE structural
   constraint. Consequences: the policy must either score options directly from board features,
   or we build our OWN partial next-state predictor for the deterministic actions (attach
   energy, evolve, attack damage) to recover a one-ply lookahead.
2. Card data is not exposed in the Python layer. `cg/__init__.py` is empty; no card DB
   ships; stats live in the native `cg.dll`. But `hp`/`maxHp` are in the observation, and
   attack damage per `attackId` is recoverable by logging defender HP before and after each
   attack across self-play games. The relevant card set is small (the default deck has ~10
   distinct Pokemon), so this table is cheap to build. Not a wall, but a prerequisite.
3. Variance is high. Card games swing on draws, prizes, and coin flips. A few hundred games
   is a wide confidence band; calling a change real needs proper sample sizes and a stated
   noise floor (this is also where the quant-style discipline pays off).

## 3. What winning realistically takes, and the field

- The cash and the talent are on the Strategy track ($240k for the Round-1 top 8). The
  contest is run with HEROZ (a Japanese game-AI company, shogi pedigree) and the Matsuo
  Institute, so the serious field is well-resourced and game-AI-literate.
- The realistic agent ceiling without an in-match forward model is a strong heuristic or a
  learned policy net, not a deep search agent. That is a leveler in our favor (no one gets
  to brute-force search either), but the top teams will have learned policies trained on
  large self-play, plus tuned decks.
- Honest probability of a top leaderboard finish: low. The value, if there is value, is the
  method and the learning, which transfer to quant, plus a real portfolio artifact.

## 4. The forks (where the decision lives)

### Fork A: how hard to push the in-match agent
- A0. Always-legal heuristic (done): beats random 0.835, ties first_agent.
- A1. Board-aware heuristic eval: score options using prizes, HP, energy, lethal/KO threat,
  status, hand/deck clocks, plus a self-play-mined attack-damage table. Effort M. This is
  the floor-raiser that should beat first_agent, and the prerequisite for everything else.
- A2. Self-built partial one-ply lookahead: reimplement the deterministic effects (attach,
  evolve, attack damage) from the mined table to approximate "best line this turn" without
  the engine's forward model. Effort M-L. Recovers most tactical value (lethal, best trade).
- A3. Offline-learned policy/value net: encode the observation, train by imitation from the
  heuristic then improve by self-play, run cheap at inference. Effort L. This is how a
  serious team actually competes given no in-match search.

### Fork B: the metagame / deck layer
- Deck selection as a payoff-matrix game: build a matchup matrix from self-play between
  candidate decks, solve for a maximin / mixed strategy. Effort M. Buys win-rate cheaply IF
  we choose decks (the FAQ says custom decks are allowed but not required; confirm on the
  live rules). External Limitless decklists are a weak prior because the card pool is
  organizer-fixed; we mostly generate our own matchups. High game-theory learning value.

### Fork C: the research / quant-aligned layer (transfers regardless of the leaderboard)
- C1. Separability / boundary-mapping: when is a deck or strategy edge recoverable from
  noisy win-rates at the game counts we can afford? Directly your `stable-grn-inference`
  exp-28 method, reused.
- C2. Opponent modeling as a belief/bandit problem: exploit weak deterministic ladder bots
  rather than play unexploitable. A ladder rewards exploitation; this is a clean online-
  learning problem.
- C3. Variance-aware evaluation: the sample-size and control-baseline discipline that keeps
  us from over-reading noisy match data. Transfers straight to backtest discipline in quant.

### Fork D: commitment level
- Light, to the gate: A1 + the decode + the self-play data pipeline. ~2-3 sessions. Decides
  whether the agent path has legs (does a real eval beat first_agent above the noise band).
- Medium, competitive agent: add A2/A3 and Fork B. Multi-week.
- Heavy, serious push: add C2 exploitation, deck optimization, and continuous ladder
  feedback. Months.

## 5. Recommended sequencing (hedged so early work is never wasted)

1. Decode the option-type legend + mine the card/attack table from self-play (serves A1,
   A2, A3, B, and the audit at once).
2. Build the self-play data pipeline and storage (serves everything downstream).
3. A1 board-aware eval. Measure vs first_agent. THIS IS THE GATE.
4. If the gate passes: branch into Fork B (deck game) and Fork C (the quant methods), and
   optionally A2/A3 for more agent strength.

Steps 1-3 are the light commitment and produce the data pipeline the rest needs, so the
cheap "is this worth it" test and the foundation for a serious attempt are the same work.

## 6. The honest read

Feasibility is not the question anymore; the engine is public, self-play is free, the
observation is rich, and the system is built. Time is the question. A top finish is
unlikely against a funded game-AI field. But this problem is an unusually clean vehicle for
the exact skills the quant target needs (self-play, learned evaluation, payoff-matrix game
theory, opponent modeling, variance-aware evaluation, recovering signal from noisy
outcomes), and you bring real TCG intuition to the one part that is hand-designed (the
evaluation). So the defensible framing is: low leaderboard EV, high learning and portfolio
EV, low remaining infrastructure cost. The light path to the gate is the cheap way to buy
information about whether the deeper investment is worth it.
