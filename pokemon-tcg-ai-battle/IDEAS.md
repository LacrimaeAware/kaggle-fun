# Idea bank

Raw ideas and directions we might pursue. This is the "park a thought so it is not lost"
surface. Nothing here is a commitment or a tested claim.

How it fits the other files: the **journal** is the daily messy dump, **this file** is the
curated list of ideas, and the **registry** is for ideas we have turned into a testable
claim and measured. An idea graduates from here into a registry hypothesis when we decide
to actually test it.

Status words: `idea` (raw) / `parked` (set aside, maybe later) / `testing` (now a registry
hypothesis) / `done` / `dropped`.

## What human data we have vs do not (updated)

Three tiers, decreasing availability:
1. **Decklists** (have, ~1670): what cards strong players run. Usable now for deck choice and
   for which cards and effects matter.
2. **Match results** (have, ~3700): who beat whom, with each player's deck archetype. A real
   human deck-vs-deck matchup matrix. Usable now for the deck-selection and "human deck vs AI
   deck" ideas. Results only, no moves.
3. **Move-by-move logs** (do NOT have): how each turn was played. No public bulk dataset;
   only per-player Pokemon TCG Live text logs, and they are the real game not cabt. Anything
   that needs "the exact move a human made" is blocked until we collect our own.

The other constant catch: all of this is the REAL game's cards, while cabt uses its own card
IDs and an organizer pool. So it is human strategy reference, not a cabt drop-in.

## Ideas

### 1. Learn from tournament players to speed up our heuristics
`idea`
- Use the 1630 downloaded decklists to learn what strong human decks look like, and let that
  shape the bot's priorities (which cards and lines matter).
- Why it could help: a head start on good play instead of discovering everything from blind
  self-play.
- What it needs / catch: this gives deck knowledge, not move knowledge (see the shared catch
  above). The win is in deck construction and "what is important," not in copying turns.

### 2. Two opposing bots, human-shaped vs pure self-play, and compare them
`idea`
- Build one bot shaped by human knowledge (human-style decks, human-informed priorities) and
  one bot grown purely from self-play with no human input, and compare them head to head and
  against the field.
- Why it could help: the comparison itself is the value. It tells us whether human knowledge
  actually buys anything over raw self-play, which is exactly the kind of "does the clever
  thing beat the plain thing" question your other repos are built around.
- What it needs / catch: a true "trained on human play" bot needs human move data we do not
  have. The doable version now is human-**deck** bot vs self-play-**deck** bot, both using
  the same in-game brain. The "learned from human play" version waits on move data.

### 3. Scenario test: "what would you do here?" scored against a reference move
`idea`
- Take a position from a real game, hide the rest, ask a bot "what is your move here?", and
  score it against a reference best move. Run this across many positions as a quality test.
- Why it could help: a fast, position-by-position way to measure move quality, instead of
  only win-rate over whole games (which is noisy). Good for catching where a bot plays badly.
- What it needs / catch: the clean version wants the human's actual move as the reference,
  which needs human move logs we do not have. The doable version now uses a **strong bot's**
  move as the reference (positions where a strong and a weak bot disagree), which is a real
  move-quality benchmark even if it is not "human optimal." Upgrade to human-move reference
  if we ever source that data.

### 4. Card embeddings to find counter-structure (not co-occurrence)
`idea` (parked, the user's framing, handled measuredly)
- The idea: give each card a vector and a weight, and look for hidden structure, specifically
  what COUNTERS what, not what is played together. The user's "hardness of rock" point: the
  real factor is often not the card identity but an underlying property (tempo, energy cost,
  speed) that cuts across decks. Rock beats scissors because of hardness, not because it is rock.
- Why it could help: better deck-building intuition and a way to prune the deck search toward
  the axes that actually decide matchups, instead of trying every combination.
- The honest caveats (the user's own, and from their repos):
  - Naive embeddings on co-play data just recover decks (they split cards by which deck they
    appear in). That is not the counter-structure we want.
  - "What counters what" needs DIRECTED data (matchup outcomes, or forced-matchup interventions
    in self-play), not symmetric co-occurrence. This is the structured-transform-discovery and
    mechanistic-model-inference lesson: intervention reveals structure co-occurrence cannot.
  - Across the user's prior repos, these methods bought diagnosis and boundary-mapping, not
    raw accuracy, and only when the signal sat above the noise floor (the separability lesson,
    H9). So the disciplined version is: factor the matchup matrix for latent counter-axes, and
    measure it against a dumb baseline (logistic regression on hand-crafted deck features, H10).
- Status: a Stage 2/3 diagnostic experiment with a clear question and a control, NOT a
  near-term win. Do not credit it with win-rate until it beats the baseline on real data.

### 5. Heuristic vs reinforcement-learning bake-off
`idea`
- Train a heuristic-tuned agent and an RL self-play agent on the same engine and play them head
  to head, to settle for THIS game whether RL actually beats a strong heuristic. The clean
  result is the deliverable either way (the structured-vs-standard question the user's research
  keeps asking).
- Caveat: RL needs a lot of self-play and careful setup; it is Stage 3 in `docs/PLAN.md`, after
  the heuristic floor and shallow search. Past RL attempts "sucked" most likely from too little
  training, weak reward/credit assignment, or no fast simulator; we now have a fast simulator,
  which removes one of those, but not the others.

## Next actions for these (not now, just noted)

- Idea 1: when we build the deck layer, mine common card counts and archetypes from the
  decklists as a prior. Low effort, do it alongside deck selection.
- Idea 2: only meaningful after the basic smart bot exists (`docs/PLAN.md` step 3); then it
  is a clean A vs B comparison.
- Idea 3: a strong-bot-reference version becomes possible once we have a strong bot and the
  practice-game recorder (`docs/PLAN.md` steps 2-3).
