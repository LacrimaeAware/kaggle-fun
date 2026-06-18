# Consensus and way forward (2026-06-18) — single source of truth

This supersedes the scattered planning in `2026-06-18-master-plan.md` and `2026-06-18-forward-plan.md`
(kept for history). It folds in the four inbox handoffs (deep-research report, codex evaluation, codex
methodology audit, card-effects/action-prior handoff) and the user's direction. Written to be read by
a reviewing model and by us. Every result names its baseline.

## 1. Where we are (results, each baseline named)

Local cabt engine, same deck both sides, seats swapped. Small-n directional reads (tight CIs are not
the point).

| matchup (A vs B) | A win-rate | n | meaning |
| --- | --- | --- | --- |
| heuristic vs random | 0.835 | 200 | heuristic = KO/energy rules, no search |
| agent_search vs first_agent (contest baseline) | 0.585 | 800 | search beats the contest baseline |
| search vs heuristic (DENPA92 deck) | 0.86 | 50 | DECK-DEPENDENT, see note |
| combine vs heuristic | 0.83 | 160 | mostly the SEARCH, not the learned value |
| combine vs search | 0.37 | 60 | learned-value BLEND leaf worse than hand leaf |
| search_v vs search | 0.25 | 20 | learned VALUE leaf worse than hand leaf |
| search2 (2-ply) vs search | 0.35 | 20 | 2-ply opponent-min over-pessimistic, worse |
| eff (effect HAND-heuristic) vs heuristic | 0.20-0.48 | 40/deck | hand-weighted effect REPLACEMENT scorer worse |

DECK NOTE: "vs heuristic" is not a fixed measure of our strength. The no-search heuristic pilots a
simple basic-attacker deck nearly as well as search (old deck gap ~0.54) but pilots an evolution deck
(DENPA92, onechan1) badly, so search beats it by more there (0.86). Cross-deck "vs heuristic" numbers
are not comparable. Absolute strength is only the LADDER, which is untested with the new deck.

Best local agent: `agent_search` (1-ply forward search + hand leaf eval). Nothing has beaten it.

## 2. What every reviewer + the data agree on (consensus)

1. **The recurring failure was process, not ambition:** build a feature artifact, run an agent that
   does not consume it, then mislabel the result. `search_v`/`combine` losing does NOT test card
   effects, because they use the OLD board-summary value, not `card_effects.json`. The effect idea has
   barely been tested.
2. **Imitation top-1 is a probe, not the gate.** It is dominated by the engine's option ordering
   (option-0 ~0.59) and is noisy. The deciding metric is WIN-RATE; the longer-term truth is the LADDER.
3. **The simulator wall-clock is the bottleneck, not CPU.** Live self-play A/Bs are slow (~5 s/game
   for search). Training and diagnostics should run OFFLINE on the replay decision dataset, with cached
   per-decision features; reserve win-rate A/B for finalists.
4. **Card effects are affordances, not values.** "Poffin = search 2 basics" is good only with bench
   space, deck targets, early game, and no KO available. The effect layer must be a conservative
   RESIDUAL on top of the baseline (`score = baseline + effect_bonus*context - opportunity_cost`), with
   state x effect interactions, NOT a replacement scorer. My hand-weighted `eff` failed because it was
   a replacement that valued setup over attacking.
5. **Build the integrated model, but make benchmarks unfakeable.** Stop testing isolated fragments
   forever; some ideas only work when the pieces are together. But require a wiring proof and
   enabled-vs-zeroed ablations so a result cannot be mislabeled again.

## 3. The way forward: ONE integrated, replay-trained action model (the user's stack)

A single neural model, trained OFFLINE on replay decisions, scoring each legal option:

```
legal option
  -> card id (area==HAND -> hand[index].id)        # the verified join
  -> card embedding (learned, per id)              # residual card identity
  + decoded card effects (card_effects.json)       # magnitude-aware semantic scaffolding (draw 8 != draw 2)
  + state context (the 47 board features)
  + state x effect interactions                    # search_2 * bench_space, draw_8 * low_hand, accel * attacker_needs_energy ...
  + forward-model option_deltas (cached per decision)  # consequence: prizes/KO/draw/board
  -> shared neural trunk (torch, CPU is fine at this scale)
  -> ACTION-RANKING head (listwise over the decision's options)   # built FIRST
  -> [later] value head (win prob) + auxiliary heads (resulting option count, board dev)  # the "separated layers"
```

Heads added in order: action-ranking first (it directly tests "can the model choose Poffin / draw /
evolve / attack in context"), then value + auxiliary heads once the option-ranker path is proven real.
Used as the policy (argmax option) and/or a prior inside search; promoted only on WIN-RATE.

### 3a. The learning signal (the part the user says we do poorly) — specify it carefully

Do NOT train on exact-next-move imitation alone. A move the winner did not make THIS decision may be a
move they make a turn later, so exact top-1 over-penalizes valid resequencing. The target per decision:

- **Outcome-weighted:** weight each decision by the game result and margin (a winner still misplays;
  decisive-game moves are better demonstrations). Optionally clone both seats with outcome weighting.
- **Horizon / sequence-aware:** a candidate option is a soft-positive if the card/action it represents
  appears in that player's moves within the next K decisions, discounted by distance (gamma^steps). So
  "right move, played a bit later" gets partial credit, not zero. This is the "within the realm of the
  next moves, and how far in the future" idea.
- **Opportunity-cost / floor:** never reward skipping a KO or a strong attack; bake the lethal/KO floor
  in as a hard prior so the learned residual only operates where the floor is indifferent.
- **Loss:** listwise (softmax over a decision's options) toward the outcome-weighted, horizon-credited
  target; this is "how we incentivize it," and it must be stated and logged, not left implicit.

### 3b. Speed plan (because the sim is the bottleneck)

- Build an OFFLINE decision dataset once: per replay decision, cache the option feature matrix (card
  ids, effects, state, interactions) and the `option_deltas` (the only simulator-touching part). Reuse
  it across every training run. `decision_id -> option_feature_matrix`.
- Fast loop = offline ranker metrics (top-1/top-3/MRR vs option-0, and the outcome-weighted target).
  Slow loop = win-rate A/B, run only for finalists, with common seeds and early-stop on clear losers.
- Keep search out of the training loop entirely; only use it for the final win-rate gate.

### 3c. Unfakeable benchmark gates (so we never mislabel again)

Before trusting any number:
1. Wiring proof: show one real Poffin decision where enabling effects changes its score and zeroing
   them removes the contribution; show one decision where the policy refuses setup because a KO exists.
2. Ablations: full model vs no-card-effects vs no-embedding vs no-option-deltas vs effects-zeroed vs
   option-0/heuristic baseline. A win counts only if it beats the EFFECTS-ZEROED ablation, not just a
   weaker baseline.
3. Multi-deck: evaluate across decks (onechan1, DENPA92, Heisei, old), not one, since the goal is a
   flexible multi-deck agent.

## 4. My ranking of the directions (asked for)

Ranked by (fit to the goal of a flexible learned agent) x (tractability given the slow sim) x (evidence):

1. **Integrated replay-trained action-ranker (Section 3).** Highest fit (it IS the user's vision),
   and training offline on cached replay features sidesteps the slow simulator. This is the main bet.
2. **Submit the best current agent for a real ladder read** (`agent_search` + DENPA92 deck). Cheap,
   parallel, and the only ground truth (local self-play does not predict the ladder). Do this now.
3. **Replay-only analysis** to build the offline dataset + extract opponent decks/meta + the
   horizon-aware target. This is the substrate for #1 and matches "process replays, not live games."
4. **Search depth/quality** (more determinizations, better rollout). Deprioritized: 2-ply already lost,
   and the user's read (shared) is that search is not the answer.
5. **Learned value at the search leaf** (`search_v`/`combine`). Parked: worse than the hand leaf, twice
   confirmed. Do not revisit as-is.

## 5. Submission (the user wants one strong agent on the ladder)

Package `agent_search` with the DENPA92 deck (8 basics, fixes the old deck's mulligan problem; our
search pilots it) as the strong submission, to get a real ladder result while #1 is built. Local
self-play does not predict ladder placement; this is the only way to know if the deck swap + search
helps against real opponents. Build with `tools/build_submission.sh`, verify with verify_submission.py.

## 6. Honest risk

The integrated model may still not beat search or the field; this game's engine ordering + a simple
floor + search is a strong combination. The value of the plan is that the offline/replay focus makes
iteration fast, and the wiring proofs + ablations mean we will KNOW what actually carried any result,
instead of mislabeling it. Ambition (build the whole stack) + discipline (prove the path, ablate) is
the corrected method.

## 7. Immediate next actions
1. Build the offline replay decision dataset with cached option features + option_deltas (no live sim
   in training).
2. Build the integrated action-ranking model (embedding + effects + state + interactions + deltas),
   listwise outcome+horizon-weighted target; report offline metrics + the required ablations.
3. In parallel: package + submit `agent_search` + DENPA92 deck for a ladder read.
4. Only after offline ablations show card effects/embeddings carry signal: win-rate A/B vs
   `agent_search` across decks.
