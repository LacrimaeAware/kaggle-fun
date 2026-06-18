# Learning plan: features -> state -> search + RL (the systemic version)

> CURRENT PLAN OF RECORD (2026-06-18): `dropoff/outbox/2026-06-18-CONSENSUS-and-way-forward.md`. The
> active build is ONE integrated, replay-trained, OFFLINE action-ranker (card embedding + decoded
> effects + state x effect interactions + forward-model deltas -> listwise outcome/horizon-weighted
> head), with wiring proofs + ablations, because the simulator is the wall-clock bottleneck. Defer to
> the consensus doc where this older framing differs.

How the classification, the card features/embeddings, the feature-encoded state, search, and
reinforcement learning fit into one pipeline, with the exact components and the evaluation
rules that prevent killing an idea on one bad experiment.

## The non-negotiable evaluation discipline (read first)

This plan is staged. Early versions are EXPECTED to be rough and sometimes worse than what
they replace. That is the plan, not a refutation. The rules (from `AGENTS.md` and the
ai-prompt-engineering research-methodology failure modes):

1. Every capability is a registry hypothesis with a named surface (self-play vs a frozen
   baseline, or the ladder), a sample size, and a stated noise band BEFORE the run.
2. To ACCEPT a gain: win rate vs the immediately-prior version, n >= 400, lower CI bound
   above the band, ideally repeated across 2 seeds.
3. To REJECT or park a component: never on a single run. Required first: (a) repeat the
   result, (b) implementation sanity-check (agent stays legal, never times out, features are
   actually populated, no NaNs, the forward model is being called), (c) eval-surface check
   (right opponent pool, enough games). Only then mark it refuted WITH a re-open gate.
4. Rejecting one component does NOT reject the architecture. If RL fails to beat the heuristic
   this cycle, that means "RL did not help yet," not "features/search are useless." The last
   working layer stands and we keep it.
5. Distinguish "the idea is wrong" from "weak implementation or weak eval." A rung-1
   observation (one run moved a number) never becomes a rung-6 decision ("park this forever").
6. No component is declared dead while a cheaper, untried variant of it exists.

If a result tempts a conclusion that violates these, that is the signal to stop and recheck,
not to write the conclusion.

## The architecture (five layers, one data flow)

```
classification (registry/card_review.json, the functional tags)
        |
        v
[L0] card features  -> tools/build_card_features.py -> card_features.json
        |   per card: multi-hot functional tags + numeric stats (hp, best-dmg, attack costs,
        |   retreat, ex/mega prize value, energy type). This is the designed card embedding.
        v
[L1] state encoder  -> agent/features.py : encode_state(obs) -> fixed-length feature vector
        |   board (my/opp active+bench hp, prizes left each side, hand size, deck count),
        |   AFFORDANCES computed from L0 over my hand+board:
        |     can_attack_now, energy_short_by_N, can_ko_opp_active_this_turn,
        |     draw_available, gust_available, energy_accel_available, heal_available,
        |     switch_available, counts per functional role in hand, deck role-composition,
        |     energy-type spread (the "3-4 colors -> need fixing/draw" signal).
        v
[L2] policy / value over the feature state   (the learner; same input across all versions)
        v0  hand-weighted linear scorer over L1 features (interpretable; the new heuristic)
        v1  small net behavior-cloned to imitate the v0+search agent (good starting weights)
        v2  RL self-play fine-tune of v1 (PPO or AlphaZero-style)
        |
        v
[L3] search  -> agent/search.py : forward model (cg.api search_begin/search_step)
        |   simulate each candidate action, score the resulting state with L2, pick the best.
        |   v0 = 1-ply; v1 = shallow MCTS guided by the L2 policy. Supplies the credit signal
        |   the sparse win/loss reward cannot.
        v
   the agent (agent/main.py) = L1 encode -> L3 search using L2 -> legal action, crash-safe.

[L4] deck layer (separate, smaller problem, same L0 features)
        self-play matchup matrix -> deck-selection as a payoff game; deck-composition and
        "resource curve / gas" features for building. Uses cabt_arena self-play, not in-match RL.
```

Data/self-play loop: `cabt_arena.py` generates games (heuristic vs heuristic, later net vs
net), logs (feature-state, action chosen, final outcome) to `data/selfplay/`. Those logs train
L2 v1 (imitation) and v2 (RL), and build the L4 matchup matrix. Better agent -> better
self-play -> repeat.

## Build stages, each with its gate

### Stage A: feature foundation (no learning)
- A1 `tools/build_card_features.py`: card_features.json from the confirmed classification + stats.
- A2 `agent/features.py`: `encode_state(obs)` + the affordance features above.
- A3 `agent/eval.py`: v0 linear scorer over the features (named, tunable weights).
- GATE A (registry hypothesis): the feature eval beats first_agent AND the current type-14
  agent in self-play, n>=400, stated band. A miss here means tune features/weights, not abandon.

### Stage B: search + imitation
- B1 `agent/search.py`: 1-ply forward-model search with the v0 eval at the leaves.
- B2 self-play logging -> behavior-clone a small net (v1) to imitate the v0+search agent.
- GATE B: search+eval beats no-search eval; v1 net matches the search agent at lower per-move
  cost. Measured across seeds.

### Stage C: RL self-play
- C1 RL fine-tune (PPO or AlphaZero+MCTS) from the v1 start, feature input, reward = win/loss
  plus a shaped prize-differential term.
- GATE C: v2 beats v1/heuristic beyond the band over several training runs. Per rule 4, a miss
  keeps v1; it does not condemn L0-L3.

### Stage D: deck layer
- D1 matchup matrix from self-play; deck-selection payoff game; deck-composition + gas-curve
  features (e.g. average turn the deck runs out of playable resources vs average game length,
  which the model can compute from logs and a human cannot eyeball).

## Embeddings: tied to the question (not co-occurrence)

L0 is a designed embedding (functional tags + stats). A LEARNED refinement is optional and
only worthwhile if tied to the right objective: train it to predict the functional tags, or to
predict matchup outcomes, NOT raw co-occurrence (which only recovers deck membership). It is a
v2+ refinement of L0, evaluated as an input representation for L2, never as a standalone
predictor (that is the structured-transform-repo lesson).

## Built so far (2026-06-17) and next

DONE:
- A1 L0 card features: `tools/build_card_features.py` -> `agent/card_features.json` (1267 cards,
  functional tags + hp/best-dmg/attack-costs-by-type/ex-mega). Reads the confirmed classification.
- A2 L1 state encoder: `agent/features.py` `encode_state(obs)` -> ~40 named features, tested on
  a real game state. Two sources as designed: the board (obs.current) and the LEGAL options
  (obs.select.option). Key pieces, verified:
  - TYPE-AWARE energy affordance (`energy_shortfall`): 4 Water attached vs an attack needing 4
    Lightning returns shortfall 4 (zero valid) even though 4 energy are held; colorless cost =
    any; Rainbow/Team-Rocket energy = wildcard. Features: active_energy_short, can_attack_now,
    active_affordable_dmg, can_ko_opp_now.
  - PLAYABILITY from the legal-option list, not the hand: draw_playable_now / tutor_playable_now
    / gust_playable_now / can_attach_energy / can_evolve / can_use_ability, so a draw card that
    is not playable this turn does not count.
  - COLOR-MATCH: needed_colors vs color_mismatch (colors my attackers need but I cannot
    currently produce), complemented by fixing_available (have draw/tutor/accel to dig out).
  - Board/resource: prize_lead, bench counts, hand size, deckout_risk, status conditions, HP.

- A3 feature heuristic: added energy-to-active-attacker rule to the conservative policy.
  GATE A (no-search): TIES first_agent, 0.513/300 (inside noise). A board-aware hand policy does
  not beat the baseline -> confirms the ceiling that motivates search. Kept (correct, harmless).
- B0 forward model CONFIRMED LOCAL (H001): cg.dll loads on Windows, the env emits
  search_begin_input, and search_begin->search_step round-trips with no reentrancy crash across
  800+ arena games. The lever is real, not theoretical.
- B1 1-ply search (`agent/search.py` + `agent/eval.py`): simulate each option, finish my turn +
  the opponent's reply with the engine default policy, score the leaf with the linear eval, pick
  best. Determinizing the opponent's hidden cards from the real deck composition (`_hidden_pool`,
  not all-energy) lifted it 0.522 -> 0.585 vs first (H022, supported).
  GATE B (partial): BEATS first_agent 0.585/800, 95% CI [0.551, 0.619] (clears the band). Does
  NOT yet beat the board-aware heuristic: 0.460/200 (within noise, point estimate behind). So the
  architecture works against the naive baseline but v0 search is at parity with the better hand
  policy. Per the discipline this is "v0 search did not beat the heuristic YET," not a refutation.

L2 v0 LEARNER SKELETON built + measured (2026-06-17, registry H023):
- `agent/datagen.py` (self-play -> feature/outcome rows), `tools/train_value.py` (value model +
  honest game-wise eval + collinearity prune + sign-warnings), `agent/value_model.py` (pure-numpy
  inference), wired into search leaves via `eval.py:evaluate_learned` (agent `search_v`).
- Adversarial review (19 findings, all verified) caught real bugs; mechanical ones FIXED
  (start-of-turn-only logging to match the leaf distribution; drop garbage setup rows; game-wise
  split; collinearity prune; neutral fallback; leaf-guard). Honest signal: value predicts winner
  0.618 acc / 0.663 AUC (game-wise) over a 0.532 baseline -- weak but real.
- RESULT: the learned MC-logistic value still LOSES in search (search_v 0.347 vs heuristic, 0.292
  vs hand-eval search). Mechanical fixes were necessary hygiene, not the cure. Diagnostic: GBM
  AUC 0.724 > logistic 0.675 (model form matters) but still modest because the data is coinflip
  heuristic self-play. So the two real levers are (a) a threshold-aware value form (tree / no
  sign-flips) and (b) stronger + iterated data so states separate. NOT an architecture refutation.

L2 v1 THE REAL LOOP built + robust (2026-06-17, registry H023/E008), on the CORRECTED tags:
- Value form = gradient-boosted tree (`tools/train_value.py`), exported as raw tree arrays for
  pure-numpy inference (`agent/value_model.py`), VERIFIED to match sklearn within 1e-16.
- Data = varied/separable corpus (`agent/datagen.py`, exploration eps + mixed heuristic/random/
  search matchups), one start-of-turn row per turn. GBM AUC 0.774 (was 0.663 logistic), sane
  importances, no sign-flips.
- Search = robust (`agent/search.py`): aggressive opponent reply (was a non-punishing default
  pushover) + average each option over N_DETERM=4 determinizations (was one noisy world).
- RESULT (CORRECTED after adversarial verification found a scale bug, then re-measured with CIs):
  search_v (tree value) vs heuristic 0.427, 95% CI [0.380, 0.476] -- ENTIRELY below 0.5, the learned
  value LOSES conclusively (n=400). vs hand search 0.467 [0.411,0.523]; hand search vs heuristic
  0.543 [0.487,0.599] (strongest agent). A scale bug (terminal +/-1e6 averaged with P(win)) had
  INFLATED search_v to 0.485; fixing it DROPPED it to 0.427 (the terminal-rate proxy out-ranked the
  clean value). LESSON (now properly supported): a tree MC value at high global AUC (0.735) is a
  WORSE 1-ply leaf eval than the hand eval, because global classification != LOCAL ranking of nearby
  candidate leaves (tree piecewise-constant; the hand eval's continuous prize/HP gradient ranks
  siblings finely). Architecture NOT refuted. Levers: smooth/monotone value; value guiding DEEPER
  search (AlphaZero policy+value, bootstrapped targets) not 1-ply leaf scoring; complex decks. Ship
  agent_search (hand eval), NOT search_v.

NEXT -- LIVE STATUS AND PRIORITIES ARE IN docs/RESEARCH.md (updated 2026-06-17); this list is the
architectural plan, RESEARCH.md is the current source of truth. DONE since: combine v1 (parity);
value-target fix = search-bootstrapped targets, expert-iteration passes 1+2 (no gain on this deck,
all CIs ~0.5); action-ranking objective on sibling-leaf data (measuring). So we ARE doing
bootstrapped/expert-iteration value learning (not "no RL"); full policy-gradient RL is not done.

1. COMBINE v1 (DONE, parity): floor search with the clean heuristics (always take a listed lethal;
   go first) AND blend leaf eval = hand_eval + lambda*value on ONE [0,1] scale.
2. VALUE-TARGET FIX (highest-leverage for the measured symptom): retrain the value on SEARCH-
   BOOTSTRAPPED targets (greedy backup / the search's own leaf value), and/or a candidate-leaf
   ADVANTAGE/RANKING model over sibling actions, not a global state classifier. (Willemsen 2022:
   MC-outcome targets cause exactly our good-AUC/poor-ranking failure; off-policy bootstrapped
   targets train faster + stronger.)
3. SEARCH KNOBS (cheap): raise N_DETERM 4 -> 20-40 (spend budget on breadth, not depth); TEST a
   weaker/random rollout vs the current aggressive one (weak rollouts beat strong-expert rollouts:
   Cowling 2012, Gelly-Silver 2007).
4. REPRESENTATION: continuous magnitude-aware card features + outcome-supervised CONTRASTIVE
   embeddings (draw-2 != draw-7); judge on generalization, not in-sample AUC (Bertram 2024).
5. EXPERT ITERATION: self-play with current-best -> search-bootstrapped targets -> retrain ->
   repeat. Game-theoretic self-play (fictitious play, ByteRL) is an alternative; ReBeL /
   Player-of-Games are the principled hidden-info frontier. Evaluate RL by head-to-head, not AUC.
6. FOLLOW-UP RESEARCH: neuro-symbolic / pseudo-linguistic learned heuristics (idea 4) is unanswered.
Live registry questions: H015 (heuristic-as-prior), H004 (expectimax > heuristic), H022 (archetype
determinization), H023 (learned value).
