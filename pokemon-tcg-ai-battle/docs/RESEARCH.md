# Research state and open ideas (LIVING doc)

In-the-moment notes on where the learned-value direction is going, the ideas behind it, and the
questions a deep-research pass should answer. LIVING and temporary: when an idea here stabilizes
into a decision or a result, MOVE it into the permanent home (docs/LEARNING_PLAN.md for the plan,
registry for hypotheses/results, AGENTS.md/conventions for rules) and delete it here. Do not let
this become a second source of truth.

## Where we are (2026-06-17)

- Strongest agent = `agent_search` (forward search + hand eval): 0.585 vs first, 0.543 vs heuristic.
- `agent_search_v` (search + learned gradient-boosted-tree value, AUC 0.735) LOSES: 0.427 vs the
  heuristic. A tree predicts win/loss globally but ranks NEARBY candidate leaves coarsely, and
  1-ply search needs local sibling ranking. Registry H023; details in MODEL_COMMUNICATION.md.
- We have moved past pure supervised MC-outcome value. Now using SEARCH-BOOTSTRAPPED /
  expert-iteration value targets (2 passes run) and an ACTION-RANKING objective (sibling-leaf
  data) -- the RL-family direction. Full policy-gradient self-play RL is NOT done. Current numbers
  in the reckoning below.

## Replay-DB findings + deck decision (2026-06-18, READ THIS — newest)

Built a replay database from real ladder games: `tools/build_replay_db.py` over 135 downloaded
replays -> 133 games, 14,442 outcome-labelled decisions, 57 distinct decks. Artifacts under
`data/replay_db/` (games.jsonl, decks.json, decisions.jsonl; gitignored). Source: top-3 public
teams (onechan1, DENPA92, Kyo_s_s) + given submissions, pulled with `fetch_episodes.py --team`.

- OUR AGENT on the ladder = "Shishio Makoto": **0.44 win rate** (18 games), 6-basic deck, and a
  **mulligan outlier** -- our old deck shuffled up to 6x at setup (field mean 0.26 mulligans; only
  2.3% of seats mulligan >=3). The opening no-basic redraw is engine-forced, not an agent bug;
  it is the basic-light deck handing the opponent a free card each time.

- HEADLINE: **deck value is POLICY-COUPLED -- copying the best deck FAILS.** Each top deck piloted
  by OUR policy vs our old deck (real cabt, seat-swapped, Wilson95):
  | deck (ladder wr) | basics | under our heuristic | under search |
  | --- | --- | --- | --- |
  | onechan1/Kyo_s_s Iono-Bellibolt (0.83) | 9 | 0.212 [0.14,0.31] n=80 | 0.350 n=40 |
  | DENPA92 Dudunsparce/Alakazam (0.79) | 8 | **0.738 [0.63,0.82] n=80** | 0.55 n=40 |
  | Heisei (0.87) | 8 | 0.500 [0.39,0.61] n=80 | -- |
  Our agent is a basic-attacker pilot; it cannot run an evolution engine (Iono-Bellibolt needs
  evolution sequencing + ability use). DENPA92's deck suits our heuristic.

- ACTION TAKEN: adopted **DENPA92's deck** as `agent.DECK` (8 basics -> fixes the mulligan outlier;
  0.738 vs the old deck under our best ladder policy, the heuristic). Old deck kept in a code
  comment + git. This is deck-first improvement WITHIN our agent's competence.

- DECISION STATS (`decisions.jsonl`): win-rate-at-decision-type and point-biserial feature-outcome
  correlation. Board development (`my_bench` +0.28), `prize_lead` +0.28, and option-availability
  (`fixing_available`, `can_use_ability`, `active_can_attack_now`, `can_retreat` all positive)
  correlate with WINNING; `opp_bench` -0.30. `evolve-to` decisions sit at 0.07 win@ (selection
  effect: the evolving side is usually behind, and our mispiloted evolution decks live here). This
  is the non-circular OUTCOME signal for the action ranker, and it matches the "more options =
  better" framing. Caveat (per the deep-research label-circularity note): these are naive
  correlations with selection effects, not causal action values.

- FORK (open): **(A) policy-first** -- imitation-learn from the strong players' trajectories so we
  can pilot the top evolution decks (onechan1); highest ceiling, the deep-research #1 path, data is
  ready (14,442 labelled decisions + full winner trajectories). **(B) deck-first** -- keep refining
  decks our current agent pilots well (DENPA92 swap is step 1). A is the larger build; awaiting steer.

## Imitation feature-ceiling diagnostic + correction (2026-06-18, adversarially verified)

`tools/diag_action_ceiling.py`: for every single-select decision the EVENTUAL WINNER faced in the
replays (6,953 decisions, avg 8.2 options), train a within-decision ranker to predict the option the
winner actually chose; metric top-1.

- A FIRST pass looked like "richer action features don't help -> our representation is the ceiling".
  A multi-agent adversarial verification (registry: this pass) REFUTED that. Two real bugs:
  1. The card-identity join was DEAD CODE (`isinstance(hand[idx], int)`, but replay hand entries are
     dicts `{'id':..}`). The whole card-feature block was all-zeros, so "+card" was bit-identical to
     "type-only". Fixed: join via `AreaType.HAND==2` + `hand[idx]['id']`, restricted to PLAY/ATTACH/
     EVOLVE.
  2. `agent/eval.py` `evaluate_blend` referenced `VM` without importing it -> NameError -> every
     blend / `agent_combine` data point silently fell back to hand-only. Fixed (likely a cause of
     sub_combine's 358 ladder score). Re-measure agent_combine before trusting any blend number.
- CORRECTED, stratified result (with the right baseline): random 0.197; **chose-option-0 (the engine's
  option-ordering prior) 0.587**; hand heuristic 0.553; pointwise GBM on our per-option features ~0.50
  (which-card 0.81 = the option-0 prior since type is constant there; mixed-strategic ~0.33, BELOW the
  ~0.46 option-0 rate). So our per-option features, even fixed, do NOT beat "pick option 0" under a
  pointwise objective.
- HONEST VERDICT: INCONCLUSIVE on representation-vs-objective. The metric is dominated by a positional
  prior, the objective is a weak pointwise GBM, and the real levers are UNTESTED: a listwise/pairwise
  objective (test first, cheapest), card-EFFECT features decoded from cards_full.json text, and
  forward-model one-ply DELTA features (simulate each option, feature the state delta). Do NOT repeat
  the earlier "it's the representation" claim; it was premature.
- HANDOFF (per the user's ask): `dropoff/outbox/2026-06-18-feature-optimization-prompt.md` (paste to
  another model to build the corrected diagnostic + the prioritised action/effect/interaction feature
  set) and `dropoff/outbox/2026-06-18-research-questions.md` (deep-research questions: representation
  vs objective vs depth; BC done right / circularity; the learned-heuristics bridge; multi-deck SOTA).

## PROGRESS RECKONING + DIAGNOSIS (2026-06-17, latest; read this for current status)

ACHIEVED:
- Infrastructure end to end: local engine, forward model (search_begin/step), full pipeline
  features -> value -> forward-model search -> expert-iteration loop; crash-safe; measurement
  discipline (Wilson CIs, n>=400-800, registry); many bugs fixed; submissions load on Kaggle.
- Signal exists: the representation + a model predict the eventual winner from a mid-game state at
  AUC ~0.74, and predict the search's state value at Pearson ~0.9.
- Decision content measured: of 400 of my decisions, 82% are real choices (>=2 options), 18%
  forced. The deck is NOT forced.

NOT ACHIEVED:
- No learned agent beats the hand-eval search in play; all cluster ~0.50 vs heuristic/hand-search.
- Learned value progression vs heuristic: 0.347 (logistic) -> 0.427 (tree, MC outcome) -> 0.517
  (tree, search-bootstrapped pass 1) -> 0.490 (pass 2, learned-value leaves). All CIs span 0.5.
  Two expert-iteration passes gave no measurable gain.
- The predictive signal does not convert into better move selection.

THE GAP (data-grounded): the three policies disagree on 45-65% of real choices (hand-search ==
heuristic 55%, learned == hand 63%, all three agree 35%) yet outcomes stay ~0.50. So most single
decisions are LOW-IMPACT on the win (different move, same result); leverage is the few high-impact
decisions, which we do not isolate. Mechanically: (1) at 1-ply one move barely changes the leaf, so
the value cannot separate sibling moves (good GLOBAL accuracy, poor LOCAL resolution); (2) we have
only ever trained a STATE VALUE (rank positions), never an ACTION objective (rank the moves out of
one position). It is OUR conversion gap, not the deck being empty.

COMPLEX DECK: data argues against it as the next move. This deck already has 82% real choices; a
complex deck adds decisions but carries the same value-to-action conversion gap. Work this deck.

## DIRECTIONS LEDGER (what we have tried and what we will try) -- keep this current

All win rates are real cabt engine, same deck both sides, Wilson CIs; canonical numbers in the
registry (BELIEFS.md / results.jsonl). Read: every LEARNED variant clusters near parity; the
hand-eval search is marginally best; NOTHING decisively separates, so the learned approach is
close, not beaten. The learned-value-at-1-ply framing is PARKED (registry H023), not abandoned.

TRIED:
| direction | what it is | result vs heuristic | note |
| --- | --- | --- | --- |
| heuristic | hand rules, no search | ~0.51 (ties first) | the floor |
| hand-eval 1-ply search (agent_search) | search, hand formula at leaves | 0.543 (0.585 vs first) | STRONGEST measured |
| learned value: logistic | state-value leaf eval | 0.347 | bug-laden (collinearity sign-flip) |
| learned value: tree, MC-outcome target | state-value | 0.427 | global AUC 0.74, poor local ranking |
| learned value: tree, search-bootstrapped (pass 1) | A0GB-style target | 0.517 | best learned; imitates hand search |
| learned value: expert-iteration (pass 2) | leaves = learned value, retrain | 0.490 | no gain over pass 1 |
| action-ranking: sibling leaves, raw value | rank the moves, not the state | 0.403 | raw target failed (keys on energy_attach_done) |
| combine = hand eval + learned value (blend) | leaf = (1-l)*hand + l*value | ~0.48-0.53 | value dilutes the hand eval |

TO TRY (not yet run; the learned approach's open paths come first per the user):
- CENTERED-ADVANTAGE action objective (within-decision ranking, value - decision mean) -- the
  untried action-model variant (registry H024 re-open gate). Same sibling data, new objective.
- DEEPER SEARCH horizon (multi-ply): the decision-content diagnosis shows 1-ply is low-resolution;
  depth is what a leaf value cannot substitute for. Improves hand AND learned leaves.
- OPPONENT-MODELLED DETERMINIZATION (H022): feed real opponent decks (from downloaded replays) into
  search_begin instead of the self-mirror. Data now available (tools/fetch_episodes.py).
- BEHAVIOR CLONING / IL from Kaggle's daily top-episode export (strong agents) -- the data Kaggle is
  providing for exactly this; pairs with the action/policy model.
- CONTRASTIVE, magnitude-aware card embeddings (draw-2 != draw-7) as the value/policy input.
- BELIEF / opponent model for determinization; neuro-symbolic / pseudo-linguistic heuristics (idea 4).
DEPRIORITIZED: a more complex deck (the diagnosis shows this deck is not the bottleneck).

## Deep-research synthesis + way forward (2026-06-18; full report: research/deep-research-report-2026-06-18.md)

An external deep-research pass reviewed the repo + the literature and AGREES with our diagnosis.
Its ranked priorities and the cautions that change how we build:

1. SIBLING-ACTION RANKING (its #1, = our H024). CRITICAL CAUTION it adds, and the reason our
   attempt 1 failed: LABEL CIRCULARITY. A ranker trained only on 1-ply SEARCH VALUES inherits the
   1-ply search's blindness, so it cannot exceed it. Fix: train on OUTCOME-aware labels with a
   WITHIN-DECISION pairwise/listwise objective (Bradley-Terry / contextual-preference / InfoNCE),
   not absolute value regression. Validate ONLY by head-to-head vs agent_search at n=800-1600 with
   Wilson CIs + a second seed. (This is exactly the user's "label positions/actions good/bad by the
   OUTCOME, naive-correlation" idea: outcome is the non-circular label.) Refs: Bertram 2023/2024.
2. BELIEF-CONDITIONED DETERMINIZATION (= H022). We now HAVE the data: 37 replays downloaded
   (tools/fetch_episodes.py), each exposing both players' decks. Build a replay ETL -> deck/meta
   priors -> seed search_begin with realistic opponent hidden states. Validate first on replay
   hidden-card likelihood (held-out), then head-to-head. Refs: Cowling 2012, Dockhorn (Hearthstone).
3. SEARCH-BUDGET SWEEP (cheap, high-info): N_DETERM {4,8,16,32}, rollout {aggressive, weak/random,
   default, heuristic-guided}, horizon {1-ply+reply, deeper}. Weak rollout may beat the aggressive
   one (Cowling/Gelly-Silver). 
4. REPRESENTATION: magnitude-aware + contrastive/decision-supervised card embeddings; judge on
   GENERALIZATION + sibling ranking, not in-sample AUC. Then expert iteration AROUND THE ACTION
   model (not state value). Full RL / ReBeL / Student-of-Games is the long-term frontier, premature now.

USER DIRECTIONS folded in: (a) outcome-based labeling = the non-circular target for #1; (b) DECK
RANDOMIZATION (randomize our deck AND the opponent deck) to prevent overfitting of the complex
models + find a policy robust ACROSS matchups -- pair this with evaluating against the REAL
opponent decks extracted from replays (re-simulate real matchups). (c) ladder note: sub_search /
sub_combine sitting below the first submission is within early-ladder noise; deck randomization +
matchup-robust evaluation is the principled guard against the overfitting the user suspects.

WAY FORWARD (build order, supersedes the to-try list above where they overlap):
1. Action ranker, OUTCOME-aware + within-decision pairwise/listwise objective, on the grouped
   candidate-action data; head-to-head validation. (Fixes attempt-1's circularity.)
2. Replay ETL: extract opponent decks + outcomes from the 37 replays -> deck/meta table + an
   outcome-labeled position/action dataset (real-ladder, not just self-play). KEY non-circular
   signal: the WINNING side's move choices in top-agent replays (onechan1 etc., score ~1300) are a
   STRONGER policy than our 1-ply search, so IMITATION-LEARNING a policy/ranker from strong winners'
   trajectories can exceed the hand search (unlike cloning our own 1-ply values, which only matches
   it). This is the high-value use of the replays + the report's IL path. (Verify the replay step
   obs exposes select.option + current so features and the chosen option are recoverable.)
3. Belief-conditioned determinization seeded from those decks (H022).
4. Deck-randomized self-play + matchup-robust evaluation (our deck x opponent deck grid).
5. Search-budget sweep; then magnitude-aware/contrastive representation; then expert iteration.

## Ideas to keep (from the user, do not lose these)

1. BAKE THE OBVIOUS HEURISTICS INTO EVERY AGENT (including search_v). Some rules are clean enough
   that there is no point learning or overriding them: if a listed attack KOs the opponent, take
   it; if "go first?" is offered, say yes. The learned value should INHERIT these, not relearn or
   contradict them. ("Attach to an energy-short active" is borderline and could be learned.)
   -> near-term: floor the search policy with these so the value only decides the genuinely open
   choices. This is part of "combine them," and it is cheap.

2. CONTINUOUS, MAGNITUDE-AWARE CARD EMBEDDINGS (not binary tags). The functional tags collapse
   magnitude: "draw 2" and "draw 7" are both `draw`. Represent each card as a soft, multi-class
   loading across all functional classes (softmax / multiple-regression style) WITH magnitudes,
   instead of a 0/1 tag set. These loadings are TRAINABLE and refinable from game outcomes; the
   current tags are only "default starts." (The other model independently flagged the same
   collapse: tags drop draw-2-vs-7, optional costs, once-per-turn, target legality, combo text.)

3. REFINE THE WEIGHTS/EMBEDDINGS WITH THE GAME ITSELF (loss / RL). The initial weights are a
   start; use outcomes + search to update the representation toward what actually wins. Train the
   value on SEARCH-improved targets (candidate-leaf / advantage / ranking over siblings), not only
   raw Monte-Carlo outcomes of weak self-play. Expert iteration (AlphaZero family): self-play with
   the current best agent -> better targets -> retrain -> repeat, for many hours.

4. PSEUDO-LINGUISTIC / LEARNED INTERPRETABLE HEURISTICS (the big speculative one; the user may
   hand this to a dedicated model to explore). Let the learner compose its OWN rules from a
   vocabulary of CONDITIONS (scenario embeddings: "it draws", "I'm behind on prizes") and ACTIONS
   (verbs/operators: "draw more", "attack first"). The embedding classifies what a scenario MEANS;
   learned statistics decide which heuristic applies there. Neuro-symbolic / program-induction /
   LLM-as-rule-composer flavor: human-readable rules conditioned on a learned state meaning,
   refined by what statistically wins.
   - HONEST FRAMING (user): this may help a HUMAN understand what the bot is doing more than it
     helps the bot WIN -- like the structured-transform-repo lesson where the value was diagnosis,
     not accuracy -- UNLESS we can leverage the embeddings well. Still a layer the user is very
     interested in.
   - WHY IT MIGHT WORK HERE, where past "fit a random equation to the problem" attempts FAILED:
     random equation/rule search is combinatorial and you are shooting in the dark (no structure
     to learn or exploit from). Here we are NOT blind: humans recognize good heuristics instantly
     and can agree on a RANKED HIERARCHY with exceptions. So seed a FLOOR of human-agreed, ranked
     rules and let the system learn (a) which rule applies in which learned scenario and (b) refine
     the ranking/exceptions, instead of inventing rules from scratch.
   - STARTER HIERARCHY (human-agreed, to encode + then refine): (1) KO the opponent -- UNLESS it
     leaves you open to being KO'd back (so "avoid being KO'd" is a real competing rule); (2) go
     first / draw when offered (more information is better); ... The learnable part is the
     ordering, the exceptions, and the scenario->rule mapping; the floor is hand-given.

5. ENSEMBLE / COMBINE the heuristic + hand eval + learned value + search. Weight each option by
   whether the evaluators AGREE; when they agree, trust the move; when they disagree, search
   deeper. Leaf eval = hand_eval (sharp local ranking) + lambda * learned_value (global judgment).

6. DO NOT TIE THE VALUE TO WIN-RATE ALONE; build a causal chain of intermediate correlates and
   ask several questions (user, 2026-06-17). A single target ("P(win) from this position") is not
   enough. The user's past work (gene-expression / equation-fitting) failed when everything was
   tied to one outcome: the combinatoric space is too large and the correlations are not
   meaningful without intermediate structure. Reference: stable GRN (gene regulatory network)
   inference, which imposes a causal-chain order.
   - The features encode two things at once: (a) what options we have NOW, (b) how those factors
     LEAD to outcomes. Example: "have I drawn my hand" has near-zero direct effect on winning (you
     usually draw your hand in this game), but the sequence/combination of plays leads to more
     options later.
   - Ask several questions, not one: how many options do I have; how many does the opponent have;
     which of my options lead to MORE options (now and in the future); which lead to states whose
     vectors correlate with winning. "Number of options/features available" is itself a value
     signal; chains that increase our options are good. With continuous feature values you can sum.
   - First-order correlates (the bootstrap value surfaced prize_lead as the top feature) ->
     second-order correlates -> a layered causal chain (the user's "backpropagation" framing).
   - Human positional rules that are not direct win-rate: more cards played from hand is usually
     good; a stronger active is usually better; do not over-extend the bench (sweep risk); a
     lower-retreat-cost active gives future switching flexibility, conditional on having a switch
     target or it being a buffer play.
   - Possibly SEPARATE / META-LEVEL embeddings: one space for "what correlates with winning",
     another for option/structure; clusters may emerge without an explicit win-rate target.
   - Point: choose the questions and intermediate signals to make the learner's job easier, even
     though a net/RL could in principle discover them with enough data. Connects to idea 4
     (linguistic heuristics) and the research's auxiliary-task / advantage-over-siblings framing.
   - Buildable piece: add auxiliary signals/features (my option count, opponent option count,
     one-ply "leads to more options" delta) and consider auxiliary value heads / multi-target
     training instead of a single win/loss target.

## Near-term concrete actions (cheap, before/around the research)

- Combine v1: floor search with the obvious heuristics (idea 1) AND blend the leaf eval
  hand_eval + lambda*value (idea 5). Measure at n>=800 with Wilson CIs vs hand search AND heuristic.
- Belief/opponent model for determinization (other model's highest-leverage pre-RL point): infer
  opponent deck/hand/prizes from revealed cards + meta priors, feed realistic hidden states to
  search_begin, instead of self-mirror sampling.
- Candidate-leaf data product: log, per root decision, every candidate first action + its leaf
  features + outcome, to train a sibling-ranking / advantage model (idea 3) rather than a state
  classifier. This is the data the expert-iteration loop needs.

## Deep-research questions (the plan for the research pass)

RQ1. Representation learning for card games: how are cards/states embedded for value/policy
  learning? Continuous multi-class card embeddings vs hand tags; magnitude-aware encodings
  (draw-2 vs draw-7); factorized / set / graph encoders; outcome-supervised vs co-occurrence
  embeddings (co-occurrence only rediscovers deck membership).
RQ2. Value functions FOR SEARCH (local sibling ranking, not global classification): what value
  forms and training targets give good leaf ranking in MCTS/expectimax? Learning-to-rank vs
  regression; calibration; monotonicity constraints; smooth models vs piecewise-constant trees;
  search-bootstrapped (TD / AlphaZero) targets vs Monte-Carlo.
RQ3. Imperfect-information card-game agents: ISMCTS / determinized search, expert iteration (ExIt),
  AlphaZero adaptations for hidden info + stochasticity (Hearthstone, MTG, Stratego, DeepMind
  Player of Games / ReBeL). What works, and data/compute requirements.
RQ4. Was AlphaZero's success specific to deep, stable, perfect-information games at massive scale?
  How much self-play/compute is needed; diminishing returns; what makes a game "learnable" this
  way; does it transfer to short stochastic imperfect-info card games (i.e., is "just run it for
  12 hours" realistic here, and under what conditions).
RQ5. Neuro-symbolic / learned interpretable policies (idea 4): differentiable rule learning,
  decision-rule/list induction, program synthesis for game policies, LLM-generated or
  LLM-composed heuristics, neuro-symbolic RL. Feasibility of rules conditioned on a learned
  scenario embedding.
RQ6. Practical combination: ensembling a hand eval + learned value + policy prior at search
  leaves; value-as-prior PUCT; how to weight/trust options when evaluators agree vs disagree.

## Research findings (2026-06-17 deep-research pass; 23/25 claims verified 2-3 votes)

All sources are peer-reviewed / competition reports; NONE is on Pokemon TCG, so treat as
strongly-motivated transfers, not proven-on-our-game. Confidence noted per item.

- [HIGH, direct hit on our symptom] Monte-Carlo-outcome value targets CAUSE "good global AUC but
  poor local sibling ranking": the self-play outcome target bakes in exploration, is biased, and
  fails to rank greedy continuations. FIX = search-bootstrapped, off-policy targets: A0GB (back up
  the value found by descending the MCTS tree GREEDILY by visit count to a leaf/terminal), or
  soft-Z / A0C. All three trained faster and stronger than the MC target on Connect-Four /
  Breakthrough. (Willemsen et al. 2022.) This is the single most-supported fix for our exact bug.
- [HIGH] Imperfect-info workhorse = ENSEMBLE determinized MCTS (a tree per determinization,
  aggregate by total visit count). Spend budget on MORE determinizations (~20-100), not deeper
  per-tree (per-tree sweet spot 100-1000 sims). We average N_DETERM=4 -> go higher. (Cowling et
  al. 2012, MtG.)
- [HIGH, counterintuitive, bears on my recent change] A WEAKER, partly-random rollout policy makes
  a STRONGER MCTS player than rollouts driven by the strong expert policy (expert rollouts rigidly
  fix outcomes and bias the stats). Same as the classic Go result. So my "aggressive opponent
  rollout" may not be optimal; TEST weaker/random rollouts. (Cowling 2012; Gelly & Silver 2007.)
- [HIGH] Move decomposition: turn a compound multi-card move into a binary play-or-not tree per
  card so MCTS accrues partial-decision stats; big speed+strength win; without it, extra budget is
  wasted on a huge branching factor. Relevant if/when our action space has subset choices.
- [HIGH] Representation: judge embeddings on GENERALIZATION / local discrimination, NOT in-sample
  fit (~67% seen accuracy was flat across representations; unseen ranged 24%->43%). Continuous,
  magnitude-aware, multi-attribute card vectors (numeric+categorical+card-TEXT embedding) beat
  binary role tags for generalization. This confirms the draw-2 != draw-7 idea. (Bertram et al.
  2024, MtG draft.)
- [HIGH] TRAIN embeddings by outcome/decision-supervised CONTRASTIVE learning (triplet / InfoNCE),
  NOT co-occurrence (which only rediscovers deck membership, as suspected). (Bertram 2024.)
- [HIGH] Scale/RL viability: pure end-to-end RL (ByteRL, optimistic smooth fictitious play, NO
  search) was SOTA on a short stochastic imperfect-info card game (Legends of Code & Magic) but is
  EXPLOITABLE by a tailored opponent. Historical arc: forward-search agents won early LOCM; RL/NN
  overtook later. So SEARCH is the stronger near-term bet; heavy RL pays off later. (Xi & Zhang
  2023; Kowalski & Miernik 2023; Krupnik 2024.)
- [HIGH, frontier, heavy] Principled hidden-info search with a learned value: ReBeL (search over
  public belief states, value/policy trained by self-play + CFR, learned value as the search leaf
  eval) and Player/Student of Games (unified search + self-play + GT-CFR). The template for
  marrying our forward model with a learned value under hidden info, but expensive and 2-player
  zero-sum. (Brown et al. 2020; Schmid et al. 2021/2023.)

REFUTED (do not rely on): "training on internal tree nodes cuts self-play games needed" (1-2);
"ByteRL beat champions by >20%" (0-3).
UNANSWERED (needs a dedicated follow-up pass): the neuro-symbolic / learned-interpretable-heuristic
direction (idea 4, the pseudo-linguistic rules) - NO verified claim surfaced; it remains
speculative. Also unquantified: the self-play/compute budget for AlphaZero/ReBeL on a small-deck
game, and the exact 3-way (hand+value+prior) ensemble recipe.

## Implied plan (priority order, UPDATED 2026-06-17 after the decision-content diagnosis)

DONE (no gain on this deck, all CIs span 0.5): combine v1 (blend); value-target fix (bootstrap
passes 1+2). The state-value path has hit its 1-ply ceiling here. CURRENT TARGETS:

1. ACTION OBJECTIVE (the untried thing the diagnosis points at): stop ranking STATES, learn to
   rank the MOVES out of a state. Log candidate-action data per decision (root features, each
   candidate's leaf features, the per-option search value, the chosen move, outcome) and train
   with a WITHIN-DECISION ranking objective (softmax/pairwise over the options), not absolute
   state-value regression. Measure search using the ranker vs hand search AND heuristic, n>=800.
2. OPTION / "future-options" SIGNAL as first-class (user idea 6): add my option count, opponent
   option count, one-ply option-count delta; these have a per-move gradient win-rate lacks.
3. DEEPER SEARCH HORIZON (complement): 1-ply makes most decisions low-impact; a longer horizon lets
   the value compound across a move sequence (the causal chain). Test multi-ply / longer rollout.
4. REPRESENTATION: continuous magnitude-aware card features + outcome-supervised contrastive
   embeddings; evaluate on generalization, not in-sample AUC. (Bertram 2024.)
5. EXPERT ITERATION loop with the ACTION model (not just state value). ReBeL / Player-of-Games are
   the principled frontier if we go deep.
6. FOLLOW-UP RESEARCH: the neuro-symbolic / pseudo-linguistic learned-heuristics idea (unanswered).
NOTE: complex deck is DEPRIORITIZED (the diagnosis shows the simple deck is not the bottleneck).

## Note for the next model

The other model's MODEL_COMMUNICATION "Methodology cautions" are good and aligned (sibling-ranking
vs global value, candidate-leaf data, magnitude collapse, outcome-supervised embeddings, belief
model first, evaluate RL by head-to-head not AUC). Treat them as inputs, re-verify before relying.
Full cited findings live in the deep-research output; key papers: Willemsen 2022 (value targets),
Cowling 2012 (ensemble determinization + weak rollouts), Bertram 2024 (card embeddings), Brown
2020 (ReBeL), Schmid 2021/2023 (Player/Student of Games), Xi & Zhang 2023 (ByteRL/LOCM).

## Live data sources / replay access (note, 2026-06-17)

- A known ladder episode: submissionId 53781334, episodeId 80408508; seed id 486837364.
  Kaggle leaderboard: kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard?submissionId=53781334&episodeId=80408508
  HEROZ visualizer (renders any episode by id): https://ptcgvis.heroz.jp/Visualizer/Replay/80408508/0
- URL pattern to view a replay (user found 2026-06-17): the share link comes as
  `kaggle.com/competitions/pokemon-tcg-ai-battle/submissions#?submissionId=<S>&episodeId=<E>` which
  is bugged; replace `submissions#` with `leaderboard` and it loads:
  `kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard?submissionId=<S>&episodeId=<E>`.
  Examples: S=53781334 E=80408508; S=53794404 E=80411394 (our replays).
- REPLAY DOWNLOAD WORKS, NO AUTH (confirmed 2026-06-17). The full replay JSON is on the public CDN
  `https://www.kaggleusercontent.com/episodes/<EpisodeId>.json`. tools/fetch_episodes.py downloads
  by id into data/external/replays/. A replay contains BOTH players' DECKS (the step-1 action is
  the 60-card list), every step's observations + actions, statuses, rewards, and the game seed.
  Real-ladder replays give opponent decks/policies -> use for opponent-modelled determinization
  (H022: seed search_begin's opponent_deck from real opponents, not the self-mirror) and meta deck
  analysis. NOT yet built: bulk episode-id DISCOVERY (the EpisodeService ListEpisodes endpoint needs
  Kaggle session auth, or scrape the leaderboard); for now pass ids from the leaderboard URL.
  Respect Kaggle rate limits / ToS.
