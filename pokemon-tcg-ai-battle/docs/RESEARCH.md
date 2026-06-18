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
- IDEA to try: pull OTHER players' ladder replays as a data source. Kaggle exposes episode JSONs
  via its API/endpoints by episode id (we already have tools/parse_replay.py + data/external/replays
  for local replay parsing). Real-ladder replays would give opponent decks/policies for opponent
  modelling and for evaluating against the real pool, not just self-play. Verify the API path and
  terms before scraping; not yet built.
