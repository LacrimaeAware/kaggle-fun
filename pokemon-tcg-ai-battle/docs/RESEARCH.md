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
- We are NOT yet doing reinforcement learning. The value is supervised (predict win/loss from
  features). The agreed direction is to move to RL / expert-iteration.

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

4. PSEUDO-LINGUISTIC / LEARNED INTERPRETABLE HEURISTICS (the big speculative one). Let the learner
   compose its OWN rules from a vocabulary of CONDITIONS (scenario embeddings: "it draws", "I'm
   behind on prizes") and ACTIONS (verbs/operators: "draw more", "attack first"). The embedding
   classifies what a scenario MEANS; learned statistics decide which heuristic applies there. A
   neuro-symbolic / program-induction / LLM-as-rule-composer flavor: discover human-readable rules
   conditioned on a learned state meaning, refined by what statistically wins. Likely hard to
   build; capture as a research direction, do not commit to it blind.

5. ENSEMBLE / COMBINE the heuristic + hand eval + learned value + search. Weight each option by
   whether the evaluators AGREE; when they agree, trust the move; when they disagree, search
   deeper. Leaf eval = hand_eval (sharp local ranking) + lambda * learned_value (global judgment).

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

## Implied plan (priority order, evidence-backed)

1. COMBINE v1 (cheap, immediate, repeatedly requested): floor search with the clean heuristics
   (always take a listed lethal; go first), and blend the leaf eval = hand_eval + lambda*value on
   ONE [0,1] scale. Measure n>=800 with CIs vs hand search AND heuristic. (Our diagnosis + user.)
2. VALUE-TARGET FIX (highest-leverage for the symptom): retrain the value on SEARCH-BOOTSTRAPPED
   targets (greedy-backup / the search's own leaf value), and/or train a candidate-leaf ADVANTAGE
   / RANKING model over sibling actions instead of a global state classifier. (Willemsen 2022;
   sibling-ranking framing.)
3. SEARCH KNOBS (cheap): raise N_DETERM (4 -> 20-40); TEST a weaker/random rollout vs the current
   aggressive one. (Cowling 2012.)
4. REPRESENTATION: continuous magnitude-aware card features + outcome-supervised contrastive
   embeddings; evaluate on generalization, not in-sample AUC. (Bertram 2024.)
5. EXPERT ITERATION loop: self-play with current-best -> search-bootstrapped targets -> retrain ->
   repeat. Consider game-theoretic self-play (fictitious play) as an alternative to search-target
   RL. ReBeL / Player-of-Games are the principled frontier if we go deep.
6. FOLLOW-UP RESEARCH: the neuro-symbolic / pseudo-linguistic learned-heuristics idea (unanswered).

## Note for the next model

The other model's MODEL_COMMUNICATION "Methodology cautions" are good and aligned (sibling-ranking
vs global value, candidate-leaf data, magnitude collapse, outcome-supervised embeddings, belief
model first, evaluate RL by head-to-head not AUC). Treat them as inputs, re-verify before relying.
Full cited findings live in the deep-research output; key papers: Willemsen 2022 (value targets),
Cowling 2012 (ensemble determinization + weak rollouts), Bertram 2024 (card embeddings), Brown
2020 (ReBeL), Schmid 2021/2023 (Player/Student of Games), Xi & Zhang 2023 (ByteRL/LOCM).
