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

## Note for the next model

The other model's MODEL_COMMUNICATION "Methodology cautions" are good and aligned (sibling-ranking
vs global value, candidate-leaf data, magnitude collapse, outcome-supervised embeddings, belief
model first, evaluate RL by head-to-head not AUC). Treat them as inputs, re-verify before relying.
