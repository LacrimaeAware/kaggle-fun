---
title: "Pokemon TCG Agent Roadblock Diagnosis and Clean Reset Plan"
date: "2026-06-18"
author: "ChatGPT"
---

# Pokemon TCG Agent Roadblock Diagnosis and Clean Reset Plan

## Purpose

This document converts the prior diagnostic response into a reusable project note. It focuses on the current Pokemon TCG Kaggle agent research project, the repeated roadblock around learned values and search, and a clean next methodology.

The central point is that the project is not merely repeating one bad idea. It is repeatedly implementing a nearby, easier surrogate for the stated research question, getting respectable prediction metrics, and then discovering that the surrogate does not improve decisions.

One public-state caveat: GitHub showed `main` and an active Claude branch with different update times when checked. I can assess the visible pushed code and documents, but not any local work that has not reached GitHub.

Relevant public sources referenced while forming this diagnosis include:

- [`docs/RESEARCH.md`](https://raw.githubusercontent.com/LacrimaeAware/kaggle-fun/main/pokemon-tcg-ai-battle/docs/RESEARCH.md)
- [`docs/LEARNING_PLAN.md`](https://raw.githubusercontent.com/LacrimaeAware/kaggle-fun/main/pokemon-tcg-ai-battle/docs/LEARNING_PLAN.md)
- [`docs/MODEL_COMMUNICATION.md`](https://raw.githubusercontent.com/LacrimaeAware/kaggle-fun/main/pokemon-tcg-ai-battle/docs/MODEL_COMMUNICATION.md)
- [`registry/experiments.jsonl`](https://raw.githubusercontent.com/LacrimaeAware/kaggle-fun/main/pokemon-tcg-ai-battle/registry/experiments.jsonl)
- [`registry/results.jsonl`](https://raw.githubusercontent.com/LacrimaeAware/kaggle-fun/main/pokemon-tcg-ai-battle/registry/results.jsonl)
- [`tools/datagen_actions.py`](https://raw.githubusercontent.com/LacrimaeAware/kaggle-fun/main/pokemon-tcg-ai-battle/tools/datagen_actions.py)
- [`agent/search.py`](https://raw.githubusercontent.com/LacrimaeAware/kaggle-fun/main/pokemon-tcg-ai-battle/agent/search.py)
- [`agent/features.py`](https://raw.githubusercontent.com/LacrimaeAware/kaggle-fun/main/pokemon-tcg-ai-battle/agent/features.py)

---

## What has genuinely been accomplished

A substantial amount of the foundation is real:

- The engine integration and forward model work.
- The agent is legal and crash-resistant.
- One-ply forward search with the hand-written leaf evaluation is the strongest measured policy.
- The data-generation, feature, value-model export, search integration, and arena-evaluation loop all exist.
- Several genuine implementation failures were found and corrected: train/test leakage, target saturation, inconsistent score scales, improper leaf distributions, submission loading, and determinization issues.
- Search-bootstrapped training repaired the learned agent's severe loss, moving it from approximately 0.427 against the heuristic to statistical parity around 0.50.

So this was not wasted work. The first bootstrap pass demonstrated something useful: a model trained to imitate the search's evaluation could recover most of the hand evaluator's playing strength. But it did not create a stronger decision signal.

Pass two raised Pearson correlation with the generated search targets from approximately 0.825 to approximately 0.904 while play remained at parity. That is strong evidence that fitting the current target more accurately is not the current bottleneck.

---

## The precise repeated failure pattern

I would call the pattern **objective slippage**.

The research document correctly states the hard question:

> Rank the legal moves arising from the same position.

But the implementation has generally answered an easier question:

> Assign an absolute scalar value to the resulting state.

Those are related, but they are not equivalent.

### 1. The "action-ranking" experiment is not yet a true action-ranking experiment

Registry hypothesis H024 specifies:

- candidate actions grouped by decision;
- a per-decision-centered advantage target;
- evaluation against hand search and the heuristic;
- at least 800 games.

Experiment E013 is marked done, but its own description says:

- raw value target;
- not centered;
- generic gradient-boosted regression;
- 400-game evaluation configuration.

The training program remains an ordinary classifier/regressor over feature vectors; it has no query-group-aware pairwise or listwise ranking loss. The candidate-data generator records each leaf's 47 state features and the hand evaluator's absolute score. Therefore, as publicly implemented, E013 is still fundamentally a **leaf-state regression/distillation experiment**, not the full H024 test described by the hypothesis.

That is probably the single most important finding from this review.

### 2. The model does not explicitly receive the action

The candidate dataset contains:

```text
definition: current candidate-data shape

leaf features
absolute hand-search value
decision group
turn
seat
```

It does not explicitly contain:

```text
definition: missing action-ranking inputs

root-state features
action identity or action attributes
root-to-leaf feature changes
uncertainty across determinizations
opponent-belief context
```

Consequently, even though the rows are grouped by decision, the learner is still being asked to infer action quality from the resulting state alone. It is not learning a proper `Q(root, action)` or `advantage(root, action)` function.

A model intended to distinguish "play this draw card," "attach this energy," and "attack now" should see both the starting circumstances and what the action changed. Merely presenting the leaf makes different causal routes that produce superficially similar leaves indistinguishable.

### 3. The target cannot reliably surpass its teacher

`datagen_actions.py` scores candidates with the current hand evaluator and plays the highest-scored hand-eval action while producing data. That is legitimate **policy/value distillation**. It could produce a cheaper or smoother approximation to the hand search.

But the project's acceptance gate asks the resulting learner to **beat the hand-search teacher**. With the teacher's score as the sole supervisory signal, there is no systematic information telling the learner where the teacher is wrong. It might occasionally generalize better through smoothing, but that is not a defensible primary mechanism for exceeding the teacher.

To surpass hand search, the target needs an independent or stronger source of information, such as:

- a longer-horizon search;
- multiple rollouts from each candidate;
- common hidden-state samples across candidates;
- eventual counterfactual outcomes;
- ladder or stronger-opponent trajectories;
- a mixture of deeper-search return, terminal result, and hand score.

The hand score can remain an auxiliary target or safety prior. It should not be the only authority.

### 4. The bootstrapping loop largely learned its own fixed point

Pass one learned the hand-search target and recovered parity. Pass two placed that learned value at the leaves, generated targets from it, and trained another model to fit those targets. Pearson improved, but gameplay did not.

That result is unsurprising in retrospect: the loop improved agreement with its own evaluator without introducing a clearly stronger policy-improvement operator. In AlphaZero-like expert iteration, the search policy must become a meaningfully improved teacher relative to the network. Here, one-ply search over a value that already has poor sibling resolution may simply reproduce and sharpen the same ordering.

That does not invalidate expert iteration. It means the current loop is missing the element that makes the "expert" stronger.

### 5. Low-impact decisions dominate the dataset

The research document's diagnosis is plausible and important: the policies disagree on roughly 45-65% of real choices, yet their outcomes remain near parity. That suggests many disagreements occur at decisions with little causal effect on winning. The project currently logs every qualifying multi-option decision approximately equally. It does not calculate or weight **decision criticality** - the spread in expected return between the best and worst candidates.

A large dataset of mostly inconsequential decisions can yield excellent global fit while teaching almost nothing about the few swing decisions that determine games.

This is not solved merely by collecting more rows. The data needs to identify:

- where candidate values meaningfully diverge;
- where policies disagree;
- where a different action changes attack availability, survival, prizes, resource flow, or future options;
- where the preferred move is stable across hidden-state samples;
- where the selected move produces measurable regret.

---

## What exists only as a plan

The project language can make several components sound further along than they are.

### Learned embeddings

The current state encoder uses designed multi-hot card tags and numeric features. The documents refer to this as a "designed embedding," but the proposed continuous, magnitude-aware, learned card embedding is not yet the operative representation-learning system. The current code still primarily consumes engineered card/state features.

So "use the embedding vectors" keeps failing to happen partly because there is not yet a well-defined learned embedding artifact with:

- a precise training objective;
- a frozen dimensionality and schema;
- an integration point;
- an ablation against the designed features;
- an evaluation based on action discrimination.

Trying to learn embeddings before fixing the action target would likely just give a more sophisticated representation of the wrong objective.

### Future-options objectives

The living document proposes option count, future-option deltas, and auxiliary heads. Those are not yet first-class training targets in the public pipeline. Current features include legal affordances such as attack, draw, tutor, attach, and evolve availability, but this is not the same as explicitly estimating how an action changes the next turn's option structure.

### Belief-modelled determinization

Current search still assumes the same deck for the opponent in local same-deck experiments. The real-replay pathway and archetype-prior idea are documented, but a learned/meta opponent belief model is not yet driving `search_begin`.

### Search methodology experiments

The code still uses four determinizations and an aggressive rollout policy. The research synthesis specifically recommends testing a larger determinization budget and weaker or partially random rollouts, but those tests are not represented in the current experiment registry.

### Pseudo-linguistic / neuro-symbolic policy

This remains a research concept. There is no implemented vocabulary, rule representation, rule-selection model, or outcome evaluation yet. The living document is honest that its prior research pass did not find sufficient direct evidence for the concept.

---

## The clean reset the project needs

The next experiment should be narrowly defined as:

> **Can an action-conditioned model rank consequential sibling actions better than the existing hand evaluator?**

Not "train another value."

Not "run another RL pass."

Not "add embeddings."

Not "fit search targets better."

---

## Phase 1: Salvage the existing E013 data

Before generating anything new:

1. Center the existing target within each `gid`:

   $$
   A_{g,i} = y_{g,i} - \operatorname{mean}_{j \in g}(y_{g,j})
   $$

2. Train the same model on this centered advantage.
3. Evaluate only within decisions:
   - pairwise accuracy;
   - top-one agreement;
   - rank correlation within each decision;
   - regret of the selected candidate relative to the teacher's best candidate.
4. Stratify by candidate-score spread.

This is still teacher imitation, not a route to surpassing hand search. Its purpose is to answer one clean engineering question:

> Does a decision-relative target materially improve local ranking over raw absolute leaf regression?

It can reuse the existing dataset and isolate one variable.

---

## Phase 2: Build the proper action dataset

Each decision should record:

```text
decision_id
game_id
root features
option/action descriptor
leaf features
leaf minus root feature delta
values across K shared determinizations
mean candidate return
variance/uncertainty
hand-eval score
eventual game result
forced/open-decision indicator
policy-disagreement indicators
```

The same determinization samples and chance seeds should be used across candidate actions within a decision wherever the engine allows it. Otherwise, candidate A may look better than candidate B merely because it received a more favorable hidden world.

The action descriptor should include at least:

- option type;
- card or attack ID;
- target class;
- whether it attacks, attaches, evolves, retreats, draws, tutors, or ends the turn;
- immediate KO, survival, or resource effects.

That is the point where continuous card embeddings can become operational: they encode the card/action entities inside `Q(root, action)`, rather than floating around as a detached representation project.

---

## Phase 3: Produce a teacher that can exceed hand search

For a mined subset of decisions, obtain stronger labels using additional computation:

- more determinizations;
- longer opponent response;
- multi-ply or longer rollout;
- multiple stochastic continuations;
- eventual outcome where it is informative.

Then define a target such as:

$$
z_{g,i} = \alpha \cdot \text{deeper-search return}
+ \beta \cdot \text{terminal/outcome estimate}
+ \gamma \cdot \text{hand score}
$$

The exact coefficients should be predeclared and ablated. More importantly, the target must contain some information not already present in the hand evaluator.

Use the candidate spread,

$$
C_g = \max_i z_{g,i} - \min_i z_{g,i},
$$

as a criticality measure. Do not necessarily discard low-criticality decisions, but down-weight them or report results separately. The model should first prove it can resolve the high-impact tail.

---

## Phase 4: Start with the simplest credible model

Do not begin with a large neural system.

First compare:

1. centered-advantage gradient boosting;
2. pairwise logistic ranking;
3. a small action-conditioned multilayer perceptron over `[root, action, delta, leaf]`.

This makes the representation question empirical. If the small neural model with card embeddings improves held-out high-criticality ranking while the tree does not, the embedding pathway has earned further investment. If it does not, the problem is probably still in the targets or search teacher.

---

## Phase 5: Integrate conservatively

The first deployment should not replace the strongest search evaluator.

Use the learned action model as:

- a candidate-ordering prior;
- a tie-breaker when hand scores are close;
- a pruning mechanism only after high recall is demonstrated;
- a trigger for additional search when learned and hand evaluators disagree.

Keep the clean forced rules - take an unambiguous lethal and make required setup decisions - outside the learned system.

This design makes failure informative. If the ranker is poor, the agent falls back to the known working policy rather than collapsing below parity again.

---

## Phase 6: Use gameplay as the final gate

Global Pearson and AUC should become diagnostics, not milestone metrics.

The main offline metrics should be:

- within-decision top-one accuracy;
- pairwise preference accuracy;
- selected-action regret;
- performance on high-criticality decisions;
- stability across determinizations;
- calibration of uncertainty/disagreement.

Then run a frozen head-to-head evaluation against both:

- `agent_search`;
- the board-aware heuristic.

Use seat swaps, fixed code and weight hashes, and the repository's predeclared confidence-interval discipline. The existing plan already says hypotheses, practical noise bands, and sample sizes should be fixed before the run.

---

## A separate near-term route for immediate gameplay gains

The action-model work is the correct learning route, but it is not necessarily the cheapest immediate strength improvement.

The strongest existing agent can be improved independently by testing:

1. a time-budget curve for 4, 8, 16, and - only if runtime permits - more determinizations;
2. default, aggressive, weak, and partially random rollout policies;
3. realistic opponent-deck determinizations from downloaded ladder replays;
4. extra search only on high-branching or evaluator-disagreement decisions.

This branch should use the hand evaluator throughout. It would determine whether search mechanics themselves can produce a stronger teacher before training another learner.

Before expanding decks or belief modelling, the public handoff also flags several correctness issues:

- Team Rocket Energy is represented as a universal wildcard despite having narrower semantics;
- the public Stadium may be duplicated in hidden-zone sampling;
- optional selections may be forced rather than declined;
- replay-derived statistics have attribution/parser concerns.

Those are potential confounders and should be closed or explicitly scoped out before relying on affected decks or replay data.

---

## Why the project feels more chaotic than it is

The experiment registry is close to being useful, but it has one significant control-plane failure:

- E013 is marked `done`.
- The results registry ends at R007, for E012.
- There is no canonical result recorded for E013.
- E013 also does not satisfy H024's documented methodology.

That means "done" currently conflates at least three states:

1. code/data generation completed;
2. model trained;
3. hypothesis properly evaluated.

Those need separate statuses. Otherwise every new session sees "action ranking done," assumes the idea was tested, and moves to the next plan.

I would make the status vocabulary:

```text
specified
implemented
data-generated
trained
offline-evaluated
arena-evaluated
accepted / inconclusive / refuted
```

A compact `CURRENT.md` should contain only:

```text
active branch and commit
current strongest agent
current live weights and their training dataset
active hypothesis
exact command being run
latest completed result
next acceptance gate
known blockers
```

The living research document can remain expansive, but it should not be used as the execution controller.

---

## Bottom-line diagnosis

The project has not demonstrated that learned action ranking fails.

It has demonstrated that:

- global Monte Carlo state value is inadequate;
- a tree with good global AUC can be a poor local leaf evaluator;
- distilling hand-search targets can recover parity;
- fitting self-generated search values more closely does not automatically improve play;
- blending a weaker learned evaluator into the hand evaluator dilutes the stronger signal;
- prediction quality and gameplay quality have decisively separated.

The current action experiment is a useful scaffold, but **the full stated action-ranking methodology has not actually been executed**.

The clean next move is not another broad plan. It is one properly controlled H024-v2 experiment with root-action-delta inputs, within-decision targets, critical-decision analysis, and a teacher containing information beyond the current hand evaluation.

Continuous embeddings, future-option concepts, expert iteration, and eventually pseudo-linguistic macro-actions can all attach to that pipeline. Until the action-level data and target are correct, those components will continue to be discussed without having a stable place to operate.
