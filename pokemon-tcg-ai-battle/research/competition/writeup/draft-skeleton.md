# Draft skeleton: Strategy Category write-up

Working title:

```text
Consequence-Aware Search and Action Ranking for Pokemon TCG
```

Working subtitle:

```text
From board-value prediction to sibling-action decisions under hidden information
```

Target length:

```text
<= 2000 words
```

## 1. Opening thesis

Word budget:

```text
150-200
```

One-liner:

> Pokemon TCG agents need to reason about dynamic action consequences, not only static board strength.

Draft points:

- The Simulation agent must choose legal actions under hidden information, deck constraints, and future tempo.
- Early value models found predictive signal but did not consistently improve decisions.
- The project pivoted toward ranking legal sibling actions from the same decision.

## 2. Deck concept and strategic objective

Word budget:

```text
250-350
```

One-liner:

> The deck is treated as part of the agent design: it should create repeated, learnable decision patterns that the policy can exploit.

Draft points:

- Name the current submitted deck when finalized.
- Explain the deck's game plan in practical terms.
- Identify key setup cards, attackers, draw/search cards, evolution paths, and energy plan.
- Explain why the deck fits the agent: search can evaluate immediate tactics, while learned/action features should improve setup and sequencing.
- Be explicit that the deck may keep changing because the competition is only days old.

Figure candidate:

```text
Deck game-plan diagram: setup -> evolve / power attacker -> take prizes -> stabilize.
```

## 3. Agent architecture

Word budget:

```text
300-400
```

One-liner:

> The agent is layered: legal fallback, heuristic floor, forward-model search, and learned action/value components.

Draft architecture:

```text
legal options
-> fallback / safety rules
-> heuristic floor
-> forward-model search
-> learned action prior / ranker
-> final action
```

Draft points:

- Legal action space comes from the official simulator.
- Hand heuristic provides a stable floor.
- Search simulates candidate consequences.
- Learned components are intended to rank sibling actions or provide priors, not just predict global winner.
- Current strongest final agent should be described honestly at submission time.

Figure candidate:

```text
Architecture diagram connecting observations, legal options, search, learned ranker, and action choice.
```

## 4. Research arc and major hypotheses

Word budget:

```text
350-450
```

One-liner:

> Negative results became useful because they isolated the conversion gap between prediction and action choice.

Hypothesis table candidates:

| Hypothesis | Method | Result | Lesson |
|---|---|---|---|
| Hand rules can beat naive play | Heuristic vs random/first | Positive floor | Legal gameplay and obvious tactics matter |
| Forward search improves tactics | Search vs heuristic | Search became strongest validated family | Engine consequences are valuable |
| Global value can replace hand leaf eval | Learned value / blend | Weak or parity | Prediction did not equal better decisions |
| Better target fit improves play | Search-bootstrapped passes | Higher fit without clear play gain | Objective/target was not the bottleneck |
| Static imitation can learn strong play | Replay/winner ranker | Limited by option-order/static features | Need richer action semantics and deltas |
| Effects/embeddings can help action choice | Current/future action ranker | In progress | Must be wired into live decision scoring |

## 5. Key insight: rank decisions, not states

Word budget:

```text
250-350
```

One-liner:

> The unit of learning should be the decision group: root state plus candidate actions plus consequences.

Draft points:

- A board evaluator can score states globally but still fail to choose between sibling moves.
- Many decisions are low-impact; high-criticality decisions matter disproportionately.
- Correct data object:

```text
root state + action descriptor + decoded card effects + root-to-leaf delta + hidden-state assumptions
```

- Correct metrics include within-decision top-1, pairwise preference, selected-action regret, and high-criticality performance.

Figure/table candidate:

```text
Decision group example: same root, multiple legal actions, different deltas and scores.
```

## 6. Card effects, embeddings, and future learned heuristic

Word budget:

```text
250-350
```

One-liner:

> Card text features are useful only when they are connected to action choice in context.

Draft points:

- Decoding effects like draw/search/heal/energy acceleration creates semantic features.
- A decoded file alone does not affect play unless the live model consumes it.
- Learned card embeddings can capture residual card identity, but should be paired with decoded effects and state context.
- The model should learn when a card effect is worth the action cost.

Key phrasing:

```text
The goal is not to learn "Poffin is good"; it is to learn when searching two basics is worth spending the action instead of attacking, drawing, evolving, or holding resources.
```

## 7. Hidden information and robustness

Word budget:

```text
200-300
```

One-liner:

> Search quality depends on plausible hidden-state assumptions, not only leaf evaluation.

Draft points:

- Pokemon TCG contains hidden hand/deck/prize information.
- Determinization and opponent deck priors affect search.
- Replay data can support more realistic opponent/deck models.
- Robustness should be tested across seeds, seats, decks, and matchups.

## 8. Evidence and final model performance

Word budget:

```text
250-350
```

One-liner:

> The final report should separate diagnostic metrics from actual head-to-head strength.

Placeholder evidence:

- Current strongest agent family:

```text
TBD near deadline
```

- Current best head-to-head result:

```text
TBD near deadline
```

- Important negative results:

```text
global AUC/Pearson improved without reliable gameplay gain
search_v did not test card effects because it did not consume card_effects.json
hand-weighted effects can over-prioritize setup over attacks
```

Table candidate:

```text
Experiment ledger compressed to 6-8 rows.
```

## 9. Conclusion

Word budget:

```text
100-150
```

One-liner:

> The main contribution is a disciplined conversion from game mechanics to decision-conditioned action learning.

Draft points:

- The project is not only a rules bot; it is an iterative system for testing strategy hypotheses.
- The strongest lesson is that card-game AI needs consequence-aware local action discrimination.
- The final agent combines whichever components survive validation by the deadline.

## Media gallery candidates

- Architecture diagram.
- Deck game-plan diagram.
- Experiment summary table.
- Critical-decision / action-ranking plot.
- Optional replay-derived hidden-information diagram.

## Keep out of the final 2000 words unless necessary

- Long debugging narratives.
- Raw registry details.
- Overly technical code implementation.
- Claims that a branch worked if it was only implemented or partially tested.
- Unverified hype around embeddings/RL if not validated.

