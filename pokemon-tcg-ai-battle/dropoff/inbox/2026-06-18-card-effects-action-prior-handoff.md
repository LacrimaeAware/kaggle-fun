# Card effects / learned heuristic handoff

Date: 2026-06-18

## Short version

Do not interpret the `search_v` deck-robustness failure as evidence that card-effect features are useless.

`search_v` does not use `agent/card_effects.json`. It uses the old learned value based on board-summary features. Therefore, `search_v` losing to `search` on onechan1/DENPA92 does not test the user's core idea.

The user's actual idea is:

> Decode what cards do, then use those decoded effects to build or learn better action-selection heuristics.

That idea has only just started being tested.

## What went wrong conceptually

The project repeatedly fell into this trap:

1. Build a feature artifact.
2. Run a model/agent that does not actually consume that artifact.
3. Treat the result as if the feature idea was tested.

For this specific case:

- `tools/build_card_effects.py` creates `agent/card_effects.json`.
- `search_v` ignores `agent/card_effects.json`.
- The learned value layer used by `search_v` is still the older board-summary value model.
- Therefore, `search_v` losing to `search` says the old learned leaf value is not useful.
- It does not say decoded card effects are useless.

## Current read on the methods

Plain `search` sees card effects implicitly because it simulates the action in the engine. If the agent plays Buddy-Buddy Poffin, the simulator can show basics appearing on the bench afterward.

But shallow/1-ply search can still undervalue setup cards because it may not see the multi-turn payoff. This is exactly where explicit card-effect features could help.

So the useful target is not:

```text
learned value at search leaf
```

The useful target is more like:

```text
effect-aware action prior / residual heuristic
```

## Important new result

A first hand-weighted effect-aware heuristic was reportedly tested against the plain heuristic and lost on every deck:

- onechan1: about 0.375
- DENPA92: about 0.20
- Heisei: about 0.475
- old: about 0.225

This is not fatal to the idea. It likely means the hand-picked effect weights overrode the baseline too aggressively, especially by choosing setup plays over attacks.

The likely failure mode:

```text
raw effect bonus > attack / KO / immediate board pressure
```

That is expected if the effect policy is built as a replacement scorer instead of a conservative residual on top of the existing baseline.

## Recommended architecture

Do not build:

```text
score(action) = weighted_sum(card_effects)
```

Build:

```text
score(action)
= baseline_score(action)
+ conservative_effect_bonus(action, state)
+ opportunity_cost_penalties(action, state)
```

The effect layer should be a residual/action prior, not a total replacement for the existing heuristic or option ordering.

## Critical distinction

Card effects are affordances, not values.

Example:

```text
Buddy-Buddy Poffin = search 2 basics to bench
```

That is only valuable if:

- bench has empty slots
- deck likely has valid basic targets
- it is early enough for setup to matter
- current board is underdeveloped
- current hand lacks better development
- playing it does not skip a KO or strong attack
- it advances the deck's actual game plan

So the learned/weighted feature should not mean:

```text
search_basics_2 => always good
```

It should mean:

```text
search_basics_2 * setup_need * bench_space * early_game * no_immediate_KO_available
```

## Required guardrails before trusting any next result

Before running another win-rate benchmark, prove the wiring and behavior.

Required checks:

1. Show the code path:

```text
legal option -> played card id/name -> card_effects.json lookup -> score contribution -> chosen action
```

2. Show one real Buddy-Buddy Poffin decision where:

```text
effects enabled: Poffin gets a visible score contribution
effects zeroed: that contribution disappears
```

3. Show at least one decision where the policy refuses to play a setup card because attacking/KO is better.

4. Track skipped attacks:

```text
How often did the effect-aware policy choose setup while a damaging attack or KO was available?
```

5. Run an ablation:

```text
same policy + card_effects enabled
vs
same policy + card_effects zeroed
```

This is more informative than only testing effect-policy vs heuristic.

6. Use common random seeds for comparisons.

7. Do not accept a win-rate gain unless it beats the effect-zeroed ablation, not merely a weaker baseline.

## If tuning weights

Tuning a small number of effect weights may be worthwhile, but only if constrained.

Avoid tuning directly to noisy full-game win-rate without ablations.

Better:

1. Start with conservative residual weights.
2. Add opportunity-cost gates.
3. Tune on train seeds/decks.
4. Report holdout seeds/decks.
5. Report enabled vs zeroed effects.
6. Log top changed decisions, not just win-rate.

The target is not to learn "Poffin is good."

The target is to learn:

```text
When is Poffin worth the action compared with attacking, evolving, drawing, or doing nothing?
```

## My recommendation

Stop spending effort defending `search_v` as currently built. Treat it as a failed old-value branch.

Continue with the decoded effect layer, but only as a verified action-prior/residual system.

The next useful agent should be something like:

```text
agent_effect_prior
```

or:

```text
agent_search_effect
```

where decoded effects affect legal action ranking directly and are tested with an enabled-vs-zeroed ablation.

## Pasteable instruction for the next model

```text
Do not claim card effects help or fail until you prove the agent actually consumes card_effects.json during action selection.

Build the effect-aware policy as a conservative residual on top of the current baseline/search, not as a replacement scorer.

Before any win-rate benchmark, show:
1. a legal option mapped to a card id/name,
2. the card_effects.json lookup,
3. the resulting score contribution,
4. the same decision with effects zeroed,
5. whether attacks/KOs were available and whether the policy skipped them.

Then run effect-enabled vs effect-zeroed ablation on common seeds.
Only after that compare against heuristic/search.
```

## Addendum: do not confuse guardrails with endless toy proof steps

The user pushed back correctly on "small proof steps." They are exhausted because the project keeps doing partial diagnostics forever, then never assembles the full idea.

The goal is not to chase toy tests indefinitely.

The goal is:

```text
Build the full intended stack, but make each benchmark unfakeable.
```

The intended stack is allowed to be holistic:

- decoded card effects
- card identity / embedding information
- state context
- state x effect interactions
- option-level features
- forward-model deltas
- learned action ranking
- eventually value / auxiliary heads
- eventually search prior or search integration

The user is right that some ideas only work when multiple pieces are present. If we test one isolated fragment and it fails, that may tell us almost nothing.

So the corrected methodology is not:

```text
test only tiny fragments forever
```

It is:

```text
build the integrated model, but require wiring proofs and ablations before interpreting results
```

## Recommended build order for the neural/effect model

The other model proposed:

1. learned card embeddings
2. per-option encoding
3. shared neural trunk
4. action-ranking head
5. value head
6. intermediate/auxiliary heads
7. replay-decision training
8. policy/search-prior usage

This is directionally correct, but do not let it become another huge unfalsifiable blob.

Recommended approach:

```text
Build one integrated model path now:

legal options
-> card id/name
-> card embedding
-> decoded card effects
-> state context
-> state x effect interactions
-> option_deltas
-> neural trunk
-> action-ranking logits
```

Start with the action-ranking head as the first live output, not because the other heads are wrong, but because action ranking is the part that directly tests whether the model can choose Buddy-Buddy Poffin / draw / evolve / attack in context.

Then add value and auxiliary heads after the option-ranker data path is proven real.

This is not "abandon the big system." It is "make the central spine real before attaching extra heads."

## Clarification on learned card embeddings

A learned card embedding is possible, but it is not magic.

It can learn useful card identity information if cards appear often enough in replay decisions and the training target is informative.

It cannot learn much about rare cards from card id alone.

That is why the model should not rely on pure card-id embeddings only. It should combine:

```text
card_id_embedding + decoded_card_effects + state_context + option_deltas
```

The decoded effect vector gives semantic scaffolding:

- draw amount
- search amount
- bench target
- discard cost
- energy acceleration
- heal
- switch
- disruption
- evolution relevance

The embedding can then learn residual card-specific meaning that the decoder missed.

So the right framing is:

```text
Do not expect embeddings to discover card text from nothing.
Use embeddings to learn residual behavior on top of decoded text/effects.
```

## What the model must learn

The model should not merely learn:

```text
Buddy-Buddy Poffin is good
```

It should learn:

```text
Buddy-Buddy Poffin is good when bench space, deck targets, setup need, turn timing, and opportunity cost make it good.
```

This requires state x effect interactions. Raw effect features alone are not enough.

Examples:

```text
search_2_basics * bench_space
search_2_basics * early_game
search_2_basics * low_bench_count
draw_8 * low_hand_size
energy_accel * attacker_needs_energy
heal_bench * damaged_bench_exists
switch * active_is_bad_or_retreat_locked
setup_bonus * no_immediate_KO_available
```

## Required ablations, but not as busywork

Ablations are not meant to slow the project down or replace the full system.

They are there to prevent another false conclusion like:

```text
search_v failed, therefore card effects failed
```

Minimum required ablations for the integrated ranker:

```text
full model
vs no card_effects
vs no card_id_embedding
vs no option_deltas
vs effects zeroed at inference
vs option-0 / current heuristic baseline
```

If the full model only wins when card effects are enabled, that is meaningful.

If it performs the same with effects zeroed, then the card effects are not actually carrying the result.

## Speed / processing plan

The slow part appears to be Kaggle/environment simulation wall-clock latency, not local CPU saturation.

Therefore, throwing more CPU at the problem may not help much unless execution is parallelized or the number of engine calls is reduced.

Recommended speed strategy:

1. Prefer replay/offline decision datasets for model training and diagnostics.

```text
Use logged legal-option decisions instead of full game rollouts wherever possible.
```

2. Cache legal-option encodings.

```text
decision_id -> option feature matrix
```

This avoids recomputing card-effect joins, state features, and option deltas repeatedly.

3. Cache or batch `option_deltas`.

`option_deltas` may invoke the simulator. If so, it can dominate runtime. Store results once per replay decision and reuse them.

4. Separate fast model iteration from slow win-rate validation.

```text
offline ranker metrics: frequent
full A/B win-rate: sparse, after wiring/ablation gates pass
```

5. Use parallel game workers only for final A/B if the engine supports it safely.

If each game is mostly waiting on environment overhead, multiple worker processes may improve wall-clock. But only do this with fixed seeds and isolated temp/log outputs.

6. Add early-stop rules for bad A/B runs.

Example:

```text
If after n=30 a candidate is clearly below baseline with a wide negative gap, stop.
```

7. Use common random numbers.

For A/B tests, compare policies on matched seeds/seats where possible. This lowers variance and reduces the number of games needed.

8. Reduce search during development.

For policy iteration, use:

```text
no-search or tiny-search smoke tests
```

Reserve full search-vs-search matchups for finalists.

## Answer to the user's current concern

The user is right that "small proof steps" can become a trap if they prevent building the whole intended model.

The corrected position:

```text
Do not stop at small proof steps.
Build the integrated learned effect/action model.
But do not trust any aggregate result unless the wiring proof and ablations show what was actually tested.
```

The project needs both:

- ambition: build the actual neural/effect/embedding/action-ranker stack
- discipline: prove the live path and ablate components so results are interpretable

The failure so far was not too much ambition. It was building partial pieces and then mislabeling what had been tested.
