# Codex evaluation summary: adversarial run, methodology, and next gates

Purpose: compact handoff of Codex's evaluation after the adversarial runs finished. Use this as a sanity-check before continuing implementation or interpreting new results.

## Bottom line

Do not conclude that learning failed.

Do not conclude that richer features are useless.

The strongest current conclusion is: the previous diagnostics were confounded, the old representation-ceiling claim was premature, and the next work must isolate objective, representation, and search-depth one at a time.

The project is at: diagnostics and conversion layer need repair. It is not at: complex learning is hopeless.

## Confidence and severity

| Finding | Confidence | Severity |
|---|---:|---:|
| Option-0 baseline must be included everywhere | Very high, about 95% | Major |
| Pointwise GBM is the wrong decisive objective for variable-size action ranking | Very high, about 90% | Major |
| The old representation-ceiling conclusion was premature | Very high, about 90% | Major |
| Old blend/combine results are invalid before the `evaluate_blend` import fix | Very high, about 95% | Major |
| `diag_action_fwd.py` may not test the intended immediate one-step delta | High, about 85% | Medium-major |
| Listwise/pairwise ranking will improve diagnostics | Medium, about 60-70% | Promising, not guaranteed |
| Action deltas/card-effect features will improve live win rate | Medium-low, about 40-60% | Worth testing, not guaranteed |

## What changed after adversarial verification

The adversarial pass found a real bug in the original diagnostic:

- The card-feature join was dead code.
- It checked whether `hand[idx]` was an int.
- Replay hand cards are dicts like `{"id": ...}`.
- Therefore the old `+card` feature rung was all zeros.
- So the old claim that card features add no signal was invalid.

The adversarial pass also exposed a major baseline confound:

- Choosing option 0 reaches about 0.587 top-1 on winner imitation.
- The hand heuristic was about 0.553.
- The pointwise GBM was about 0.50 overall.
- Therefore any imitation diagnostic must beat option-0, not random.

The blend/combine bug was also real:

- `evaluate_blend()` referenced `VM` without importing `value_model as VM`.
- That made blend/combine silently fall back through broad exception handling.
- Old combine results should not be trusted.
- The worktree now appears to have the local import fix, but combine must be remeasured.

## Major interpretation warnings

1. Random is not the relevant imitation baseline.

The relevant baseline is choose-option-0, because engine option ordering explains a large share of winner moves.

2. Pointwise binary classification is not enough.

The decision is listwise: one chosen option among a variable-size set of legal options. Use grouped pairwise/listwise ranking.

3. Winner imitation is noisy.

Winner moves are demonstrations, not ground truth. Winners can make weak moves, win by matchup/variance, or have multiple equivalent choices.

4. Global value AUC is not move quality.

A value model can predict eventual winner from a state while still ranking sibling legal actions badly.

5. Static replay diagnostics are not the full test.

If static options do not expose action meaning, use card-effect decoding and immediate forward-model deltas.

6. Do not add all feature families at once.

If objective, card effects, forward deltas, belief, and neural layers all change together, the result will be uninterpretable.

## Major concerns with the current plan

The plan is directionally right but still risks going in circles unless it uses hard gates.

Main risks:

- overvaluing imitation top-1 instead of live win rate
- learning engine option ordering instead of intelligence
- adding too many feature ideas at once
- confusing immediate one-step deltas with rollout leaf deltas
- assuming multi-deck generalization is easy
- using stale diagnostics as proof

## Recommended near-term gates

### Gate 1: objective-only test

Keep corrected static features fixed.

Change only the objective:

- replace pointwise GBM with grouped pairwise/listwise ranker
- group by decision id
- compare to option-0
- stratify which-card, mixed, and trivial decisions
- report top-1, top-3, MRR, and confidence intervals

Pass criterion:

- beat option-0 on mixed strategic decisions

Interpretation:

- If it passes, the previous plateau was largely objective/metric artifact.
- If it fails, move to richer features.

### Gate 2: immediate action-delta test

For each legal option, simulate exactly one step with the forward model.

Feature the immediate changes:

- prize delta
- damage / HP delta
- hand-size delta
- deck-count delta
- energy delta
- energy-shortfall delta
- attacker-online delta
- evolution/engine-online delta
- future legal-option count delta

Do not roll through the entire turn or opponent reply for this test.

Pass criterion:

- improve mixed-decision ranking over option-0 and static-feature ranker

Interpretation:

- If it passes, immediate forward deltas are the bridge.
- If it fails, static replay imitation may be too noisy or the label may be weak.

### Gate 3: live win-rate test

Only after Gate 1 or Gate 2 shows diagnostic improvement, plug the learned component into live play as:

- option prior
- tie-breaker
- deeper-search trigger
- budget allocator when heuristic/search/model disagree

Evaluate against `agent_search`, not only imitation.

Use seat alternation and confidence intervals.

## Feature priorities

Tier 1: diagnostic hygiene

- option-0 baseline
- stratification
- grouped train/test splits
- top-1, top-3, MRR
- confidence intervals
- no pointwise-only conclusions

Tier 2: static action semantics

- KO this target
- KO prize value
- overkill
- attack affordability
- energy-shortfall delta
- attach completes attack cost
- evolve stage / HP / attack gain
- retreat to better attacker

Tier 3: card-effect features from `cards_full.json`

- draw amount
- cycle/discard amount
- tutor target class and count
- energy acceleration source/target/amount/type
- heal amount and bench-heal amount
- gust/switch effect
- evolution enabler
- ability unlock
- disruption amount
- coin-flip expected value and variance

Tier 4: within-decision relative features

- only KO option
- highest damage option
- best affordable attack
- only engine-unlock option
- highest future-option gain
- rank of option under hand heuristic

Tier 5: immediate forward-model deltas

- state changes after one `search_step`
- no full rollout in the first diagnostic
- distinguish this from rollout leaf evaluation

## Specific caution on `diag_action_fwd.py`

Before relying on `diag_action_fwd.py`, check two issues:

1. It appears to copy the old dead hand-card join in its static feature block. Fix it to use `AreaType.HAND == 2` plus `hand[idx]["id"]`.

2. It currently uses `search.option_evals()`, which rolls through the turn plus opponent reply. That is not the same as the intended immediate one-step action delta test.

Both experiments can be useful, but they answer different questions.

## Should work be interrupted?

Interrupt if the current work is doing any of these:

- claiming representation ceiling is proven
- using random as the main baseline instead of option-0
- treating pointwise GBM as decisive
- trusting old combine/blend results
- treating `diag_action_fwd.py` as a clean one-step-delta test while it uses rollout leaf features
- adding many feature families at once without gates

Do not interrupt if it is doing these:

- grouped listwise/pairwise action ranking
- option-0 baseline plus stratified metrics
- fixing `diag_action_fwd.py`
- immediate one-step forward deltas
- remeasuring combine after the blend fix
- narrow pass/fail experiment design

Urgency: moderate-high. Not disaster, but pause before trusting the next result.

## Final recommendation

The next move should be one clean axis at a time:

1. objective-only listwise ranking
2. immediate action-delta features
3. card-effect features
4. live win-rate integration

Do not let the project drift into another broad, ambiguous experiment. The antidote to going in circles is hard gates, correct baselines, and one changed variable at a time.
