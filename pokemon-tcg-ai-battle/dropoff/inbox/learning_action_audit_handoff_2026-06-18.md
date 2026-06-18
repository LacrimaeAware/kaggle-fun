# Learning/action audit handoff

Purpose: give another model a compact, concrete handoff on the current Pokemon TCG AI Battle learning/search/classification situation. This is not a broad project summary. It preserves likely bugs, suspect measurements, and the recommended research direction.

## Short summary

Do not conclude that complex learning has failed. The stronger conclusion is that the current conversion layer from complex information to move choice is still broken or under-specified.

The project has found real value signal, such as global value AUC and replay/deck correlations, but that signal has not reliably become better action selection. The likely causes are action representation gaps, noisy/circular labels, and several concrete code issues that can make measurements misleading.

## Likely bugs or suspect measurements

- `agent/eval.py`: `evaluate_blend()` appears to call `VM.score_obs(obs)` without importing `value_model as VM` in module/function scope. If true, `agent_combine` / blend measurements may silently fall back through broad exception handling.

- `tools/diag_action_ceiling.py`: the hand-card join is suspicious. It may assume `hand[idx]` is an int, while other code often treats hand cards as dicts with an `id`. If true, the claim that card features add no imitation signal is not trustworthy yet.

- `agent/features.py`, `agent/eval.py`, and `tools/diag_action_ceiling.py`: HP vs damage semantics need hard verification. Some logic may treat `hp` as remaining HP while other logic subtracts `damage`. KO detection, board HP eval, and imitation features can all be wrong if this is inconsistent.

- `agent/search.py`: hidden-zone reconstruction still pads missing cards with card `3` Water Energy. This avoids crashes, but contaminates determinizations and can make belief/search results unrealistic.

- `agent/search.py`: hidden pool accounting may fail to strip public Stadium or other public-zone cards from sampled hidden states. This can duplicate visible cards in hidden decks/prizes/hands.

- `agent/features.py`: Team Rocket Energy is treated as a universal wildcard. The docs already flag this as wrong; it matters once decks using that energy become relevant.

## Methodology diagnosis

- State-value AUC does not imply good sibling-action ranking. A model can predict who eventually wins from a board state while still being bad at choosing between the legal options in one decision.

- Winner imitation top-1 is useful but harsh and noisy. Strong players can have multiple equivalent choices, make imperfect moves, or choose lines that require hidden context not encoded in the current features.

- Search-value action labels can be circular. If the action ranker is trained mostly on current hand-search values, it learns to copy current search, not beat it.

- Current action features likely do not encode card effects deeply enough. Important effects include draw amount, heal amount, bench healing, exact tutor target, evolution target, ability unlock, energy acceleration, discard costs, gust/switch effects, prize swing, and KO-back risk.

- The result from `diag_action_ceiling.py` should not be accepted as a final representation ceiling until the hand-card join and HP/damage semantics are verified.

## Recommended next direction

- Fix the suspect bugs first, especially blend import, hand-card join, and HP/damage semantics.

- Build richer action-effect features. Each legal option should have features that describe what the move means, not only its option type.

- Train within-decision action ranking, not only global state value. The model should compare sibling legal moves from the same root decision.

- Use real replay decisions from strong players, but treat them as noisy demonstrations. Prefer grouped pairwise/listwise ranking over exact top-1 imitation.

- Add action delta features from forward simulation: prize lead delta, board HP delta, hand size delta, deck count delta, future legal option count, energy shortfall delta, attacker online, ability online, and KO risk.

- Use learned models as priors, tie-breakers, or search guidance before letting them replace the reliable heuristic floor.

## Practical build order

1. Fix `evaluate_blend()` import / fallback issue.
2. Verify card shape in replay observations and repair `tools/diag_action_ceiling.py` hand-card joins.
3. Verify whether `hp` is remaining HP or max HP plus damage, then standardize KO and HP calculations.
4. Re-run the action-ceiling diagnostic only after those fixes.
5. Extend the action dataset to include root features, option features, simulated leaf/delta features, chosen option, outcome, deck id, player/team id, and episode id.
6. Train a grouped action ranker from real replay decisions and compare head-to-head against `agent_search`.
7. Add opponent/deck belief determinization after the action representation is trustworthy.

## Bottom line

The current evidence does not say learning is useless. It says the project has been trying to use global value and incomplete option features to solve a local action-ranking problem. The bridge is learned heuristics/action ranking: keep obvious hard heuristics, add richer action semantics, and let the learned system decide where the heuristic is blind.

---

# Deep methodology addendum

This addendum responds to the later concern: are we failing because learning is impossible, or because the project is failing to leverage complex information correctly?

## Main judgment

Do not conclude that complex learning has failed. The better conclusion is that global value signal is being converted into action choice incorrectly.

The project has found useful information, including global value AUC, search-target correlations, replay/deck correlations, and deck-policy coupling. The repeated failure is that these signals do not reliably choose better legal moves. That points to a representation/conversion problem, not proof that learning is useless.

## Read of the latest diagnostic claim

The latest model said a within-decision imitation ranker plateaued around 0.50 top-1 and that this proves a representation ceiling. That is partly useful but too strong.

What is likely true:

- State-only features cannot distinguish sibling legal moves.
- Action-level semantics are required.
- Static replay options often expose only option type, indexes, targets, and attack IDs, not full card-effect meaning.
- Simulating each option with the forward model is probably the cleanest way to get action features.

What is not proven:

- It is not proven that card effects carry no signal, because the option-to-hand-card join was unvalidated/inconclusive.
- It is not proven that learning cannot beat the heuristic.
- It is not proven that the bottleneck is only features and not also labels/objective/evaluation.
- Exact top-1 imitation of winner moves is a noisy and harsh target.

The honest conclusion is: current static decodable option features are insufficient. Rich action semantics and simulated deltas have not been properly tested yet.

## Exact methodological errors we may be making

1. Confusing global state value with local move quality.

A model can predict eventual winner from a board state and still be poor at choosing between the legal moves in that exact decision. Search needs sibling-action ranking, not only state classification.

2. Training on circular labels.

If action labels come from current hand-search values, the model learns to copy current search. It cannot exceed that search unless the labels include non-circular signal such as real replay choices, outcome-aware preferences, or better/deeper search targets.

3. Missing action semantics.

The model needs to know what a move means: draw 7, heal bench 30, evolve into engine, unlock an ability, search a missing piece, attach extra energy, gust a target, spend Supporter for the turn, discard resources, or expose us to KO-back risk.

4. Treating all decisions as equally important.

Many legal choices may have little effect on outcome. Training should identify and weight high-leverage decisions: policy disagreement, large simulated delta, prize swing, engine unlock, lethal/KO-back, future-option swing.

5. Treating winner imitation as ground truth.

Winner moves are useful demonstrations, not perfect labels. Strong players can make multiple equivalent choices, choose a stylistic line, or win despite mistakes. Use pairwise/listwise ranking and soft labels, not only exact top-1 imitation.

6. Letting bug-prone blend/diagnostic paths influence conclusions.

Concrete issues remain suspect: `evaluate_blend()` may reference `VM` without import; `diag_action_ceiling.py` has an unreliable hand-card join; HP/damage semantics need verification; hidden-zone sampling can inject fake Water Energy; public zones may leak into hidden pools; Team Rocket Energy is treated as a universal wildcard.

7. Overvaluing AUC and target fit.

AUC, Pearson-to-search-target, and Brier score are diagnostics. The real metric is head-to-head policy improvement against `agent_search`, with seat alternation and confidence intervals.

## Feature-set improvements

The project likely needs layered features, not just a bigger flat vector.

### Layer 1: card semantics

For each card, encode:

- card type: Pokemon, Item, Tool, Supporter, Stadium, Basic Energy, Special Energy
- stage: Basic, Stage 1, Stage 2, Mega, ex, Tera
- evolution links: evolves from, evolves into, line completeness
- attack damage, costs by type, colorless costs, self-costs
- draw amount, cycle amount, discard cost
- heal amount, bench-heal amount, damage prevention/reduction
- tutor/search target class and count
- energy acceleration target, source, amount, type restrictions
- gust/switch/retreat effects
- bench damage / spread / snipe amount
- hand disruption, energy disruption, ability lock
- once-per-turn or rule-cost implications
- coin flip EV and variance

Binary tags are useful starts, but they collapse magnitude. `draw 2` and `draw 7` should not be the same feature.

### Layer 2: state features

Keep board/state features but make them more causal:

- prize lead and prize liability
- HP remaining, damage, KO thresholds
- attacker online, backup attacker online
- energy shortfall by active and bench attackers
- evolution line completeness
- engine online/offline
- hand/deck/bench/resource counts
- status and retreat constraints
- support/stadium/energy attach already spent
- deckout risk
- opponent likely response/KO-back exposure

### Layer 3: action features

For each legal option, encode:

- option type and select context
- card/effect identity when recoverable
- target identity and target role
- resource spent: Supporter, Energy attach, Stadium, Tool, discard, retreat cost
- immediate tactical value: KO, prize gain, damage, heal, draw, search, attach
- whether it develops board, engine, energy, evolution, or attacker
- whether it preserves or spends scarce options

### Layer 4: simulated action-delta features

Use the forward model to simulate each option and record deltas:

- prize lead delta
- board HP delta
- active/bench damage delta
- hand size delta
- deck count delta
- discard/resource delta
- energy shortfall delta
- attacker online delta
- engine/ability online delta
- future legal option count delta
- future draw/tutor/evolve/attack availability
- KO-back risk after opponent reply

This is probably the most important bridge. If static replay options do not expose card meaning, simulate the action and feature the result.

### Layer 5: sequence features

Some good moves are only good because they open chains:

- draw/cycle now -> find evolution -> use ability -> attack
- tutor now -> complete line next turn
- attach now -> enable next-turn attack
- bench now -> create pivot or attacker
- retreat/switch now -> unlock attack or avoid KO

Represent at least one-turn and two-turn chain affordances where possible.

### Layer 6: belief/opponent features

Search quality depends on realistic hidden states:

- opponent archetype/deck prior from replays
- revealed cards and known deck composition
- likely hidden hand/deck/prize cards
- likely opponent response to our line
- whether our move is punished by common opponent lines

## Do we need neural network layers?

Probably yes eventually, but not as the first fix.

A neural network can help combine many interacting features, but it cannot invent missing inputs. The correct progression is:

1. engineered action and delta features
2. transparent pairwise/listwise ranker
3. small MLP or set model over legal options
4. optional card embedding / effect embedding
5. multi-head model with auxiliary targets

The most useful neural architecture would not be a plain state-value network. It should be a sibling-option model:

- encode root state
- encode each legal action
- encode simulated delta / leaf state
- score each option relative to the other options
- softmax over legal options

Possible heads:

- policy/ranker head: which option is best among siblings
- value head: eventual win probability from resulting state
- future-options head: predict future legal option count or option quality
- tactical heads: KO, KO-back, attacker online, engine online
- belief head: infer opponent hidden/deck state

This lets the model learn intermediate structure instead of forcing everything through final win/loss.

## On treating result as a feature

This idea makes sense if `result` means simulated consequence of a candidate action.

Valid examples:

- after this option, prize lead changes by +1
- after this option, future legal options increase by 4
- after this option, active can attack next turn
- after this option, opponent can KO us back

Those are action-result features and should absolutely be used.

Invalid/leaky example:

- final replay win/loss is used as an input feature

Final game result should be a label/target, not an input feature. Otherwise the model cheats and will not work in live play.

## Recommended build direction

1. Fix the concrete suspect bugs before trusting combine/diagnostic results.

2. Build a proper action dataset:

- root state features
- every legal option
- card/effect semantics where recoverable
- simulated one-step / one-turn / opponent-reply deltas
- chosen move from replay or current policy
- final outcome
- deck id, player/team id, episode id
- decision group id

3. Train grouped action ranking:

- pairwise differences between chosen and unchosen options
- listwise softmax over legal options
- outcome-weighted or player-strength-weighted demonstrations
- decision-leverage weighting

4. Use learned model as search guidance first:

- option prior
- tie-breaker
- selective deeper-search trigger
- budget allocation when heuristic/search/model disagree

5. Keep hard heuristic floors:

- take clear winning KO
- go first when correct
- avoid illegal/time-risk paths
- preserve crash-safe fallback

6. Evaluate by head-to-head, not AUC:

- vs `agent_search`
- vs heuristic
- seat alternation
- 800+ games for serious claims
- confidence intervals

## Bottom line

The project should not give up on learning. It should stop expecting global state value to become action quality automatically.

The next real unlock is a learned heuristic/action-ranker that understands legal move semantics and simulated consequences. Keep hand heuristics as the safety floor, add rich action/delta features, and let learning decide where the heuristic is blind.

---

# Adversarial-run update: superseding corrections

This update reflects the later significant adversarial verification pass. It supersedes any earlier wording that treated the static imitation plateau as a proven representation ceiling, and it updates the status of the blend bug.

## What changed

The earlier claim "richer action features do not help, therefore representation is the ceiling" was too strong and partly based on a real diagnostic bug.

The adversarial pass found that `tools/diag_action_ceiling.py` had dead card-feature code: it checked `isinstance(hand[idx], int)`, but replay hand cards are dicts like `{"id": ...}`. Therefore the `+card` feature block had been all zeros, which made the earlier "card features add nothing" result invalid.

The corrected diagnostic now joins card identity through `AreaType.HAND == 2` plus `hand[idx]["id"]`, restricted to actual hand-card options.

## Corrected diagnostic interpretation

The corrected stratified result does NOT prove that features are useless.

The corrected result instead shows the current experiment is confounded by engine option ordering and by the weak pointwise objective:

- random top-1 baseline: about 0.197
- choose-option-0 positional baseline: about 0.587
- hand heuristic: about 0.553
- pointwise GBM on current per-option features: about 0.50 overall
- which-card decisions: about 0.81, but this mostly reflects the option-0 prior because option type is constant there
- mixed strategic decisions: about 0.33, below the about 0.46 option-0 rate there

Honest verdict: the static diagnostic is inconclusive on representation vs objective. The earlier representation-ceiling conclusion should not be repeated.

## Confirmed/fixed bug update

`agent/eval.py:evaluate_blend()` previously referenced `VM` without importing `value_model as VM`. That caused `NameError` and made blend / `agent_combine` measurements silently fall back through broad exception handling.

This has now been fixed in the worktree: `evaluate_blend()` imports `value_model as VM` locally before calling `VM.score_obs(obs)`.

Any old blend/combine result is suspect and should be re-measured before being trusted. This may explain weak `sub_combine` ladder performance.

## Updated main conclusion

The project should say:

- State-only/global-value learning is still the wrong conversion path.
- Static pointwise imitation diagnostics are currently confounded.
- The correct next experiments are still action ranking, but with better objectives and better action features.
- The big unresolved question is not "can features help?" but "which action semantics and ranking objective beat the option-order prior and `agent_search` in live play?"

## Updated priorities

1. Re-measure `agent_combine` after the blend import fix.

2. Do not use random as the main imitation baseline. Always include `choose option 0` because engine option ordering is a very strong prior.

3. Replace pointwise GBM imitation with grouped pairwise/listwise objectives. The model must compare sibling options inside a decision.

4. Add card-effect features from `cards_full.json`, not just coarse tags:

- draw/cycle amount
- heal amount and bench-heal amount
- search/tutor target class and count
- energy acceleration source, target, amount, and type restrictions
- evolution target / engine unlock
- switch/gust effects
- discard/resource cost
- damage spread / bench snipe
- hand disruption / energy disruption
- coin-flip EV and variance

5. Add forward-model delta features by simulating each option:

- prize delta
- board HP / damage delta
- hand size delta
- energy shortfall delta
- attacker-online delta
- engine/ability-online delta
- future legal-option count delta
- KO-back risk after opponent reply

6. Treat winner imitation as noisy demonstration data, not perfect labels. Prefer soft, grouped, outcome-aware ranking.

7. Use the learned model first as a prior/tie-breaker/search-budget allocator, not as an immediate replacement for the heuristic/search floor.

## New related files in this worktree

- `dropoff/outbox/2026-06-18-feature-optimization-prompt.md`: handoff prompt for feature/action-representation work.
- `dropoff/outbox/2026-06-18-research-questions.md`: handoff document for research questions.
- `tools/diag_action_fwd.py`: forward-model diagnostic intended to test whether simulated action-delta features lift imitation beyond static features and heuristic/option-order baselines.

## Bottom line after adversarial runs

The discouraging conclusion got less certain, not more certain.

The right update is: the previous diagnostic was flawed, option ordering is a huge confound, blend was broken and is now fixed, and the untested high-value path is still pairwise/listwise sibling-action ranking with card-effect and forward-model delta features.

---

# Final deep-pass update after finished adversarial run

This update reflects the completed adversarial workflow and the follow-up deep pass over the generated outbox documents.

## Final corrected story

The earlier "representation ceiling" conclusion is retracted. The finished workflow correctly says the current evidence is inconclusive, not hopeless.

What is now supported:

- State-only features cannot distinguish sibling legal actions.
- The old card-feature diagnostic was invalid because the hand-card join was dead.
- Option ordering is a huge confound: choosing option 0 reached about 0.587 top-1, beating the hand heuristic at about 0.553.
- The pointwise GBM imitation diagnostic is the wrong objective for a variable-size option set.
- Old blend/combine measurements are invalid because `evaluate_blend()` was broken before the import fix.
- The right next work is listwise/pairwise action ranking, card-effect features, and forward-model action-delta features.

What is NOT supported:

- It is not proven that richer features cannot help.
- It is not proven that card-effect features are useless.
- It is not proven that behavior cloning is doomed.
- It is not proven that the current plateau is purely representation rather than objective/depth/data-quality.

## New caution from this pass

`tools/diag_action_fwd.py` appears to copy the old static hand-card join logic:

- it checks `isinstance(hand[idx], int)`
- replay hand entries are dicts with `id`
- therefore the static card-feature rung inside the forward diagnostic may still be dead or stale

Before using `diag_action_fwd.py` as evidence, repair its `static_features()` hand-card join the same way `tools/diag_action_ceiling.py` was repaired: use `AreaType.HAND == 2` plus `hand[idx]["id"]`, and restrict this to actual hand-card option types rather than target/in-play options.

This caution does not refute forward-model delta features. It only means the forward diagnostic's static-vs-forward comparison must be cleaned before trusting any result.

## Strongest next experiment

The cheapest high-information experiment is not a bigger model. It is:

1. Keep the current fixed static features.
2. Replace pointwise GBM with a grouped listwise/pairwise ranker.
3. Score against the correct baselines, especially option-0, on stratified slices.
4. Ask whether objective alone beats option-0 on mixed strategic decisions.

If yes, the previous plateau was largely objective/metric artifact.

If no, add the next feature tiers:

- card-effect features from `cards_full.json`
- state x option interaction features
- forward-model one-step delta features

## Feature priorities after the finished run

Tier 1: fix diagnostic/evaluation hygiene

- option-0 baseline everywhere
- stratify which-card vs mixed vs trivial decisions
- top-1, top-3, MRR, and confidence intervals
- grouped train/test splits by game/decision
- no pointwise-only conclusion on a listwise task

Tier 2: static action semantics

- KO this target
- KO prize value
- overkill amount
- attack affordability and energy-shortfall delta
- attach completes an attack cost
- evolve stage / HP / attack gain
- retreat to better attacker
- target already powered or vulnerable

Tier 3: card-effect features

- draw amount
- cycle/discard amount
- tutor target class and count
- energy acceleration source/target/amount/type
- heal amount and bench-heal amount
- gust/switch effect
- evolution enabler, especially Rare Candy-style effects
- ability unlock
- disruption amount
- coin-flip EV and variance

Tier 4: within-decision relative features

- is only KO option
- best damage among legal attacks
- lowest-cost setup option
- highest future-option gain
- only evolution/engine-unlock option
- action rank within decision by static heuristic

Tier 5: forward-model one-step deltas

- prize delta
- board HP/damage delta
- hand-size delta
- deck-count delta
- energy attached / energy shortfall delta
- attacker online delta
- engine online delta
- future legal-option count delta

## Methodology recommendation

Treat imitation as a representation discovery tool, not the final proof of agent strength. Ground truth remains head-to-head win rate against `agent_search` with seat alternation and confidence intervals.

A learned model should first be used as:

- an option prior
- a tie-breaker
- a selective deeper-search trigger
- a budget allocator when heuristic/search/ranker disagree

Do not replace the heuristic/search floor until the learned component proves itself in live games.

## Bottom line

The finished adversarial run made the situation less discouraging, not more. It found real bugs and a huge metric confound. The project is not at "learning failed"; it is at "our diagnostics and conversion layer were not yet good enough." The next correct push is clean listwise action ranking plus richer action/effect/delta features.

## Additional diagnostic mismatch to fix

`dropoff/outbox/2026-06-18-feature-optimization-prompt.md` recommends immediate one-step forward-model delta features: apply the candidate option with one `search_step`, then feature the immediate state delta without a rollout.

However, `tools/diag_action_fwd.py` currently calls `search.option_evals()`, which uses the existing search simulation path: first option, then finish the turn and opponent reply up to `DEPTH_CAP`. That is a rollout-contaminated leaf feature, not the intended immediate one-step action delta.

Before treating `diag_action_fwd.py` as testing the recommended feature rung, decide which experiment is intended:

- immediate action-delta diagnostic: one `search_step` only, no rollout, best for decoding what the action actually did
- rollout leaf diagnostic: current `option_evals()` behavior, best for testing whether the existing search leaf/rollout predicts winner choices

Both are useful, but they answer different questions. The immediate one-step delta is the cleaner next representation experiment.
