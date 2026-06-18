# Codex deep code + methodology audit

Purpose: sanity-check the current code/methodology after the confusing `combine vs heuristic`, `combine vs search`, Gate 2, and card-effect-decoder results.

## Executive summary

The project is not going crazy; the confusion comes from baseline mixing.

The current evidence says:

- `agent_combine` after the `evaluate_blend()` fix decisively beats the no-search heuristic.
- `agent_combine` appears worse than `agent_search`, so it is not the new best agent.
- The `0.831 vs heuristic` result is real enough as a local policy-vs-policy result, but it does not prove learned value improved search.
- The learned-value-at-leaf conclusion is being stated too broadly. What failed is this specific current blend/value setup against `agent_search`, not every possible learned component.
- The card-effect decoder is useful infrastructure, but it is not yet wired into the live policy. It did not cause the `0.831` result.
- The next model should stop using vague labels like "dead end" without saying "dead relative to which baseline."

## What is actually true about the strong result

`combine vs heuristic`:

- result: 133-27
- win rate: 0.831
- Wilson95: [0.766, 0.881]
- draws/errors: 0
- same deck for both sides
- seats swapped each game

This is a real positive result against the no-search heuristic.

It was not an asymmetric deck matchup. The A/B runner wraps agents so both sides use `agent.DECK`, and `cabt_arena.run()` alternates seats with `a_seat = g % 2`.

However, `combine` is a stronger policy class than `heuristic`:

- heuristic = no forward search
- search = forward search + hand eval
- combine = forward search + blended hand/learned eval

So `combine vs heuristic` cannot isolate the learned component. It mostly says "a search-based policy crushed a no-search policy under current code/deck."

## What is actually true about the bad result

`combine vs search` was stopped at:

- 22-38 at n=60
- win rate about 0.367 for combine
- no draws/errors

This is strong evidence that current `agent_combine` is worse than `agent_search`, but it is not as clean as a completed 160/400 game result.

Wilson CI for 22/60 is roughly below or near 0.5, so stopping early is not insane, but if this result becomes canonical it should be rerun to the planned n with a CI and recorded.

Correct conclusion:

- `agent_search` remains best unless a completed A/B proves otherwise.
- current learned blend does not add value over hand-eval search.

Incorrect overstatement:

- "learning is dead"
- "all learned value is a confirmed dead end"
- "feature learning cannot help"

Better wording:

- "The current learned-value-at-search-leaf blend, with this value model and lambda, appears worse than hand-eval search. Do not submit combine."

## VM/blend bug status

Codex previously flagged the likely bug:

- `evaluate_blend()` used `VM.score_obs(obs)` without importing `value_model as VM`.

That was a real bug.

The worktree now appears to contain the fix:

- `evaluate_blend()` imports `value_model as VM` locally before calling `VM.score_obs(obs)`.

Old blend/combine measurements before the fix are invalid.

New post-fix measurements are interpretable, but only against the specific baseline being tested.

## Did the agent learn Buddy-Buddy Poffin?

No, not in the live policy yet.

The new card-effect decoder can decode effects such as:

- Buddy-Buddy Poffin -> search 2, to bench
- Lillie's Determination -> draw 8
- Ultra Ball -> search 1, discard cost 2

That is useful feature infrastructure.

But unless `agent/card_effects.json` is wired into the live ranker/search policy, this is not the agent learning to play Buddy-Buddy Poffin. It is a decoder building semantic features that future models can use.

Do not attribute the `combine vs heuristic` win to card-effect features unless those features are actually used by the live policy being tested.

## Code audit: A/B runner and arena

`tools/run_ab.py` and `agent/cabt_arena.py` look basically sound for local A/B:

- same deck wrappers for random/first
- project agents already return `agent.DECK`
- seat swapping every game
- winner read from final rewards
- draws/errors tracked
- progress logging flushes
- no obvious deck asymmetry

Remaining concerns:

1. The `combine vs search` run was stopped at n=60.

This is enough to be a warning, but if used as a permanent result it should be completed or rerun.

2. Same-deck self-play is not ladder/generalization.

Good for policy isolation; not enough for multi-deck claims.

3. Need current `search vs heuristic` baseline.

The huge `combine vs heuristic` result is hard to attribute without the same-run/current-code `search vs heuristic` result.

If current `search vs heuristic` is also near 0.8, then combine's result is mostly search. If current search is much lower, then combine did something interesting against heuristic even though it loses to search.

## Code audit: search/combine methodology

`agent_search` and `agent_combine` are identical except leaf mode:

- `agent_search`: `leaf_mode="hand"`
- `agent_combine`: `leaf_mode="blend"`

Both use `_forced_move()` before search, then `search.best_option()`.

Therefore `combine vs search` isolates the effect of replacing the hand leaf eval with blended hand/learned eval, plus any runtime overhead.

Important possible confound:

- `search._search()` is time-budgeted.
- `evaluate_blend()` calls the learned value model and is slower than hand eval.
- Under the same 0.6s budget, combine may evaluate fewer candidate options/determinizations than hand search.

So a `combine vs search` loss may reflect:

- worse learned value signal
- dilution of a good hand eval
- slower leaf eval causing worse search coverage
- or all of the above

To isolate this, log per-decision candidate coverage / counts for hand vs blend, or run an offline fixed-count comparison without time cutoff.

## Code audit: blend formula

`evaluate_blend()` uses:

- hand score squashed by `sigmoid(hand / BLEND_SCALE)`
- learned `P(win)`
- `(1 - lambda) * hand01 + lambda * p`
- `BLEND_LAMBDA = 0.4`
- `BLEND_SCALE = 2000`

This can easily dilute the sharp local hand eval. A one-prize swing becomes only about sigmoid(0.5), and learned value can pull strong local decisions toward global noise.

This does not mean learned value is hopeless. It means this blend formula/weight/value model is not beating hand search.

If revisited, test lambda sweep and candidate-coverage logs, but do not prioritize this over deeper search unless there is a clear reason.

## Code audit: search bugs / technical debt still relevant

These issues still matter, especially before deeper search:

1. Hidden-zone padding with card `3` Water Energy.

`search._hidden_pool()` pads missing hidden cards with `[3]`. This avoids crashes but creates fake hidden cards.

This is especially dangerous for deeper search and opponent modeling.

2. Public-zone accounting.

Prior audits flagged public Stadium / possibly other public zones not being stripped from hidden pools. This can duplicate visible cards into hidden states.

3. Optional prompts.

Rollout policy can pick options even when `minCount == 0`, because the default fallback often returns up to maxCount. This can force optional choices that should sometimes be declined.

4. Time-budget governor.

Search uses a fixed per-decision budget and does not appear to adapt to match-level remaining time. Deeper search will need budget control.

These should be cleaned before or during 2-ply work, not after.

## Code audit: Gate 2 / action-delta diagnostic

`tools/diag_action_delta.py` is closer to the intended Gate 2 than `diag_action_fwd.py`:

- it calls `search.option_deltas()`
- one engine step per option
- no rollout
- listwise LightGBM ranker
- stratifies which-card vs mixed
- compares to option-0

Good.

But there are still issues:

1. It uses only the winner's deck.

`winner_deck(d, win)` returns the winner's deck, then `S.option_deltas(obs, deck)` uses that same deck for both players' hidden zones. In real replays the opponent deck may differ.

For immediate self-action deltas this may often be harmless, but it is not clean. Effects touching opponent hand/deck, forced reveals, disruption, or opponent-zone consequences can be wrong.

Better: pass both players' real decks into determinization, or extend `option_deltas()` to accept separate my/opponent decks.

2. Option-0 baseline is reported on all collected decisions, not necessarily the exact same test split/subset as the model.

For a clean comparison, report option-0 on the same test gids and same sim-success subset.

3. Single split, no confidence intervals.

Use multiple game-wise splits or bootstrap CIs before treating the diagnostic as canonical.

4. Top-1 may be the wrong yardstick.

Top-3 mixed was strong enough to show signal. If the model can put good options in top-k, it may still be useful as a search prior or budget allocator even if top-1 loses to option-0.

Correct conclusion from Gate 2:

- top-1 action-delta imitation did not beat option-0 on mixed decisions
- the delta features still carry some signal
- this does not prove action deltas are useless for live search guidance

## Code audit: card-effect decoder

`tools/build_card_effects.py` is useful but still rough.

Good:

- produces quantified features for many stereotyped effects
- catches obvious draw/search/energy/heal/switch/disrupt patterns
- creates the needed semantic layer from card text

Concerns:

- coverage number is not quality
- `switch_gust` conflates own switch and opponent gust
- `energy_accel` regex can catch attack text and may not distinguish source/target/type/amount
- heal is too generic and may not distinguish bench heal
- draw regex misses draw-until effects or conditional draws
- search lacks target class detail beyond bench flag
- disruption regex may miss important shuffle-hand effects depending on wording
- attacks and abilities are combined, losing context

Use it as a starter feature layer, not trusted ground truth. Add overrides for meta cards and validate against known cards.

## Methodology audit: current plan

The plan's best part:

- win-rate is the real judge
- `agent_search` remains best unless beaten
- deeper search/quality is the near-term performance path
- learned action-ranking remains a research side branch

The plan's weak part:

- it says "learned-value-at-leaf branch is confirmed dead end" too broadly
- it risks abandoning useful top-k/action-prior signals because top-1 imitation failed
- it risks moving to 2-ply before cleaning determinization/time-budget bugs
- it has not yet completed current baseline attribution via `search vs heuristic`

## Answer to the current strategic choice

The best answer is not pure option 1 or pure option 2.

Choose both in parallel, but with strict prioritization:

- near-term performance: deeper/better search
- learned path: small, gated, no broad claims

Concrete instruction:

1. Finish current baseline table first:

- search vs heuristic
- combine vs heuristic
- combine vs search

2. Keep `agent_search` as current best.

3. Before 2-ply, fix or at least account for:

- fake Water padding
- public-zone hidden-pool leakage
- optional prompt handling
- time-budget/candidate-coverage logging

4. Build 2-ply/deeper search as the main win attempt.

5. Continue learned work only as:

- card-effect feature validation
- action-ranker as top-k prior/tie-breaker/budget allocator
- live win-rate gate before promotion

## Final verdict

The codebase has real progress, but the communication around results is too absolute.

Precise truth:

- fixed combine is strong vs heuristic
- fixed combine is worse than search
- card-effect decoder is useful but not live proof
- Gate 2 top-1 failed but still shows signal
- `agent_search` remains the best agent
- next highest-probability win is search depth/quality, but only after cleaning determinization and budget issues

The user is right to feel whiplash. Every future result should name the exact baseline in the headline.
