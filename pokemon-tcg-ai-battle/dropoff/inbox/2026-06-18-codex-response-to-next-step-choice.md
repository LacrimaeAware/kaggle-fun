# Codex response to next-step choice prompt

Purpose: answer the current question being asked by the other model:

"Given that learning on our current features doesn't beat the hand-eval search, where should I put effort next?"

## First: clarify the apparent contradiction

There is no contradiction if the baselines are separated.

`agent_combine` after the `evaluate_blend()` fix beat the no-search heuristic very hard:

- combine vs heuristic: 133-27, win rate 0.831, Wilson95 [0.766, 0.881]

That is a real positive result against the no-search heuristic.

But `agent_combine` then appeared to lose to the current best hand-eval search:

- combine vs search: 22-38 at n=60, about 0.367 when stopped

So the correct interpretation is:

- The fixed blend/combine path is not trash against the heuristic.
- It is trash or at least clearly worse relative to `agent_search`, which remains the best local agent.
- The result is useful diagnostically, but not a new best agent.

This matters because earlier learned/combine results were often bad even against the heuristic. The bug fix did improve the situation, but not enough to beat the best search baseline.

## VM bug timeline

Codex had previously flagged a likely `evaluate_blend()` bug: `VM.score_obs(obs)` was used without importing `value_model as VM` in `evaluate_blend()`.

That was a real bug.

The worktree now appears to have fixed it with a local import inside `evaluate_blend()`.

Old combine/blend measurements before that fix are invalid. New combine measurements after the fix are valid enough to interpret, subject to normal A/B caveats.

## Was the strong result caused by asymmetric decks?

No, not in the obvious deck-vs-deck sense.

The A/B runner uses the same `agent.DECK` for both sides and swaps seats each game. So `combine vs heuristic = 0.831` was a policy comparison on the same deck, not a stronger deck beating a weaker deck.

However, it is still an asymmetric policy matchup: combine has search/blend and heuristic does not. That is expected. The missing attribution question is how much of the 0.831 comes from search alone.

Therefore the key missing same-run baseline is:

- search vs heuristic, current code/current deck/current runner

If search also crushes heuristic around 0.8, then combine's big win is mostly just search. If search is much lower, then combine did something interesting against heuristic even though it still loses to search.

## Evaluation of the other model's proposed options

The other model asks whether to choose:

1. Deeper search (2-ply)
2. Big learned bet
3. Both in parallel
4. Other

Codex recommendation: choose `3`, but constrain it hard.

Recommended answer:

"Both in parallel, but with deeper search as the near-term scoring path and the learned path narrowed to a disciplined representation/policy experiment. Do not let the learned path sprawl. First finish/record the current A/B baseline, especially search vs heuristic. Then build 2-ply/deeper search with determinization cleanup as the concrete win attempt. In parallel, continue the learned bet only as clean action-ranking infrastructure: card-effect features + grouped/listwise objective + forward delta features, evaluated by win-rate only after it beats the right diagnostic baselines."

## Why not choose only deeper search?

Deeper search is the most concrete near-term path to beating `agent_search`. It directly attacks the known weakness: 1-ply search plus rollout is shallow.

But choosing only deeper search risks abandoning the user's real goal: a flexible learned multi-deck agent.

So deeper search should be the main near-term performance branch, while learned action-ranking remains a smaller, gated research branch.

## Why not choose only the big learned bet?

Because three fair tests are now negative or inconclusive:

- learned value/blend loses to hand search
- static listwise/pointwise diagnostics are confounded by option-0 and do not clear the bar
- immediate delta ranker did not beat option-0 on mixed top-1, even though top-3 signal exists

That does not mean learning is impossible. It means making learning the only next path is high variance and likely to produce more emotional whiplash.

## Major problems to avoid going forward

1. Do not say "learning is dead."

What failed is specifically learned value at the search leaf on current features/depth. Learned move-ranking with richer action semantics has not been fully tested as a live policy improvement.

2. Do not say "combine was useless" without naming the baseline.

Combine beat heuristic decisively. Combine appears worse than search. Both are true.

3. Do not keep asking broad strategic questions after every noisy result.

Use gates. The user is exhausted by frequent pivots and ambiguous conclusions.

4. Do not trust imitation top-1 as the final target.

It is a diagnostic. Win-rate vs `agent_search` is the actual gate.

5. Do not add all learned features at once.

Objective, card effects, deltas, belief, deck conditioning, and neural models need separate gates.

## Concrete next sequence

1. Finish or re-run the full current baseline table with current code:

- search vs heuristic
- combine vs heuristic
- combine vs search

Use same deck, seat-swapped, Wilson intervals, no errors.

2. If `agent_search` remains best, keep it as the submission/default candidate.

3. Build deeper search branch:

- determinization cleanup first
- search budget governor
- more determinizations if affordable
- 2-ply or opponent-reply branching
- evaluate vs `agent_search`, not heuristic

4. Continue learned branch only as a gated side branch:

- listwise/pairwise ranker grouped by decision
- option-0 baseline always
- card-effect features from `cards_full.json`
- immediate action deltas, not rollout leaf deltas unless explicitly testing rollout
- use as prior/tie-breaker/search-budget allocator first

5. Only promote learned component if it beats `agent_search` in live win-rate.

## Suggested exact answer to paste into the choice prompt

Choose option 3, but with constraints:

"Both in parallel, but prioritize deeper search as the near-term win path. Before building, finish and record the current A/B baseline table, especially search vs heuristic, because combine vs heuristic alone is not enough attribution. Keep `agent_search` as best unless combine beats it. For the learned path, do not do another broad learned-value leaf experiment. Narrow it to listwise action-ranking with card-effect/action-delta features and option-0 baselines, then only integrate it as a search prior/tie-breaker if it passes diagnostics."

## Emotional/communication note

The user is not confused because they are missing something simple. The project has produced contradictory-sounding statements because the comparisons changed:

- combine vs heuristic: good
- combine vs search: bad
- learned value alone/search_v: bad
- card-effect decoder: useful infrastructure, not yet live proof

Future updates should name the baseline every time and avoid vague phrases like "dead end" without specifying "dead end relative to what."
