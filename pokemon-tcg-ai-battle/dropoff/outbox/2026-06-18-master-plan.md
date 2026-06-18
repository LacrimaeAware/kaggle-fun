# Master plan: beat the heuristic, then the ladder (independent synthesis, 2026-06-18)

> My own synthesis after incorporating the deep-research report, the adversarial workflow, the audit
> handoff, and the Codex review. Where they conflict with my judgment I say so. Supersedes the
> single-track framing in `2026-06-18-forward-plan.md` (that doc's gates are folded in as Track B).

## Reframe (what is actually true)

The framing "we can't beat random/heuristic" is not what the measurements say:
- heuristic beats random 0.835; `agent_search` beats `first_agent` 0.585 and beats the heuristic 0.543.
- So SEARCH already beats the heuristic, marginally. We are not stuck at random.

The two real gaps:
1. **No LEARNED component has added to search** (every learned value/ranker ties the heuristic ~0.50).
   Part of why: two bugs (blend NameError, dead card-join) meant some "learned" results were never
   validly measured; the objective was pointwise; the features were state-only.
2. **We lose to the strong LADDER agents** (onechan1 ~1308 vs our ~615-630). Their edge is deck +
   policy tuned together; our generic policy can't pilot their evolution decks (measured: 0.21).

My judgment: the fastest way to "clearly beat the heuristic" is to make the SEARCH we already have
stronger (cheap, proven, no learning). The higher-ceiling, what-the-user-wants path is a learned
component that finally ADDS to search. Run both; lead with whichever yields a measured win first.
The single judge for everything is WIN-RATE on the real engine (cabt_arena, Wilson CIs, seat-swapped,
2-3 decks), not imitation top-1 (our agent wins 0.585 while scoring ~0.42 imitation — they diverge).

## Track A — make the search stronger (highest-probability near-term win)

A1. **Re-measure `agent_combine` (blend) now that the NameError is fixed.** It was crashing to
    hand-only on every decision, so the hand+learned blend has NEVER been validly tested. Cheap; a
    potential learned win hiding behind a bug. vs heuristic AND vs `agent_search`, n>=200.
A2. **Determinization sweep:** N_DETERM {4, 8, 16}. More worlds cut single-draw variance in the leaf
    average. Cheap. Watch the 0.6s/decision budget. Pass = Wilson lower bound > 0.50 vs `agent_search`.
A3. **Depth: branch on the opponent's reply (2-ply), not just a greedy rollout.** The known 1-ply
    weakness is that one of my moves barely changes the greedily-rolled leaf. Expectimax over my
    option x the opponent's best reply directly attacks that. More expensive; implement carefully and
    keep the time cap. This is the report's #3 (horizon) and the most principled non-learning lever.
A4. **Rollout policy variants** (current max-damage vs heuristic-as-rollout). Caution (Gelly-Silver,
    cited in the report): a "stronger" rollout can make search WORSE; measure, don't assume.

## Track B — a learned component that ADDS to search (the flexible-agent goal; Codex gates)

B1. **Gate 2 (immediate action-delta diagnostic).** Wire `search.option_deltas` (built, validated)
    into a grouped LISTWISE ranker, real per-game decks, stratified, vs the option-0 baseline. PASS =
    beat option-0 on MIXED decisions. This tests whether consequence features carry the signal the
    static features lacked. (Gate 1 / objective-only already ran and did not beat option-0.)
B2. **The learned-win bet I most believe in: a leaf evaluator trained on REAL replay OUTCOMES with
    RICH features, used inside the existing search.** Past learned values tied the hand eval because
    they used self-play + state-only features + (sometimes) circular search-value targets. The new
    ingredients: (a) labels = real game outcomes from strong-player replays (non-circular), (b)
    features = consequence deltas (`option_deltas`) + card-effect tags decoded from `cards_full.json`,
    (c) used as the search leaf (the framework that already beats the heuristic). If a leaf eval
    trained this way beats the hand formula at the leaf, the whole search gets stronger. This is the
    cleanest route to a LEARNED agent that beats the heuristic, and it is deck-agnostic by construction.
B3. **Gate 3 (live win-rate).** Any learned component that passes a diagnostic gate goes into play as
    an option prior / tie-breaker / leaf eval and is A/B'd vs `agent_search` across 2-3 decks. Win-rate
    decides; nothing is promoted on a diagnostic alone.

## Why not just imitate the winners?
Imitation top-1 is dominated by the engine's option ordering (option-0 = 0.587) and the label is
noisy (winners pilot different decks, make weak moves, win by variance). Use winner replays for the
OUTCOME signal (B2) and as a noisy demonstration prior, not as a top-1 target to optimize.

## Execution order (my call)
1. A1 (re-measure combine) + B1 (Gate 2 diagnostic) — both cheap, run first this week.
2. A2 + A3 (determinization + 2-ply) — the likeliest clear win over `agent_search`.
3. B2 (learned leaf on real outcomes + rich features) — the high-ceiling learned bet.
4. Re-evaluate; promote only what passes a win-rate A/B; then belief-determinization (report #2) and
   deck/meta robustness (multiple decks) layered on the winner.

## Hard rules (from the reviews, kept)
WIN-RATE is the judge. Option-0 baseline + stratification on any imitation diagnostic. One feature
family / one variable at a time. Real per-game decks for replay determinization, never `M.DECK`. KO
vs current `hp`. Wilson CIs, seat-swap, multiple decks; no single-split or single-deck conclusions.
Train on OUTCOMES, never on the search's own values (circularity).

## Results log
- **A1 done (2026-06-18), and it's a clean negative on the learned VALUE.** Now that the
  `evaluate_blend` NameError is fixed, `agent_combine` (search + hand/learned blend leaves) was
  validly measured for the first time:
  - combine vs heuristic: **0.831** [0.766, 0.881], n=160 -- but this is mostly the SEARCH (plain
    search also beats the no-search heuristic), not the learned value.
  - combine vs `agent_search`: **0.37** (22-38, n=60, stopped early) -- combine does NOT beat plain
    search; the learned-value blend at the leaf is a net negative, not a gain.
  - Conclusion: the learned-VALUE-at-leaf branch is confirmed a dead end (it was the bug hiding a
    non-result, not a working method hidden by a bug). `agent_search` (hand eval) remains best.
    This kills Track-A's A1 as a win path and sharpens the bet: the remaining live levers are
    forward-model ACTION features for a move-RANKER (B1/Gate 2 -- a different use of the model than a
    leaf value) and search DEPTH/quality (A2/A3). Do not keep tuning the leaf value.

## Status / next
NEXT: B1 (Gate 2 -- option_deltas move-ranking vs option-0) and A2/A3 (determinization + 2-ply search).
Do NOT submit `agent_combine` (loses to search). `agent_search` stays the best submission.
