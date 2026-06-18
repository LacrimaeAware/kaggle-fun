# Forward plan: a learned move-ranker on forward-model consequence features (win-rate judged)

> For review by another model. Grounded in `dropoff/inbox/2026-06-18-deep-research-report.md`,
> `docs/RESEARCH.md` (the 2026-06-18 sections), and the adversarial-verification findings.

## Why this, and what the evidence says

The recurring failure is that learned models never beat the hand-eval 1-ply search (`agent_search`,
0.585 vs the baseline opponent). The 2026-06-18 diagnostics established WHY the past learned work
stalled, and what is left to try:

1. **Imitation top-1 is the wrong yardstick.** Predicting a strong player's exact move is dominated
   by the engine's option ORDERING (chose-option-0 = 0.587 all / 0.468 mixed), which beats both our
   heuristic (0.553) and a learned ranker. Even a proper LISTWISE objective (LightGBM lambdarank)
   only reached 0.495 top-1 / 0.327 mixed — below the ordering prior. So the plateau is not just the
   pointwise objective. And our own agent wins 0.585 on WIN-RATE while scoring ~0.42 on imitation:
   the two metrics diverge. **Win-rate is the judge; imitation is at most a representation probe.**
2. **Static features carry only partial signal** (top-3 mixed 0.624 vs ~0.375 random) and cannot
   represent move consequences (evolve-into-X, this-attack-KOs, this-card-draws-3). The 47
   `encode_state` features are constant across the options of a decision, so they cannot rank moves.
3. **The one untested lever is forward-model CONSEQUENCE features** — simulate each option one ply in
   the real engine and feature what it actually DID. This is the deep-research report's #1 (sibling-
   action ranking, "validate only by head-to-head play") fused with its #4 (representation), and it
   is the concrete form of the user's "learned heuristics + richer features" bridge.

Two bugs were found and fixed along the way (do not reintroduce): the diagnostic's dead card-join,
and `eval.py` `evaluate_blend` NameError (blend/`agent_combine` silently fell back — a likely cause
of sub_combine's 358).

## The build (incremental; each phase gated by measurement)

### Phase 1 — forward-model consequence-delta extractor  [foundation]
A function `option_deltas(obs, deck)` that, for each legal option of a single-pick decision, applies
ONLY that option (`A.search_step(root.searchId, [i])`, NO full rollout) and returns structured
consequence features computed as the post-option state minus the root state, from my seat:
- `prizes_taken` (Δ my prize pile shrinking = I knocked something out),
- `opp_ko` (an opponent Pokemon left play / active HP hit 0; compare to CURRENT hp, not maxHp),
- `dmg_dealt` (Δ damage on the opponent active),
- `cards_drawn` (Δ my hand size), `energy_attached` (Δ my attached energy),
- `board_dev` (Δ my Pokemon in play), `my_hp_left`, `deck_left`, `discard_gain`,
- `ends_turn` (does this option pass control), and an optional bounded `opp_punish` (one opponent
  reply ply) only for the top-k candidates, to capture "this leaves me dead next turn".
Reuse `search.py`: `_api`, `_hidden_pool`, `search_begin`/`search_step`, `_obs_dict`. A single
determinization is enough for one-ply self-action deltas (my immediate effect barely depends on the
opponent's hidden cards). VALIDATE before trusting: print decoded deltas for sample decisions and
sanity-check (a KO attack shows `prizes_taken>0`; a draw supporter shows `cards_drawn>0`).

### Phase 2 — wire deltas into the imitation diagnostic (secondary, cheap)
Add the delta features to `tools/diag_action_ceiling.py` as a feature tier and re-run stratified vs
the option-0 baseline. This is a SECONDARY check (imitation is the wrong yardstick); its only purpose
is to confirm the deltas carry signal the static features lacked. Do not gate the project on it.

### Phase 3 — the move-ranker policy (the "learned heuristic")
A small, interpretable model: learned weights over the Phase-1 delta features, scoring each option;
the agent picks argmax. Train the weights on OUTCOME (did the seat win the game) from the replay
decisions and/or self-play — NOT on search values (avoids the label-circularity trap the report and
registry flag). Start linear/logistic (interpretable = the "learned heuristic"); only escalate to a
GBT/listwise model if the linear floor is promising. Determinize replay-based features with each
game's REAL deck (`data/replay_db/decks.json`), never the agent's own `DECK`.

### Phase 4 — WIN-RATE A/B  [the real gate]
Drop the ranker into a move-selection agent and A/B on the real engine via `agent/cabt_arena.py`:
- vs the heuristic, and vs `agent_search`,
- Wilson CIs, n ≥ 400–800, seats swapped,
- across 2–3 DIFFERENT decks (DENPA92's adopted deck + at least one other our agent can pilot) to
  measure matchup robustness, per the user's overfitting concern — not a single mirror.
Success = Wilson lower bound > 0.50 vs `agent_search` on the deck average. A clean negative is also a
result: it would exhaust the representation lever and point to the report's #2 (belief-conditioned
determinization with real opponent decks) or #3 (search-budget sweep) next.

### Phase 5 — iterate or pivot
If Phase 4 clears the bar, fold the ranker into search as a learned leaf/prior and run one expert-
iteration pass (report's roadmap endpoint). If not, pivot to belief-determinization (#2) / search
sweep (#3) with the same win-rate discipline.

## Guardrails (from the docs and the verifiers)
- WIN-RATE is the judge; imitation top-1 is a probe, not the target.
- Train on OUTCOME, never on the search's own values (circularity).
- Real per-game decks for replay determinization; never `M.DECK`.
- KO vs current `hp`, not `maxHp`. Don't trust `registry/card_review.json` tags as truth.
- Multiple decks + Wilson CIs + seat swap; no single-deck, single-split conclusions.
- Keep the cost inside the ~0.6s/decision budget (one-ply deltas are cheap; reserve any opp-reply
  rollout for top-k options only).

## Status
Phase 1 in progress (extractor + validation). Subsequent phases follow this file; updates appended here.
