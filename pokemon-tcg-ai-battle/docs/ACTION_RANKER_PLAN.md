# H024-v2: the action-conditioned sibling ranker (the plan we follow; do not deviate)

Authority: `dropoff/inbox/2026-06-18-roadblock-diagnosis.md` (ChatGPT) +
`dropoff/inbox/2026-06-18-methodology-compliance-review.md` +
`dropoff/inbox/2026-06-18-external-current-state-methodology-review.txt` + the deep-research report.
This is the execution plan. `docs/CURRENT.md` is the live controller. The living docs (RESEARCH.md
etc.) are NOT the controller.

## The one question (do not slip off it)

> Can an action-CONDITIONED model rank the consequential sibling actions of ONE decision better than
> the hand evaluator?

NOT "train another value." NOT "fit search targets better." NOT "add embeddings" as a side project.
NOT "assign a scalar to the resulting state." The failure so far was **objective slippage**: we kept
answering "score the resulting state" instead of "rank the sibling moves." E013 was marked done but
never satisfied H024 (raw uncentered target, generic regressor, no ranking loss, the model never even
received the action).

## Binding rules (a result that breaks these is mislabeled, not real)

1. Not "card-effect learning" unless `card_effects.json` is consumed by the live action model.
2. Not "action ranking" unless options are grouped by decision AND trained/scored with a
   pairwise/listwise objective AND reported with within-decision metrics.
3. Not "embedding learning" unless card vectors are trainable / represented inside the action scorer.
4. Not "validated" without the ablation: full vs component-removed (no-effects / no-embedding / no-deltas).
5. Global AUC / Pearson are diagnostics only, never an action-quality success claim.
6. Hand-eval-only labels cannot beat hand search (circular). The target MUST carry information beyond
   the hand evaluator.
7. Win-rate is the FINAL gate; within-decision offline metrics are the fast gate. Small-n directional
   reads, not tight CIs (full A/B only for finalists).

## Status vocabulary (replace "done"; "done" used to conflate three states)

`specified -> implemented -> data-generated -> trained -> offline-evaluated -> arena-evaluated ->
accepted | refuted | inconclusive`

## The model spine (the central object, built once, kept auditable)

```
per legal option of one decision:
  root-state features (47)                         # the starting circumstances
  + action/option descriptor                       # type; card/attack id; target class; attacks/attaches/
                                                    #   evolves/retreats/draws/tutors/ends; immediate KO/survival
  + learned card-id embedding                       # trainable vector per card (residual identity)
  + decoded card effects (card_effects.json)        # magnitude-aware scaffolding: draw N, search M, accel, ...
  + state x effect interactions                     # search_M*bench_space, draw_N*low_hand, accel*needs_energy
  + option_deltas (root->leaf consequence)          # prizes/KO/cards/energy/board, SHARED determinizations
  -> shared trunk
  -> action-ranking logits over the decision's siblings   # listwise/pairwise, grouped by decision_id
  -> [later] value head + auxiliary future-option head    # the separated heads
```

## Phases (in order; each gated; small models before big ones)

- **Phase 1 (cheap, sim-free, reuse E013 if it exists):** center the existing candidate target within
  each decision, A_{g,i} = y_{g,i} - mean_j(y_{g,j}); retrain the SAME model on the centered advantage;
  evaluate WITHIN decisions only (pairwise accuracy, top-1 agreement, rank corr, regret vs the teacher's
  best), stratified by candidate-score spread. Answers one isolated question: does a decision-relative
  target beat absolute leaf regression for local ranking? (Still teacher imitation, not a route to beat
  hand search; it isolates the centering variable.)
- **Phase 2 (data, sim only here, cached once):** build the proper action dataset. Per decision record:
  decision_id, game_id, root features, option/action descriptor, leaf features, leaf-minus-root delta,
  values across K SHARED determinizations (same hidden worlds across candidates so A doesn't win by a
  luckier draw), mean candidate return, variance, hand-eval score, eventual game result, forced/open
  flag, policy-disagreement flag, criticality spread C_g = max_i z - min_i z. Cache it; reuse across runs.
- **Phase 3 (non-circular target):** for a mined high-criticality subset, stronger labels via extra
  compute (more determinizations, longer opponent reply, multi-ply, multiple continuations, eventual
  outcome). Target z = a*deeper-search + b*outcome + c*hand (predeclare + ablate a,b,c). The target MUST
  contain info not in the hand eval. Use C_g to down-weight low-criticality; prove the high-impact tail.
- **Phase 4 (models, simplest first):** compare (1) centered-advantage GBM, (2) pairwise logistic
  ranking, (3) small action-conditioned MLP over [root, action, delta, leaf] with card embeddings +
  effects. The representation question is EMPIRICAL: embeddings earn investment only if the small NN
  beats the tree on held-out high-criticality within-decision ranking. Report all ablations (rule 4).
- **Phase 5 (integrate conservatively):** the learned ranker as a candidate-ordering prior / tie-breaker
  when hand scores are close / trigger for extra search on disagreement / pruning only after high recall.
  Forced rules (clean lethal/KO, required setup) stay OUTSIDE the learned system, so failure falls back
  to the known policy instead of collapsing below parity.
- **Phase 6 (gameplay gate):** offline metrics are top-1/pairwise/regret/high-criticality/stability;
  then a FROZEN head-to-head vs `agent_search` AND the heuristic, seat-swapped, predeclared noise band.

## Separate near-term gameplay route (optional, hand-eval throughout, does not touch the learner)

Search-mechanics sweep as a controlled matrix (not ad hoc long runs): determinizations {4,8,16},
rollout {default, aggressive, weak, partial-random}, realistic opponent decks from replays, extra
search only on high-branching / evaluator-disagreement decisions. First close correctness confounders:
Team Rocket Energy wildcard semantics, public Stadium duplicated in hidden sampling, optional
selections forced rather than declined, replay attribution/parser.

## Deep-research pipeline (the loop this plugs into)

collect (self-play + replays) -> state + sibling-action rows -> representation (card embeddings/effects)
-> [belief model] + [action ranker + value heads] -> belief-conditioned search -> selected move ->
head-to-head eval (Wilson) -> promote best -> repeat. Action ranking is the missing central piece; build
it first, attach belief/value/auxiliary heads after the ranker path is proven real.
