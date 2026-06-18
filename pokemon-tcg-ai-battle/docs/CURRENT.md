# CURRENT — execution controller (check every step; the living docs are NOT the controller)

Updated: 2026-06-18

- **Branch:** claude/optimistic-proskuriakova-8800d3
- **Strongest agent:** `agent_search` (1-ply forward search + hand leaf eval). Nothing beats it. It is
  the submission default and the baseline every learned thing must beat.
- **Live weights:** `agent/value_weights.json` = the OLD board-summary value (used by search_v/combine,
  which LOSE). NOT used by the new ranker. The new ranker has no weights yet.
- **Submission ready:** `submissions/sub_search.tar.gz` = agent_search + DENPA92 deck (verified, 8/8 vs
  first). Upload is the user's call.
- **Active hypothesis:** H024-v2 (see `docs/ACTION_RANKER_PLAN.md`) — an action-CONDITIONED ranker
  (root + action descriptor + card embedding + decoded effects + state x effect interactions +
  option_deltas) ranks SIBLING actions within a decision better than the hand evaluator, with a
  NON-CIRCULAR target, judged within-decision then by win-rate.
- **Phase / status:** A first sim-free run was a STATIC-IMITATION SMOKE TEST (not a ceiling): static
  features only, evaluated over ALL decisions -> tied option-0 (0.549 vs 0.553). It skipped the two
  things the plan requires: forward-model DELTAS and CRITICALITY/non-option-0 stratification. Now doing
  the REAL Phase 2: `action_imit.jsonl` rebuilt WITH one-step option_deltas (prizes/KO/dmg/draw/board)
  per option, and the ranker (tools/train_action_ranker.py) reports STRATIFIED top-1 (all / non-option-0
  / high-criticality) with no-deltas / no-effects / no-embedding ablations. Training (Phase 4).
- **LABEL AUDIT (2026-06-18, tools/audit_action_labels.py) -- the prior imitation result was on
  CONFOUNDED labels:** the KanNinomiya-"deck" winner set mixes 18 distinct players (not one policy);
  72% of decisions have an EQUIVALENT sibling to the chosen option (exact top-1 mislabels them); only
  57% are genuinely strategic. On the STRATEGIC subset option-0 is 0.322 (NOT the 0.55 wall; 0.55 was
  inflated by trivial decisions). So "imitation ties option-0" is not a clean result. Clean version
  (one coherent player + canonicalized options + strategic-only, bar 0.322) is the next test. Per the
  other model's plan: imitation is an AUXILIARY prior, not the sole target; the main learned object is
  action-ADVANTAGE from multi-turn counterfactual search values. Performance lane = a bounded search
  sprint (determinizations / rollout / belief / continuation) frozen as the teacher first.
- **LANGUAGE RULE (from the methodology reviews):** do NOT call a direction failed / a "ceiling" /
  "the exact stack" until the experiment includes the consequence signal (forward-model deltas) and is
  reported on the non-option-0 + high-criticality strata. Aggregate top-1 vs option-0 is not the headline.
- **Exact next command:** Phase 2 -- build the proper ACTION-CONDITIONED dataset: rewrite
  `tools/datagen_actions.py` (or a v2) to log per option: root features + action descriptor (type,
  card/attack id, target, draws/tutors/evolves/attacks/ends, immediate KO/survival) + leaf features +
  leaf-minus-root delta + values across K SHARED determinizations + hand score + eventual result +
  criticality spread. Then Phase 4 model (centered-advantage GBM / pairwise logistic / small
  action-conditioned MLP with card embeddings + effects), reported with the component ablations.
- **Next acceptance gate:** offline within-decision top-1 / pairwise / regret on HIGH-CRITICALITY
  decisions, beating option-0 AND the hand-eval teacher, WITH ablations (no-effects / no-embedding /
  no-deltas). Only then a frozen win-rate A/B vs agent_search + heuristic.
- **Known blockers / confounders to close:** Team Rocket Energy wildcard semantics; public Stadium
  possibly duplicated in hidden-zone sampling; optional selections may be forced not declined; replay
  attribution. Simulator wall-clock is the bottleneck -> train OFFLINE on cached replay decisions.
- **Data:** ~622 replays in `data/external/replays/` (gitignored; see that folder for the live count); `tools/fetch_episodes.py
  --top-teams N` auto-pulls more (no manual linking). `agent/card_effects.json` decoded (581/1267).

## Status vocabulary (no more "done" = script ran)
specified -> implemented -> data-generated -> trained -> offline-evaluated -> arena-evaluated ->
accepted | refuted | inconclusive

## Binding rules (full list in ACTION_RANKER_PLAN.md)
1 card-effect claim needs card_effects.json consumed live. 2 action-ranking needs grouping + ranking
loss + within-decision metrics. 3 embedding claim needs trainable vectors in the scorer. 4 no
"validated" without the ablation. 5 AUC/Pearson are diagnostics only. 6 hand-eval-only labels cannot
beat hand search. 7 the objective is RANK SIBLING ACTIONS, never score a state (objective slippage).
