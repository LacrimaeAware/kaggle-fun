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
- **Phase / status:** Phase 1 = `specified`. Plan written; no action-ranker code yet.
- **Exact next command:** check whether E013 candidate data exists (`data/` action dataset); if yes run
  Phase 1 (center target within decision, retrain, within-decision metrics); if no, go to Phase 2
  (build the proper action dataset with root+action+delta+shared-determinization).
- **Next acceptance gate:** offline within-decision top-1 / pairwise / regret on HIGH-CRITICALITY
  decisions, beating option-0 AND the hand-eval teacher, WITH ablations (no-effects / no-embedding /
  no-deltas). Only then a frozen win-rate A/B vs agent_search + heuristic.
- **Known blockers / confounders to close:** Team Rocket Energy wildcard semantics; public Stadium
  possibly duplicated in hidden-zone sampling; optional selections may be forced not declined; replay
  attribution. Simulator wall-clock is the bottleneck -> train OFFLINE on cached replay decisions.
- **Data:** ~275+ replays in `data/external/replays/` (gitignored); `tools/fetch_episodes.py
  --top-teams N` auto-pulls more (no manual linking). `agent/card_effects.json` decoded (581/1267).

## Status vocabulary (no more "done" = script ran)
specified -> implemented -> data-generated -> trained -> offline-evaluated -> arena-evaluated ->
accepted | refuted | inconclusive

## Binding rules (full list in ACTION_RANKER_PLAN.md)
1 card-effect claim needs card_effects.json consumed live. 2 action-ranking needs grouping + ranking
loss + within-decision metrics. 3 embedding claim needs trainable vectors in the scorer. 4 no
"validated" without the ablation. 5 AUC/Pearson are diagnostics only. 6 hand-eval-only labels cannot
beat hand search. 7 the objective is RANK SIBLING ACTIONS, never score a state (objective slippage).
