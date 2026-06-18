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
- **Phase / status:** Phase 1 `offline-evaluated` (leaf-only data can't rank; centering doesn't fix it,
  below option-0). Phase 2 `data-generated`: single deck = KanNinomiya (top by winner-decisions),
  action-conditioned imitation dataset `data/replay_db/action_imit.jsonl` (96,561 option-rows /
  11,628 winner-decisions; root + action descriptor + card_id + 11 decoded effects; target = the
  winner's actual move, non-circular). Phase 4 `trained/training` (tools/train_action_ranker.py): torch
  ranker = card EMBEDDING + effects + action + root -> listwise per decision, with no-embedding /
  no-effects / no-root ablations. Awaiting offline within-decision metrics vs option-0.
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
