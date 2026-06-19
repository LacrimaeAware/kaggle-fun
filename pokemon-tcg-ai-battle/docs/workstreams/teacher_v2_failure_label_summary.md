# Teacher V2 targeted labels for Model B's failure/test states

Model A labeled **exactly** the 15 states Model B requested in
`data/manifests/teacher_v2_label_request_for_A.json` (from B's failure analysis) -- not a generic batch.

Branch `exp/planner-teacher-v2`; `agent_search` remains the live baseline; no live agent consumed anything;
no arena screen.

## Result

- **Requested:** 15. **Labeled:** **15/15.** **Failed / unrecoverable:** 0.
- All states self-contained: `observation`, `legal_options`, `decision_id`, `obs_hash`, plus per-option
  `index`, `semantic_action_key`, `eq_class`, `hand_norm_advantage`, `hand_value_variance`,
  `completed_determinizations`, `outcome_winrate`, `outcome_playouts`, `outcome_se`; per-decision
  criticality, soft policy, acceptable set, margin/spread, coverage, timing, seed, paired flag, and the
  echoed `request_id` / `request_reason`.
- **Decks:** B-provided for all 15 (no production-deck fallback).
- **Criticality:** labeled regardless of criticality (B selected these), incl. one at 0.275 below the
  default gate.

| metric | value |
|---|---|
| hand-vs-outcome argmax disagreement | **8/15** (agree 7) -- ~0.53, matches the scaled batch's 0.52 |
| mean per-option outcome SE (k=16) | **0.014** (notably lower than the general batch's 0.044 -- the outcome is more decisive/reliable on these states) |
| all-siblings-completed | **15/15** (full coverage) |

## Artifacts

- `data/manifests/teacher_v2_labels_for_B_failures.jsonl` (15 labels)
- Generator/consumer: `tools/label_requested_states.py`

## Recommendation for Model B

Align these by `decision_id` / `obs_hash` + per-option `index` + `semantic_action_key` + `eq_class` (same
join as the scaled batch). **Primary target:** `hand_norm_advantage` (weight by criticality, inverse
`hand_value_variance`, coverage). **Auxiliary:** `outcome_winrate` -- and here the outcome SE is low
(0.014), so it is more trustworthy than in the general batch; still confidence-weight by `outcome_se`, do
not use its argmax as a hard label.

Diagnostic worth checking: the outcome disagrees with hand on ~half of these failure states. If B's previous
mis-predictions concentrate where hand and outcome disagree, that points to the hand-only label being
misleading on those states (label-source issue) rather than a pure model/objective failure -- in which case
the SE-weighted outcome auxiliary is exactly the signal to add.
