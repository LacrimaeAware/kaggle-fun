# Teacher V2 residual/risk labels (narrowed target) -- for Model B's residual/risk model

The learned model no longer replaces search's ranking; it learns a residual correction + catastrophic-risk
flags on top of `agent_search`. These labels supply that target. Branch `exp/planner-teacher-v2`;
`agent_search` remains the live baseline; no live agent, no screen, `main` untouched.

Tools: `agent/teacher_api_v2.residual_risk_label`, `tools/label_residual_risk.py`.
Artifact: `data/manifests/teacher_v2_residual_risk_labels.jsonl` (50 decisions, 451 options, 453 KB).

## How each option is labeled (paired worlds, one N=32 run)

`current_search_value` = mean over the first 8 worlds (the live N=8 search estimate); `stronger_value` =
mean over all 32 worlds (low-noise). Because they share the same worlds, `delta_to_search = stronger -
current` is paired and isolates where the live search is wrong (its MC error), not a different sample.

Per option: `current_search_value`, `stronger_value`, `delta_to_search`, `delta_to_search_norm` (centered
within decision), `value_variance`, `value_se`, `regret` (vs stronger best), `high_regret_flag`
(regret > threshold), `unacceptable_flag` (not CI-tied with best), `outcome_winrate`, `outcome_se`,
`outcome_playouts`, `completed_determinizations`, `index`, `semantic_action_key`, `eq_class`.
Per decision: `search_selected_option`/`search_argmax_option`/`stronger_argmax_option`,
`current_acceptable_set`, `criticality`, `observation`, `legal_options`, `decision_id`, `obs_hash`,
`source`, `state_tag` (B_failure_state | high_criticality), `coverage.all_siblings_completed`, `timing`.

## Summary

| metric | value |
|---|---|
| decisions / options | 50 / 451 (15 B-failure + 35 high-criticality) |
| residual `delta_to_search` | mean 3608, stdev 26717, **p05 -65, p50 0.0, p95 +163**, abs-mean 4326 |
| high-regret threshold | 5000 (hand-eval scale; recalibratable) |
| high-regret options | 13 |
| unacceptable options | 261 / 451 |
| hand-vs-outcome argmax disagreement | **36/50** (0.72 -- higher than the general 0.52, as expected on the hardest states) |
| mean value SE / outcome SE | 2850 (hand scale) / 0.035 |
| all-siblings-completed | 50/50 |

## Reading it

- **Residual is median-0, heavy-tailed.** Most options need ~no correction (trust search); a few have huge
  `delta` (live N=8 badly off, typically lethal/terminal-value states on the +-1e6 scale). This is exactly
  the residual-correction target: predict ~0 usually, a large correction on the tail.
- **Risk targets are populated** (13 high-regret, 261 unacceptable), so catastrophic-risk recall is learnable.
- **The outcome auxiliary carries information here** (72% argmax disagreement on these critical states), with
  low SE (0.035) -- usable as an SE-weighted auxiliary / risk signal, not a hard argmax label.

## Recommendation for B (complete enough to train)

- **Residual head:** target `delta_to_search_norm`; **clip the tail** (the 26717 stdev is a few huge
  lethal-scale deltas -- clip or robust-loss so the model doesn't overreact), weight by criticality and
  coverage. Integrate as `final = search_score + gate * predicted_residual`, gate initialized small (the
  median-0 distribution justifies a conservative gate).
- **Risk head:** target `high_regret_flag` / `unacceptable_flag` (catastrophic-risk recall); use to trigger
  extra search / deprioritize, not to bulldoze search.
- **Outcome:** SE-weighted auxiliary only.
- The 15 `B_failure_state`-tagged labels are the priority eval slice (where old ranker/option-0 beat the
  prior full model).

Status: implemented, offline; labels self-contained and training-ready. No live agent consumed them.
