# Selected-Action Extra-Search Benefit Audit V1 (Model A)

Replaces the misleading all-options high-regret target. The useful question is NOT "can live search identify
bad siblings it already avoids" (tautological), but **"can we detect when the action live N=8 search actually
SELECTS is secretly high-regret, and would extra search fix it?"** Answer: **yes, with deployable live signals
-- verdict A (selective compute supported), pending a win-rate A/B.**

Offline audit only. No representation trained, no arena, `agent_search` unmodified, main untouched.
Reproducible: `tools/audit_selected_action_extra_search_v1.py`, `selected_action_extra_search_audit_v1.json`.
Data: `data/manifests/continuous_terrain_v1.jsonl` (700 non-eval decisions, 160 games).

## 1. All-options tautology recap (do not repeat the mistake)

`high_regret` over ALL options is ~tautological: it is defined as "value far below the best", and `live_gap`
(best live value - option live value) is the N=8 estimate of exactly that. An untrained `live_gap` scores
AUROC ~0.95 / AP ~0.94; on NON-selected options AUROC ~1.0; 95% of high-regret options are trivially flagged
by the live search. That number measures the search re-ranking its own options, not a deployable risk signal
(see `SEARCH_METADATA_DOMINANCE_AUDIT_ROUND2.md`). **All headline metrics below are selected-action-level.**

## 2. Decision-level / selected-action target definitions

For each decision, focus only on the action live N=8 search SELECTS (`search_selected_option`; forced moves
respected). Targets may use stronger-teacher info (that is the point):
- `selected_high_regret` -- selected action's `high_regret_prob >= 0.5` under the N=32 repeated teacher.
- `selected_unacceptable` -- selected action's (corrected, CI-overlap) `unacceptable_prob >= 0.5`.
- `extra_search_changes_action` -- `stronger_argmax_option != search_selected_option`.
- `extra_search_beneficial` -- changes AND the live pick is materially worse than the stronger best
  (selected high_regret, or selected regret beyond the paired noise band `z*(best_se + sel_se)`).
- `selected_action_instability` -- continuous, from the 8 live repeats: `1 - modal_action_stability`,
  `live_action_entropy`, `P(selected action changes across repeats)`.

## 3. Allowed / forbidden feature audit

Features must be available BEFORE the stronger search. **Allowed (deployable N=8 + free):** selected live
value, live top-2 margin, live value spread computed from `mean_live_value` (NOT `value_spread`), live
selected-action distribution / entropy / modal stability, selected live variance, criticality (no N=32),
option count, KO / KO-back / endgame flags, acting action type. **Forbidden (fail-loud guard in the script):**
`mean_stronger_value`, `stronger_value_variance`, `value_se`, `value_spread`, `stronger_soft_policy`,
`delta_to_search`, `delta_to_search_norm`, `hand_norm_advantage`, `*_prob` targets, `regret`, outcome fields.
**Guard status: passed** (no forbidden key read as a feature source).

## 4. Selected-action positives + game diversity (grouped by game)

| target | positives | games | top-game share | rate |
|---|---:|---:|---:|---:|
| `selected_high_regret` | **31** | 18 | 0.129 | 0.044 |
| `selected_unacceptable` | 54 | 34 | 0.093 | 0.077 |
| `extra_search_changes_action` | 221 | 85 | 0.086 | 0.316 |
| `extra_search_beneficial` | **72** | 42 | 0.069 | 0.103 |

Positives are well distributed (no game > 13%); not single-game-clustered. **Not underpowered:** both gating
targets exceed 20 (31 and 72), across 18 and 42 games -- the existing terrain dataset is adequately powered, so
no Step-7 mining was run.

## 5. Trigger table (operational; deployable live features only)

`xCompute` = average compute multiplier if a triggered decision is re-searched at N=32 (~4x N=8). SHR =
selected_high_regret, BEN = extra_search_beneficial.

| trigger | rate | SHR recall | SHR prec | BEN recall | BEN prec | xCompute |
|---|---:|---:|---:|---:|---:|---:|
| high live entropy | 0.46 | 1.00 | 0.096 | 0.986 | 0.220 | 1.84 |
| low modal stability (<0.99) | 0.46 | 1.00 | 0.096 | 0.986 | 0.220 | 1.84 |
| high selected live variance | 0.50 | 0.871 | 0.077 | 0.806 | 0.166 | 2.00 |
| high live value spread | 0.50 | 1.00 | 0.089 | 0.681 | 0.140 | 2.00 |
| high criticality (>0.3) | 0.71 | 0.323 | 0.020 | 0.625 | 0.090 | 2.85 |
| crit>0.3 AND modal<0.99 | 0.29 | 0.323 | 0.049 | 0.611 | 0.215 | 1.17 |
| crit>0.3 AND low margin | 0.34 | 0.097 | 0.013 | 0.278 | 0.083 | 1.37 |
| low top-2 margin | 0.50 | 0.194 | 0.017 | 0.333 | 0.069 | 2.00 |
| **learned logistic (grouped OOF)** | **0.31** | **0.871** | 0.125 | **0.653** | 0.218 | **1.23** |

Learned, leak-free, grouped out-of-fold (no game crosses folds): **selected_high_regret AUROC 0.895, 95% CI
[0.804, 0.960]; extra_search_beneficial AUROC 0.839, 95% CI [0.776, 0.893]** (game-clustered bootstrap, both
CIs exclude 0.5). AP is modest (0.32/0.33) because the positive class is rare (4-10%), so AUROC + the trigger
table are the honest read, not AP.

The deployable signal is real: instability (modal flips / entropy) and selected-action value variance flag the
secret failures. A conservative learned trigger firing on ~31% of decisions (1.23x average compute) recovers
87% of selected high-regret and 65% of beneficial cases; the instability rule (modal<0.99) recovers ~99% of
beneficial at 1.84x. Precision is low (0.05-0.22) -- but a false trigger only spends extra search, it does not
change the action wrongly (search still decides).

## 6. Oracle upper bound

| quantity | value |
|---|---:|
| % decisions where N=32 changes the action | 31.6% |
| % decisions where the change is beneficial | **10.3%** |
| % of selected_high_regret fixed by switching | 96.8% |
| mean regret avoided per beneficial switch | ~19,024 (hand-eval scale) |
| oracle trigger rate (= beneficial rate) | 10.3% |

A perfect trigger would re-search ~10% of decisions and materially improve the action on each (mean ~19k
regret avoided). This is a real ceiling: selective compute has something to win, and a learned live trigger
recovers most of it at 1.2-1.8x compute.

## 7. Mining

Not required -- the dataset is not underpowered (Step 4). Matched safe controls already present (the terrain
rings + 594 safe-selected decisions). Skipped Step 7.

## Recommendation

Implement a **conservative selective-compute trigger** from deployable live signals (the learned logistic on
live instability + selected-value-variance + criticality, or the simple `modal_action_stability < 0.99` rule),
re-searching the triggered decisions at N=32. Then -- and this is required before any deploy claim -- run a
**paired, seat-swapped win-rate A/B at equal AVERAGE wall-clock**: `agent_search` (uniform N=8) vs
`agent_search + selective N=32 on trigger`, budget-matched so the trigger's extra compute is paid for by
running N=8 elsewhere. Offline regret-reduction (this audit) is necessary but not sufficient; only the
win-rate A/B accepts a policy. Do NOT train another representation model on the all-options target.

Caveats kept honest: 31 selected_high_regret positives is modest (AUROC CI [0.80, 0.96] is wide-ish); the
beneficial target uses the N=32 teacher (itself a 4-repeat estimate) as ground truth; offline regret is not a
win-rate proof.

## Verdict

**A. SELECTIVE COMPUTE SUPPORTED.** Live deployable triggers catch the selected-action failures and
extra-search-beneficial cases at acceptable trigger rate (learned trigger: 87% SHR / 65% BEN recall at 1.23x
compute; instability rule: ~99% BEN at 1.84x). The signal is leak-free (forbidden-feature guard, grouped CV,
AUROC CI excludes 0.5) and non-tautological (selected-action target). Next step is a budget-matched win-rate
A/B, not another representation run.

Status: audit only, offline, reproducible, committed to `exp/planner-teacher-v2`. `agent_search` unchanged;
main untouched; no arena.
