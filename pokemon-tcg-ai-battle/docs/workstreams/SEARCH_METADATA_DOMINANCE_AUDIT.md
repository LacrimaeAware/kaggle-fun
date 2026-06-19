# Search Metadata Dominance Audit (Model A audit of the Continuous Terrain V1 experiment)

Question: **is R1 (search-metadata-only) strong because it is a real deployable signal, or because the labels
leak from the same N=32 computation?** Verdict: **LEAKAGE.** R1's near-perfect score is label leakage; the
deployable live-only signal is far weaker. Three further defects (a dead-feature schema mismatch, a duplicated
target that is my bug, and degenerate task weights) make the representation gate **invalid as run**.

Reproducible: `tools/audit_search_metadata_dominance.py` + `data/manifests/audit_search_metadata_dominance.json`.
My replication of B's R1 matches B's reported numbers exactly (AP 0.742 vs 0.747, AUROC 0.990, recall 1.0),
which validates the audit harness. External auditors can re-run the script.

## 1-3. Leakage: which R1 fields leak, and what survives when they are removed

B's R1 (`train_continuous_terrain_v1.py`) is 8 features: `[margin, spread, value_var, value_se, entropy,
modal, criticality, is_selected]`. Provenance in the terrain schema:

| R1 field | source | deployable at inference (N=8 only)? |
|---|---|---|
| `spread` = `value_spread` | from `hand_norm_advantage` = **STRONGER N=32 eval** | NO -- leak |
| `value_se` | `sqrt(stronger_variance / 4)` = **STRONGER N=32 eval** | NO -- leak |
| `margin` = `current_search_value` top-2 | **key absent in schema -> always 0** | dead |
| `value_var` = `value_variance` | **key absent in schema -> always 0** | dead |
| `entropy` = `live_action_entropy` | live N=8 repeats | yes |
| `modal` = `modal_action_stability` | live N=8 repeats | yes |
| `criticality` | obs only (no engine) | yes |
| `is_selected` | live argmax | yes |

The high_regret TARGET is itself computed from the stronger N=32 runs (`regret = best_stronger -
option_stronger > 5000`). So `spread`/`value_se` are inputs derived from the same computation as the label. At
deployment the live agent has only the N=8 search; computing the N=32 stats IS the expensive work the risk
flag was meant to TRIGGER, so feeding them as inputs is non-deployable circularity.

High-regret prediction on B's held-out test games (same balanced-logistic probe, 763 test options, 33 positive):

| feature set | AP | AUROC | recall@FPR5 | recall@FPR10 |
|---|---:|---:|---:|---:|
| **R1_FULL** (B's R1: leak + live) | 0.742 | **0.990** | 1.00 | 1.00 |
| **LEAK_ONLY** (spread, value_se) | 0.612 | 0.985 | 1.00 | 1.00 |
| **JUST `spread`** (one leak field) | 0.632 | **0.988** | 1.00 | 1.00 |
| **LIVE_ONLY** (crit+entropy+modal+live_var+live_margin+sel) | 0.476 | **0.690** | 0.485 | 0.576 |
| **LIVE_MINIMAL** (entropy, modal, crit) | 0.053 | 0.367 | 0.121 | 0.121 |

`spread` ALONE reproduces R1 (AUROC 0.988). Strip the two stronger-derived fields and the deployable signal
collapses to **AUROC 0.690 / AP 0.476**. **R1's dominance is leakage, not a deployable signal.** (LIVE_MINIMAL
is near-random because entropy/modal/criticality are decision-level -- they say whether a decision is unstable,
not which option is the high-regret one.)

## 4. Extra-search trigger: underpowered, inconclusive

A deployable trigger must fire on LIVE signals and concentrate the decisions where extra search helps -- i.e.
where the live N=8 modal pick is itself high-regret (N=8 systematically picks a catastrophe that N=32 would
correct). On the 90 held-out test decisions there are **only 2 such "beneficial" decisions** (2.2%). The live
trigger `modal_action_stability < 0.99` fires on 31/90 and catches both (recall 1.0, ~2.9x lift) but at 6.5%
precision over n=2; `< 0.75` catches 1 of 2. The leaked `spread` trigger finds **0 of 2** (it flags decisions
with dangerous SIBLINGS, not decisions where the SELECTED option is bad). **No trigger conclusion is
supportable** on 2 positives. The 2 beneficial decisions both have stronger_argmax != live pick with mean
|selected regret| ~38,925, so extra search WOULD change the move when these states occur -- but the held-out
count is far too small to validate a policy.

## 5. The unacceptable table -- duplicated target (MY bug, now fixed)

B's `unacceptable` predictive table is byte-identical to the `high_regret` table for R0/R1/R2. Root cause:
`tools/label_terrain_v1.py` computed BOTH `high_regret_prob` and `unacceptable_prob` from the identical
condition `reg > hrt`, so `unacceptable_prob == high_regret_prob` for 100% of the 5,743 options. Round-2's
`residual_risk_label` used a DISTINCT CI-overlap criterion (`regret > z*(best_se + opt_se)`); the terrain
labeler dropped it. Fixed in the labeler; the existing dataset is patched (re-derived from stored
`regret` + `value_se`): the two targets now coincide on 48.7% of options, with `unacceptable` (CI-overlap)
flagging 3,348 options vs `high_regret` 599 -- a genuinely different, broader target.

## Further defects found

- **Schema-key contract mismatch (dead features).** B reads `opt["current_search_value"]` and
  `opt["value_variance"]`, neither of which exists in the terrain schema (the keys are `mean_live_value`,
  `live_value_variance`, `stronger_value_variance`). Both `margin` and `value_var` were silently 0, so R1 ran
  on 6 of 8 features. A simple "no feature column is all-zero" assertion would have caught it. The A-side doc
  should publish the exact option-field names so B reads `mean_live_value` for the live estimate.
- **Degenerate task weights.** The learned homoscedastic uncertainty weights all stayed ~0.99-1.01 (their
  init), so the multi-task balancing was effectively a no-op. The "learned weights" table carries no signal.
- **Neural R4 (AP 0.252) underperforms a logistic probe on the leaked features (AP 0.742).** Even WITH the
  leak available in its search-metadata branch, the trained encoder did not exploit it as well as a 2-feature
  logistic probe -- evidence of underfitting / a weak frozen-encoder probe, separate from the leakage.

## Reinterpreting B's verdict

B concluded **E. CURRENT SEMANTIC REPRESENTATION NOT VALIDATED**, comparing semantic R3 (AP 0.222, AUROC
0.757) against the dominant R1. But R1's dominance was leakage. The semantic representation is itself
LEAK-FREE: `semantic_vector` is card identity + static decoded effects + one-step forward-model deltas (a
single cheap `search_step`) + free `encode_state` context -- no N=32-derived field. Against the **deployable**
baseline (live-only metadata, AUROC 0.690 / AP 0.476), R3 semantic is **not dominated** (R3 AUROC 0.757 > 0.690;
live-metadata AP 0.476 > R3 0.222 -- genuinely mixed, both weak on a 4.3% positive class). So the fair read is
**inconclusive / underpowered, with the prior gate confounded by leakage** -- not "semantic decisively
dominated by search metadata."

## Recommendations (for the re-run)

1. **Re-run the gate with a deployable metadata baseline:** drop `spread`, `value_se`, and any N=32-derived
   field from R1; keep `criticality` + live N=8 stats only (entropy, modal, live_value_variance, live margin,
   is_selected). The honest R1 baseline is AUROC ~0.69, not ~0.99.
2. **Fix the schema-key contract** (read `mean_live_value` / `live_value_variance` / `stronger_value_variance`)
   and **assert no all-zero feature column**.
3. **Use the patched dataset** (distinct `unacceptable_prob`); re-derive both targets at label time with the
   corrected labeler.
4. **The extra-search-trigger question is underpowered** on this split (2 beneficial test decisions). It needs
   either many more held-out catastrophe decisions or a cross-validated/all-games estimate before any trigger
   claim; do not build a live screen (B's own conclusion, reinforced).
5. The leakage does **not** invalidate the dataset's distributions -- the leak was in how a stronger-derived
   field was used as a deployable INPUT. The terrain artifact's stronger-derived fields remain valid as
   TARGETS / analysis, just not as features.

Status: audit only; offline; `agent_search` unchanged; main untouched. Labeler fixed, dataset patched, both
copied to Model B's worktree. This audit and the harness are themselves open to the external auditor
(`tools/audit_search_metadata_dominance.py`).
