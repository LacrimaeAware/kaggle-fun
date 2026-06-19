# Clean-Rerun Re-Audit (Round 2): is B still making mistakes?

Audits Model B's clean rerun (`exp/robust-learner-v2` @ `cfcf207`, verdict **C: LIVE METADATA IS THE USEFUL
SIGNAL**, strict-live high-regret AP 0.985). Short answer: **B fixed the gross leak correctly, but verdict C
misreads a tautology as a discovery.** Independently verified by a 3-lens workflow; reproducible via
`tools/audit_clean_rerun_decomposition.py`.

## What B got right (credit)

B genuinely removed the N=32 leak. The rerun forbids every stronger-derived field (`value_spread`, `value_se`,
`mean_stronger_value`, `stronger_value_variance`, `regret`, `delta_to_search`, `*_prob` targets), builds R1
only from N=8/free fields (`mean_live_value`, `live_value_variance`, live distribution/entropy, modal
stability, criticality, rank, index), and adds the assertions I recommended (correct schema keys, no all-zero
column, no near-perfect target column) -- all passing. The narrow factual claim "semantic R3 does not beat the
honest live baseline on this gate" is true and reproduces (R1 0.985 vs R3 0.328). No `agent_search` change, no
arena, no merge.

## The mistake: the headline gate is near-tautological

`high_regret` is DEFINED as "value far below the best option" (regret > 5000 under N=32). `live_gap` = (best
live value - this option's live value) is the N=8 estimate of exactly that quantity, and N=8 ~ N=32 on easy,
large-margin options (corr(live_value, stronger_value) ~ 0.999; corr(live_gap, stronger_gap) ~ 0.998). So
"predict high_regret across all options from live metadata" mostly re-expresses the label in free units.

| slice (held-out test games) | AP | AUROC | pos/n |
|---|---:|---:|---:|
| strict-live R1, ALL options (B's headline) | 0.984 | 0.999 | 33/763 |
| **`live_gap` ALONE (one untrained feature)** | **0.939** | 0.948 | 33/763 |
| strict-live R1, NON-SELECTED options only | **0.992** | **1.00** | 31/673 |
| decision-live feats, SELECTED option only (the useful task) | 0.75* | 0.99* | **2**/90 |

`live_gap` alone (zero learning) nearly matches the trained R1 (median live_gap ~994,388 for positives vs ~27
for negatives, a ~37,000x separation). **95% of high_regret options (564/592) have live_gap >= 5000** -- the
cheap N=8 search already screams they are bad; only **5% are "sneaky"** (live thinks them playable). The gate
is decided ~95% by cases the experiment question does not depend on, and R1 is recovering the search's OWN
value ordering, not a learned risk signal. (* the selected-only row has 2 test positives -- not a real number.)

**Consequence: the semantic gate is unwinnable by construction.** The target is near-monotone in value and the
baseline uses value directly, so R3 "failing" says nothing about whether semantics carry orthogonal signal.
Verdict C should be downgraded to: *live value restates the target in free units; neither the live baseline nor
the semantic encoder is shown to carry risk signal beyond what the search already exploits.*

## Three further issues in the rerun

1. **New self-leak on the `instability` target.** R1 scores instability AP 1.000 / AUROC 1.000 -- because the
   instability target IS `live_action_entropy`, which is an R1 INPUT. Predicting entropy from entropy is
   circular; that row is meaningless.
2. **Base-rate-dominated targets.** `acceptable` (734/763 = 96% positive) gives AP ~1.0 for everything;
   `unacceptable` (467/763 = 61%) is similarly inflated. AP across targets with 4%-96% base rates is not
   comparable; report lift / AUROC / calibration, not raw AP.
3. **The one non-tautological table is ignored.** B's own `selected_high_regret` table (live-argmax option
   secretly high-regret -- the only slice where live value is blind, `live_gap = 0` by construction) has just
   **2 held-out positives**, where R1 AP is 0.333. The verdict rests on the trivial all-options table, not this.

## Power: the split cannot answer the question

2 held-out selected-catastrophe positives give SE(AUROC) ~0.17 (95% CI half-width ~0.34) -- no representation
difference is detectable. The corpus has ~31 selected-catastrophe positives across 18 games; a single 23-game
test split puts ~2 in test. You need grouped **leave-one-game-out CV over all 160 games** to pool ~15-30
held-out positives.

## Recommended next experiment (supersedes another representation run)

Re-run R1/R3/R4 on the **SELECTED-option catastrophe** target (and the value-residual slice: only options with
live_gap < 5000) under **grouped leave-one-game-out CV over all 160 games**, scored as **paired delta-AUROC
over a fixed cheap-live baseline** (selected `live_value` + `live_value_variance` + `live_action_entropy` +
`modal_action_stability`), with a game-clustered bootstrap CI on the delta. A semantic representation validates
only if its delta-CI **excludes 0** against that baseline. On train evidence this is a genuine contest (live
instability AUROC ~0.75 vs raw selected value ~0.80), so one properly-powered run is warranted before
concluding "live metadata is the only signal." B's own next-step instinct (selective-compute / instability
trigger over another representation run) is right; this is how to power it.

Status: audit only; offline; reproducible (`tools/audit_clean_rerun_decomposition.py`); independently
verified. `agent_search` unchanged; main untouched. Copied to Model B.
