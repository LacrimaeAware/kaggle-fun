# Pokemon TCG Research Synthesis, 2026-06-19

Status: current synthesis across the pre-split methodology docs, Branch A planner/teacher work, and Branch B robust-learner work. This is a navigation and interpretation document, not a replacement for raw artifacts. When this conflicts with `docs/workstreams/BRANCH_PLAN.md`, the branch plan remains the methodological authority.

Live baseline: `agent_search` with DENPA92 and `N_DETERM=8`. No learned model, tactic floor, tactic prior, Teacher V2 model, or risk model has been promoted. `agent_search` remains the submission baseline until equal-budget evidence beats it.

## Executive Read

The project has converged on a clearer diagnosis:

- The strongest thing we have is still forward-model search plus a hand leaf evaluator.
- The old learning failures do not prove learning is hopeless. They mostly prove that global state values, hard top-1 imitation, and weak action features do not reliably turn into good move selection.
- The central object is now explicit: score legal sibling actions from the same root, using root context, action descriptors, card identity/embedding, decoded effects, target/entity features, state-effect interactions, immediate option deltas, and uncertainty.
- The final system is still search-guided or search-authoritative. The learner proposes, orders, allocates budget, triggers abstention/extra search, or breaks ties. Search evaluates and decides until equal-budget arena evidence says otherwise.
- Teacher labels are noisy. A single hard argmax is often the wrong label. The safe target family is soft policy, advantage/regret, acceptable set, confidence/stability, and risk flags.
- Terminal outcome is useful but weak. It is an auxiliary critic, not a hard primary label.

The most important current finding is uncomfortable but useful: the pieces are now finally being built close to the intended methodology, and the failures are more specific. We are no longer at "we did not build the thing." We are at "the data path works, but objective weighting, calibration, and rare tail-risk supervision are still not good enough to improve `agent_search`."

## Methodology That Should Remain Binding

The methodology docs and later coordination prompts agree on these rules.

1. Keep immutable data boundaries.
   - Rolling replay fetches can append raw data.
   - Experiments consume dated snapshots, fixed splits, hashes, and recorded deck/search config.
   - DENPA92 stays fixed for method comparisons unless a separate deck promotion gate is run.

2. Treat action choice as grouped sibling ranking.
   - The unit is one root decision with all legal sibling options.
   - Never split rows from the same decision across train/test.
   - Report within-decision top-k, pairwise accuracy, MRR/NDCG, acceptable-set agreement, and teacher-relative regret.
   - Option-0 is a real baseline because engine ordering is strong.

3. Use richer action-conditioned inputs.
   - Root state alone is insufficient.
   - The model must consume action descriptors, acting card/entity, decoded effects, target/entity properties, state x effect interactions, option deltas, and optionally short public history.
   - Card embeddings/effects/deltas are trainable inputs, not fixed final score weights.

4. Avoid circular success claims.
   - A learner trained only to clone the current search target is mostly a compression or stabilization device.
   - To improve policy, the target needs stronger information: higher-N/offline teacher, repeated counterfactual advantages, criticality/risk labels, terminal-outcome auxiliary, on-policy recovery states, or future integration into search.

5. Use soft/noisy labels correctly.
   - Teacher V1 and parts of Teacher V2 are unstable enough that hard argmax labels are often misleading.
   - Train with soft policy, action advantage, acceptable set, confidence weights, and stable/unstable partitions.
   - Outcome winrate is auxiliary and confidence-weighted by SE.

6. Evaluate on deployment-like states.
   - Replay-state accuracy is not enough.
   - Track on-policy learner/search states, before and after first unacceptable action, action type, high-criticality slices, high-regret tail, held-out game/player/deck, and group-held-out labels where applicable.

7. Promote only on gameplay evidence.
   - Offline metrics are fast gates.
   - A live candidate needs same deck, same wall-clock budget, seat swaps, multiple seeds, errors/timeouts recorded, and enough games for the stated gate.

## Major Methodology Sources

These docs explain why the plan changed.

- `dropoff/inbox/2026-06-18-methodology-compliance-review.md`: the project had often implemented nearby proxies, not the true root-action sibling ranker.
- `dropoff/inbox/2026-06-18-external-current-state-methodology-review.txt`: the cleanest diagnosis was objective slippage, meaning scoring resulting states instead of ranking sibling actions.
- `dropoff/inbox/2026-06-18-card-effects-action-prior-handoff.md`: card effects are affordances, not values. They must be state-conditioned action inputs or residuals.
- `dropoff/outbox/2026-06-18-CONSENSUS-and-way-forward.md`: the integrated stack should be built, but benchmarks must be unfakeable.
- `docs/workstreams/BRANCH_PLAN.md`: final split plan: Branch A builds a stronger offline teacher/search evidence; Branch B builds robust contextual learning and recovery machinery.

## Timeline And Findings

### 1. Pre-split learning and search lessons

Finding: `agent_search` became the live baseline. Older learned value/blend/ranker efforts did not beat it.

Key lessons:

- Global state-value prediction is not action quality.
- Search-value distillation can fit the current teacher without improving play.
- Winner imitation is noisy and option-order-confounded.
- Card-effect and embedding claims were sometimes invalid because the tested agent did not consume those artifacts.
- Broad "done" statuses were misleading.

Practical status: keep `agent_search` as baseline. Do not resurrect old state-value or old-objective DAgger paths without a new concrete bug fix or target.

### 2. Shared split base

The split base froze the core assumptions:

- DENPA92 deck for method comparison.
- Immutable replay snapshot/split.
- State/action schema and golden fixtures.
- Teacher API V1 with variance, soft policy, acceptable set, advantage, and semantic action keys.
- Two branches from the same base: `exp/planner-teacher-v2` and `exp/robust-learner-v2`.

Practical status: this structure should remain. Do not merge either branch into main casually.

### 3. Branch A A2: Teacher V1 stability

Replay audit:

- 1094 replay decisions.
- Stable fraction: 0.505.
- Near-tie fraction: 0.212.
- Unstable fraction: 0.282.
- Mean cross-seed top-action stability: 0.783.
- Engine-only stability: 0.772.
- Determinization extra instability: about -0.011.

Production self-play audit:

- 999 self-play decisions.
- Stable fraction: 0.402.
- Near-tie fraction: 0.259.
- Unstable fraction: 0.338.
- Mean cross-seed top-action stability: 0.748.
- Engine-only stability: 0.774.
- Determinization extra instability: about +0.026.

Interpretation:

- Teacher V1 is noisy on both replay and production-self-play states.
- The noise is mostly engine rollout RNG, not hidden-world determinization sampling.
- A hard argmax teacher label is unsafe for roughly half of non-forced strategic decisions.
- More/better averaging and selective computation matter more for stability than belief/world sampling, though this does not refute opponent belief as a strength hypothesis.

### 4. Branch A practical search/tactic work

Hard floors:

- Draw floor: 10-20, discarded.
- Evolve floor: 14-16, wash.
- Gust+evolve pooled: 33-37, wash.

Tactic Miner V1:

- Built ontology and miner.
- Mined 46,889 winner sibling decisions.
- Found sensible patterns, including gust-when-KO and attack-when-no-setup.
- Implemented a near-tie soft prior.
- Screen: `agent_search_prior` 13-17, wash.

Interpretation:

- Unconditional floors are too blunt.
- Winner-replay tactics are partly correlation/survivorship, not causal optimality.
- Search is decent on average; its problem is noise/tail cases, not a simple tactical blind spot.
- Tactic ontology/miner may still help as features or labels for Branch B.

### 5. Branch A Teacher V2 labels

Teacher V2 path:

- Criticality-gated high-N hand advantage.
- Terminal-outcome auxiliary.
- Self-contained records with root observation, legal options, semantic keys, eq classes, variance, coverage, timing, seeds.

Outcome validation:

- Pilot: outcome argmax disagreed with hand argmax 6/8, but k=4 was noisy.
- Higher-k validation: disagreement persisted around 0.58, but outcome argmax stability across two k=32 runs was only 0.50.
- Scaled batch: 50 high-criticality decisions, k_outcome=16, hand-vs-outcome disagreement 26/50 = 0.52, mean outcome SE 0.044, all siblings completed.

Interpretation:

- Outcome carries real information beyond the hand leaf.
- Outcome argmax is not reliable enough to be the primary label.
- Primary target for B is `hand_norm_advantage`, weighted by criticality, inverse hand variance, and coverage.
- Outcome is only an SE-weighted auxiliary.

### 6. Branch B B1 diagnostics

B1.1 representation ceiling:

- On stable Teacher V1 labels, root-only was weak.
- Root plus action nearly memorized stable labels: expected top-1 about 0.958/0.979/0.979 train/val/test.
- Conclusion: action identity/semantic key is the necessary discriminative ingredient.

B1.2 teacher noise:

- Teacher V1 top-1 labels were frequently unstable.
- Stable labelled decisions: 64 per partition.
- Unstable labelled decisions: 162/189/106 train/val/test.
- Conclusion: use soft policy, advantage, acceptable sets, and confidence weighting.

B1.3 on-policy shift:

- Old `rank` vs heuristic, 100 games: 18-82.
- Student visited decisions: 2980.
- Teacher-applicable: 2939.
- Acceptable agreement: 0.688.
- Hard top-1: 0.439.
- Mean regret: 30694.7.
- High-regret decisions >=1000: 335.

Before/after first unacceptable action:

- Before: acceptable 0.965, mean regret 1187.8, high-regret 5.
- First unacceptable: acceptable 0.103, mean regret 53275.6, high-regret 21.
- After: acceptable 0.634, mean regret 38202.4, high-regret 308.

Interpretation:

- The old ranker has a real compounding on-policy failure mode.
- It works much better before the first unacceptable action than after.
- This supports DAgger/recovery-state training as a valid diagnostic path, not as a guaranteed fix.

### 7. Branch B DAgger rounds

Round 1:

- Kept old ranker architecture.
- Mixed base distillation, stable replay, and recovery states.
- On-policy metrics improved materially.
- Old B1.3 mean regret 30694.7 -> Round 1 6463.3.
- High-regret 335 -> 82.
- After-first-unacceptable high-regret 308 -> 74.
- But mixed offline target fidelity got worse.

Round 2:

- Tried loss/weight calibration, lower LR, anchoring, lower recovery weights.
- Recovered some offline damage vs Round 1 but did not recover to Round 0.
- On-policy metrics worsened vs Round 1: mean regret 6463.3 -> 8631.6, high-regret 82 -> 101.

Interpretation:

- DAgger Round 1 proved recovery-state training can improve the intended failure mode.
- Round 2 failed its gate.
- Do not run Round 3 on the old ranker/objective.

### 8. Branch B contextual action ranker

Contextual Action Ranker V1 finally built the intended integrated model:

- root state features;
- option/action descriptor;
- acting-card embedding;
- decoded card effects;
- target/entity features;
- state x effect interactions;
- immediate option deltas;
- short public history;
- grouped sibling-action targets.

It deployed conservatively as `agent_search_ctx`: search remained authority, model ordered candidates and broke exact-value ties.

Initial V1:

- Offline replay test did not beat old ranker.
- 20-game `search_ctx` vs `search`: 11-9, directional only.
- On-policy diagnostics were mixed.

Calibration pass:

- Found normalization bug: many near-zero std features caused huge normalized live values.
- Fixed with `min_std=0.05` and clipping.
- Calibrated `search_ctx` vs `search`: 10-10.
- On-policy p90/p95 looked low, but mean regret was dominated by extreme outliers.

Interpretation:

- The contextual model path is real and wired.
- Decoded effects, embeddings, and target features are actively consumed.
- Option deltas remain suspect/calibration-sensitive.
- The live result washed. No promotion.

### 9. Teacher V2 contextual retrains

Data path:

- Old 160-decision contextual dataset only matched 8/50 Teacher V2 decisions.
- Direct featurization path solved this: 50/50 roots reconstructed, 404/404 options aligned.
- Targeted B-failure labels later made all held-out test rows Teacher V2-labelled.

First Teacher V2 retrain:

- Teacher V2 model top1 0.333, acceptable 0.667, mean regret 108.9.
- Old ranker top1 0.429, acceptable 0.857, mean regret 74.1.
- Option-0 top1 0.476, acceptable 0.714, mean regret 89.8.

Post-label failure analysis:

- All held-out rows eventually had Teacher V2 labels.
- Failure no longer looked like label-source mismatch.
- It looked like objective/weighting/model-calibration failure.

Objective v2:

- Revised full model still failed:
  - revised full top1 0.238, acceptable 0.619, mean regret 80.42.
  - old ranker top1 0.429, acceptable 0.810, mean regret 66.97.
  - option-0 top1 0.476, acceptable 0.667, mean regret 82.19.
  - no decoded effects mean regret 33.47, warning that full feature mix overreacts or miscalibrates.

Decision:

- Pause this Teacher V2 contextual-ranker retrain path.
- More blind objective tuning is not justified.

### 10. Residual/risk path

Rationale:

- A broad residual correction is dangerous because the residual distribution is median-zero and heavy-tailed.
- Most actions need no correction; a few are catastrophic.
- Risk-only should be understood before residual correction.

B-bootstrap risk prototype:

- Risk-only improved a small bootstrap held-out slice.
- Caveat: labels were B-bootstrap, not Model A high-compute labels.

A-label risk-only:

- A produced 50 decisions / 451 options, 13 high-regret options, 261 unacceptable.
- B ingested 50/50 decisions and 451/451 options.
- New A-label risk-only classifier had detection signal:
  - high-regret recall 0.833 on held-out test options.
  - unacceptable recall 0.692.
  - false-positive risk rate 0.200.
- But selected-action safety got worse:
  - `agent_search` mean regret 1425.57, p95 20.82, acceptable 1.000.
  - new A-label risk-only mean regret 1430.51, p95 104.12, acceptable 0.955.
- It missed the crucial catastrophic search-selected action and falsely blocked one safe search choice.

Decision:

- Do not integrate `agent_search_risk`.
- Request more/different targeted labels.

Round-2 targeted risk labels:

- A mined targeted enrichment around two failures: missed catastrophic search-selected high-regret and safe false-positive block.
- Delivered 60 labels / 588 options.
- Search-selected high-regret states: 9.
- High-regret options: 127 / 588.
- Unacceptable options: 404 / 588.
- Safe-search false-positive class: 49 states.
- Both seed states present and hash-verified.
- Group/provenance fields added.

Critical finding:

- Search-selected catastrophic states are rare and concentrated in the highest-criticality tier.
- The c1 label is only about 53-56% reproducible; the selected option can flip across independent labels because engine rollout RNG is not fully seedable.
- Raw regret magnitude is noisy; high-regret flag is the more stable target.
- c1 positives cluster by game, so B must use `group_id` for group-held-out splits.

Recommended use:

- Primary risk target: `high_regret_flag`.
- Auxiliary: `unacceptable_flag`.
- Treat c1 as soft/upweighted, not perfect truth.
- Prefer recall plus abstain/extra-search fallback.
- Never let a risk model freely reorder all actions.
- Hold out the two seed states as priority eval, especially the exact missed catastrophic and safe false-positive cases.

## Current State By Branch

Branch A, `exp/planner-teacher-v2`:

- Latest visible commit in worktree: `ad03f34`, round-2 risk-label enrichment for B.
- Current contribution: Teacher V1 stability audits, tactical screens/miner, Teacher V2 labels, residual/risk labels, targeted round-2 risk enrichment.
- Status: no live agent change, no arena promotion, no main merge.

Branch B, `exp/robust-learner-v2`:

- Latest committed B model package before copied A labels: `de17f7f`, A-label risk-only evaluation.
- It has copied round-2 risk-label artifacts available in the worktree for the next B task.
- Current next work, if authorized: consume round-2 risk labels and retrain exactly one high-regret-primary risk-only model with threshold calibration. No live screen unless the missed catastrophic case is fixed and offline safety improves.

## What We Should Not Do Next

- Do not promote any learned model.
- Do not run DAgger Round 3 on the old ranker/objective.
- Do not run another broad Teacher V2 contextual objective tuning pass without a new targeted reason.
- Do not add a big architecture, affordance heads, or independent embedding optimization before the current risk/contextual gates are resolved.
- Do not treat outcome argmax as ground truth.
- Do not treat raw regret magnitude in the rare c1 risk labels as clean regression truth.
- Do not change deck and method together.
- Do not merge to main without a review gate.

## Most Plausible Next Steps

### Immediate Branch B step

Consume `data/manifests/teacher_v2_risk_labels_for_B_request.jsonl` after verifying it is the round-2 content.

Required checks:

- decisions loaded;
- options aligned;
- high-regret positives present;
- c1 seed included;
- c2 seed included;
- `group_id`, `eval_only`, `c1_candidate`, and `c1_reproduced_this_label` fields available;
- group-held-out split is used.

Train one risk-only model:

- high-regret primary;
- unacceptable auxiliary;
- class weighting;
- threshold calibration;
- optimize high-regret recall first, then false positives;
- hold out seed states as eval-only.

Offline decision rule:

- A: catches catastrophic actions with acceptable false positives and improves safety -> tiny live screen proposal.
- B: improves but needs more independent c1 labels -> request targeted mining from A.
- C: still misses key catastrophes -> pause risk-only path.
- D: useful diagnostic only -> no live guide.

### If B needs more labels

Ask A for more independent c1 games, not more near-duplicates from the same top games. The round-2 mine found c1 is roughly top-criticality-only and rare, so scaling should be disjoint/sharded and group-aware.

### Integration direction, later

The intended final shape remains student/risk-guided search:

- search chooses normally;
- model may order candidates, allocate budget, or trigger extra search;
- risk model may abstain or reduce priority of dangerous actions;
- model must not bulldoze search or freely reorder everything without evidence.

Final gate remains equal-budget `agent_search_ctx` or `agent_search_risk` vs `agent_search`.

## Document Hygiene

The docs now fall into categories:

- Current authority: `docs/workstreams/BRANCH_PLAN.md`.
- Current synthesis: this file.
- Current workstream index: `docs/workstreams/README.md`.
- Branch A source docs: `PLANNER_TEACHER_V2.md`, `A2_SELFPLAY_ADDON.md`, `TACTICAL_SCREENS_V1.md`, `PLANNER_TEACHER_V2_A3.md`, `teacher_v2_residual_risk_summary.md`, `teacher_v2_risk_label_request_summary.md`.
- Branch B source docs: `ROBUST_LEARNER_V2.md`, `ROBUST_LEARNER_V2_DAGGER_ROUND1.md`, `ROBUST_LEARNER_V2_DAGGER_ROUND2.md`, `CONTEXTUAL_ACTION_RANKER_V1*.md`, `CONTEXTUAL_ACTION_RANKER_TEACHER_V2.md`, `teacher_v2_*failure_analysis*.md`, `contextual_*risk*.md`.
- Methodology source material: `dropoff/inbox/*methodology*`, `dropoff/inbox/*deep*`, `dropoff/inbox/*card-effects*`, and `dropoff/outbox/2026-06-18-CONSENSUS-and-way-forward.md`.
- Historical/superseded: older master/forward plans, old H024/E013 framing, first-pass risk labels where round-2 explicitly supersedes them, old learned-value/leaf/blend paths.

Do not delete the old docs. They are useful provenance. Use the index to keep future work from treating stale plans as current instructions.
