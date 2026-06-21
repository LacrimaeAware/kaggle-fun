# Workstreams Document Map

Status: cleanup index added 2026-06-19. This file does not delete or replace older docs; it marks which ones are current, historical, or superseded so the workstream folder is easier to navigate.

## Read First

1. `BRANCH_PLAN.md`
   - Authoritative methodology for the split: Branch A planner/teacher, Branch B robust learner, student-guided search as final integration.
2. `RESEARCH_SYNTHESIS_2026-06-19.md`
   - Current high-level synthesis of findings, experiments, methodology, and next gates across both branches.
3. This file
   - Navigation map for the workstream document pile.

## Current Branch A Sources

- `PLANNER_TEACHER_V2.md`: A1/A2 Teacher V1 stability audit and initial A3 rationale.
- `A2_SELFPLAY_ADDON.md`: production self-play Teacher V1 stability comparison.
- `TACTICAL_SCREENS_V1.md`: tactical floors, Tactic Miner V1, and soft-prior screen. Current read: no tactical submission candidate.
- `PLANNER_TEACHER_V2_A3.md`: Teacher V2 label generator, outcome auxiliary validation, scaled n=50 Teacher V2 batch.
- `teacher_v2_residual_risk_summary.md`: first residual/risk label batch, 50 decisions / 451 options.
- `teacher_v2_risk_label_request_summary.md`: round-2 targeted risk-label enrichment. Current read: trust this over the first risk request for B's next risk-label work.

## Current Branch B Sources

- `ROBUST_LEARNER_V2.md`: B1 diagnostics, representation ceiling, teacher stability, on-policy shift.
- `ROBUST_LEARNER_V2_NEXT.md`: post-B1 memo. Historical now, but useful for the diagnostic conclusions.
- `ROBUST_LEARNER_V2_DAGGER_ROUND1.md`: Round 1 DAgger pilot, directionally positive on on-policy regret/high-regret.
- `ROBUST_LEARNER_V2_DAGGER_ROUND2.md`: Round 2 calibration attempt, failed continuation gate. Do not run old-objective Round 3.
- `CONTEXTUAL_ACTION_RANKER_V1.md`: first integrated contextual sibling-action model and search-guided screen.
- `CONTEXTUAL_ACTION_RANKER_V1_CALIBRATION.md`: normalization fix, ablations, washed 20-game confirmation.
- `CONTEXTUAL_ACTION_RANKER_TEACHER_V2.md`: Teacher V2 direct featurization path and first retrain.
- `teacher_v2_failure_analysis.md`: analysis before targeted Teacher V2 labels. Historical because labels were later supplied.
- `teacher_v2_post_label_failure_analysis.md`: post-label analysis. Current read: not a label-source mismatch; objective/calibration issue.
- `contextual_action_ranker_teacher_v2_objective_v2_summary.md`: revised objective failed; Teacher V2 contextual-ranker path paused.
- `contextual_residual_risk_v1_summary.md`: B-bootstrap residual/risk prototype. Historical/caveated.
- `contextual_risk_only_v1_summary.md`: A-label risk-only evaluation. Current read: detection signal but selected-action safety did not improve; requested targeted labels.
- `teacher_v2_risk_label_request_summary.md`: A's targeted risk-label response copied into B. Use for the next B risk-only pass.

## JSON Artifacts Worth Knowing

- `robust_learner_v2_b1_3_rank_100g.json`: old-ranker on-policy shift diagnostic.
- `contextual_action_ranker_v1_calibrated_train_eval.json`: calibrated contextual model ablations.
- `contextual_action_ranker_teacher_v2_objective_v2_eval.json`: failed Teacher V2 objective v2 offline eval.
- `contextual_risk_only_v1_eval.json`: failed A-label risk-only offline eval.
- `teacher_v2_residual_risk_labels_round2.jsonl` and `teacher_v2_risk_labels_for_B_request.jsonl`: byte-identical round-2 residual/risk labels copied under both useful names.
- `teacher_v2_residual_risk_labels_round2_summary.json` and `teacher_v2_risk_labels_for_B_request_summary.json`: byte-identical round-2 targeted risk-label class balance summaries.

## Methodology Source Material Outside This Folder

- `dropoff/inbox/2026-06-18-methodology-compliance-review.md`
- `dropoff/inbox/2026-06-18-external-current-state-methodology-review.txt`
- `dropoff/inbox/2026-06-18-card-effects-action-prior-handoff.md`
- `dropoff/inbox/learning_action_audit_handoff_2026-06-18.md`
- `dropoff/outbox/2026-06-18-CONSENSUS-and-way-forward.md`
- `docs/ACTION_RANKER_PLAN.md`

Use these as methodology provenance. The current controller is still the branch plan plus the latest coordination prompt.

## Superseded Or Historical

- `dropoff/outbox/2026-06-18-master-plan.md` and `2026-06-18-forward-plan.md`: superseded by the consensus document and then by `BRANCH_PLAN.md`.
- Early old-ranker/value/leaf/blend docs: useful for the "objective slippage" diagnosis, not current implementation targets.
- First risk-label request/summary with 51 labels: superseded by round-2 targeted risk-label enrichment if the round-2 artifacts are present.
- Teacher V2 alignment failure docs before direct featurization: historical. Alignment was solved by Path B direct featurization.

## Practical Rule

If a doc says a result is "done," translate that into the status vocabulary before acting:

`specified -> implemented -> data-generated -> trained -> offline-evaluated -> arena-evaluated -> accepted | refuted | inconclusive`

Only `accepted` changes production direction. Most documents here are `inconclusive`, `refuted`, or `offline-evaluated only`.
