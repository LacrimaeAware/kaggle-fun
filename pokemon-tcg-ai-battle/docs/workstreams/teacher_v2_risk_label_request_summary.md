# Teacher V2 Risk Label Request Summary

Status: targeted residual/risk labels for Branch B. No live agent change and no arena screen.

## Artifact

- Source path: `C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\.claude\worktrees\planner-teacher-v2\pokemon-tcg-ai-battle\data\manifests\teacher_v2_risk_labels_for_B_request.jsonl`
- B access path: `C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\.claude\worktrees\robust-learner-v2\pokemon-tcg-ai-battle\data\manifests\teacher_v2_risk_labels_for_B_request.jsonl`
- Labeled decisions: 51
- Options: 490
- Failed/unrecoverable states: 3
- All siblings completed: 51/51

## Seed Coverage

- 80251230.json:12: true
- 80252701.json:56: true

## Selection Reasons

- near_miss_risk_boundary: 46
- safe_search_choice_false_positive_analogs: 32
- search_selected_high_regret_analogs: 1
- seed_example: 2

## Class Balance

- High-regret positives: 19
- High-regret negatives: 471
- Unacceptable positives: 234
- Unacceptable negatives: 256
- Search-selected high-regret decisions: 1
- Densifies sparse high-regret class versus prior 13 positives: true

## Recommendation For B

Use this as a targeted high-regret recall calibration batch; verify the two seed cases first, then retrain one high-regret-primary risk-only model with threshold calibration.
