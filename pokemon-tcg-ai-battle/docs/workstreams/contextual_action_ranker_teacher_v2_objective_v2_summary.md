# Teacher V2 Objective V2 Offline Evaluation

Status: offline only. No live arena/screen and no `agent_search` change.

Decision: **D** - offline did not improve; pause Teacher V2 contextual-ranker path

The revised model does not beat old ranker or option-0 on the required safety metrics. More objective tuning would be another blind pass, and the labels are no longer the blocking issue.

## Held-Out Mixed Test

| model                     | top1  | top3  | acceptable | mean regret | p95 regret | hi-regret | pairwise | mrr   |
| ------------------------- | ----- | ----- | ---------- | ----------- | ---------- | --------- | -------- | ----- |
| revised_full              | 0.238 | 0.476 | 0.619      | 80.42       | 221.56     | 0         | 0.62     | 0.453 |
| previous_teacher_v2_model | 0.333 | 0.667 | 0.714      | 77.19       | 267.06     | 0         | 0.61     | 0.522 |
| old_ranker                | 0.429 | 0.81  | 0.81       | 66.97       | 229.38     | 0         | 0.775    | 0.64  |
| option0                   | 0.476 | 0.762 | 0.667      | 82.19       | 221.56     | 0         | 0.65     | 0.638 |
| no_decoded_effects        | 0.381 | 0.714 | 0.667      | 33.47       | 139.38     | 0         | 0.666    | 0.561 |
| no_card_embedding         | 0.238 | 0.476 | 0.619      | 80.42       | 221.56     | 0         | 0.61     | 0.447 |
| no_option_deltas          | 0.238 | 0.476 | 0.619      | 80.42       | 221.56     | 0         | 0.623    | 0.461 |

## Slice Summary

| slice                              | n   | rev acc | rev mean | rev p95   | prev mean | old mean  | opt0 mean |
| ---------------------------------- | --- | ------- | -------- | --------- | --------- | --------- | --------- |
| all_heldout_mixed_test             | 21  | 0.619   | 80.42    | 221.56    | 77.19     | 66.97     | 82.19     |
| teacher_v2_targeted_failure_states | 16  | 0.688   | 92.62    | 369.08    | 89.63     | 77.47     | 98.7      |
| high_criticality_states            | 129 | 0.86    | 11821.34 | 1859.8    | 13120.23  | 36707.5   | 35658.38  |
| high_regret_tail                   | 19  | 0.737   | 79627.48 | 523872.25 | 79749.21  | 125759.79 | 99849.37  |
| recovery_states                    | 80  | 0.925   | 18959.22 | 2138.71   | 21043.22  | 59047.06  | 51886.78  |
| hand_outcome_disagreement_states   | 36  | 0.833   | 62.78    | 177.67    | 61.84     | 111.3     | 130.15    |
| stable_low_variance_labels         | 68  | 0.912   | 16.27    | 81.62     | 13.03     | 14824.0   | 14842.78  |
| noisy_labels                       | 132 | 0.894   | 11506.66 | 1540.3    | 12750.23  | 28215.84  | 27153.28  |

## Notes

- Targeted Teacher V2 overlays applied to held-out replay-test rows: 16
- Revised model artifact: `C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\.claude\worktrees\robust-learner-v2\pokemon-tcg-ai-battle\agent\contextual_ranker_teacher_v2_objective_v2.json`
- Previous Teacher V2 model: `C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\.claude\worktrees\robust-learner-v2\pokemon-tcg-ai-battle\agent\contextual_ranker_teacher_v2.json`
- High-regret tail report: `C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\.claude\worktrees\robust-learner-v2\pokemon-tcg-ai-battle\docs\workstreams\contextual_action_ranker_teacher_v2_objective_v2_tail_report.json`
