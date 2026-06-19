# Teacher V2 Failure Analysis

Branch: `exp/robust-learner-v2`

Status: analysis only. No retrain and no arena screen.

## Summary

- Held-out mixed test decisions: 21
- Teacher V2 labels available on test rows: 21
- Teacher V2 labels missing on test rows: 0
- Replay-test rows reinterpreted with targeted Teacher V2 labels: 16
- Hand/outcome argmax disagreement on labelled test rows: 8/21
- Recommendation: **C** - revise objective/weighting before retraining

All held-out rows now have Teacher V2 labels, so the failure no longer looks like a label-source mismatch. The saved full model still trails old-ranker on top-1, acceptable agreement, and mean regret, while the zero-effects ablation is much better on regret. The next change should recalibrate the objective/weights around hand_norm_advantage and regularize decoded-effect/delta influence before another retrain.

## Performance

| model                   | top1  | acceptable | mean regret | p90 regret | p95 regret | >=100 regret |
| ----------------------- | ----- | ---------- | ----------- | ---------- | ---------- | ------------ |
| teacher_v2_model        | 0.333 | 0.714      | 77.19       | 203.88     | 267.06     | 4            |
| old_ranker              | 0.429 | 0.81       | 66.97       | 146.88     | 229.38     | 3            |
| option0                 | 0.476 | 0.667      | 82.19       | 203.88     | 221.56     | 4            |
| full_model_zero_effects | 0.524 | 0.714      | 29.55       | 83.12      | 139.38     | 2            |
| full_model_zero_deltas  | 0.238 | 0.571      | 85.98       | 203.88     | 267.06     | 4            |

## Classifications

| class                                             | count |
| ------------------------------------------------- | ----- |
| all_acceptable                                    | 10    |
| all_wrong                                         | 7     |
| old_ranker_correct_teacher_v2_model_wrong         | 5     |
| option0_correct_teacher_v2_model_wrong            | 5     |
| teacher_v2_label_or_current_label_ambiguous_noisy | 20    |
| teacher_v2_model_correct_old_ranker_wrong         | 3     |

## Likely Causes

| likely cause                                                | count |
| ----------------------------------------------------------- | ----- |
| decoded-effect / interaction overreaction                   | 4     |
| label noise                                                 | 20    |
| no clear failure; acceptable or correct under current label | 1     |
| option-0 prior issue                                        | 5     |
| overfitting small n                                         | 14    |

## Decision Table

| decision           | src               | classes                                                                                                                            | model | old | opt0 | best | model regret | causes                                                                     |
| ------------------ | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----- | --- | ---- | ---- | ------------ | -------------------------------------------------------------------------- |
| 80459198.json:15:1 | replay_test       | all_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                                                        | 1     | 3   | 0    | 2    | 8.75         | label noise,overfitting small n                                            |
| 80459198.json:16:1 | replay_test       | all_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                                                        | 2     | 2   | 0    | 3    | 6.56         | label noise,overfitting small n                                            |
| 80459198.json:17:1 | replay_test       | -                                                                                                                                  | 1     | 1   | 0    | 1    | 0.0          | no clear failure; acceptable or correct under current label                |
| 80459198.json:18:1 | replay_test       | all_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                         | 0     | 0   | 0    | 2    | 4.38         | label noise,decoded-effect / interaction overreaction,overfitting small n  |
| 80459198.json:21:1 | replay_test       | option0_correct_teacher_v2_model_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                            | 2     | 2   | 0    | 0    | 3.12         | label noise,option-0 prior issue,overfitting small n                       |
| 80459198.json:22:1 | replay_test       | all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                                   | 1     | 1   | 0    | 1    | 0.0          | label noise                                                                |
| 80459198.json:23:1 | replay_test       | all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                                   | 0     | 0   | 0    | 0    | 0.0          | label noise                                                                |
| 80459198.json:23:1 | replay_test       | all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                                   | 0     | 0   | 0    | 0    | 0.0          | label noise                                                                |
| 80459198.json:35:1 | replay_test       | all_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                                                        | 0     | 6   | 0    | 5    | 203.88       | label noise,overfitting small n                                            |
| 80459198.json:36:1 | replay_test       | all_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                                                        | 0     | 0   | 0    | 7    | 811.62       | label noise,decoded-effect / interaction overreaction,overfitting small n  |
| 80459198.json:38:1 | replay_test       | all_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                         | 0     | 2   | 0    | 1    | 11.25        | label noise,overfitting small n                                            |
| 80459198.json:39:1 | replay_test       | old_ranker_correct_teacher_v2_model_wrong,option0_correct_teacher_v2_model_wrong,teacher_v2_label_or_current_label_ambiguous_noisy | 5     | 0   | 0    | 0    | 267.06       | label noise,option-0 prior issue,overfitting small n                       |
| 80459198.json:41:1 | replay_test       | teacher_v2_model_correct_old_ranker_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                         | 0     | 2   | 0    | 0    | 0.0          | label noise                                                                |
| 80459198.json:42:1 | replay_test       | option0_correct_teacher_v2_model_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                            | 1     | 1   | 0    | 0    | 91.69        | label noise,decoded-effect / interaction overreaction,option-0 prior issue |
| 80459198.json:44:1 | replay_test       | old_ranker_correct_teacher_v2_model_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                        | 4     | 9   | 0    | 3    | 5.5          | label noise,overfitting small n                                            |
| 80459198.json:45:1 | replay_test       | old_ranker_correct_teacher_v2_model_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                        | 4     | 9   | 0    | 9    | 20.29        | label noise,overfitting small n                                            |
| 80252701.json:55   | teacher_v2_scaled | old_ranker_correct_teacher_v2_model_wrong,option0_correct_teacher_v2_model_wrong,teacher_v2_label_or_current_label_ambiguous_noisy | 1     | 0   | 0    | 0    | 29.38        | label noise,option-0 prior issue,overfitting small n                       |
| 80253882.json:154  | teacher_v2_scaled | teacher_v2_model_correct_old_ranker_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                        | 0     | 5   | 0    | 0    | 0.0          | label noise                                                                |
| 80251230.json:17   | teacher_v2_scaled | all_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                         | 2     | 0   | 0    | 1    | 18.12        | label noise,decoded-effect / interaction overreaction,overfitting small n  |
| 80253882.json:173  | teacher_v2_scaled | teacher_v2_model_correct_old_ranker_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                         | 0     | 1   | 0    | 0    | 0.0          | label noise                                                                |
| 80252701.json:58   | teacher_v2_scaled | old_ranker_correct_teacher_v2_model_wrong,option0_correct_teacher_v2_model_wrong,teacher_v2_label_or_current_label_ambiguous_noisy | 3     | 0   | 0    | 0    | 139.38       | label noise,option-0 prior issue,overfitting small n                       |

## Request For Model A

Request file: `C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\.claude\worktrees\robust-learner-v2\pokemon-tcg-ai-battle\data\manifests\teacher_v2_post_label_request_for_A.json`

Requested states: 0

Reason: none; targeted labels cover all held-out mixed-test rows in this analysis.
