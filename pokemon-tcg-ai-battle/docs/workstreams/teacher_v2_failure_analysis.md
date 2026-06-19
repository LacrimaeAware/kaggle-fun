# Teacher V2 Failure Analysis

Branch: `exp/robust-learner-v2`

Status: analysis only. No retrain and no arena screen.

## Summary

- Held-out mixed test decisions: 21
- Teacher V2 labels available on test rows: 5
- Teacher V2 labels missing on test rows: 16
- Recommendation: **B** - ask Model A for labels on the specific failure/test states

The first Teacher V2 retrain failed the offline gate, but the failure analysis is not yet enough to justify
another objective tweak. Most held-out failures are on rows that still have old Teacher V1-style targets, not
Teacher V2 targets. The clean next move is to label those exact failure/test roots with Teacher V2.

## Classifications

| class                                             | count |
| ------------------------------------------------- | ----- |
| all_acceptable                                    | 11    |
| all_wrong                                         | 6     |
| old_ranker_correct_teacher_v2_model_wrong         | 5     |
| option0_correct_teacher_v2_model_wrong            | 5     |
| teacher_v2_label_or_current_label_ambiguous_noisy | 19    |
| teacher_v2_model_correct_old_ranker_wrong         | 3     |

## Likely Causes

| likely cause                              | count |
| ----------------------------------------- | ----- |
| decoded-effect / interaction overreaction | 3     |
| label noise                               | 19    |
| old-ranker teacher-alignment issue        | 16    |
| option-0 prior issue                      | 5     |
| overfitting small n                       | 3     |

## Decision Table

| decision                       | src               | classes                                                                                                                            | model | old | opt0 | best | model regret | causes                                                                                   |
| ------------------------------ | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----- | --- | ---- | ---- | ------------ | ---------------------------------------------------------------------------------------- |
| 80459198.json:f97cbce0c9407192 | replay_test       | teacher_v2_model_correct_old_ranker_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                        | 1     | 3   | 0    | 1    | 0.0          | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:b6b923d3b78763de | replay_test       | all_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                         | 2     | 2   | 0    | 3    | 50.0         | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:25d7fc98e34acc49 | replay_test       | all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                                   | 1     | 1   | 0    | 1    | 0.0          | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:47dace3568e45815 | replay_test       | all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                                   | 0     | 0   | 0    | 0    | 0.0          | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:ce0691111c953258 | replay_test       | option0_correct_teacher_v2_model_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                            | 2     | 2   | 0    | 0    | 2.5          | label noise,old-ranker teacher-alignment issue,option-0 prior issue                      |
| 80459198.json:8e74004879647369 | replay_test       | option0_correct_teacher_v2_model_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                            | 1     | 1   | 0    | 0    | 0.0          | label noise,old-ranker teacher-alignment issue,option-0 prior issue                      |
| 80459198.json:ca083ee58ab52bae | replay_test       | all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                                   | 0     | 0   | 0    | 0    | 0.0          | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:ca083ee58ab52bae | replay_test       | all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                                   | 0     | 0   | 0    | 0    | 0.0          | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:f5b724f91efe6f35 | replay_test       | old_ranker_correct_teacher_v2_model_wrong                                                                                          | 0     | 6   | 0    | 6    | 123.0        | old-ranker teacher-alignment issue                                                       |
| 80459198.json:d611fd2e8e215a0e | replay_test       | all_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                                                        | 0     | 0   | 0    | 7    | 797.5        | label noise,old-ranker teacher-alignment issue,decoded-effect / interaction overreaction |
| 80459198.json:b038053a08b986aa | replay_test       | all_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                         | 0     | 2   | 0    | 3    | 45.0         | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:868c45838611644f | replay_test       | all_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                                                        | 5     | 0   | 0    | 1    | 695.0        | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:afc4fd8179f65a5a | replay_test       | all_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                                                        | 0     | 2   | 0    | 3    | 80.0         | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:e56d4d21aebddea8 | replay_test       | option0_correct_teacher_v2_model_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                            | 1     | 1   | 0    | 0    | 211.0        | label noise,old-ranker teacher-alignment issue,decoded-effect / interaction overreaction |
| 80459198.json:74d0ba20b067c588 | replay_test       | old_ranker_correct_teacher_v2_model_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                        | 4     | 9   | 0    | 9    | 38.83        | label noise,old-ranker teacher-alignment issue                                           |
| 80459198.json:8b77118073e4277b | replay_test       | old_ranker_correct_teacher_v2_model_wrong                                                                                          | 4     | 9   | 0    | 9    | 58.0         | old-ranker teacher-alignment issue                                                       |
| 80252701.json:55               | teacher_v2_scaled | old_ranker_correct_teacher_v2_model_wrong,option0_correct_teacher_v2_model_wrong,teacher_v2_label_or_current_label_ambiguous_noisy | 1     | 0   | 0    | 0    | 29.38        | label noise,option-0 prior issue,overfitting small n                                     |
| 80253882.json:154              | teacher_v2_scaled | teacher_v2_model_correct_old_ranker_wrong,teacher_v2_label_or_current_label_ambiguous_noisy                                        | 0     | 5   | 0    | 0    | 0.0          | label noise                                                                              |
| 80251230.json:17               | teacher_v2_scaled | all_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                                                         | 2     | 0   | 0    | 1    | 18.12        | label noise,decoded-effect / interaction overreaction,overfitting small n                |
| 80253882.json:173              | teacher_v2_scaled | teacher_v2_model_correct_old_ranker_wrong,all_acceptable,teacher_v2_label_or_current_label_ambiguous_noisy                         | 0     | 1   | 0    | 0    | 0.0          | label noise                                                                              |
| 80252701.json:58               | teacher_v2_scaled | old_ranker_correct_teacher_v2_model_wrong,option0_correct_teacher_v2_model_wrong,teacher_v2_label_or_current_label_ambiguous_noisy | 3     | 0   | 0    | 0    | 139.38       | label noise,option-0 prior issue,overfitting small n                                     |

## Request For Model A

Request file: `data/manifests/teacher_v2_label_request_for_A.json`

Requested states: 15

Reason: these held-out replay-test states lack Teacher V2 labels, so current comparisons cannot cleanly
separate label mismatch from model/objective failure.
