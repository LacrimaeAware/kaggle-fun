# Contextual Action Ranker Teacher V2 Pass

Branch: `exp/robust-learner-v2`

Status: Teacher V2 direct featurization succeeded; one narrow contextual-ranker retrain completed; no live
screen and no promotion.

## Inputs

Live baseline remains `agent_search`.

Teacher V2 source artifact:
`data/manifests/teacher_v2_labels_scaled.jsonl` from Branch A.

Model A summary:

- 50 high-criticality decisions;
- `k_outcome = 16`;
- hand-vs-outcome disagreement 26/50 = 0.52;
- mean outcome SE 0.044;
- all siblings completed 50/50;
- no live agent consumed these labels.

Interpretation held in Branch B:

- primary target: `hand_norm_advantage`;
- auxiliary target: `outcome_winrate`, confidence-weighted by `outcome_se`;
- no outcome argmax hard primary label.

## Alignment

Old 160-decision contextual dataset alignment:

- Teacher V2 decisions loaded: 50;
- matched decisions: 8;
- unmatched decisions: 42;
- required checked fields missing: 0;
- training-ready against old rows: false.

This is only an artifact-join result, not a Teacher V2 quality failure.

Path B direct featurization:

- reconstructed Teacher V2 roots: 50/50;
- option-level index alignment: 404/404;
- semantic-key alignment: 404/404;
- eq-class exact/remappable alignment: 404/404;
- training-ready after direct featurization: true.

Direct dataset: `docs/workstreams/teacher_v2_contextual_scaled_dataset.json`

Mixed dataset: `docs/workstreams/contextual_action_ranker_v1_teacher_v2_mixed_dataset.json`

The mixed dataset has 200 decisions after replacing 10 old rows with Teacher V2 rows sharing the same
replay-file plus ordered semantic sibling signature.

## Retrain

Model artifact: `agent/contextual_ranker_teacher_v2.json`

Train/eval report: `docs/workstreams/contextual_action_ranker_teacher_v2_train_eval.json`

Tail report: `docs/workstreams/contextual_action_ranker_teacher_v2_tail_report.json`

Training used the existing contextual architecture:

- root state;
- action descriptor;
- acting-card embedding;
- decoded card effects;
- target/entity features;
- one-step option deltas;
- state/effect interactions;
- short public history.

The embeddings/effects/deltas remain trainable inputs, not fixed hand weights.

## Offline Result

Held-out mixed test, 21 decisions:

| model | top1 | acceptable | mean regret | p95 regret | high regret | NDCG |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.333 | 0.667 | 108.9 | 695.0 | 0 | 0.849 |
| no decoded effects | 0.238 | 0.619 | 95.1 | 212.5 | 0 | 0.827 |
| no card embedding | 0.333 | 0.571 | 111.9 | 695.0 | 0 | 0.809 |
| no option deltas | 0.333 | 0.667 | 108.9 | 695.0 | 0 | 0.849 |
| old ranker baseline | 0.429 | 0.857 | 74.1 | 255.0 | 0 | 0.878 |
| option-0 baseline | 0.476 | 0.714 | 89.8 | 212.5 | 0 | 0.869 |

Teacher V2 held-out test slice, 5 decisions:

- top1: 0.400;
- acceptable agreement: 0.600;
- mean regret: 37.4;
- p95 regret: 117.4;
- high-regret count: 0.

High-regret reference rows are concentrated in the train/recovery split under this deterministic split, so
the held-out high-regret-tail evidence is thin. On the train high-regret-reference slice, full model still
has 11 high-regret predictions over 19 rows; no-card-embedding lowers that to 7/19, which suggests the full
input mix may still overfit or overreact on sparse card/effect identity.

## Recommendation

Do not promote and do not run a live screen for this model. The direct Teacher V2 feature path is now real and
training-ready, but this narrow retrain does not beat the offline old-ranker baseline on the mixed held-out
test. The next useful step is objective/weight calibration or a larger Teacher V2 batch, not live screening
this artifact.
