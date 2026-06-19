# Robust Learner V2 DAgger Round 2

Branch: `exp/robust-learner-v2`

Status: one bounded Round 2 completed. No model promotion, submission, main merge, deck change, large architecture, embedding ablation, successor-affordance head, or RL run.

## Calibration Change

Round 1 improved the intended on-policy failure mode, but damaged offline/stable teacher-target fidelity. Round 2 therefore changed only the loss/training calibration:

- start from `agent/ranker_model_dagger_round1.json`;
- collect recovery states with `CABT_RANKER_MODEL=ranker_model_dagger_round1.json`;
- anchor base/stable sources to the original `agent/ranker_model.json` with KL regularization;
- lower learning rate from `5e-4` to `1e-4`;
- reduce epochs from `12` to `5`;
- strengthen base/stable source weights;
- down-weight recovery labels, especially unstable recovery labels;
- reduce advantage and acceptable-set auxiliary weights;
- add gradient clipping;
- add p95 regret to on-policy summaries.

This keeps the same deployed ranker architecture.

## Artifacts

- `agent/ranker_model_dagger_round2.json`
- `docs/workstreams/robust_learner_v2_dagger_round2_train.json`
- `docs/workstreams/robust_learner_v2_dagger_round2_on_policy_20g.json`
- `docs/workstreams/robust_learner_v2_dagger_round2_offline_compare.json`
- `tools/train_dagger_round1.py`
- `tools/measure_on_policy_shift.py`

## Training Run

Command shape:

```text
python tools/train_dagger_round1.py --dagger-round 2 --collection-ranker-model ranker_model_dagger_round1.json --model-in agent\ranker_model_dagger_round1.json --model-out agent\ranker_model_dagger_round2.json --report docs\workstreams\robust_learner_v2_dagger_round2_train.json --recovery-games 30 --max-recovery-decisions 800 --epochs 5 --lr 0.0001 --base-weight 1.0 --stable-replay-weight 1.2 --recovery-stable-weight 0.45 --recovery-unstable-weight 0.10 --lam-rank 0.05 --lam-accept 0.03 --anchor-model agent\ranker_model.json --anchor-weight 0.75 --anchor-sources base_distill,stable_replay --max-grad-norm 1.0
```

Round-2 recovery collection: 30 games, Round-1 student went `19-11`, 0 draws/errors, 1820 traced decisions, 670 kept after filters.

Dataset:

| Source / stability | Decisions |
|---|---:|
| base_distill / original | 1218 |
| recovery / stable | 442 |
| recovery / unstable | 228 |
| stable_replay / stable | 18 |
| stable_replay / unstable | 27 |
| total | 1933 |

## Offline Stable/Replay Fidelity

Common offline comparison set: 1218 base distill decisions plus 45 reconstructed stable-replay decisions.

| Model | N | Acceptable | Hard top-1 | Mean regret | P90 | P95 | High-regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Round 0 | 1263 | 0.963 | 0.770 | 1608.1 | 20.3 | 90.0 | 7 |
| Round 1 | 1263 | 0.840 | 0.470 | 19264.5 | 593.3 | 1149.2 | 93 |
| Round 2 | 1263 | 0.840 | 0.466 | 18568.8 | 535.1 | 1090.2 | 86 |

Round 2 recovers some Round-1 offline damage: mean regret, tail regret, and high-regret count improve versus Round 1. It does not recover close to Round 0.

## Arena Screen

Round-2 alternate model:

```text
CABT_RANKER_MODEL=ranker_model_dagger_round2.json
python agent/cabt_arena.py --games 20 --a rank --b heuristic
```

Result: `9-11`, 0 draws/errors, win rate `0.450`.

For context, the Round-1 cheap arena screen was `7-13`. The Round-2 on-policy diagnostic trace also went `9-11`.

## On-Policy Diagnostics

Round-2 command:

```text
CABT_RANKER_MODEL=ranker_model_dagger_round2.json
python tools/measure_on_policy_shift.py --student rank --opponent heuristic --games 20 --progress 25 --n-determ 3 --time-budget 4.0 --teacher-repeats 3 --min-teacher-agreement 1.0 --min-acceptable-rate 0.5 --reference-max-files 160 --reference-max-decisions 3000 --output docs\workstreams\robust_learner_v2_dagger_round2_on_policy_20g.json
```

Overall:

| Model | Games | Trace | Decisions | Applicable | Acceptable | Hard top-1 | Mean regret | P90 | P95 | High-regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 0 | 100 | 18-82 | 2980 | 2939 | 0.688 | 0.439 | 30694.7 | 1110.0 | 2416.9 | 335 |
| Round 1 | 20 | 10-10 | 1363 | 1351 | 0.723 | 0.452 | 6463.3 | 427.8 | 1117.6 | 82 |
| Round 2 | 20 | 9-11 | 1177 | 1163 | 0.701 | 0.461 | 8631.6 | 873.9 | 1355.6 | 101 |

After first unacceptable action:

| Model | N | Acceptable | Hard top-1 | Mean regret | P90 | P95 | High-regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Round 0 | 2205 | 0.634 | 0.405 | 38202.4 | 1240.7 | 166031.7 | 308 |
| Round 1 | 1148 | 0.698 | 0.431 | 6527.0 | 557.2 | 1127.8 | 74 |
| Round 2 | 1028 | 0.683 | 0.457 | 8191.7 | 929.3 | 1399.8 | 95 |

## Decision

Round 2 should not be called successful. It partially reduced the Round-1 offline collapse and improved the cheap arena screen, but it worsened the main on-policy regret and high-regret metrics versus Round 1:

- acceptable-set agreement fell from `0.723` to `0.701`;
- mean regret rose from `6463.3` to `8631.6`;
- p90/p95 regret rose;
- high-regret actions rose from `82` to `101`;
- after-first-unacceptable high-regret rose from `74` to `95`.

It still preserves much of the Round-1 gain over Round 0, but the stated continuation gate was stricter: improve regret/high-regret while recovering offline target quality. Round 2 did not meet that gate.

Recommendation: stop DAgger here and reassess the learner objective before any Round 3. The next discussion should focus on why the base/recovery objectives conflict: label weighting, calibration of acceptable-set targets, and whether the current ranker scoring scale can absorb recovery data without breaking base fidelity. Do not move to embeddings, affordance heads, larger networks, or production promotion from this result.
