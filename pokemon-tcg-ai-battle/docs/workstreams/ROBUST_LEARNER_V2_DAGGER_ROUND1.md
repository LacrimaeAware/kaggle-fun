# Robust Learner V2 DAgger Round 1

Branch: `exp/robust-learner-v2`

Status: one bounded DAgger pilot completed. No Student V2 architecture, embedding ablation, successor-affordance head, RL run, deck change, shared-schema edit, or main merge.

## Implementation

Files:

- `tools/train_dagger_round1.py`
- `agent/ranker_model_dagger_round1.json`
- `agent/ranker.py`
- `docs/workstreams/robust_learner_v2_dagger_round1_train.json`
- `docs/workstreams/robust_learner_v2_dagger_round1_on_policy_20g.json`

`agent/ranker.py` now honors `CABT_RANKER_MODEL`; default behavior still loads `ranker_model.json`. This lets the pilot evaluate the alternate round-1 model without overwriting the production ranker artifact.

The trainer keeps the existing architecture fixed:

```text
card embedding + action/root/delta dense features -> 2-layer MLP -> option score
```

It starts from `agent/ranker_model.json` and fine-tunes once with:

- original fixed-deck distillation rows from `action_adv.jsonl`;
- stable replay rows reconstructed from the B1.2 train artifact where Teacher V1 remained applicable;
- old-ranker learner-visited recovery states from the same B1.3 rank-vs-heuristic on-policy process.

Training target:

- class-level Teacher soft policy;
- advantage-weighted pairwise ranking;
- acceptable-action-set binary target;
- confidence weights, with unstable recovery labels down-weighted.

Important caveat: the committed B1.3 JSON stores hashes/metrics, not raw observations or full option targets. The exact B1.3 states therefore cannot be losslessly rehydrated for training. Round 1 uses the same learner-visited collection process and settings to collect full training rows, rather than relabelling unrelated replay states.

## Training Run

Command:

```text
python tools/train_dagger_round1.py --recovery-games 30 --max-recovery-decisions 800 --progress 50 --epochs 12 --teacher-repeats 3 --n-determ 3 --time-budget 4.0
```

Dataset:

| Source / stability | Decisions |
|---|---:|
| base_distill / original | 1218 |
| recovery / stable | 373 |
| recovery / unstable | 298 |
| stable_replay / stable | 18 |
| stable_replay / unstable | 27 |
| total | 1934 |

Recovery collection trace: 30 games, old ranker went `9-21`, 0 draws/errors, 809 traced decisions, 671 kept after forced-floor/applicability/feature filters.

Internal mixed-target eval got worse after fine-tuning:

| Eval | Acceptable agreement | Hard top-1 | Mean regret | P90 regret | High-regret |
|---|---:|---:|---:|---:|---:|
| before | 0.851 | 0.600 | 11374.6 | 234.4 | 93 |
| after | 0.782 | 0.435 | 18051.1 | 888.9 | 170 |

This is a warning that the round-1 loss/weights are not yet well-calibrated on the mixed offline target.

## Arena Screen

Alternate model:

```text
CABT_RANKER_MODEL=ranker_model_dagger_round1.json
python agent/cabt_arena.py --games 20 --a rank --b heuristic
```

Result: `7-13`, 0 draws/errors, win rate `0.350`.

The after-training on-policy diagnostic trace also ran 20 games and went `10-10`, 0 draws/errors. The old B1.3 100-game baseline was `18-82`.

## On-Policy Diagnostic

After-training command:

```text
CABT_RANKER_MODEL=ranker_model_dagger_round1.json
python tools/measure_on_policy_shift.py --student rank --opponent heuristic --games 20 --progress 25 --n-determ 3 --time-budget 4.0 --teacher-repeats 3 --min-teacher-agreement 1.0 --min-acceptable-rate 0.5 --reference-max-files 160 --reference-max-decisions 3000 --output docs\workstreams\robust_learner_v2_dagger_round1_on_policy_20g.json
```

Baseline is the committed B1.3 100-game old-ranker diagnostic.

| Metric | Old ranker B1.3 100g | DAgger R1 20g |
|---|---:|---:|
| games | 100 | 20 |
| arena trace | 18-82 | 10-10 |
| visited decisions | 2980 | 1363 |
| teacher-applicable decisions | 2939 | 1351 |
| acceptable-set agreement | 0.688 | 0.723 |
| hard top-1 agreement | 0.439 | 0.452 |
| mean regret | 30694.7 | 6463.3 |
| p90 regret | 1110.0 | 427.8 |
| high-regret decisions >=1000 | 335 | 82 |

After first unacceptable action:

| Metric | Old ranker B1.3 100g | DAgger R1 20g |
|---|---:|---:|
| acceptable-set agreement | 0.634 | 0.698 |
| hard top-1 agreement | 0.405 | 0.431 |
| mean regret | 38202.4 | 6527.0 |
| p90 regret | 1240.7 | 557.2 |
| high-regret decisions >=1000 | 308 | 74 |

## Decision

Round 1 passes the stated bounded-pilot gate directionally: at least one on-policy metric improved materially, and arena play was not worse than the old B1.3 baseline.

Caveats:

- the after diagnostic is only 20 games, so win rate and tail-regret estimates are noisy;
- the offline mixed-target eval worsened, so the loss/weighting needs a small calibration pass before scaling;
- the DAgger recovery trace has many more decisions per game than the old 100-game B1.3 average, so direct per-decision counts should be read with sample-size caution;
- this does not prove DAgger is guaranteed to fix the ranker.

Recommendation: authorize exactly one second bounded DAgger collection/retrain round, but keep it narrow. The next round should first adjust the loss/weights to avoid the offline regression, then repeat the same cheap arena/on-policy gate. Do not start embeddings, successor-affordance heads, or a large new network yet.
