# Experiment 04: model diversity

## Question

Whether a second model type (CatBoost) adds anything over the LightGBM baseline on balanced accuracy, alone or blended. This is the last standard lever after features (experiments 01, 03), capacity, neighbors, and thresholds (experiments 02, 03).

## Setup

- 200,000-row subsample, fixed 5-fold StratifiedKFold (seed 42).
- LightGBM as in experiments 01 to 03. CatBoost: 400 iterations, depth 8, learning rate 0.05, MultiClass loss, auto_class_weights Balanced, the two categoricals passed natively.
- Blend: the average of the two models' out-of-fold class probabilities; the blend weight searched on a grid from 0 to 1.
- Code: experiments4.py.

## Results

| model | bal_acc | delta vs lightgbm |
|---|---|---|
| lightgbm (baseline) | 0.96248 | 0.00000 |
| catboost | 0.95510 | -0.00738 |
| blend 0.5/0.5 | 0.96106 | -0.00142 |
| blend best (lightgbm weight 1.0) | 0.96248 | 0.00000 |

## Findings

1. CatBoost underperformed LightGBM by 0.00738 at this configuration. CatBoost was not tuned (one quick configuration), so this is the result for an untuned drop-in, not a verdict on CatBoost in general.
2. The blend did not help. The even blend was worse than LightGBM alone (-0.00142), dragged down by the weaker CatBoost, and the grid search placed the optimal blend weight at 1.0 on LightGBM, that is, no CatBoost. The second model added no useful diversity here.

## Caveats

- CatBoost used one untuned configuration. A tuned CatBoost could match LightGBM, but the blend grid put the optimum at pure LightGBM, so even matched diversity would not raise the blend at this configuration.
- Subsample (200,000 rows), fast configurations.

## Outputs

results/experiments4_blend.csv (gitignored).

## Run

```
python experiments4.py
```
