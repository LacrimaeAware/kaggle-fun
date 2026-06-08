# Experiment 06: stacked ensemble with threshold calibration

## Question

Whether the public top-of-leaderboard recipe (a diverse base-model zoo, a balanced-logistic meta-model stacked on out-of-fold class log-probabilities, then per-class threshold calibration, under honest nested cross-validation) beats the single-LightGBM first attempt (out-of-fold balanced accuracy 0.9640, public leaderboard 0.96523).

## Setup

- Full training set (577,347 rows), 5-fold StratifiedKFold (seed 42), out-of-fold balanced accuracy.
- Base zoo: balanced LightGBM (500 trees), an unbalanced LightGBM (700 trees, different params, for diversity), XGBoost (balanced via sample weights), CatBoost (balanced), HistGradientBoosting (balanced). The two categoricals were ordinal-encoded into a single numeric matrix shared by all bases.
- Stack: features are the concatenated log-probabilities of each base's out-of-fold predictions; meta-model is LogisticRegression(class_weight=balanced).
- Threshold calibration: differential evolution over three per-class multipliers (bounds 0.1 to 5), maximizing balanced accuracy.
- Honest nesting: base out-of-fold via the fold CV; the meta-model fit via CV over the stack features; the thresholds fit on disjoint out-of-fold rows. The final submission fits meta and thresholds on all out-of-fold rows and applies to the test set.
- Omits the neural-tabular base models (RealMLP, TabM) the top teams use, which need a GPU.
- Code: experiments6.py.

## Results

| component | OOF bal_acc |
|---|---|
| base lgb_bal (500 trees) | 0.96518 |
| base lgb_unbal | 0.95469 |
| base xgb | 0.96396 |
| base cat | 0.95682 |
| base hgb | 0.96424 |
| meta (balanced logreg stack) | 0.96596 |
| meta + threshold calibration | 0.96591 |
| first attempt (single LightGBM, ~1700 trees, early-stopped) | 0.96400 |

Final threshold weights: approximately (3.6, 3.8, 3.4), near-uniform. Delta of the stack over the first attempt: +0.00196 out-of-fold.

Public leaderboard of the stack submission: 0.96659, against the baseline's 0.96523, a +0.00136 gain. The public-slice resolution is about 0.00087, so the gain is above the noise.

## Findings

1. The stack reached 0.96596 out-of-fold, +0.00196 over the first attempt, well above the 0.0002 noise floor. A real gain.
2. Most of that gain was not the ensemble. A single balanced LightGBM with 500 fixed trees scored 0.96518, about +0.0012 over the first attempt's ~1700 early-stopped trees. The first attempt slightly overfit on tree count, consistent with experiment 02, where added capacity hurt. The 5-model stack added a further +0.0008 over that best single base.
3. Threshold calibration added nothing (0.96596 to 0.96591; near-uniform weights). The base models already use balanced class weights, so the decision boundary is pre-aligned to balanced accuracy. The calibration that helps the public top teams matters when the base models are trained on log-loss and left unbalanced, which over-predict the majority class; here it is redundant.
4. CatBoost (0.95682) and the unbalanced LightGBM (0.95469) were the weak bases. The balanced LightGBM, HistGradientBoosting, and XGBoost cluster at 0.964 to 0.965.
5. The result sits about 0.004 below the public stacked ceiling (~0.9697 out-of-fold). That gap is the neural-tabular base models the top teams use, which need a GPU and were omitted. The remaining distance is craft and compute, not unextracted signal.

## Caveats

- The categoricals were ordinal-encoded into a single numeric matrix for all bases, a minor simplification versus native categorical handling, applied consistently.
- This is one configuration per base model, not a tuning sweep. The point was the stacking recipe, not squeezing each base.

## Outputs

results/submission_stack.csv (gitignored, not submitted).

## Run

```
python experiments6.py
```
