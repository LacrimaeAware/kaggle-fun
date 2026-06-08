# Experiment 02: levers beyond features

## Question

Experiment 01 found no feature transform of the raw columns beats the baseline above the fold noise. This batch tests the levers experiment 01 did not, and pins the noise floor that decides what counts as a real change.

## Setup

- Data: stratified 200,000-row subsample. Metric: out-of-fold balanced accuracy.
- Baseline model as in experiment 01 (LightGBM multiclass, balanced class weights, 350 trees, learning rate 0.05, num_leaves 63).
- Noise floor: the baseline run under three fold seeds (42, 1, 7); the standard deviation of the three out-of-fold scores is the floor.
- Capacity: 1500 trees, learning rate 0.03, num_leaves 255.
- kNN neighbor features: leak-free, k=50, in standardized color-redshift space. Per fold, NearestNeighbors is fit on the training fold; the features are mean neighbor distance, nearest-neighbor distance, and the three training-label class fractions among the neighbors.
- Per-class thresholds: class weights applied to the baseline out-of-fold probabilities before argmax, fit on 60 percent of the out-of-fold rows and evaluated on the disjoint 40 percent. Optimizer: Nelder-Mead.
- Code: experiments2.py.

## Results

| lever | bal_acc | delta vs baseline |
|---|---|---|
| baseline (canonical folds) | 0.96248 | 0.00000 |
| capacity (1500 trees, lr 0.03, 255 leaves) | 0.95653 | -0.00595 |
| kNN neighbor features (k=50, leak-free) | 0.96195 | -0.00053 |
| per-class thresholds (eval split) | 0.96286 | +0.00000 |

Noise floor: baseline across seeds 42, 1, 7 scored 0.9625, 0.9620, 0.9625; mean 0.9623, standard deviation 0.0002.

## Findings

1. The noise floor is about 0.0002, the seed-to-seed standard deviation of out-of-fold balanced accuracy. This is tighter than the within-run fold standard deviation of about 0.002 and is the correct bar for comparing out-of-fold scores. A change below roughly 0.0002 to 0.0004 is not real.
2. More capacity hurt. 1500 trees with 255 leaves scored 0.9565, down 0.00595, far beyond the noise floor. The baseline's 350 trees and 63 leaves are already past the point where added capacity helps balanced accuracy; the larger model degrades it. One configuration only, not a tuning sweep.
3. Leak-free kNN neighbor features did not help: -0.00053, a small change but beyond the noise floor, so a real decrease. The local joint-distribution structure they encode is already captured by the trees on the raw columns.
4. Per-class threshold tuning produced no change and returned the identity weights. With balanced class weights already applied during training, the argmax decision is already aligned with balanced accuracy. Caveat: balanced accuracy as a function of the weights is piecewise-constant, and the Nelder-Mead search did not move from its identity start, so this is rechecked with a grid search in experiment 03.

## Caveats

- Subsample (200,000 rows), one fast model configuration per lever. The capacity result is a single large configuration, not a sweep.
- The threshold result rests on an optimizer that may have stalled on a flat objective; experiment 03 rechecks it with a grid.

## Outputs

results/experiments2_levers.csv (gitignored).

## Run

```
python experiments2.py
```
