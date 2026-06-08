# Experiment 03: physics-motivated features and error diagnostics

## Question

Whether the stellar-locus distance and a UV-excess indicator, the principled features flagged in the literature note, improve balanced accuracy; and where the baseline's errors and signal actually are. The literature expects the quasar class to be the limit; this checks that against per-class recall and feature importance on the dataset, which, unlike the photometry-only literature, includes redshift.

## Setup

- 200,000-row subsample, fixed 5-fold StratifiedKFold (seed 42), the same fast LightGBM as experiments 01 and 02.
- Features tested: signed perpendicular distance from the stellar-locus line u-g = 2.15 (g-r) + 0.26 (dperp); a UV-excess region indicator (u-g < 0.6 and g-r > 0); both together.
- Diagnostics: out-of-fold confusion matrix and per-class recall; baseline feature importance (gain, mean over folds).
- Threshold recheck: grid search over per-class probability weights, fit on 60 percent of out-of-fold rows, evaluated on the disjoint 40 percent.
- Code: experiments3.py.

## Results

Features:

| experiment | bal_acc | delta vs baseline |
|---|---|---|
| baseline | 0.96248 | 0.00000 |
| +dperp (stellar-locus distance) | 0.96198 | -0.00050 |
| +uvx indicator | 0.96233 | -0.00015 |
| +dperp+uvx (locus pack) | 0.96213 | -0.00035 |

Baseline out-of-fold confusion matrix (rows true, columns predicted):

|  | GALAXY | QSO | STAR |
|---|---|---|---|
| GALAXY | 125110 | 1793 | 3861 |
| QSO | 690 | 39473 | 417 |
| STAR | 1033 | 172 | 27451 |

Per-class recall: GALAXY 0.9568, QSO 0.9727, STAR 0.9579. Balanced accuracy 0.96248.

Feature importance (gain, normalized): redshift 0.554, spectral_type 0.097, z 0.083, g 0.067, u 0.056, galaxy_population 0.044, alpha 0.033, i 0.024, delta 0.023, r 0.020.

Grid thresholds: eval argmax 0.96286 to optimized 0.96278 (delta -0.00008), best fit-split weights (0.7, 0.8) for (QSO, GALAXY).

## Findings

1. The stellar-locus features did not help. dperp scored -0.00050 (beyond the 0.0002 noise floor, a small real decrease), uvx -0.00015 (within noise), both together -0.00035. The optical color geometry the line encodes is already captured by the trees.
2. The quasar class is not the limit on this dataset. QSO recall is the highest of the three at 0.9727; GALAXY (0.9568) and STAR (0.9579) are lower. The errors concentrate on the galaxy/star boundary (3861 galaxies labelled star, 1033 stars labelled galaxy). This reverses the photometry-only literature, in which the quasar class limits accuracy.
3. The reversal is explained by redshift. Redshift carries 0.554 of the feature-importance gain, more than all five magnitudes combined. Redshift identifies quasars by their high redshift, removing the optical star/quasar degeneracy that limits the no-redshift regime. The remaining confusion is galaxy versus star, where the discriminating feature in real surveys is morphology (resolved versus point-like), which this dataset does not contain.
4. Threshold tuning gives nothing, now confirmed by grid search. The fit-split optimum did not transfer to the evaluation split (-0.00008). Balanced class weights during training already align the decision with balanced accuracy.
5. Sky position has nonzero importance (alpha 0.033, delta 0.023, about 0.056 together). Position cannot causally determine class, so this is a minor flag consistent with a synthetic-data artifact. It is small and not pursued.

## Caveats

- Subsample and fast model. The morphology explanation for the galaxy/star confusion is inference from the literature; this dataset has no morphology feature, so it cannot be tested directly here.
- Feature importance by gain is model-specific and can split credit among correlated features. The redshift dominance is large enough to be robust to that.

## Outputs

results/experiments3_locus.csv (gitignored).

## Run

```
python experiments3.py
```
