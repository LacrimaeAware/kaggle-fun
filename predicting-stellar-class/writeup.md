# Predicting Stellar Class (playground-series-s6e6)

Kaggle Playground Series, Season 6 Episode 6. https://www.kaggle.com/competitions/playground-series-s6e6
Deadline 2026-06-30.

## Question

Predict `class`, the type of an astronomical object, one of GALAXY, QSO (quasar), or STAR. One row is one object, described by sky coordinates (alpha, delta), five-band photometric magnitudes (u, g, r, i, z), redshift, and two categorical fields (spectral_type, galaxy_population). The metric is balanced accuracy, the mean of per-class recall, which weights each of the three classes equally regardless of how common the class is.

## Method

Baseline: LightGBM multiclass (objective multiclass, three classes) with 5-fold StratifiedKFold cross-validation on the class label and balanced class weights (LightGBM class_weight balanced). The two categorical fields (spectral_type, 4 values; galaxy_population, 2 values) are passed as pandas category dtype; the eight numeric fields are used as is. The score is out-of-fold balanced accuracy, which mirrors the competition metric. The submission uses the mean test-set class probabilities across the five folds, with the predicted label as the argmax. Code in baseline.py.

No feature engineering yet. The five magnitudes invite color features, the differences between bands (for example u minus g, and g minus r), which are standard in photometric classification and are the next step. Per-class decision thresholds are a later option.

## Result

Numbers first: local out-of-fold balanced accuracy 0.9640, public leaderboard balanced accuracy 0.96523. The two agree closely (the leaderboard is higher by 0.0012), so the validation scheme tracks the leaderboard and shows no sign of leakage. Per-fold balanced accuracy ranged 0.9632 to 0.9645.

| Model | Features | Local CV | Public LB |
| --- | --- | --- | --- |
| LightGBM multiclass, 5-fold, balanced weights | 10 raw (8 numeric, 2 categorical) | 0.9640 | 0.96523 |
| Stacked GBDT zoo + balanced-logreg + thresholds (experiment 06) | 10 raw | 0.96596 | 0.96659 |

## Caveat

The out-of-fold balanced accuracy uses early stopping on each validation fold, which makes it mildly optimistic as an estimate of generalization. The public leaderboard score is computed on a subset of the test set; the private score (currently blank) decides the final standing. This baseline has no feature engineering and no tuning, so 0.9640 is a floor, not a tuned result.

## Lesson

The local cross-validation tracked the public leaderboard (0.9640 versus 0.96523), which establishes that the StratifiedKFold scheme is trustworthy here and shows no obvious leakage. Balanced class weights produced a predicted class mix (GALAXY 0.642, QSO 0.205, STAR 0.152) close to the training mix.

Experiments 01 to 04 (see experiments/) tested feature engineering (color indices, SED moments, spectral curvature, Gaussian-mixture density, kNN neighbor features, the stellar-locus distance), added model capacity, per-class threshold tuning, and a CatBoost blend. None beat the baseline above the 0.0002 out-of-fold noise floor. Feature importance shows redshift carries 0.554 of the gain, the quasar class has the highest recall (0.973), and the errors fall on the galaxy/star boundary, where the real-survey discriminator (morphology) and the strongest photometry-only one (infrared color) are both absent from this dataset. The optical-plus-redshift baseline is at its feature ceiling, and the public leaderboard top near 0.971 is consistent with that ceiling.

A model-level gain remains, separate from the feature ceiling. A stacked GBDT ensemble (experiment 06) reached out-of-fold balanced accuracy 0.96596, +0.0019 over the first attempt, of which about +0.0012 is correcting the first attempt's tree count (it slightly overfit) and about +0.0008 is the ensemble. Threshold calibration added nothing because the base models were already balanced. The remaining gap to the public cluster (~0.9697) is the neural-tabular base models that need a GPU. Per the competition forum, the top cluster is within the metric's own noise, so the ranking there is the private draw.

## Leaderboard dynamics

From the competition's public forum, a variance analysis by a participant corroborated by the first-place and a top-grandmaster competitor: the metric's resolution is about 0.00087 on the public slice (49,500 rows) and about 0.00035 on the private slice (198,000 rows). The top 25 public teams sit within a 0.0002 window, below the private resolution, so the final ranking among them is dominated by which way the private slice falls rather than by model quality. The honest stacked ceiling reported there is about 0.9697 out-of-fold, mapping to about 0.9707 public, reached with a diverse base zoo (including neural-tabular models), a balanced-logistic stack on out-of-fold log-probabilities, and per-class threshold calibration under nested cross-validation. No exploitable leak was found: the one clean structural rule (redshift = 0.0001 implies QSO, pure in the training set) is already predicted by a competent model, so forcing it adds nothing. The generalization of these signals is in [docs/recognizing-leverage.md](../docs/recognizing-leverage.md).

## Data

- Metric: balanced accuracy. Confirmed by web search of the Evaluation section. The live page is JavaScript-rendered and was not read directly, so reconfirm on the logged-in Evaluation tab before final submission.
- train.csv: 577,347 rows, 12 columns. test.csv: 247,435 rows, 11 columns.
- Target `class` distribution in train: GALAXY 0.654, QSO 0.203, STAR 0.143. Imbalanced across three classes.
- No missing values in train.
- Columns: id, alpha, delta, u, g, r, i, z, redshift, spectral_type (categorical), galaxy_population (categorical), class (target).
