# Lessons learned

Rules that carry across competitions, each with the evidence that supports it. The first entries are starting principles from general practice. Later entries cite the specific competition that produced them.

## Validation

### Naive K-fold leakage on grouped or time-ordered data

Use a grouping-aware or time-aware split instead of a plain shuffled K-fold when rows are not independent.

Evidence: when several rows share a group (the same passenger group, household, or user) or follow a time order, a plain K-fold places related rows in both the training and validation folds. The validation score then measures memorization of the group rather than generalization, and it overstates the leaderboard score. Splitting by group (GroupKFold) or by time keeps related rows on one side of the split.

### Out-of-fold score as the truth meter

Select final submissions by the best local cross-validation score, not the best public leaderboard score.

Evidence: the public leaderboard is computed on a subset of the test set and is a single noisy number, so selecting on it overfits to that subset. A well-constructed out-of-fold score averages over the whole training set and is more stable. The agreement between local cross-validation and the public leaderboard is checked once, early. If the two diverge, the validation scheme is fixed before anything else, because divergence usually indicates leakage.

Observed in playground-series-s6e6 (Predicting Stellar Class): a 5-fold StratifiedKFold out-of-fold balanced accuracy of 0.9640 matched the public leaderboard 0.96523 (leaderboard higher by 0.0012), confirming the local score tracked the leaderboard with no sign of leakage.

## Feature engineering and diagnostics

### Confirm the discriminative information is present before engineering features

For a tabular gradient-boosted model, transforms of the existing columns rarely help, because the model already splits on those relationships. Check first whether the information that would separate the classes is in the data at all, and whether the model already extracts the structure that is there, before investing in feature engineering.

Evidence: in playground-series-s6e6 (Predicting Stellar Class), color indices, SED moments (including skew and kurtosis), spectral curvature, a Gaussian-mixture density, leak-free kNN neighbor features, and the physically-derived stellar-locus distance all stayed within or just below the 0.0002 out-of-fold noise floor (experiments 01 to 03). The strongest real-world discriminator, infrared (WISE) color, is absent from the dataset, so no transform of the optical columns could supply it. Added model capacity and per-class threshold tuning also did not help.

### Diagnose the error structure before optimizing

Compute per-class recall, the confusion matrix, and feature importance before choosing what to improve. The result can overturn the prior about which class or boundary is the limit.

Evidence: the photometry-only literature expects the quasar class to limit accuracy, but in playground-series-s6e6, which includes redshift, the quasar class had the highest recall (0.973) and the errors concentrated on the galaxy/star boundary. Redshift carried 0.554 of the feature-importance gain, which explains the reversal: redshift identifies quasars and removes the optical star/quasar degeneracy that limits the no-redshift regime.

## When to invest effort

### Estimate the noise floor and classify the regime before investing

Before engineering or modeling, estimate the metric's noise floor and decide whether the problem has exploitable leverage or is signal-exhausted. The full checklist is in [recognizing-leverage.md](recognizing-leverage.md).

Evidence: in playground-series-s6e6 the seed-to-seed out-of-fold standard deviation was 0.0002; the competition's public-forum bootstrap put the metric at about 0.00087 (public) and 0.00035 (private); the top 25 teams sat within a 0.0002 window, below the private resolution. Diverse strong methods clustered in a tight band, the strongest discriminators (infrared color, morphology) were absent, and no lever beat the baseline above the floor. The problem was in the exhausted regime, identified by the floor and the cluster before further effort.

### Ensembling reduces variance; it does not add signal

A stacked ensemble lowers the variance of the prediction by averaging uncorrelated model errors. It does not supply information the inputs lack, and its gain shrinks as the base models correlate. Metric-aligned post-processing (per-class thresholds for an imbalanced metric) helps only when the base model is not already aligned.

Evidence: in playground-series-s6e6 (experiment 06) a 5-model stack added about 0.0008 over the best single base, while correcting the first attempt's tree count added about 0.0012; threshold calibration added nothing because the base models already used balanced class weights.

## Explainability as validation

### High metrics can hide leakage, feature dominance, and fragility; check why the model decides, not only how well it scores

A model with strong accuracy or AUC can rely on leaked, biased, or unstable features. Beyond the score, inspect the reasoning: feature importance (SHAP for global attribution, LIME for individual predictions), leakage (features that encode the target or information unavailable at prediction time), fragility (does the score collapse when the top feature is removed), and over-reliance on attributes that should not drive the decision (protected attributes, or features with no causal link to the target).

Evidence (external, from a practitioner writeup): a LendingClub credit model scored 0.946 AUC but ranked post-application payment features (out_prncp, last_pymnt_amnt) at the top; removing the 23 leakage features dropped AUC to 0.869, and removing the single top feature dropped it another 0.077. An insurance-cost model had good R-squared but age drove 42 percent of predictions. The metrics did not reveal these; the feature attribution did.

Tool notes: SHAP is deterministic, fast, and gives exact attributions, but is unreliable under highly correlated features. LIME is local and sampling-based, so its explanations vary run to run and use binned ranges, which can read more clearly for non-technical stakeholders but is less precise. Both reflect the model: if the model learned a biased pattern, the explanation reports the bias, it does not correct it.

Applied here: the same checks ran in playground-series-s6e6. Feature importance showed redshift at 0.554; the agreement between cross-validation and the leaderboard was the leakage check (it passed); and the nonzero importance of sky position (alpha, delta), which cannot causally determine class, was flagged as a warning sign of a generation artifact. The named tools formalize a practice already in use.
