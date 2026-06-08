# Experiments

Index of the feature and method experiments for predicting-stellar-class. Each experiment compares against the LightGBM baseline on fixed 5-fold StratifiedKFold out-of-fold balanced accuracy, on a stratified subsample. The seed-to-seed noise floor is 0.0002 (experiment 02), so a change below about 0.0002 to 0.0004 is within noise.

| Experiment | Tests | Result |
|---|---|---|
| [01 feature_screen](01_feature_screen/feature_screen.md) | color indices, SED moments (skew, kurtosis), spectral curvature, centered bands, redshift transforms and interactions, per-fold PCA, per-fold Gaussian-mixture density; a Gaussian-noise null | No feature set beat the baseline above noise. Best (PCA) +0.00018. Adding more columns trended toward the null. |
| [02 levers](02_levers/levers.md) | multi-seed noise floor, model capacity, leak-free kNN neighbor features, per-class thresholds | Noise floor 0.0002. Capacity -0.00595. kNN -0.00053. Thresholds no change. |
| [03 locus_and_diagnostics](03_locus_and_diagnostics/locus_and_diagnostics.md) | stellar-locus distance, UV-excess indicator, grid thresholds; confusion matrix, per-class recall, feature importance | Locus features -0.0005 to -0.0001. QSO recall highest (0.973); errors on the galaxy/star boundary. Redshift 0.554 of feature importance. |
| [04 model_diversity](04_model_diversity/model_diversity.md) | CatBoost, LightGBM and CatBoost blend | CatBoost -0.00738 (untuned). Best blend weight 1.0 on LightGBM (no gain). |
| 05 full_scale_seedbag (experiments5.py) | full-data tuning and seed bagging of a single LightGBM | Started, then stopped before completion once the forum variance analysis showed the real lever is multi-model stacking, not single-model seed bagging. |
| [06 stacked_ensemble](06_stacked_ensemble/stacked_ensemble.md) | 5-model GBDT zoo, balanced-logreg stack, DE threshold calibration | OOF 0.96596, +0.00196 over the first attempt. Mostly a tighter single config (+0.0012) plus a small ensemble lift (+0.0008); calibration null (bases pre-balanced). About 0.004 below the public cluster (neural-tabular models omitted). |

## Conclusion

No feature transform or single-model lever beat the baseline above the 0.0002 noise floor (experiments 01 to 04): on the optical-plus-redshift features the model is at its feature ceiling. The diagnostics (experiment 03) explain why: redshift carries most of the signal and makes the quasar class the easiest of the three, the residual errors fall on the galaxy/star boundary where the real discriminator is morphology (absent here), and the strongest photometry-only discriminator, infrared color, is also absent.

A model-level gain exists and is separate from the feature ceiling. The stacked GBDT ensemble (experiment 06) reached out-of-fold 0.96596, +0.00196 over the single-LightGBM first attempt (0.9640). About +0.0012 of that corrects the first attempt's tree count (it slightly overfit), and about +0.0008 is the ensemble. Threshold calibration added nothing because the base models were already balanced. The result is about 0.004 below the public stacked ceiling (~0.9697), a gap that is the neural-tabular base models (GPU) the top teams use. Per the competition's own forum variance analysis, the top cluster is separated by less than the metric's noise, so beyond ~0.9697 the ranking is the private draw, not skill.
