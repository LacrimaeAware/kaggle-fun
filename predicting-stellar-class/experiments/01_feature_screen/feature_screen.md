# Experiment 01: feature screen against the raw-feature baseline

## Question

Whether hand-crafted features improve out-of-fold balanced accuracy over a LightGBM baseline trained on the ten raw features. The set spans standard ideas (color indices, redshift transforms, redshift-color interactions) and non-standard ones (SED moments including skew and kurtosis, second-difference spectral curvature, per-fold PCA, per-fold Gaussian-mixture density and responsibilities). A Gaussian-noise feature set is the null: a useful idea must beat it.

## Setup

- Data: stratified subsample of 250,000 training rows. Target class (GALAXY, QSO, STAR), the competition metric balanced accuracy.
- Model: LightGBM multiclass, balanced class weights, 350 trees, learning rate 0.05, num_leaves 63, fixed rounds (no early stopping). Identical for every experiment.
- Validation: one fixed 5-fold StratifiedKFold split (seed 42), reused across all experiments, so each comparison is paired against the same held-out rows.
- Each experiment is the baseline ten features plus one feature set. Per-fold transformers (PCA, Gaussian mixture) are fit on the training fold only and applied to the held-out fold, so they do not leak.
- Code: experiments.py.

## Results

Out-of-fold balanced accuracy, ranked, with the change against the baseline and the fold standard deviation.

| experiment | bal_acc | delta vs baseline | fold_std | n_features |
|---|---|---|---|---|
| pca_bands_foldfit | 0.96312 | +0.00018 | 0.0016 | 10 |
| baseline | 0.96294 | 0.00000 | 0.0020 | 10 |
| gmm_density_foldfit | 0.96284 | -0.00010 | 0.0017 | 10 |
| centered_bands | 0.96269 | -0.00025 | 0.0020 | 15 |
| sed_moments | 0.96267 | -0.00027 | 0.0018 | 14 |
| colors_adjacent | 0.96266 | -0.00028 | 0.0017 | 14 |
| redshift_transforms | 0.96266 | -0.00028 | 0.0020 | 12 |
| colors_adjacent+sed_moments | 0.96251 | -0.00043 | 0.0020 | 18 |
| colors_all_pairs | 0.96241 | -0.00053 | 0.0020 | 20 |
| redshift_color_interactions | 0.96241 | -0.00053 | 0.0020 | 14 |
| colors_adjacent+gmm | 0.96228 | -0.00066 | 0.0017 | 14 |
| sed_curvature | 0.96225 | -0.00069 | 0.0018 | 13 |
| colors_all+curvature+moments | 0.96212 | -0.00082 | 0.0019 | 27 |
| null_gaussian | 0.96174 | -0.00119 | 0.0022 | 14 |

## Findings

1. No engineered feature set beat the baseline above the noise. The best, per-fold PCA at +0.00018, sits an order of magnitude below the fold standard deviation of about 0.0020, so it is not evidence of a gain.
2. The Gaussian-noise null was the worst at -0.00119, and adding more engineered columns trended toward it (the 27-feature combination reached -0.00082). On this data the model extracts the available structure from the ten raw features, and explicit deterministic transforms of those features add dimensions without information and dilute slightly.
3. The non-Gaussian and manifold ideas behaved like the standard ones: within noise of the baseline and below zero. The Gaussian-mixture density, the only added feature that summarizes the joint color-redshift distribution rather than restating a row-wise function, reached -0.00010, the closest to baseline of the added-information features, but still not a detectable gain.

## Caveats

- This is a screen: a 250,000-row subsample, a fast fixed model, and a single fold seed. With a fold standard deviation near 0.0020 it can detect gains larger than roughly 0.002 but cannot rule out a true gain smaller than that. The conclusion is "no detectable gain," not "no gain."
- Every engineered feature here is a deterministic function of the raw columns, which a tree can already split on. Features carrying information not determined by the raw columns (neighbor or density structure from the joint distribution, post-processing aligned to the metric, model diversity) are tested in experiment 02.
- Per-fold PCA and the Gaussian mixture were fit on the training fold only; the small PCA value is not leakage.

## Outputs

results/experiments_screen.csv (gitignored).

## Run

```
python experiments.py
```
