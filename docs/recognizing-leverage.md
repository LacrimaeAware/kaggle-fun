# Recognizing exploitable leverage

A checklist for deciding, before investing effort on a prediction problem, whether a real gain is available (exploitable leverage) or the signal is exhausted and only standard craft and luck remain. It is stated generally. The evidence instance is the playground-series-s6e6 competition and its public forum variance analysis (see predicting-stellar-class/).

## Two regimes

For a given metric and dataset a problem is in one of two regimes:

- Exploitable: a real improvement is available that exceeds the metric's noise. Effort, including non-standard method, can pay.
- Exhausted (noise-limited): the available signal is already extracted by standard methods, and the differences between strong solutions are smaller than the metric's resolution. Effort buys variance reduction and a favorable draw, not skill.

The first step on any problem is to estimate the noise floor and locate the regime. Investing structural effort in an exhausted problem is the common waste.

## Estimating the noise floor

- Multi-seed cross-validation: rerun the same model under several fold seeds; the standard deviation of the out-of-fold score is the floor for comparing scores. (s6e6: 0.0002.)
- Bootstrap the metric on held-out labels to get its standard deviation at the public and private slice sizes. (s6e6: about 0.00087 on the 49,500-row public slice, about 0.00035 on the 198,000-row private slice.)
- A null control: add a feature of pure noise; its effect bounds what an uninformative change does. (s6e6: a Gaussian-noise feature scored -0.00119.)

A change smaller than the floor is not evidence of improvement.

## Signals that leverage exists

- A discriminative input known to carry signal is present but unused or under-used by the current model.
- Error analysis shows a concentrated failure mode tied to a feature that is held or buildable, not to a feature that is absent.
- The gap between the current score and the best known score exceeds the metric's noise, and the better solutions are not yet in hand.
- The problem sits near or below a recoverability floor that a stronger method could cross (a low signal-to-noise regime where the current method, not the data, is the limit).

## Signals that the signal is exhausted

- Diverse strong methods cluster in a tight band, with no family meaningfully ahead.
- The top of a leaderboard sits within the metric's own resolution: the spread from rank 1 to rank N is below the private-slice standard deviation. (s6e6: the top 25 within 0.0002, below the private 0.00035.)
- The strongest known discriminators for the domain are absent from the data, a structural ceiling no transform can pass. (s6e6: infrared color and image morphology are absent; redshift is present and already used.)
- Controlled experiments show added features, added capacity, neighbor structure, and metric-aligned post-processing do not beat a baseline above the floor.
- Cross-validation and leaderboard scores anti-correlate near the top, a sign of optimizing past the signal into noise.

## What standard craft buys in the exhausted regime

- Ensembling diverse models reduces the variance of the prediction; it does not add signal. The gain is bounded and shrinks as the models correlate.
- Metric-aligned post-processing (per-class thresholds for an imbalanced metric) helps only if the base model is not already aligned. A model trained with balanced class weights has the alignment built in, so calibration on top adds little. (s6e6: threshold calibration was null on a balanced-weighted pipeline; it is the main lever for teams whose bases are log-loss-trained and unbalanced.)
- Correcting an over-fit configuration can recover more than the ensembling. (s6e6: most of the gain over the first attempt was reducing the tree count.)

## Real edge versus noise edge

A small gain is worth pursuing only if it is real: it exceeds the noise floor and reproduces across seeds and splits. A small gain within the floor is noise, and pursuing it is overfitting to one slice. Where a real small edge is scaled, by capital and leverage in quantitative finance or by repeated application, distinguishing a real edge from a noise edge is the load-bearing skill, more than the size of the edge.

## Decision rule

1. Estimate the noise floor first.
2. Classify the regime from the signals above.
3. In the exploitable regime, invest in the lever the diagnosis points to: the missing-but-buildable feature, the addressable error mode, or the method that crosses the recoverability floor.
4. In the exhausted regime, apply bounded standard craft (a tighter configuration, a small ensemble) to reach the cluster, then stop. The remaining ranking is the draw, not skill.
