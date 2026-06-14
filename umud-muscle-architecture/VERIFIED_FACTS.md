# Verified facts (code + leaderboard only)

Every line here is backed by a source `file:line` or an exact leaderboard number. Nothing else is
allowed in this file. If a fact cannot be re-derived from code or an LB score, it belongs in a
hypothesis note, not here. Pair with `docs/CURRENT_STATE.md`.

## Metric

- Score = (1/3)[MAE(PA)/6 + MAE(FL)/12 + MAE(MT)/3]; tolerances PA=6, FL=12, MT=3, equal weights.
  Median/RMSE side terms weighted 1e-6/1e-9 (tie-breakers only). `metric.py:10-19, 81-88`.
- Official scorer defaults to per-image (`eval_unit="image"`) but supports per-subject aggregation.
  `refs/umud-score.ipynb`. (Per-image assumed; not re-confirmed on the live host.)
- Consequence: a global mean shift on any term is a first-order lever.

## Leaderboard results (2026-06-14, the only oracle)

- PA flat shift on burn_13 base: +0 -> 0.58910, +2 -> 0.55075, +2.5 -> 0.55033, +3 -> 0.55168.
- FL global scale on the PA+2.5 base (0.55033 at x1.00): **x1.05 -> 0.52570** (current best).
- FL x1.10/x1.15/x1.20/x1.25 and MT x0.95/x1.05 are built (`results/submission_fl_x*.csv`,
  `results/submission_mt_x*.csv`) but not yet scored.

## Pipeline (segment_then_measure.py)

- `PRIOR = {fl_mm:74.424, mt_mm:18.628, pa_deg:15.105}` `:96`. Clips PA[5,45] `:97`, FL[30,200] `:98`,
  MT[10,50] `:99`.
- `USE_FL_RECENTER` default ON `:109`; applied `:1144-1145` as `fl_mm *= PRIOR['fl_mm']/mean(fl_mm)`.
  On the live pipeline the raw FL mean is 91.596 so the factor is ~0.81252 (a ~19% shrink) and it
  changes 308/309 rows by mean ~17 mm. It is an active shortening device. (The "0/309 no-op" in older
  docs came from re-applying the pin to an already-pinned file, where the factor is ~1.)
- MT path: `mt_px / px_per_mm`, clip [10,50] `:1104-1107`. There is NO MT recenter anywhere in main().
- Scale router `USE_SCALE_ROUTER` default ON `:102`; `calibrate_image` `:993-1011`; per-family logic in
  `scale_ticks.py:264-298`, incl. `family_b_signature` fixed 134.5 px/cm at conf 1.0 `:235,295`.
  (Stale inline comment at `:102` says "54% coverage"; real coverage is 295/309.)

## Live per-image data (results/calibration_measurement_debug.csv, 309 rows)

- This file is PRE-recenter per-image geometry (recenter touches only the submission frame, not these rows).
- Raw FL mean 91.596 mm, std 27.07 mm (= 2.26 FL tolerances). PA mean 14.627 deg. MT mean 21.836 mm.
- Scale coverage: 295/309 rows have a recovered px/mm; 14 fall back to prior. Methods: right_ruler_5mm 87,
  bottom_ticks 59, png_left_ruler 58, left_ruler_1cm 50, family_b_signature 41, none 14.

## Validation surfaces (why they are blind to global FL/scale)

- 35-image benchmark scorer feeds TRUE scale `experiments/score_weights.py:42` and recenters predicted
  FL to truth mean `:54` -> structurally cannot see a global FL scale/mean error.
- 19 hand labels are self-measured by the same geometry engine; reported FL bias ~-0.46 mm (~unbiased)
  and did not surface the 0.025 FL win. `results/human_benchmark/target_scores.csv`,
  `benchmark_lab/score_labels.py`.

## External reference points (recorded, not host-verified)

- Competition: "UMUD Challenge: Muscle Architecture in Ultrasound Data" (Kaggle, host Paul Ritsche);
  recorded deadline 2026-11-14; CHF 5000; top-3 must release FAIR open-source reproducible code.
- SOTA muscle-US: top tools reach PA <1-1.5 deg, MT <1 mm, FL ~2-6 mm; FL is everyone's fragile term;
  FL = thickness/sin(angle) straight-line extrapolation is the universal convention.
