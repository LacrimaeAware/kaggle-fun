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

## Leaderboard results (the only oracle)

- PA flat shift on burn_13 base: +0 -> 0.58910, +2 -> 0.55075, +2.5 -> 0.55033, +3 -> 0.55168.
- FL global scale on the PA+2.5 base: x1.05 -> 0.52570.
- family_b scale constant 134.5 -> 147 px/cm (41 rows; found by the user hand-reading ticks): ~0.488.
- aponeurosis band fix: 0.46076; band fix + FL x1.05: **0.46041 (current best, `submission_bandfix_flx105.csv`)**.
- reproducible median pipeline, one `local_infer.py` run: 0.47473 (`submission_reproduced.csv`); the
  ~0.014 gap to 0.460 is old per-row CSV residue, not a model difference.
- min_extrap_top3 FL on that pipeline: **0.49983 (REFUTED 2026-06-15)**. The benchmark scored it 0.39
  (below the human floor) and it regressed the LB by 0.025: the benchmark does not predict the LB.

## Pipeline (segment_then_measure.py)

- `PRIOR = {fl_mm:81.866, mt_mm:18.628, pa_deg:15.105}` `:96` (fl raised from 74.424 to bake in the
  LB-fitted FL level). Clips PA[5,45], FL[30,200], MT[10,50].
- `USE_FL_RECENTER` default ON; applied in main() as `fl_mm *= PRIOR['fl_mm']/mean(fl_mm)`, and
  `local_infer.py:98` applies the same pin regardless of the flag. Raw FL mean ~91.6, so the factor is
  ~0.89 (a ~11% shrink to mean 81.866). It is an active shortening device. (The old "0/309 no-op" came
  from re-applying the pin to an already-pinned file.)
- `FL_FRAGMENT_MODE` default `median` `:107`. `min_extrap_top3` was tried and REFUTED on the LB
  (0.49983 vs median 0.47473, 2026-06-15); do not re-flip without a test-distribution gate.
- MT path: `mt_px / px_per_mm`, clip [10,50] `:1104-1107`. There is NO MT recenter anywhere in main().
- Scale router `USE_SCALE_ROUTER` default ON; `calibrate_image`; per-family logic in `scale_ticks.py`,
  incl. `family_b_signature` now **147 px/cm** (was 134.5; the user hand-read ticks at ~147 and the LB
  confirmed the fix, ~0.488). bottom_ticks reads were corrected ~+9.5% the same way. Real coverage 295/309.

## Live per-image data (results/calibration_measurement_debug.csv, 309 rows)

- This file is PRE-recenter per-image geometry (recenter touches only the submission frame, not these rows).
- Raw FL mean 91.596 mm, std 27.07 mm (= 2.26 FL tolerances). PA mean 14.627 deg. MT mean 21.836 mm.
- Scale coverage: 295/309 rows have a recovered px/mm; 14 fall back to prior. Methods: right_ruler_5mm 87,
  bottom_ticks 59, png_left_ruler 58, left_ruler_1cm 50, family_b_signature 41, none 14.

## Validation surfaces (why they are blind to global FL/scale)

- 35-image benchmark scorer `experiments/score_weights.py` feeds TRUE scale `:42` and recenters
  predicted FL to truth mean `:54` -> structurally cannot see a global FL scale/mean error.
- `benchmark_lab/honest_validate.py` (built 2026-06-15) removes the recenter and reports per-image vs
  the 7-rater consensus + per-term human floor. With true scale: PA 0.1505 (floor 0.2445), MT 0.0840
  (floor 0.0810), FL 0.5218 (floor 0.4026, over-reads +5.8 mm). EVEN un-blinded it did not predict the
  LB: it ranked min_extrap_top3 best (FL 0.39) and that submission regressed to 0.49983. Different
  distribution than test. Use for measurement-bug detection, not submission gating.
- 19 hand labels are self-measured by the same geometry engine; reported FL bias ~-0.46 mm (~unbiased)
  and did not surface the 0.025 FL win. The only per-image TEST truth is the correction-UI labels.

## External reference points (recorded, not host-verified)

- Competition: "UMUD Challenge: Muscle Architecture in Ultrasound Data" (Kaggle, host Paul Ritsche);
  recorded deadline 2026-11-14; CHF 5000; top-3 must release FAIR open-source reproducible code.
- SOTA muscle-US: top tools reach PA <1-1.5 deg, MT <1 mm, FL ~2-6 mm; FL is everyone's fragile term;
  FL = thickness/sin(angle) straight-line extrapolation is the universal convention.
