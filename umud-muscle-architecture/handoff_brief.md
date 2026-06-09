# UMUD handoff brief (for a collaborating model)

Current-state briefing so another model can get caught up and extend or cross-check this work.
Read it with the canonical docs at the end. Last substantive update: 2026-06-09.

## The competition

UMUD Challenge: Muscle Architecture in Ultrasound Data (Kaggle community competition, deadline
2026-11-14). For each B-mode skeletal-muscle ultrasound image, predict three numbers: pennation
angle PA (deg), fascicle length FL (mm), muscle thickness MT (mm). One row per image.

- Test set: 309 images, `IMG_00001..IMG_00251` (.tif) then `IMG_00252..IMG_00309` (.png).
- Metric (UMUD Score): tolerance-normalized MAE, tolerances PA 6 deg / FL 12 mm / MT 3 mm, equal
  weight, **lower is better**. Submission: comma CSV `image_id,pa_deg,fl_mm,mt_mm`.
- Geometry: segment the two aponeuroses (bright bands) + fascicles (thin diagonal fragments), fit
  lines, then PA = angle of fascicle to the deep aponeurosis, MT = perpendicular gap between the
  aponeuroses, FL = fascicle length between them. Identity used: **FL = MT / sin(PA)**.
- Full official facts + host forum clarifications: **`competition_reference.md`** (read it).

## Where we stand (2026-06-09)

- Best **submitted** public LB: **1.09194** (U-Net PA + tick-calibrated MT on the 58 PNGs, prior FL/MT
  elsewhere). Nothing newer has been submitted yet.
- Leaderboard leader **0.378**; provided DL-Track benchmark **0.679**; a careful BY-HAND human
  labeling of the test set scored **0.459** on the public LB (forum, "PatrickAIForFun").
- The big lever turned out NOT to be a secret method. DL-Track uses MANUAL scaling; the gap from our
  constants to a real score is automated per-image pixels-to-mm calibration + real FL/MT measurement.
  Do not speculate about the leader's private method.

## How we iterate WITHOUT submitting (the most important tool)

The user downloaded a published **35-image expert benchmark** (OSF, gitignored at
`data/osf_arch_benchmark/...`, with `Results_benchmark_architecture_v0.1.0.xlsx`). The xlsx gives,
per image: the TRUE pixels-per-cm scale, all 7 experts' PA/FL/MT, and DL-Track's + SMA's automated
outputs. This is our **local scoreboard**:

- `benchmark_validate.py` - `load_truth()` (expert consensus, human floor, DLTrack/SMA refs) and
  `score(pred_df, truth)` (tolerance-normalized MAE). Human floor 0.307, DL-Track-with-true-scale
  0.331, SMA 0.409.
- `score_on_benchmark.py` - run the saved weights' geometry on the 35 images with TRUE scale.
- `local_infer.py` - regenerate the full 309-row submission from saved weights on CPU (~110 s with
  TTA), VERIFIED to reproduce the Kaggle PA/MT exactly. **Test changes here; never submit to test.**

Caveat: the benchmark is different devices than the Kaggle test set, so FL/MT *magnitudes* don't
transfer, but the *rankings* and the measurement-logic conclusions do.

## Per-target status (measured on the 35 experts, with TRUE scale)

Current end-state geometry: **overall 0.368** (PA 0.164, FL 0.449, MT 0.490) vs human 0.307,
DL-Track 0.331. So the MEASUREMENT is DL-Track-competitive given scale.

- **PA - effectively solved (0.164, beats DL-Track's 0.242).** Total-least-squares (PCA) fragment
  orientation + length-weighted median + a min-6-deg filter that rejects aponeurosis-parallel
  fragments. Wired. Leave it.
- **MT - good given scale (0.490), ~+1 mm high.** Apo-gap geometry works; the only lever is scale.
- **FL - the bottleneck (0.449), scatter-limited.** It is NOT bend or systematic bias (correcting
  bias did not help). The fascicle TRAINING masks are sparse dashes by design (~14 fragments/image,
  median 59 px, ~6 % of image width - host confirmed they label only clear-contrast fascicles), so
  per-image FL is noisy. FL = MT/sin(PA), recentered to the trusted mean, is the wired estimator.
  Every bend/curve/tracking attempt failed (parabola, streamlines, banded - see experiments log).

## The pipeline and its switches (`segment_then_measure.py`)

Trains apo + fascicle smp U-Nets (ResNet34, 0.5 Dice + 0.5 BCE, flips/rotations, 384 px), predicts
masks, measures geometry, writes `submission_segmentation.csv`. Runs on a Kaggle GPU via
`kaggle_segment_notebook.ipynb` (pulls the script from the public repo; no API token here). Saved
weights `results/seg_apo.pt` / `seg_fasc.pt` let everything else run on CPU.

Env switches (defaults in parens):
- `UMUD_TTA` (on) - mirror+scale test-time aug, averages sigmoids then thresholds; denoises the sparse
  masks (benchmark 0.383 -> 0.370). The only FL gain that is not a mean-fit.
- `UMUD_USE_IDENTITY_FL` (on) - FL = MT/sin(PA), recentered to the prior mean.
- `UMUD_USE_CALIBRATED_MT` (on) - MT_px / px_per_mm where scale is found.
- `UMUD_SCALE_ROUTER` (on) - the validated per-family scale router (below). `CALIBRATION_MIN_CONF` 0.5.
- `FASC_MIN_AREA` 40, `FASC_MIN_ANG` 6 - the PA-polish post-processing (wired constants).
- `UMUD_FASC_POS_WEIGHT` (0) - >0 adds a pos_weight to the FASCICLE BCE to push recall (the user's
  idea to make the model draw more of each dash). Kaggle-GPU retrain only. UNTESTED.
- `UMUD_CLAHE` (0) - CLAHE contrast-normalize input in read_rgb; surfaces more fragments but MUST
  retrain BOTH models with it on (inference-only hurts, train/test mismatch). UNTESTED on GPU.

## Scale calibration - the leaderboard lever (NEW, the main 2026-06-09 work)

The 251 "unscaled" TIFFs ARE scalable. The host confirmed pixels are always square and bottom ticks
are 1 cm apart. The test set is **four device families by UI** (the 800x1200 size hides two devices);
each family's scale was validated by reading its actual ruler (no test labels exist):

| family | n | scale | source |
| --- | ---: | ---: | --- |
| PNG left numbered ruler | 58 | 150 px/cm | left ruler, 5 mm minor ticks (the feared 2x bug does NOT exist) |
| 644x1088 left depth ruler | 50 | 126 px/cm | left ruler 0-50 mm, 1 cm ticks |
| Telemed 800x1200 ("De 50 mm") | 49 | 134 px/cm | bottom ticks AND left 0-50 mm ruler agree |
| clean cropped/other | ~10 | varies | bottom ticks (IMG_00040 -> 78 px/cm) |
| **German Siemens 800x1200 ("12L3 Quadriceps")** | **~132** | **UNSOLVED** | scale is a measured BRACKET, not ticks |

- `scale_ticks.py` - `recover_scale` (bottom ticks), `recover_scale_left_ruler`, and
  `recover_for_image(gray, name)` the per-family ROUTER. Wired into `calibrate_image`.
- Coverage: **167/309 = 54 % scaled** with validated detectors (was 58 PNG = 19 %). Regenerating
  `submission_local.csv`: calibrated MT on 167, FL now per-image (std 25.6, range 30-158, was flat
  74.4), MT mean 20.8 mm range 13-31 with ZERO clipping (the tell that scales are sane).
- Harness: `experiments/scale_coverage.py`, `experiments/scale_qa.py` (overlays in
  `results/calibration_qa/`), `experiments/check_submission.py`. Full map: `competition_reference.md`
  sections 3, 3a, 3b.

This is the first change that targets the actual LEADERBOARD (the test set) vs the local benchmark.
The Kaggle gain is UNMEASURABLE locally (no test labels); it is submission-ready and the user decides
whether to spend a submission.

## The two open fronts

1. **German Siemens scale-bar reader (~132 imgs, 43 %).** The remaining scale gap; its depth is a
   bracket/label, not periodic ticks. Solving it pushes coverage past 90 %.
2. **Kaggle-GPU fascicle retrain** with `UMUD_FASC_POS_WEIGHT` (recall) +/- `UMUD_CLAHE` (contrast),
   to fill out the sparse masks and reduce FL scatter. Evaluate the downloaded `seg_fasc.pt` locally
   on the 35-expert board; zero submissions to test. FL payoff uncertain (it is scatter-limited).

## Constraints and values

- The artifact that matters is the thinking and writeup, not the rank. Claims proportional to
  evidence; label hypotheses; cite or drop.
- No manual labeling of the 309 test images. External data (the 35-expert benchmark, used for local
  validation only) must be DECLARED in any writeup. Pipeline must be reproducible.
- Public GitHub repo (github.com/LacrimaeAware/kaggle-fun): no secrets, no personal data, no
  Co-Authored-By trailer. Do not submit without explicit say-so. Local AMD 5700 XT cannot train on
  Py3.13 (no torch-directml); CPU training works but is slow; Kaggle GPU is the training path.

## Canonical docs to cross-check

- `competition_reference.md` - official rules + host clarifications + the full scale family map (3a/3b). **Most current.**
- `synthesis.md` - the canonical methods synthesis (folds in the older codex_review / forward_plan / ranked_research_directions).
- `experiments/README.md` - the experiment log (exp01-11 + scale tools), with method/result/read each.
- `benchmark_findings.md` - the 35-expert benchmark analysis.
- `segment_then_measure.py` + `kaggle_segment_notebook.ipynb` - the pipeline and its Kaggle runner.
- `scale_ticks.py`, `benchmark_validate.py`, `score_on_benchmark.py`, `local_infer.py` - the calibration router and the local scoreboard.
- Older/partly-superseded: `leader_playbook.md`, `forward_plan.md`, `ranked_research_directions.md`, `codex_review.md`, `strategy_brief.md`, `plan.md`, `rundown.md`, `writeup.md`.
