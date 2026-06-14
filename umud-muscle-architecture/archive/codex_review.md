# UMUD Codex review

Public-safe review of the current UMUD state, written for another model or collaborator to
pick up without needing private context. This is a hypothesis map, not a claim that the
leader's private UMUD method is known.

## Sources inspected

- Current repo docs: `handoff_brief.md`, `writeup.md`, `strategy_brief.md`,
  `leader_playbook.md`, `rundown.md`, and `plan.md`.
- Current runner: `kaggle_segment_notebook.ipynb`, which only installs dependencies,
  downloads `segment_then_measure.py` from GitHub, and runs it on Kaggle GPU.
- Current scripts: `segment_then_measure.py`, `mask_geometry.py`,
  `train_segmentation.py`, `train_pseudo_baseline.py`, and `metric.py`.
- Ignored local artifacts: result CSVs, pseudo-label metrics, trained segmentation weights,
  `file_manifest.csv`, and ignored public reference notebooks in `refs/`.
- Kaggle CLI check on 2026-06-09: public leader `suguuuu | no hand labelling` at `0.37766`;
  `DLTrack_0.3.1_benchmark_solution.csv` at `0.67944`.
- Follow-up pasted note from another agent: DL-Track-US normally relies on a manual scaling
  step where the user enters/clicks a known distance. If the public benchmark was forced
  into an automated Kaggle submission, it likely used a fixed or assumed scale, which is a
  plausible reason it trails the leader.
- Public `sugupoko` notebook index from Kaggle CLI. High-signal public refs include
  `baseline-2d-segmentation-with-smp`, `1st-place-hms-inference-code`,
  `1st-place-all-train-code`, `10th-place-inference-yolov7ensemble-pseudo-res640`, and
  `v30-4-group-ensemble-blend-blur-bright-lowco`.
- External background: DL_Track_US and UMUD papers/documentation indicate that the natural
  task shape is segmentation followed by geometric post-processing for PA, FL, and MT.

## Current repo state

The submission loop is solved. The repo knows the valid 309 IDs and writes comma CSVs with
actual `.tif` / `.png` suffixes.

Current score ladder:

| Approach | Public LB |
| --- | ---: |
| ExtraTrees pseudo-label regressor, inflated model FL/MT | 1.23135 |
| ExtraTrees PA, prior FL/MT constants | 1.11066 |
| U-Net segmentation PA, prior FL/MT constants | 1.12324 |
| DLTrack 0.3.1 benchmark | 0.67944 |
| Current public leader | 0.37766 |

The latest notebook is not a research notebook; it is a Kaggle launcher for
`segment_then_measure.py`. The actual model is a two-U-Net pipeline:

```text
test image -> aponeurosis mask + fascicle mask -> line geometry -> PA
```

It still writes constant `fl_mm = 74.424` and `mt_mm = 18.628`, so it is not measuring the
two millimetre targets yet.

## Main diagnosis

Hypothesis 1: the current score is mostly stuck because FL and MT are constants.

Evidence for:

- The best scored file changed FL/MT from a noisy model output to constants and improved
  from `1.23135` to `1.11066`.
- The segmentation run changed only PA and scored `1.12324`, slightly worse than the
  ExtraTrees-PA plus constants run.
- The metric weights PA, FL, and MT equally after tolerance normalization. A flat MT value
  is especially expensive because MT tolerance is only `3 mm`.
- DLTrack's published task shape explicitly measures muscle thickness and fascicle length
  from segmented aponeuroses/fascicles, then reports values in millimetres.
- DL-Track's usual workflow includes manual scale calibration, so automated per-image scale
  recovery is a credible missing ingredient between the benchmark and the current leader.

Evidence against / uncertainty:

- We do not have hidden labels, so the per-target leaderboard error split is unknown.
- A much better PA estimate could still help, but PA-only movement has so far been small.

Next test:

- Build a calibration table and a first real FL/MT submission, even if PA is unchanged. If
  the score drops materially, this hypothesis is confirmed.

Hypothesis 2: the current U-Net PA result underperformed because the segmentation masks are
not yet good enough for geometry.

Evidence for:

- The first segmentation run reported weak fascicle validation quality in the handoff
  context, and fascicle masks are the fragile part of PA.
- The segmentation PA distribution was more spread out than the ExtraTrees PA distribution.
  Under normalized MAE, a weak high-variance predictor can lose to a mean-reverting one.
- The current training is a single split, one backbone, simple thresholding, no TTA, no
  fold ensemble, no visual QA gate, and no confidence-aware fallback beyond total geometry
  failure.

Evidence against / uncertainty:

- The validation Dice is not the actual scoring target. A low Dice can still yield usable
  angles if the line orientation is right.
- The public score difference between `1.11066` and `1.12324` is small enough that it could
  be ordinary leaderboard noise or a few bad PA cases.

Next test:

- Add overlay QA and geometry validation on held-out labeled masks. Evaluate derived PA
  error from predicted masks, not only Dice.

Hypothesis 3: the public leader's UMUD edge is likely a stronger segment-then-measure system
with calibration, test-domain handling, and ensembling, not a pure direct regressor.

Evidence for:

- The leader's display name says `no hand labelling`, which points toward an automated
  method rather than manual test annotation.
- The problem itself exposes segmentation supervision, not a normal label CSV.
- DLTrack benchmark already reaches `0.67944`, far beyond our constant-FL/MT files, using
  the same broad automated measurement concept.
- The leader is far better than the DLTrack benchmark, and one concrete way to beat an
  automated DLTrack run is to recover per-image pixels-per-millimetre instead of relying on
  a fixed assumed scale.
- Public `sugupoko` notebooks show a consistent reusable playbook: segmentation baselines
  with SMP/UNet, fold models, TTA, OOF-based ensembling, model soups or checkpoint soups,
  pseudo-labeling, and quality-specific specialist models.

Evidence against / uncertainty:

- The leader has not published the UMUD solution. The public notebooks are from other
  competitions, so they only show habits and transferable craft.
- It is possible the current leader uses extra public UMUD/DLTrack/FALLMUD data, a private
  pretrained model, or a clever competition-specific postprocess that is not visible.

Next test:

- Treat the leader playbook as experiment inspiration only. Reproduce the DLTrack benchmark
  first, then measure which extra components actually help.

## What the leader-style playbook means here

The public pattern is not "try a giant model first." It is closer to:

1. Use the task's native structure.
2. Build a reliable validation signal.
3. Inspect predictions visually.
4. Add domain-shift handling.
5. Use multiple models/checkpoints when each base has real signal.
6. Blend simply or with OOF evidence.
7. Add quality-aware fallback/specialists for failure regimes.

Translated to UMUD:

- Native structure: segment aponeuroses and fascicles, then measure.
- Validation signal: held-out mask-derived PA/FL_px/MT_px, plus overlay QA.
- Visual inspection: save predicted masks, fitted lines, intersections, tick detections, and
  final numeric predictions for representative and worst-confidence images.
- Domain shift: ultrasound brightness/contrast/speckle/depth/ruler differences; train
  augmentations and normalization around those shifts.
- Ensembling: fold/seed/backbone ensembles only after one base segment-then-measure model
  produces real FL/MT signal.
- Quality-aware fallback: detect failed segmentation, failed calibration, unstable fascicle
  geometry, and sequence outliers; fall back per target rather than for the whole row.

## Priority experiments

### Experiment A: DLTrack reproduction

Goal: get an automated baseline near the public `0.67944` benchmark.

Why first:

- It proves the full unit-aware path.
- It exposes the expected DLTrack preprocessing, aponeurosis/fascicle postprocess, and
  calibration assumptions.
- It creates a real baseline to improve, rather than comparing against constant FL/MT.

Risk:

- Environment friction and hidden GUI assumptions.
- The benchmark CSV may use contest-specific calibration or preprocessing not obvious from
  the public package.

### Experiment B: tick/ruler calibration

Goal: estimate pixels per millimetre for every test image with confidence.

Why high leverage:

- PA does not need scale, but FL and MT do.
- Without calibration, `fl_mm` and `mt_mm` remain priors, and the score ceiling is poor.

Candidate method:

1. Crop likely ruler regions.
2. Use threshold/edge detection to isolate tick marks.
3. Estimate repeated spacing using connected components, Hough lines, or 1D projection peaks.
4. Convert to `px_per_mm`; bottom tick ambiguity can use the public-discussion assumption
   that ambiguous bottom ticks are `1 cm` apart.
5. Smooth/fill within adjacent 5-frame groups.
6. Save overlays for manual QA of the algorithm, not manual labels.

Failure modes:

- Text overlays or ultrasound UI marks may mimic ruler ticks.
- Different devices may put scale marks on different sides.
- Some PNG test images may have different crop/layout conventions.

### Experiment C: geometry-first FL/MT before better PA

Goal: use current or DLTrack masks to compute MT_px and FL_px, then convert via calibration.

Why:

- The current code already has part of the geometry in `mask_geometry.py`.
- Even rough FL/MT signal may beat constants if calibration is sane.

Important detail:

- FL can be computed two ways: direct line intersection length between aponeuroses, or
  `MT / sin(PA)` style straight-line extrapolation. Both should be tried on local
  pseudo-label validation because they fail differently.

### Experiment D: segmentation QA and fold ensemble

Goal: make segmentation reliable enough for geometry.

Steps:

- Train fold models rather than one random 85/15 split.
- Track derived PA/FL_px/MT_px validation error, not Dice alone.
- Add TTA with correct inverse transforms for masks.
- Save overlay panels for the worst validation geometry errors.
- Add confidence features: number of apo bands, line fit residuals, fascicle component count,
  angle dispersion, calibration confidence, and sequence deviation.

### Experiment E: sequence smoothing and fallback

Goal: reduce large errors without flattening real signal.

Candidate rules:

- Group likely 5-frame sequences by filename order and image similarity.
- Smooth calibration and final predictions within groups.
- Use Hampel/median filters for outliers.
- Fall back per target: e.g. keep PA if geometry is stable but replace MT when calibration
  confidence is low.

## What not to over-prioritize yet

- A bigger direct regressor. It can be useful later as an ensemble member or fallback, but
  it has no native way to discover millimetre scale unless labels or calibrated pseudo-labels
  are available.
- Heavy ensembling before FL/MT exist. Ensembles of constant or uncalibrated targets mostly
  average the same blind spot.
- Public-LB hill climbing. The leaderboard is useful for sanity checks, but the submission
  budget is small and the public set can be misleading.

## Suggested immediate next work order

1. Add a calibration prototype that outputs `image_id, px_per_mm, confidence` plus overlays.
2. Add MT first: use aponeurosis masks to measure thickness in pixels, convert with
   `px_per_mm`, and keep PA/FL as the current best fallbacks.
3. Add FL geometry from fascicle/aponeurosis intersections and compare direct line length
   against `MT / sin(PA)` style extrapolation.
4. Reproduce or port enough of DLTrack to compare its masks/geometry against our script.
5. Run one Kaggle GPU experiment that changes FL/MT, not only PA.
6. If that improves, add fold/TTA segmentation and sequence smoothing.

## First calibration prototype

`tick_calibration.py` now implements a diagnostic tick/ruler detector. It is deliberately
standalone so it can be improved without touching the current Kaggle training script.

Run:

```bash
python umud-muscle-architecture/tick_calibration.py --overlay-limit 309 --overlay-failures
```

Outputs:

- `results/calibration_debug/tick_calibration.csv`
- `results/calibration_debug/overlays/*.jpg`
- overlay contact sheets created during local QA

First local diagnostic run:

| Split | Detected | Confidence >= 0.5 |
| --- | ---: | ---: |
| All 309 | 203 / 309 | 88 |
| 251 TIFF | 145 / 251 | 30 |
| 58 PNG | 58 / 58 | 58 |

After the PNG-left-ruler fix, the more relevant confidence-0.7 gate is:

| Split | Confidence >= 0.7 |
| --- | ---: |
| All 309 | 68 |
| 251 TIFF | 10 |
| 58 PNG | 58 |

Interpretation:

- PNGs are the cleanest first win: the detector now locks onto the left depth ruler and
  gives depth-aware scales, e.g. `IMG_00252.png` at 4.0 cm gives `15.0 px/mm`, while
  `IMG_00280.png` at 3.0 cm gives `20.0 px/mm`.
- Some TIFFs with dark UI borders have usable side or bottom ruler detections.
- Many cropped TIFFs expose no obvious border ruler to this first detector; those need a
  second strategy before any all-image MT submission should trust the calibration blindly.
- The detector should be used behind a confidence gate and sequence/group fallback, not as
  a blanket replacement for all 309 rows yet.

Implementation status:

- `segment_then_measure.py` now imports `tick_calibration.py` when available.
- Calibrated MT is on by default behind `UMUD_CALIBRATION_MIN_CONF=0.7`.
- Calibrated FL is off by default (`UMUD_USE_CALIBRATED_FL=0`) because fascicle geometry is
  more fragile and should be tested after MT.
- The Kaggle notebook now downloads both `segment_then_measure.py` and `tick_calibration.py`.
- The segment script writes `calibration_measurement_debug.csv` beside the submission so the
  exact rows using calibrated MT can be audited after a run.

## Questions for another model

1. Given binary aponeurosis/fascicle masks, what is the most robust way to compute FL and
   MT for noisy partial fascicles?
2. How should tick marks be detected across heterogeneous ultrasound UI layouts without
   confusing text, frame borders, or artifacts for scale marks?
3. Can DLTrack's preprocessing and post-processing be run headless on these 309 images, and
   what assumptions does it make about pixel spacing?
4. Does the 5-frame sequence structure hold exactly by filename order, and how much should
   predictions be smoothed without erasing true motion?
5. Which local validation metric best predicts leaderboard movement: Dice, PA/FL_px/MT_px
   geometry error, visual failure categories, or a composite confidence score?
