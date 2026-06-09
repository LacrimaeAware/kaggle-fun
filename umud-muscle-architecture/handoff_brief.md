# UMUD handoff brief (for a collaborating model)

A current-state briefing so another model can review or extend this work and cross-check
the conclusions. Read it alongside the canonical docs listed at the end. Where it draws a
conclusion it states the hypothesis and what would confirm or break it, because several of
these conclusions are not yet verified.

## The competition

UMUD Challenge: Muscle Architecture in Ultrasound Data (Kaggle community competition,
deadline 2026-11-14). For each skeletal-muscle ultrasound image, predict three numbers:
pennation angle PA (degrees), fascicle length FL (millimetres), muscle thickness MT
(millimetres). One row per image. The test set is 309 images: 251 `.tif`
(IMG_00001..IMG_00251) then 58 `.png` (IMG_00252..IMG_00309). Metric (UMUD Score):
tolerance-normalized mean absolute error, tolerances PA 6 deg, FL 12 mm, MT 3 mm, equal
weight, lower is better. Submission is a comma CSV with columns `image_id,pa_deg,fl_mm,mt_mm`.
Plain-language explainer: `rundown.md`. Competition writeup: `writeup.md`.

## Where we stand

- Best public score so far: **1.11066**.
- Leaderboard leader: **0.37766** (handle sugupoko, marked "no hand-labelling").
- Official DL-Track benchmark: **0.67944**.
- Lower is better, so there is a large real gap. This is not a noise-limited leaderboard.

Submission history (also in `writeup.md`):

| What | Public LB |
| --- | --- |
| ExtraTrees on mask pseudo-labels, with inflated model FL/MT | 1.23135 |
| ExtraTrees pennation + prior FL/MT constants | **1.11066** (best) |
| Segmentation U-Net pennation + prior FL/MT constants | 1.12324 |

## What has been built

- `segment_then_measure.py`: the main pipeline. Auto-discovers the data wherever it is
  mounted, trains aponeurosis and fascicle U-Nets (segmentation_models_pytorch, ResNet34,
  Dice+BCE, flips/rotations, AMP), predicts masks on the 309 test images, fits lines,
  computes pennation geometrically, writes `submission_segmentation.csv`. FL and MT are
  left at prior constants (74.424 mm, 18.628 mm) because pixels-to-mm calibration is not
  built. Runs on a Kaggle GPU via `kaggle_segment_notebook.ipynb` (no API token needed; the
  notebook pulls the script from the public repo).
- `mask_geometry.py`: derives PA / FL_px / MT_px from ground-truth masks (the pseudo-labels).
- `tick_calibration.py`: first standalone calibration prototype. It writes
  `results/calibration_debug/tick_calibration.csv` plus overlays. First diagnostics:
  203/309 images detected; after the PNG left-ruler fix, 58/58 PNGs and 10/251 TIFFs are
  above confidence 0.7. PNG coverage is now depth-aware; cropped TIFF coverage needs a
  second strategy.
- `metric.py`: a local UMUD scorer.
- `train_pseudo_baseline.py`, `submission_variants.py`: the earlier ExtraTrees regressor path.
- Data layout (the local mirror equals the Kaggle mount): aponeurosis 1048 image/mask pairs,
  fascicle 2761 image/mask pairs, test 309. Images are uint8. Masks arrive at different
  resolution and aspect than their images, so they are resized to the image; this alignment
  is a known quirk, not yet shown to matter.

## The problem we are stuck on

The score is dominated by fascicle length and muscle thickness, and we have no per-image
signal for either: both are flat constants in every submission so far. PA is the only target
we can currently estimate per image, and PA is the smallest of the three levers (tolerance
6 deg vs FL 12 mm, MT 3 mm, but FL and MT are completely unmodelled). Moving toward the
benchmark most likely needs pixels-to-millimetre calibration (tick marks on the images) so
FL and MT become real measurements. That calibration is unsolved and is the central
difficulty. PA is scale-free, so it does not need calibration.

## What the latest result suggests (hypotheses, not settled)

1. Swapping the regressor's pennation for segmentation-geometry pennation made the score
   slightly worse (1.11066 -> 1.12324). The only column that changed was pennation; FL and MT
   were identical constants in both files.
   - Hypothesis: the segmentation models are undertrained (fascicle Dice ~0.25), so their
     per-image angles are more dispersed (std 3.74 deg), and the tolerance-normalized MAE
     penalizes confident wrong angles more than the regressor's mean-reverting ones.
   - Supports it: only the PA column changed; the new PA is more spread; the metric is L1
     with a 6 deg tolerance, which rewards staying near the centre when true signal is weak.
   - Would break it: a better-trained segmentation (higher Dice, TTA, more epochs) that
     scores below 1.11066, or evidence the spread is correct and the regressor was biased.
2. Bigger picture: PA looks near its achievable floor with simple methods, and the FL/MT
   constants account for most of the gap.
   - Supports it: we are at ~1.11 vs ~0.68 with FL/MT untouched; the benchmark gets FL/MT
     right by measuring in mm.
   - Would break it: a large score drop from a PA-only change (would mean PA had headroom).

## The open question we are actively researching

Why is the leader (~0.378) about twice as good as the official DL-Track tool (~0.679)?
Candidate explanations, none verified yet:

- He solved calibration well (tick-mark pixels-to-mm), so FL and MT drop sharply.
- He trained on external labeled data with real measured PA/FL/MT (e.g. the DL-Track-US
  dataset or FALLMUD), which we have not used.
- Self-training / domain adaptation toward the test images' appearance.
- Temporal smoothing across the 5-frame test sequences plus outlier control.
- A carefully tuned full segment-then-measure pipeline (the benchmark may be a default run).
- Some data-structure signal (sequence grouping, file metadata) we have not examined.

`leader_playbook.md` reconstructs his general method from his other public notebooks, but it
is explicitly NOT his UMUD solution. Verifying which of the above is true is in progress.

### What a 2026-06 web check found (calibration is the leading hypothesis)

Could not read the leader's UMUD notebook or the competition discussion (Kaggle was behind a
browser check), so leader-specific points stay inference. What was verifiable:

- DL-Track-US (the tool behind the 0.679 benchmark) uses a **manual scaling tool**: a human
  enters the scale or clicks a known distance. UMUD requires fully automated prediction over
  309 images, so the benchmark run almost certainly used one fixed/assumed scale, which is
  wrong for images shot at different depths and inflates FL and MT error.
- DL-Track's published accuracy is ~5 mm FL, <1 mm MT, <1.5 deg PA. A rough decomposition of
  the leader's 0.378 (illustrative, per-term errors not observed) lands near those numbers,
  which is only reachable with a correct per-image scale.
- So the leading hypothesis: the leader's edge is **automated per-image pixels-to-mm
  calibration** (tick-mark detection), not anything about pennation. Supports it: DL-Track is
  manual-scale, the numbers line up, FL/MT are our flat constants. Against it: the score
  decomposition is assumed not observed, and his actual code was not seen. Secondary
  hypothesis: external labeled data + direct mm-regression (weighted lower; DL-Track ships no
  public labeled set, it says bring your own).
- Actionable consequence: the whole gap from our ~1.11 (constants) to ~0.68 is the value of
  real per-image FL and MT; PA is a rounding error. We already segment the aponeuroses, so
  muscle thickness in pixels is measurable now and waits only on scale (tightest tolerance,
  3 mm, so high value). Next experiment: a tick-mark detector returning pixels-per-mm per
  image, then verify the "bottom ticks ~1 cm apart" assumption on real images before relying
  on it.

## Constraints and values

- CV discipline, no leakage, claims proportional to evidence.
- No manual labeling of the 309 test images.
- Any external data or pretrained model must be documented.
- Public GitHub repo: no secrets, no personal data.

## Canonical docs to cross-check

- `rundown.md` - plain-language explainer of the task.
- `writeup.md` - competition writeup and result table.
- `strategy_brief.md` - improvement levers (DLTrack repro, calibration, segmentation,
  geometry, temporal, ensemble, outlier) and compute status.
- `leader_playbook.md` - reconstructed method of the leader (not his actual UMUD code).
- `codex_review.md` - second-pass hypothesis map and prioritized next experiments.
- `forward_plan.md` - post-1.09194 interpretation, no-GPU ablations, and larger plan.
- `plan.md` - staged plan and verified data facts.
- `segment_then_measure.py` - the current pipeline; `kaggle_segment_notebook.ipynb` runs it.
- `mask_geometry.py`, `metric.py` - geometry and scoring.
