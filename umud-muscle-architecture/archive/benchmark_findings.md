# UMUD expert benchmark: what the measured ground truth tells us

You downloaded the OSF expert-analysed benchmark (`data/osfstorage-archive/`, gitignored).
The useful payload is `Expert Analysed Benchmark Image Datasets/benchmark_dataset_architecture_v0.1.0.zip`
(extracted to `data/osf_arch_benchmark/`, also gitignored): **35 muscle-architecture images** plus
`Results_benchmark_architecture_v0.1.0.xlsx`. The aponeurosis/fascicle benchmark *model* folders in
the archive came down empty, so we did not get DL-Track's pretrained segmenters from this download.

## What the spreadsheet contains (per image, 35 rows)

- `Scale_pixel_per_cm` - the **true scale** for each image (range 74-126, so scale varies ~1.7x).
- `R1..R7_{MT,FL,PA}` - all **seven experts'** thickness/length/angle measurements (in mm/mm/deg).
- `DLTrack_{MT,FL,PA}` - **DL-Track's own automated output** on these images (with correct scale).
- `SMA_{MT,FL,PA}` - the SMA tool's automated output.
- Methodology (Readme): ImageJ, 3 thickness lines + 3 fascicles + 3 angles per image, averaged;
  muscles GM/soleus/VL; four devices (Hitachi Aloka, Telemed EchoBlaster, Philips HD11, Telemed ArtUs).

## The scored picture (tolerance-normalized error vs the robust expert consensus, on these 35)

`benchmark_validate.py` now leaves the source spreadsheet untouched but drops a single rater only
when that value is at least two competition tolerances from the other-rater mean and the remaining
raters span no more than two tolerances. This currently removes exactly two obvious tails:

- `im_19_arch` MT: drop `R7=80.13`; raw mean `30.03` -> robust mean `20.01`.
- `im_26_arch` FL: drop `R7=33.88`; raw mean `64.11` -> robust mean `70.16`.

| | UMUD-style score | per-target detail |
| --- | ---: | --- |
| Human floor (one careful expert vs the other six) | **0.243** | after removing the two clear single-rater tails |
| DL-Track, correct scale | **0.299** | automated output on the same 35 rows |
| SMA tool | 0.375 | automated output on the same 35 rows |
| Kaggle DL-Track benchmark (reference) | 0.679 | different images/devices/hidden labels - NOT directly comparable to local 0.299 |
| Public leader (reference) | 0.378 | |
| Us public LB, best known | **0.61918** | hidden test, not locally decomposable |
| Us public LB, rejected blend probe | ~0.64 | changed FL only; PA/MT identical to 0.61918 file |
| Us public LB, old run | 1.092 | pre-scale-router history |
| Our constant-prior, scored on these 35 | 0.886 | pa 0.573, fl 1.158, mt 0.928 (within-set baseline) |

Inter-expert spread (the noise): PA SD ~1.6 deg, FL SD ~5.4 mm, MT SD ~1.0 mm. FL is the noisiest
even for experts; MT the most reliable. This matches the reliability literature, now measured.

## What this establishes (and what it does not)

1. **The ceiling is about 0.24 on this cleaned reference.** A careful human expert "scores" 0.243
   against the other experts after the two clear tails are removed. You essentially cannot beat the
   local human floor, because that is how much the human ground truth disagrees with
   itself. The public leader at 0.378 is near the same order but is on a different hidden exam.
   (Caveat: the hidden test is
   different images, same methodology, so these are strong estimates, not exact transfers.)
2. **DL-Track with correct scale is already near human-level (0.299) on this set.** A good
   segment-then-measure pipeline reaches the floor; nothing exotic is required to be competitive.
3. **What the scale claim actually rests on (corrected - the earlier version was wrong).** An
   earlier draft compared local DL-Track scores (DL-Track on these 35, vs these experts) with the
   Kaggle 0.679
   (different images, devices, hidden labels) and blamed the whole gap on scale. That comparison is
   invalid - two different evaluations. What IS supported, all within this same set and truth:
   replacing constants with real per-image measurement is worth a lot here - our constant-prior
   baseline scores **0.886**, DL-Track's real measurement **0.299**, a ~0.59 gap. Scale is a
   *necessary part* of that, and per-image scale varies enough (74-126 px/cm) that, with otherwise
   perfect measurement, a single fixed scale would add ~0.6 to FL/MT error (an upper bound). What is
   NOT supported is a number for what scale ALONE is worth to our Kaggle score: our one real scale
   experiment (PNG MT) moved 0.03, because it fixed one target on 58/309 images with imperfect masks.
   So real measurement (segmentation + geometry + scale together) has large headroom; scale is part
   of it; the magnitude realized for us is unmeasured.
4. **We are at 1.092 because FL and MT are mostly constants.** FL is constant on all 309 rows, MT on
   251 - and our constant prior scores ~0.92 on the benchmark too, so the scoreboard is consistent
   with reality. The road forward is real per-image FL/MT measurement; the benchmark shows a good
   pipeline reaches ~0.33 vs experts on similar muscles, on a different device set.
5. **Our tick detector does not yet generalize.** On these 35 it returned a scale on 17/35 and was
   about 2x off (it assumes 5 mm side ticks; these devices use 1 cm ticks), at low confidence. The
   value is that we now have the true scale to fix it against - but per device (see the caveat).

## Current measured result: where our pipeline is now (with TRUE scale)

The current default full local harness is `experiments/score_weights.py`, which applies the wired
choices that matter now: TTA, min-area/min-angle filtering, inner-edge MT, fragment-extrapolated FL,
and FL recentering to the benchmark mean. Current default result:

| target | our term | read |
| --- | ---: | --- |
| PA | **0.1498** | effectively solved; orientation is strong on the clean set |
| MT | **0.0840** | inner-edge aponeurosis measurement is stronger once the `im_19` MT typo is removed |
| FL | **0.3390** | fragment-only FL is the current default after the blend failed publicly |
| Overall | **0.1909** | strong on this small/clean set, but do not transfer literally to hidden LB |

The rejected blend harness is **0.1507** under the robust reference (PA 0.1498, FL 0.2183, MT
0.0840), but it worsened the
public LB from `0.61918` to about `0.64`. This is now the clearest warning that FL-method changes
validated on the 35-image benchmark can fail on hidden target data. The simpler `score_on_benchmark.py`
is a raw sanity check and does not apply the same FL recentering as the full harness. Use
`experiments/score_weights.py` for the current local diagnostic, not as a submission oracle.

## Historical first measured result: where the pipeline initially failed (with TRUE scale)

Ran our trained U-Nets on the 35 images and converted pixels to mm with the TRUE per-image scale,
to isolate measurement quality from calibration (`score_on_benchmark.py`). Overall **0.634**, but
the per-target split is the real finding:

| target | our term | DL-Track | read |
| --- | ---: | ---: | --- |
| PA | **0.225** | 0.242 | competitive, near human - pennation is effectively solved |
| MT | **0.489** | 0.438 | decent WITH true scale - the aponeurosis-gap geometry works; the lever is scale |
| FL | **1.188** | 0.312 | broken, and no better than a constant (~1.17) even with perfect scale |

So, measured: PA is good, MT works once scale is right, and **fascicle length is the bottleneck -
and it is a geometry problem, not a scale problem** (true scale, FL still 1.19). Our fascicle-length
geometry (straight line fit + aponeurosis intersection on fragmented masks) does not recover real
length. This corrects the earlier scale over-emphasis: scale is the lever for MT, but the big gap is
FL, which needs better fascicle geometry (extrapolation / curves) - the path-geometry intuition.
Caveat: different devices, and the FL constant here (Kaggle mean 74.4) is mismatched to this set's
~61 mm, so the FL *magnitude* does not transfer cleanly; the geometry-is-broken conclusion does (it
holds with true scale, which removes calibration from the picture).

## Why this also answers the "can we even iterate locally" worry

Yes. We now have **35 images with true measured PA/FL/MT and the true scale**, plus DL-Track's and
SMA's outputs for reference. That is a local validation bench: run our geometry/calibration on these
images, score against the experts, and iterate - on CPU, in seconds, no training per loop. The slow
GPU training is a once-in-a-while step; the oracle/visual/measurement iteration runs locally against
real numbers. The loop the project needs does not depend on fast local training.

## Caveat: different devices, so do NOT transfer the tick convention to Kaggle

The 35 benchmark images come from four devices (Hitachi Aloka, Telemed EchoBlaster, Philips HD11,
Telemed ArtUs) that are not the same as the Kaggle test devices (the test `.tif` family shows a
"12L3 MSK" interface, the `.png` family a German "Tiefe X.X cm / L12-4" interface). Therefore:

- The specific finding that THESE devices use 1 cm ticks does **not** transfer to the Kaggle test
  set. The Kaggle `.png` family reads as ~5 mm ticks, cross-checked against its own depth readout,
  so the two image sources have different ruler conventions. Tick spacing must be determined **per
  device**, never assumed globally.
- What DOES transfer is device-independent: the score landscape (human floor ~0.31, a good
  segment-then-measure pipeline reaching ~0.33 vs experts) is about the scoring methodology. So
  "real per-image measurement has large headroom over constants" holds regardless of device; only
  *how* to get the scale (and the masks) is per-device.
- Running our Kaggle-trained segmentation on these benchmark images is a cross-device test, so weak
  masks here would reflect domain shift, not necessarily our test-set quality. Treat this set as a
  check on the measurement LOGIC and the metric, plus a calibration target where we happen to know
  the true scale - not as a stand-in for the Kaggle devices.

Practical consequence: prefer scale signals that do not require guessing the tick spacing - the
depth readout (Kaggle PNGs) or the labelled major ruler span (0 to N cm) - over a fixed "5 mm" or
"10 mm" assumption. The detector being 2x off here is exactly the failure of a fixed tick-mm guess.

## Immediate uses

- Build `benchmark_validate.py`: given our predicted PA/FL/MT for these 35, print the UMUD-style
  score vs experts next to the DL-Track and human-floor references. That is our local scoreboard.
- Fix calibration against truth: the `Scale_pixel_per_cm` column lets us learn the real tick spacing
  per device and stop assuming 5 mm. The detector being 2x off is a solvable, now-measurable bug.
- Study DL-Track's per-image outputs (already in the sheet) to see where the benchmark tool matches
  or misses experts, without running anything.
