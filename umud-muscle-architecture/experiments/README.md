# UMUD experiments log

Mini-experiments in the style of structured-transform-discovery: each has a question, a method, a
result, and a read. They run locally on CPU against the expert-benchmark scoreboard
(`benchmark_validate.py`), so results are measured versus expert ground truth, not the leaderboard.

Caveat throughout: the 35 benchmark images are different devices than the Kaggle test set, so FL/MT
*magnitudes* do not transfer cleanly. The *rankings* and the qualitative conclusions do, because
they are about the measurement logic.

## exp01 - fascicle-length estimators (`exp01_fl_estimators.py`)

Question: our straight-line fragment FL is broken (term 1.19 even with true scale). PA and MT are
good. Can we derive FL from them via the identity FL = MT / sin(PA)?

Result:
- Identity vs the experts' OWN FL: MAE 6.5 mm (0.54 tol), corr 0.70, expert FL ~1.06x the straight
  value. The straight identity is lossy, and the gap is fascicle bend.
- FL estimator on our pipeline (PA/MT held at our values, only FL swapped), FL-term:

  | estimator | FL-term | overall |
  | --- | ---: | ---: |
  | fragment line (current) | 1.188 | 0.634 |
  | **MT / sin(PA)** | **0.680** | **0.465** |
  | constant prior | 1.172 | 0.629 |
  | 0.6 frag + 0.4 identity | 0.874 | 0.530 |

  DL-Track FL-term is 0.312 for reference.

Read: MT/sin(PA) is the obvious standard win and it works - it nearly halves the FL error and is
ready to wire in where scale is trustworthy. The remaining gap to DL-Track is the **bend**; a
straight model cannot close it. Curvature is the frontier.

## exp02 - FL input sensitivity + test sequence detection (`exp02_fl_sensitivity_and_sequences.py`)

(a) Which input limits FL = MT/sin(PA)? Swap in the experts' MT or PA one at a time (FL-term):

  | inputs | FL-term |
  | --- | ---: |
  | our MT, our PA | 0.680 |
  | expert MT, our PA | 0.637 (our MT costs ~0.04) |
  | our MT, expert PA | 0.409 (our PA costs ~0.27) |
  | expert MT, expert PA (floor) | 0.540 (the bend) |

Read: **PA precision is the amplified limiter**, because 1/sin(PA) magnifies small angle errors at
low PA. Curiously our MT is biased slightly high, which accidentally compensates for the bend
(our-MT + expert-PA = 0.409 beats the perfect-input floor 0.540) - a fragile coincidence, not a
method to rely on.

(b) Are the test images time-series clips? Over the 309 test images, **112/308 consecutive pairs are
>0.9 similar** (clearly the same moving muscle). Clips exist; the simple 0.6 threshold splits them
roughly (median run 2, max 10 - needs tuning), but the sequential structure is real and matches the
"same muscle moving" observation.

Read: temporal smoothing of our *own* predictions within a clip is a legitimate, non-leakage lever
(we use the test set's structure, not its labels) to reduce per-frame noise. Worth a submission once
a base FL/MT method is set.

## exp03 - curved FL, texture PA, no-scale FL (`exp03_curved_fl_and_texture_pa.py`)

Two orthogonal ideas, both tested, both FAILED (logged because negative results are the point):

(A) Curved/arc-length FL: fit a parabola to all fascicle pixels, take arc length not chord.
    FL-term **0.688 vs the straight 0.680 - slightly worse**. A parabola over all pixels captures
    the arrangement of many fascicles, not one fascicle's bend. Real bend needs per-fascicle tracking.

(B) Texture-orientation PA (the dark-space / complement idea): structure-tensor orientation of the
    belly, independent of segmentation. **PA MAE 8.2 deg vs the segmentation's 1.35**; blending makes
    PA worse. The raw belly structure tensor is too noisy (speckle, aponeurosis edges). The instinct
    (fascicles and dark gaps share orientation) is right; a raw structure tensor is the wrong tool. A
    Gabor / coherence-weighted version might do better, not pursued.

(C) Kaggle-relevant: does FL = MT/sin(PA) help even WITHOUT scale? Constant thickness (18.628 mm)
    over our PA gives FL-term **0.799 vs constant-FL 1.172** - it captures per-image FL variation via
    PA. With true scale, 0.680. Caveat: benchmark FL mean (~61) differs from Kaggle's (74.4), so the
    constant looks worse here than on Kaggle; the real Kaggle gain is uncertain, a submission decides.

Standing best (benchmark): PA seg 0.225, FL = MT/sin(PA) 0.680, overall 0.465. Neither new idea beat
it. Bend stays uncracked without real fascicle tracking.

## exp04 - real curved fascicle tracking (`exp04_fascicle_tracking.py`)

Question (ranked idea #5, done properly): trace streamlines through a per-pixel orientation field
(structure tensor on the image) from the deep aponeurosis to the superficial one; arc length = FL,
capturing bend.

Result: **FAILED.** FL-term 2.27 raw / 1.78 recentered - far worse than even a good constant (0.682).
The ultrasound orientation field is too noisy (speckle), so streamlines wander and lengths are junk.

Read: tracking that beats the straight line is genuinely hard - it is what DL-Track does with trained
fascicle models and tuned tracking (0.312), not a structure-tensor streamline hack. The recentered
straight identity (0.528) stands as our best FL. Beating it needs DL-Track's models (the architecture
model folders in the OSF download were empty) or a serious tracking build, not a quick experiment.

## exp05 - sharper PA (PCA + length weighting), banded FL (`exp05_better_pa_and_banded_fl.py`)

Iterating on exp03/04 instead of bailing.

(A) PA: `polyfit` minimizes the vertical residual, biased for steep fascicles. Total-least-squares
    (PCA) orientation, weighted by fragment size, gives **PA MAE 1.10 deg / term 0.184** (from
    polyfit's 1.35 / 0.225) - and beats DL-Track's PA (0.242). WIRED into `segment_then_measure.py`.
(B) Banded curved FL (per-depth angle from the clean mask, integrated): 0.658 - still worse than the
    straight line (0.514). Bend stays uncracked, but the sharper PA dropped the straight FL to 0.514.

**Combined end-state**, our full pipeline on the 35 experts WITH good scale (wpca PA + recentered
identity FL + true MT): **overall 0.383** (pa 0.184, fl 0.476, mt 0.489) - vs human 0.307, DL-Track
0.331, SMA 0.409, and our start-of-night 0.634. So the MEASUREMENT is now DL-Track-competitive given
scale. HONEST caveat: true scale, benchmark devices; our real Kaggle score is ~0.9-1.09 because the
251 TIFFs have NO scale and fall back to constants. **The real Kaggle bottleneck is now per-image
scale on the TIFFs (ranked #3)** - that is what unlocks the now-good measurement.

## exp06 - visual diagnosis of FL errors (`../benchmark_overlay.py`)

Drew our predicted geometry + measured PA/FL/MT vs the experts on all 35 images
(`results/benchmark_overlay/`). Findings:
- Good cases (im_01): our PA/FL match the experts closely (28.1/35 vs 27.5/36).
- Misses (im_06, im_20): PA runs slightly LOW, MT slightly HIGH; since FL=MT/sin(PA) both inflate FL.
- BUT across all 35 the bias is small (PA -0.5 deg, MT +0.95 mm) and CORRECTING it does NOT help FL
  (0.476 -> 0.488). So FL is **per-image scatter-limited, not bias-limited** - the noise comes from
  the fragmentary fascicle masks (6 tiny segments per image, never a full fascicle, per the user's
  observation). FL will not yield to a formula tweak; it needs fuller/cleaner fascicle masks (better
  segmentation) or real tracking. The MT ~+1 mm overestimate is a real aponeurosis-line-placement
  issue (the bands are thick) worth fixing - it helps the MT term even if not FL.

Human-in-the-loop: the overlays exist to eyeball where the model misses fascicles or mis-places the
aponeuroses; that feedback guides the next segmentation/geometry fix.

## exp07 - testing the user's two visual observations (`exp07_depth_angle_and_filter.py`)

The user eyeballed the overlays and offered two domain observations. Tested both vs the 35 experts
instead of guessing:

| PA angle source | PA-term | FL-term |
| --- | ---: | ---: |
| all fragments (current) | 0.184 | 0.476 |
| near-superficial (the "bend toward upper apo") | 0.250 | **0.458** |
| near-deep | 0.200 | 0.535 |
| depth-weighted toward superficial | 0.193 | 0.481 |
| **stricter horizontal filter, min 6 deg (the "mid-line is not a fascicle")** | **0.182** | **0.473** |

Read: **both observations were directionally correct, both tiny - exactly as the user predicted
("nitpicky", "not a crazy result").**
- The bend-toward-superficial does shorten FL (0.476 -> 0.458) but worsens PA, because the experts
  define PA at the *deep* aponeurosis, not the top. So the steeper top angle is the wrong thing to
  report as PA, though it hints the straight FL is a touch long.
- The mid-line/horizontal-artifact filter (min 6 deg) is a clean free win on BOTH terms
  (PA 0.184 -> 0.182, FL 0.476 -> 0.473). The user's instinct that near-aponeurosis-parallel
  fragments are not fascicles is right; they were mildly poisoning the angle.

Conclusion: this confirms exp06 from the other side - the geometry is at its ceiling. Sub-0.02 left in
angle-picking. The FL gap to DL-Track (0.47 vs 0.31) is the **mask quality** the user flagged (sparse,
under-segments fascicles), which is a *segmentation* problem, not a geometry one. The overlay
(`benchmark_overlay.py`) now renders the predicted fascicle mask as a clear dilated red blend so a
human can actually see the under-segmentation.

## exp08 - test-time augmentation denoises the masks (`exp08_tta_masks.py`)  [WIRED]

Following exp06/07's conclusion that the bottleneck is mask quality (not geometry), tested the
cheapest mask fix that needs no retraining: TTA - average the sigmoid over the image, its horizontal
mirror, and a second scale (448), then threshold. Also swept the threshold. Scored vs the 35 experts,
true scale, recentered-identity FL (the wired pipeline):

| config | overall | PA | FL | MT |
| --- | ---: | ---: | ---: | ---: |
| baseline (single pass, 0.5) | 0.383 | 0.185 | 0.476 | 0.490 |
| **TTA, threshold 0.5** | **0.370** | **0.171** | **0.449** | 0.490 |
| TTA, threshold 0.4 | 0.377 | 0.180 | 0.460 | 0.490 |
| TTA, threshold 0.35 | 0.397 | 0.196 | 0.507 | 0.490 |
| TTA, threshold 0.30 | 0.386 | 0.184 | 0.484 | 0.490 |

Read: TTA at the standard 0.5 threshold is a **real, measured win** - the first FL improvement that
is not a mean-fit (0.476 -> 0.449) and a solid PA gain (0.185 -> 0.171, now well under DL-Track's
0.242). Crucially, *lowering* the threshold to add pixels makes it WORSE - so TTA is not filling
gaps, it is **denoising** (the mirror/scale ensemble cancels spurious fragments, leaving a cleaner
fascicle set to fit). Cost: 3 forward passes/image (~a few minutes for the full 309), no retraining.
**WIRED into `segment_then_measure.py` as `UMUD_TTA` (default on);** PA term transfers exactly
(0.171) in `score_on_benchmark.py`, confirming the wiring.

New combined end-state on the 35 experts (true scale): **overall 0.370** (pa 0.171, fl 0.449,
mt 0.490) - vs human 0.307, DL-Track 0.331, our prior end-state 0.383. The measurement is now
*between* DL-Track and the human floor on PA, and closing on DL-Track overall. The Kaggle bottleneck
is still the 251 unscaled TIFFs.

## exp09 - post-processing sweep on the TTA masks (`exp09_postproc_sweep.py`)  [partly WIRED]

Grounding work (a 4-agent investigation) confirmed the segmenter facts: apo + fascicle are already
TWO separate models; loss is 0.5 Dice + 0.5 BCE with NO class weighting; the local fascicle TRAINING
masks (2761 files) are dash-style - mean 13.7 fragments/image, median 59 px, each ~6% of image width,
within-image orientation std 2.7 deg. Apo masks are clean (exactly 2 bands in 48/50). So bend cannot
be supervised into the fascicle model (no curve in the labels), and the fascicle model under-draws
because thin sparse positives under unweighted BCE bias toward background.

Swept the only no-retrain levers on the already-TTA'd probability maps: threshold {.45,.50,.55} x
fascicle min-area {20,40,60} x min-orientation filter {3,6,9} deg. Best vs current wired (thr .5,
area 20, ang 2 = 0.370):

| thr | area | ang | overall | PA | FL | MT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 40 | 6 | **0.368** | **0.164** | 0.449 | 0.490 |
| 0.50 | 20 | 6 | 0.368 | 0.168 | 0.447 | 0.490 |
| 0.50 | 20 | 2 (current) | 0.370 | 0.171 | 0.449 | 0.490 |
| 0.55 | 60 | 6 | 0.390 | 0.180 | 0.500 | 0.490 |

Read: the whole gain is PA (0.171 -> 0.164), from the min-6-deg filter rejecting apo-parallel
fragments (the user's mid-line observation). **FL does not move - it sits at 0.447-0.451 across every
combo.** The local post-processing lever is now spent; FL is pinned at the straight-mask floor, as
the scatter analysis (exp06) predicted. WIRED the free PA win: `FASC_MIN_AREA=40`, `FASC_MIN_ANG=6`
(confirmed PA 0.164 in `score_on_benchmark.py`). New end-state ~0.368 (pa .164, fl .449, mt .490).

The only untested FL lever left is the user's idea-2: bias the FASCICLE model toward recall so it
draws more of each dash, giving the geometry more/fuller fragments. Wired as `UMUD_FASC_POS_WEIGHT`
(>0 adds a pos_weight to the fascicle BCE; apo untouched). It is a Kaggle-GPU retrain (local AMD has
no CUDA). Expected: better fascicle recall and PA stability; FL payoff uncertain because FL is
scatter-limited (exp06), and exp08 warned that merely adding pixels at a low threshold hurt FL.

## exp10 - image preprocessing (contrast / brightness / bleed) (`exp10_preprocessing.py`)  [flag WIRED]

User question: do contrast (CLAHE), brightness, or "brightness bleed" (blur) help the fascicle mask
catch more? Tested at INFERENCE on the current model over the 35 experts (apo input kept raw, only
fascicle input preprocessed):

| preprocess | overall | PA | FL | fragments/img |
| --- | ---: | ---: | ---: | ---: |
| none (current) | 0.368 | 0.164 | 0.449 | 16.9 |
| CLAHE | 0.398 | 0.197 | 0.507 | **19.3** |
| CLAHE strong | 0.422 | 0.229 | 0.547 | 20.6 |
| histogram equalize | 0.394 | 0.260 | **0.432** | 20.1 |
| gamma brighten | 0.391 | 0.205 | 0.479 | 14.9 |
| brightness bleed (Gaussian blur) | 0.386 | 0.189 | 0.480 | 12.5 |

Read: the user's intuition is half-right. **Contrast DOES surface more fascicle fragments** (CLAHE/
equalize: ~20 vs 17 on raw). But every variant HURT the score at inference, because the model was
trained on RAW images and contrast-boosting creates a train/test MISMATCH - the extra fragments are
noisier and PA degrades. One signal: equalize *improved* FL (0.449 -> 0.432) while wrecking PA, so
contrast genuinely changes the FL signal; it is just unusable as an inference-only bolt-on. Blur
("bleed") merges fragments into fewer blobs (12.5) and hurts.

Conclusion: preprocessing must go into TRAINING so train matches inference. Wired CLAHE as
`UMUD_CLAHE` inside read_rgb (applies to both train and inference; default OFF because turning it on
requires retraining BOTH models). It is the second knob to test in the one Kaggle GPU run, alongside
`UMUD_FASC_POS_WEIGHT` (recall bias). The FL improvement under equalize is the reason to try it.

## exp11 - per-image scale recovery for the test TIFFs (`exp11_calibration_coverage.py`, `../scale_ticks.py`)  [WIRED]

Prompted by the host's forum clarifications (pixels always square; test images carry tick marks;
bottom ticks = 1 cm). The 251 "unscaled" TIFFs ARE scalable. Findings:

- The test set is FOUR device families by UI, not one - the 800x1200 size hides TWO devices.
- Each family's scale was validated by reading its actual ruler (no test labels exist to score it):
  PNG (58) = 150 px/cm (5 mm minor left ruler; the feared 2x bug does NOT exist); 644x1088 (50) =
  126 px/cm (left ruler 0-50 mm); Telemed-800x1200 "De 50 mm" (49) = 134 px/cm (bottom ticks AND left
  ruler agree); clean cropped (~10) = bottom ticks (IMG_00040 -> 78 px/cm, ticks land on real marks).
- The German Siemens 800x1200 ("12L3 Quadriceps") "bracket" was a red herring: its real scale is a
  faint RIGHT-edge 5 mm depth ruler (`recover_scale_right_ruler`, thr 90, x~1150). The 5 mm interval
  is pinned three ways (MT physiology: 1cm->49mm absurd vs 0.5cm->24.7mm; the "4.5 cm" depth label:
  span ~141 vs detected 136; 4.5cm/9ticks=5mm). Detected on 87 of ~132; QA overlay confirms ticks
  land on real marks.
- Detection ACCURACY where a ruler is found is good (benchmark spacing MAE 1.7 px/cm).

Built `scale_ticks.py` (`recover_scale` bottom ticks, `recover_scale_left_ruler`,
`recover_scale_right_ruler`, and `recover_for_image` the per-family router). WIRED into
`segment_then_measure.calibrate_image` (`UMUD_SCALE_ROUTER`, default on; `CALIBRATION_MIN_CONF` 0.3).
Current router check: **295/309 scaled (95% coverage)**, with method counts:
right_ruler_5mm 87, bottom_ticks 59, png_left_ruler 58, left_ruler_1cm 50,
family_b_signature 41, none 14. The `family_b_signature` path is instrument recognition plus an
assigned validated family scale, not per-image ruler reading; keep that method visible in outputs.
Harness: `experiments/{scale_coverage,scale_qa,siemens_ruler,check_submission}.py`
(overlays `results/calibration_qa/`). Full map: `competition_reference.md` 3a/3b.

Latest handoff/context public score after the scale work: **0.619** (#7 at the time), now better than
the provided DL-Track benchmark (0.679). That leaderboard value is not locally decomposable because
the 309 target labels are hidden.

## exp12 - temporal smoothing within sequence clips (`temporal_check.py`)  [built, OFF by default]

The test set has clips (exp02: ~112/308 consecutive frame-pairs >0.9 similar = same moving muscle).
Within a clip the true PA/FL/MT are ~constant, so median-smoothing our per-frame predictions is
variance reduction using the test set's STRUCTURE, not labels (non-leakage; DL-Track does the same
with Hampel/Savitzky-Golay). Built `fingerprint` + `temporal_smooth` in segment_then_measure
(`UMUD_TEMPORAL_SMOOTH`, default off), wired into main() and local_infer.

Clip detection is stable and safe (thr 0.88-0.95 all give ~28 clips covering ~140 images, longest
clip only 5 - no mega-clips that would blur different muscles). Smoothing moves predictions a small,
safe amount vs the tolerances: PA 0.16/6, FL 1.28/12, MT 0.21/3 mm on ~140 images. So it is a modest
positive-EV lever, NOT a big mover. Cannot score it locally (the 35-expert benchmark is not
sequences). Kept OFF for the first scale submission to isolate the scale value; flip on for the next.

## exp13 - real train-vs-target domain gap (`domain_gap_real.py`)  [diagnostic]

Question: was the remaining gap caused by an unseen-device segmentation domain shift, and should the
next expensive run be augmentation/CLAHE/self-training?

Result on real frames (400 sampled train, 309 target):

- Global appearance gap is modest: mean |standardized-mean-diff| = **0.44**.
- No global feature crosses the |SMD| > 1 "real domain axis" line.
- k=5 clusters are mixed train/test; there is no clean test-only cluster.
- Normalization does not close the gap: raw 0.44, CLAHE 0.44, z-score 0.60, minmax 0.57.
- The 1069x853 cropped family is bright/far from train, but it is only 11 images.

Read: the synthetic "unseen instrument isolates into its own class" result did **not** reproduce on
real data. The augmentation/domain-adaptation retrain is therefore demoted. Do not spend a GPU slot
on that premise unless a later correctness check points back at segmentation.

## exp14 - target mask-presence quality (`seg_quality_test.py`)  [diagnostic]

Question: even if global appearance is not far OOD, do the real trained masks collapse on target
families?

Result from the CPU probe (sampled families, no labels needed):

| group | n | apo_bands | frags | geom_ok | pa_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN-fasc | 40 | 3.42 | 14.55 | 100% | 19.42 |
| BENCHMARK | 35 | 2.37 | 14.14 | 100% | 18.02 |
| TEST-1200x800 | 40 | 2.50 | 18.35 | 100% | 14.34 |
| TEST-1088x644 | 40 | 2.67 | 23.25 | 100% | 11.75 |
| TEST-1069x853 | 11 | 3.27 | 20.55 | 100% | 17.17 |

Smaller cropped singleton/two-image families also returned 100% geometry success in this probe, but
their n is too small to over-interpret.

Read: segmentation **presence** does not collapse on target. Target fragments are comparable or
higher than controls. Caveat: this is not correctness; extra fragments may be coherent signal or
texture/noise. That becomes an FL/orientation coherence question, not a generic domain-gap question.

## exp15 - term2/FL geometry (`term2_geometry.py`)  [synthetic diagnostic]

Question: why is FL the largest term, and is there a trivial combiner bug?

Result: FL is a geometric amplifier:

```text
FL = MT / sin(PA)
```

A one-degree PA error becomes ~9.9% FL error at 10 deg, 6.5% at 15 deg, 4.8% at 20 deg, and 3.7% at
25 deg. A naive **mean** of per-fragment lengths is Jensen-biased high in simulation (+7.7% clean
fragments). A median or aggregate-orientation combiner avoids the large bias; MAD-gated aggregate
orientation is most robust under 30% texture outliers (4.62% abs error vs 8.85% for the mean combiner).

Read: current production code already uses **median** per-fragment FL, not a mean, so this is not a
confirmed one-line live bug. The useful next experiment is to test MAD-gated aggregate orientation
and orientation coherence on the 35-expert harness and target families. This directly answers whether
the higher target fragment count is signal or texture pickup.

## exp16 - live FL combiner test (`exp16_fl_combiner.py`)  [WIRED, PUBLIC TRANSFER FAILED]

Question: does the synthetic term2 idea help the actual 35-expert benchmark?

Result, with the same TTA/inner-edge/current weights and true scale:

| variant | overall | PA | FL | MT |
| --- | ---: | ---: | ---: | ---: |
| current fragment median | 0.2274 | 0.1498 | 0.3528 | 0.1795 |
| identity, weighted-median PA | 0.2701 | 0.1498 | 0.4811 | 0.1795 |
| identity, MAD-gated PA | 0.2772 | 0.1678 | 0.4843 | 0.1795 |
| 25% fragment / 75% gated identity | 0.2228 | 0.1498 | 0.3390 | 0.1795 |
| **50% fragment / 50% gated identity** | **0.1873** | **0.1498** | **0.2326** | **0.1795** |
| 75% fragment / 25% gated identity | 0.1962 | 0.1498 | 0.2594 | 0.1795 |

No-recenter sanity also improves: current fragment median 0.2877 vs 50/50 blend 0.1998.

Read after public test: this was a real local score improvement and a bad leaderboard proxy. The
50/50 blend worsened public LB from `0.61918` to about `0.64` while PA and MT stayed identical to
the `0.61918` file. The blend remains in `segment_then_measure.py`, but the default is now
`UMUD_FL_IDENTITY_BLEND=0` to recover fragment-only FL.

Caveat turned into evidence: this 35-image clean benchmark, even with the no-recenter sanity check,
is not a reliable submission oracle for FL-method changes. Do not submit future FL changes because
they improve this table alone.

## exp17 - target blend/centering sensitivity (`exp17_blend_sensitivity.py`)  [diagnostic]

Question: after wiring the 50/50 FL blend, does the target-set CSV look stable under nearby blend and
centering choices?

Result from cached local inference, no U-Net rerun:

| variant | mean | std | min | p50 | p95 | max | at_min | mean abs delta vs current |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **blend 0.00, center 74.424** | **74.45** | **21.94** | **30.00** | **74.45** | **113.69** | **130.91** | **2** | **0.00** |
| blend 0.25, center 74.424 | 74.45 | 22.71 | 30.00 | 74.50 | 113.42 | 134.39 | 2 | 1.77 |
| blend 0.50, center 74.424 | 74.45 | 23.66 | 30.00 | 73.74 | 115.06 | 138.30 | 2 | 3.53 |
| blend 0.75, center 74.424 | 74.46 | 24.78 | 30.00 | 72.28 | 118.95 | 142.21 | 2 | 5.26 |
| blend 1.00, center 74.424 | 74.46 | 26.03 | 30.00 | 70.14 | 121.36 | 146.08 | 3 | 6.98 |
| blend 0.50, no center | 92.44 | 29.47 | 30.00 | 91.59 | 142.91 | 171.77 | 1 | 17.99 |
| blend 0.50, center 70.000 | 70.04 | 22.24 | 30.00 | 69.36 | 108.22 | 130.07 | 3 | 4.79 |
| blend 0.50, center 78.000 | 78.02 | 24.82 | 30.00 | 77.28 | 120.58 | 144.94 | 2 | 4.91 |
| blend 0.50, center 82.000 | 82.01 | 26.11 | 30.00 | 81.25 | 126.77 | 152.37 | 2 | 7.88 |

Read after public test: blend 0.50 was not a distribution outlier, but that still did not predict
leaderboard transfer. Nearby blends move rows modestly; centering dominates much more than blend
choice: without FL centering the target mean jumps to 92.4 mm. The current production centering is
therefore an important assumption, not decoration. Variant CSVs and `summary.csv` are written under
`results/blend_sensitivity/` (gitignored result artifacts).

## exp18 - orientation coherence by target family (`exp18_orientation_coherence.py`)  [diagnostic]

Question: target images produce more fascicle fragments than controls; are those fragments coherent
signal or texture/noise?

Result: all target groups have 100% geometry success and high orientation coherence. Selected rows:

| group/family | n | frag_mean | coherence mean | coherence p10 | coherence min | PA mean | identity/fragment FL ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BENCHMARK mixed | 35 | ~16.7 | ~0.998 | >=0.994 | 0.992 | ~17-19 | ~0.82-0.97 |
| TEST bottom_ticks | 59 | 17.36 | 0.997 | 0.995 | 0.982 | 13.96 | 1.023 |
| TEST family_b_signature | 41 | 22.76 | 0.997 | 0.996 | 0.995 | 11.79 | 1.093 |
| TEST left_ruler_1cm | 50 | 25.94 | 0.998 | 0.997 | 0.995 | 11.70 | 1.071 |
| TEST none | 14 | 19.93 | 0.992 | 0.978 | 0.976 | 16.86 | 0.856 |
| TEST png_left_ruler | 58 | 15.14 | 0.996 | 0.991 | 0.989 | 19.11 | 0.941 |
| TEST right_ruler_5mm | 87 | 22.33 | 0.997 | 0.995 | 0.990 | 14.75 | 0.971 |

Read after public test: the higher target fragment counts look mostly coherent, not random texture
scatter, but coherence did not guarantee the blend would transfer. The `none` family has the weakest
coherence (still high: p10 0.978), so it is the obvious visual-audit candidate. Use this diagnostic
for orientation/texture audit, not as approval for the rejected blend.

## exp19 - target scale cross-check (`exp19_scale_crosscheck.py`)  [diagnostic]

Question: after the failed FL blend, can we at least bound whether the current 295/309 scale router
contains a broad hidden scale mistake, without target labels and without using global output means?

Method: run every scale cue independently on all 309 target images, then compare only rows where two
strict cues fire on the same image. Writes `results/scale_crosscheck.csv` and
`results/scale_crosscheck_pairs.csv` (gitignored result artifacts).

Result:

| pair | n | median abs % disagreement | p95 | max | rows >2% | rows >5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottom_ticks vs family_b_signature | 49 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| bottom_ticks vs left_ruler_1cm | 29 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| left_ruler_1cm vs png_left_ruler | 36 | 0.921 | 2.532 | 2.532 | 4 | 0 |

Strict cue multiplicity: 14 images with zero strict cues, 181 with one strict cue, and 114 with two
strict cues. Router counts are unchanged: right_ruler_5mm 87, bottom_ticks 59, png_left_ruler 58,
left_ruler_1cm 50, family_b_signature 41, none 14.

Read: this is a strong label-free sanity check for the multi-cue scale families. It specifically
validates the risky `family_b_signature` assignment against bottom ticks on 49 images, with exact
agreement. It does **not** prove the single-cue `right_ruler_5mm` family is correct, and it leaves
the 14 `none` rows unresolved. Of those 14, five have weak/non-strict cues in the audit output and
may be recoverable; nine have no current cue.

## Fair-test correction (important)

The exp01 "MT/sin(PA) halves FL (1.188 -> 0.680)" was misleading: it beat a *mean-mismatched* constant
(74.4 on a set whose mean is 61). Against a constant centered at the RIGHT mean (0.682), raw MT/sin(PA)
(0.680) only TIES. The per-image shape helps **only after recentering the mean** (0.528 < 0.682).
Current production FL prefers fragment-extrapolated FL when a scaled fragment exists, falls back to
identity only when needed, and then recenters. The rejected blend reinforces the enduring lesson:
recentering is a major part of the clean local score and therefore cannot be treated as hidden-test
evidence.

## Next

- Keep `results/submission_local.csv` restored to the downloaded `0.61918` baseline. Compare any new
  candidate row-by-row against it before submitting.
- Scale cross-check is now partly done: multi-cue families agree well; focus remaining scale work on
  the single-cue `right_ruler_5mm` family and the 14 `none` rows.
- The blend is rejected as a submission default despite its local win.
- Remaining no-submission work: two-cue scale-error bound, a visual audit of the 14 `none` scale rows,
  and a classical orientation-correctness audit.
- Generate a temporal-smoothing variant only after it can be compared cleanly against the restored
  baseline, not stacked with another experimental change.
- Keep augmentation/self-training demoted unless a correctness audit, not a presence audit, points
  back at segmentation.
