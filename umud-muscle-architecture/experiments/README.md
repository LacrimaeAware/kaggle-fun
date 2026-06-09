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

## Fair-test correction (important)

The exp01 "MT/sin(PA) halves FL (1.188 -> 0.680)" was misleading: it beat a *mean-mismatched* constant
(74.4 on a set whose mean is 61). Against a constant centered at the RIGHT mean (0.682), raw MT/sin(PA)
(0.680) only TIES. The per-image shape helps **only after recentering the mean** (0.528 < 0.682). So the
wired FL is the recentered identity, and the realized gain is ~0.05 on the benchmark, not a halving.

## Next

- Score FL ideas against the 35 experts locally - never submit to "test". The recentered identity is
  wired (UMUD_USE_IDENTITY_FL); its Kaggle transfer is unmeasured and small.
- Real FL beyond the straight floor needs DL-Track's trained fascicle models (re-download the OSF
  architecture-model folders) or a serious tracking pipeline. Quick heuristics (parabola, texture,
  streamlines) all failed.

- **exp03 (frontier): measure the curved fascicle path.** Fit a polynomial/spline to the fascicle
  fragment pixels (capturing bend), integrate arc length between the two aponeuroses, and compare to
  the straight identity. This is the only thing that can push FL below the ~0.54 straight-model floor
  toward DL-Track's 0.31. It is the practical form of the path-geometry / "level of bend" idea.
- Wire the chosen FL estimator (MT/sin(PA) first) into `segment_then_measure.py` for a Kaggle test,
  gated to images with trustworthy scale (PNG family first).
- Tune sequence grouping, then test within-clip median smoothing.
