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

## Next

- **exp03 (frontier): measure the curved fascicle path.** Fit a polynomial/spline to the fascicle
  fragment pixels (capturing bend), integrate arc length between the two aponeuroses, and compare to
  the straight identity. This is the only thing that can push FL below the ~0.54 straight-model floor
  toward DL-Track's 0.31. It is the practical form of the path-geometry / "level of bend" idea.
- Wire the chosen FL estimator (MT/sin(PA) first) into `segment_then_measure.py` for a Kaggle test,
  gated to images with trustworthy scale (PNG family first).
- Tune sequence grouping, then test within-clip median smoothing.
