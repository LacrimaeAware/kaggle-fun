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

## Next

- **exp03 (frontier): measure the curved fascicle path.** Fit a polynomial/spline to the fascicle
  fragment pixels (capturing bend), integrate arc length between the two aponeuroses, and compare to
  the straight identity. This is the only thing that can push FL below the ~0.54 straight-model floor
  toward DL-Track's 0.31. It is the practical form of the path-geometry / "level of bend" idea.
- Wire the chosen FL estimator (MT/sin(PA) first) into `segment_then_measure.py` for a Kaggle test,
  gated to images with trustworthy scale (PNG family first).
- Tune sequence grouping, then test within-clip median smoothing.
