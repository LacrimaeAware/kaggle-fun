# Benchmark Error Taxonomy

Date: 2026-06-12

This pass classifies the 35-image expert benchmark by likely geometry failure mode using robust
expert consensus. The point is not just "FL is wrong"; the point is to identify which upstream
assumption made the derived number wrong.

Generated local outputs:

- `results/benchmark_error_taxonomy.csv`
- `results/benchmark_error_taxonomy.md`

Those files are diagnostic artifacts; they are not submission files.

## Naive Variant Scores

All scores below use the raw true-scale benchmark CSV as the baseline candidate in the expert viewer.
Lower is better.

| variant | overall | PA | FL | MT | read |
| --- | ---: | ---: | ---: | ---: | --- |
| current raw true-scale | 0.251 | 0.150 | 0.519 | 0.084 | current expert-viewer candidate |
| signed-angle wrong-way prune | 0.252 | 0.150 | 0.522 | 0.084 | remove fragments whose signed angle opposes the area-weighted majority |
| literal raw-slope prune | 0.252 | 0.150 | 0.522 | 0.084 | remove fragments whose raw line slope opposes the area-weighted majority |
| projected-FL p10 | 0.233 | 0.150 | 0.465 | 0.084 | use the 10th percentile of projected FL spans |
| projected-FL p25 | 0.172 | 0.150 | 0.281 | 0.084 | use the 25th percentile of projected FL spans |
| projected-FL p35 | 0.182 | 0.150 | 0.312 | 0.084 | use the 35th percentile of projected FL spans |
| user's robust triangle top boundary | 0.170 | 0.150 | 0.278 | 0.083 | upper boundary as low-left / high-middle / low-right, using robust 5% anchors |
| user's exact triangle top boundary | 0.182 | 0.150 | 0.314 | 0.083 | upper boundary as lowest left quartile / highest middle / lowest right quartile |
| top-boundary chord | 0.231 | 0.150 | 0.460 | 0.084 | replace upper-boundary fit with an outer-quartile chord |
| top-boundary parallel-to-lower | 0.230 | 0.150 | 0.458 | 0.084 | make upper boundary parallel to the lower boundary at image center |

## Interpretation

The simple "remove opposite-direction fragments" idea is not supported as a broad fix on this
35-image reference set. It is still a useful QA tag: `im_03_arch` has visually opposite raw-slope
fragments, but the aggregate pruning variants slightly worsen the local score.

The user's "tails/extremes" concern is real, but the strongest version is not Tukey-style outlier
removal. On cases such as `im_12_arch`, there are no isolated statistical outliers; instead, the
whole projected-FL distribution is broad and the expert value sits below the current median. Using
the 25th percentile of projected spans is the strongest local benchmark signal in this pass
(`0.251 -> 0.172`, FL term `0.519 -> 0.281`). Treat this as a geometry/aggregation clue, not as a
submission-ready rule yet.

The user's earlier triangle idea was not tested before this pass; that was a miss. The actual idea is
piecewise, not a straight chord: low/deep left anchor, high/shallow middle anchor, low/deep right
anchor. On the 35-image expert benchmark it is the strongest local boundary-shape result so far:
robust triangle `0.170` overall, FL `0.278`; exact triangle `0.182` overall, FL `0.314`. On
`im_29_arch`, exact triangle changes median FL from `102.87mm` to `75.81mm` against expert
`75.30mm`.

The boundary-shape hypothesis is better supported locally. The naive upper-boundary chord and
upper-parallel-to-lower variants both improve the raw true-scale reference score, mostly by reducing
FL overestimation in extrapolation-heavy images. This does not make them submission-ready: previous
FL/geometry ideas improved the clean benchmark and failed publicly, so any production change needs
target-label or submission evidence.

The dominant pattern is coupled geometry:

- FL is usually extrapolation-dominated: the visible fragment often supports less than 10 percent of
  the projected full length.
- Many images sit in a shallow-angle regime where a small PA or boundary-angle shift can move FL by
  many millimeters.
- MT is now strong after robust expert-tail cleanup; the residual fight is mostly boundary/fragment
  geometry feeding FL, not raw scale on the expert set.

## Images, Worst First

These are diagnostic classes, not final causes. The deltas are our raw true-scale benchmark
candidate minus the robust expert consensus. Positive FL means we are overprojecting length.

| rank | image | score | dPA | dFL | dMT | tags | local read |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `im_29_arch` | 0.985 | +0.95 | +27.57 | -1.50 | projected FL statistical tail; broad projected FL spread; expert FL sits below our median; curved apo; severe low visible support; PA-sensitive shallow angle; FL error not explained by PA delta alone; sup_chord helps FL | FL is extrapolation-dominated; boundary shape is likely central |
| 2 | `im_12_arch` | 0.658 | -2.41 | +14.38 | +1.12 | broad projected FL spread; expert FL sits below our median; severe low visible support; PA-sensitive shallow angle; PA/FL coupled error; sup_parallel_deep helps FL | not one isolated bad tail; the whole projected distribution is high/broad |
| 3 | `im_05_arch` | 0.466 | -1.19 | +13.66 | -0.19 | broad projected FL spread; expert FL sits below our median; severe low visible support; PA-sensitive shallow angle; FL error not explained by PA delta alone | extrapolation-dominated; median aggregation is likely too high |
| 4 | `im_10_arch` | 0.456 | -4.36 | +7.55 | +0.04 | low visible support; PA-sensitive shallow angle | PA error matters more here |
| 5 | `im_27_arch` | 0.440 | -0.05 | +13.35 | -0.60 | low visible support; PA-sensitive shallow angle; FL error not explained by PA delta alone | FL error is not explained by PA delta alone |
| 6 | `im_21_arch` | 0.415 | +0.26 | +13.97 | +0.12 | severe low visible support; PA-sensitive shallow angle; FL error not explained by PA delta alone | extrapolation-dominated FL overshoot |
| 7 | `im_22_arch` | 0.393 | -2.46 | +8.83 | -0.09 | severe low visible support; PA-sensitive shallow angle | small PA shifts move FL materially |
| 8 | `im_03_arch` | 0.339 | -1.08 | +9.46 | +0.14 | opposite raw-slope fragments; severe low visible support; sup_parallel_deep helps FL | slope-sign cleanup is visually relevant, but broad pruning does not help |
| 9 | `im_31_arch` | 0.318 | +0.25 | +9.78 | -0.30 | multi-gap/band risk; severe low visible support; PA-sensitive shallow angle; sup_parallel_deep helps FL | possible band mixing plus low support |
| 10 | `im_26_arch` | 0.307 | -0.57 | +9.59 | -0.09 | severe low visible support; PA-sensitive shallow angle | extrapolation-dominated FL overshoot |
| 11 | `im_18_arch` | 0.302 | -1.25 | +8.29 | +0.02 | severe low visible support; PA-sensitive shallow angle; sup_parallel_deep helps FL | top-parallel-to-lower helps locally |
| 12 | `im_08_arch` | 0.286 | -2.25 | +5.02 | -0.19 | severe low visible support; PA-sensitive shallow angle | PA-sensitive but less severe FL miss |
| 13 | `im_20_arch` | 0.278 | -0.17 | +7.62 | -0.51 | multi-gap/band risk; severe low visible support; PA-sensitive shallow angle; sup_parallel_deep helps FL | possible band mixing plus low support |
| 14 | `im_25_arch` | 0.273 | +0.40 | +7.36 | +0.42 | severe low visible support; PA-sensitive shallow angle; sup_parallel_deep helps FL | top-parallel-to-lower helps locally |
| 15 | `im_23_arch` | 0.237 | -0.00 | +7.84 | -0.17 | multi-gap/band risk; low visible support; PA-sensitive shallow angle; sup_parallel_deep helps FL | PA can be numerically right while FL remains structurally wrong |
| 16 | `im_34_arch` | 0.209 | +0.27 | +5.67 | -0.33 | low visible support | extrapolation-dominated but moderate |
| 17 | `im_13_arch` | 0.207 | +1.14 | +4.64 | +0.13 | severe low visible support; PA-sensitive shallow angle | low support without a decisive local fix |
| 18 | `im_30_arch` | 0.201 | -1.31 | +4.33 | +0.07 | low visible support; PA-sensitive shallow angle | moderate PA/FL coupling |
| 19 | `im_04_arch` | 0.195 | -0.31 | +6.35 | +0.01 | low visible support; PA-sensitive shallow angle | moderate FL overshoot |
| 20 | `im_01_arch` | 0.193 | +0.62 | +5.25 | -0.12 | sparse fragments; low visible support | segmentation density may be limiting |
| 21 | `im_35_arch` | 0.181 | +0.86 | -3.02 | -0.44 | PA-sensitive shallow angle | no obvious geometry pathology |
| 22 | `im_28_arch` | 0.176 | +0.50 | +5.27 | +0.01 | multi-gap/band risk; low visible support; PA-sensitive shallow angle; sup_parallel_deep helps FL | possible band mixing; local boundary variant helps |
| 23 | `im_06_arch` | 0.143 | -1.56 | -1.43 | -0.14 | multi-gap/band risk; low visible support; PA-sensitive shallow angle | decent overall despite risk tags |
| 24 | `im_09_arch` | 0.142 | -0.54 | +1.48 | -0.64 | multi-gap/band risk; severe low visible support; PA-sensitive shallow angle | MT contributes more than FL here |
| 25 | `im_24_arch` | 0.137 | +0.97 | +2.17 | -0.21 | opposite raw-slope fragments; multi-gap/band risk; low visible support; PA-sensitive shallow angle | wrong-way fragments are visible, but error is small |
| 26 | `im_19_arch` | 0.136 | +1.38 | +1.87 | -0.07 | severe low visible support; PA-sensitive shallow angle | robust MT tail cleanup makes this mostly fine |
| 27 | `im_16_arch` | 0.131 | +0.92 | -2.85 | -0.01 | low visible support; PA-sensitive shallow angle | mild FL undershoot |
| 28 | `im_11_arch` | 0.118 | -0.55 | +2.46 | -0.17 | PA-sensitive shallow angle | no obvious geometry pathology |
| 29 | `im_32_arch` | 0.105 | -0.63 | +2.35 | +0.04 | multi-gap/band risk; low visible support; PA-sensitive shallow angle | risk tags present but small error |
| 30 | `im_17_arch` | 0.088 | +0.94 | +0.32 | -0.25 | severe low visible support; PA-sensitive shallow angle | good overall |
| 31 | `im_14_arch` | 0.080 | -0.34 | +1.73 | -0.12 | sparse fragments; low visible support; PA-sensitive shallow angle | sparse but accurate enough |
| 32 | `im_02_arch` | 0.061 | +0.33 | +1.50 | +0.01 | low visible support; PA-sensitive shallow angle | good overall |
| 33 | `im_33_arch` | 0.059 | -0.30 | -0.15 | -0.34 | low visible support; PA-sensitive shallow angle | good overall |
| 34 | `im_15_arch` | 0.042 | +0.33 | +0.69 | +0.04 | severe low visible support; PA-sensitive shallow angle | good overall despite low support |
| 35 | `im_07_arch` | 0.031 | -0.00 | -0.38 | -0.18 | multi-gap/band risk; sparse fragments; severe low visible support; PA-sensitive shallow angle | risk tags present but very accurate |

## Next Use

Use the expert viewer with the `lines` layer on. The cyan lines are projected fragment spans from
boundary to boundary, while yellow shows the visible fitted fragment. The metadata panel now shows
taxonomy tags and likely failure notes for each expert image.

This pass argues for testing boundary-shape handling before spending more time on broad
opposite-direction pruning. The safer next experiment is a gated, very simple boundary variant,
tested first on target human labels and synthetic geometry before any submission.
