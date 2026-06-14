# EXP56 - Term Route Ablation

Date: 2026-06-13

Purpose: ablate the corrected EXP55 term-level class route so we know which pieces of the
`0.131264` benchmark result are load-bearing.

Harness: `experiments/exp56_term_route_ablation.py`

Ignored output bundle: `results/exp56_term_route_ablation/`

## Important Framing

This is still a 35-image expert-benchmark route. It does **not** use `DLTrack`, `SMA`, or
`our_pipeline_true_scale` as route sources. It starts from `median_weight_blend_best` and routes
terms only among our own research candidates.

It is not directly submit-ready until the reducers and class gates are implemented for all 309 test
images.

## Results

| variant | overall | PA | FL | MT | gain vs base | read |
|---|---:|---:|---:|---:|---:|---|
| base `median_weight_blend_best` | 0.143337 | 0.148942 | 0.210820 | 0.070248 | 0.000000 | best single local benchmark model |
| full term route / prefix 11 | **0.131264** | **0.137020** | 0.192133 | **0.064640** | **0.012072** | benchmark-best allowed route |
| prefix 5 / first 5 large steps | 0.132442 | 0.138622 | 0.192133 | 0.066570 | 0.010895 | almost all route gain without tiny late gates |
| PA + FL route only | 0.133133 | 0.137020 | 0.192133 | 0.070248 | 0.010203 | most of the win; leaves MT untouched |
| FL route only | 0.137108 | 0.148942 | 0.192133 | 0.070248 | 0.006229 | largest single-term route family |
| PA route only | 0.139363 | 0.137020 | 0.210820 | 0.070248 | 0.003974 | real but smaller |
| MT route only | 0.141467 | 0.148942 | 0.210820 | 0.064640 | 0.001869 | small and more public-risky |

## Route Step Takeaways

The route does not depend on one tiny suspicious late rule. The first five steps already capture
about 90% of the full benchmark gain. The late steps 6-11 improve `0.132442 -> 0.131264`, useful but
not the core story.

Largest leave-one-out damage:

1. Step 2 (`not_low_support_any`, FL from `weighted_story_fl`) matters most for FL.
2. Step 1 (`low_support_any`, PA from `story_stack`) matters most for PA.
3. Step 3/5 (`strong_lower_curve_any_direction` and `lower_side_angle_changes_strongly`, FL route)
   are the rest of the FL route.
4. MT route helps locally, but the public history for vertical-style MT was bad, so treat MT splits
   as lower confidence.

## Next

Production translation should be split:

1. Core benchmark model: `median_weight_blend_best` as the conceptual anchor.
2. High-confidence non-core: FL support route.
3. Medium-confidence non-core: PA route.
4. Low-confidence non-core: MT route.

For immediate leaderboard burns, only already-wired production proxies can be submitted. See EXP57.
