# EXP71 - Local-Benchmark Proxy Plus Safe Scale

Date: 2026-06-13

## Straight Answer

This is the intended "best local benchmark story plus scale" production proxy.

Output:

`results/submission_burn_28_local_benchmark_proxy_plus_missing_scale.csv`

## What It Is

The exact EXP53/55/56 best local benchmark route is still benchmark-only. EXP71 builds the closest available 309-row production-wired proxy from existing submission artifacts:

1. start from current public best anchor:
   `submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv`
2. add robust-triangle geometry delta from:
   `submission_robust_triangle_only.csv`
3. add visibility-weighted FL proxy delta from:
   `submission_burn_03_fl_visibility_weighted.csv`
4. add vertical-MT proxy delta from:
   `submission_host_mt_vertical3_no_subpixel.csv`
5. add EXP70 missing-scale-only 3 cm repair from:
   `submission_burn_26_public_best_missing_scale_only_3cm.csv`

## Why This Exists

Burn #27 was only robust triangle plus safe scale. That was not the full local-benchmark story the user was asking for. EXP71 combines the production-wired pieces that correspond to the local benchmark stack:

- robust upper-boundary geometry;
- support/visibility FL weighting proxy;
- vertical MT proxy;
- safe missing-scale repair.

## Caveat

This is still a proxy, not the exact EXP53/55 term route. The exact local-best route depends on benchmark viewer models and class/term routing that have not been fully production-wired for the 309 test rows.

Also, the individual public probes for these proxy pieces were bad:

- robust triangle #15: `0.60102`;
- robust + visibility FL #16: `0.64511`;
- robust + vertical MT #17: `0.60720`.

So EXP71 is the correct answer to "best local benchmark proxy plus scale," but it is not high-confidence for public improvement.

## Movement

Versus current public best anchor:

- PA changed: `0` rows;
- FL changed: `307` rows, mean absolute movement `6.347 mm`, max `35.898 mm`;
- MT changed: `299` rows, mean absolute movement `0.153 mm`, max `5.434 mm`.

## Recommendation

If the goal is to submit the requested best-local-benchmark proxy with the safe scale fix, submit:

`results/submission_burn_28_local_benchmark_proxy_plus_missing_scale.csv`

If the goal is highest expected public score, this is lower confidence than the current public-best anchor because the component public probes already regressed.

