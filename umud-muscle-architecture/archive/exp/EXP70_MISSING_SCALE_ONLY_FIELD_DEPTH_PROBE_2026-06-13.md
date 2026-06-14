# EXP70 - Missing-Scale-Only Field-Depth Probe

Date: 2026-06-13

## Straight Answer

The actual submission-safe scale fix is not the broad field-height override. It is:

> use field-depth scale only when the existing scale partition has no usable px/cm.

Existing tick/ruler/text-confirmed scales should not be overwritten by a field-rectangle estimate.

## What Changed

`experiments/exp61_oracle_scale_patch.py` now uses the user-proposed scan-outward field heuristic:

- start from a textured point inside the scan;
- move outward;
- detect sustained constant-color runs as UI/background;
- return the visible scan field between those edges.

This improves the field heuristic itself. Example outcomes from the diagnostic pass:

- `IMG_00198.tif`: field height `478 px`, matching the 3 cm family used for the safe correction.
- `IMG_00031.tif`: no longer appears in the broad scale proposal.
- `IMG_00040.tif`: remains full field height `513 px`, which is appropriate for a cropped/no-overlay frame.

However, the improved heuristic still proposes broad overrides for many rows with existing scales, such as `IMG_00066.tif` (`134.2 -> 151.0 px/cm`). Burn #22 already proved that this broad behavior is leaderboard-dangerous.

## Outputs

`experiments/exp70_missing_scale_only_field_depth_probe.py` writes:

- `results/submission_burn_26_public_best_missing_scale_only_3cm.csv`
- `results/submission_burn_27_robust_triangle_missing_scale_only_3cm.csv`
- `results/submission_burn_26_27_missing_scale_only_3cm_summary.csv`

Burn #26 equals the older conservative burn #19 byte-for-byte.

Burn #27 equals the older conservative burn #21 byte-for-byte.

## Recommendation

Submit this if testing scale right now:

`results/submission_burn_26_public_best_missing_scale_only_3cm.csv`

Do not submit #24 as the preferred scale fix. It is cleaner than #22, but it still changes rows that already have a scale.

Use #27 only if deliberately testing robust triangle plus the safe 3 cm scale repair.

