# Submission Review Result

Date: 2026-06-10

This is the current handoff after the MT vertical-3 Kaggle submission.

## Public Result

Submitted file:

`results/submission_host_mt_vertical3_no_subpixel.csv`

Public score: **0.62561**

Known better anchor: **0.61918** from `results/submission_local.csv`

Decision: **reject MT vertical-3**. Do not resubmit it and do not stack anything on top of it.

## What It Tested

The host described MT as three straight global-image lines from upper to lower aponeurosis at
left/middle/right, averaged. The old production code measured the center gap perpendicular to the
deep aponeurosis. `UMUD_MT_MODE=vertical_3` aligns the code with the host description.

The candidate was generated with:

- `UMUD_MT_MODE=vertical_3`
- `UMUD_FL_FRAGMENT_MODE=median`
- `UMUD_FL_IDENTITY_BLEND=0`
- `UMUD_SCALE_SUBPIXEL=0`

The `UMUD_SCALE_SUBPIXEL=0` part matters: it keeps this candidate isolated from the later sub-pixel
scale precision pass and makes it comparable to the restored 0.619 baseline.

## Local Evidence

Benchmark sanity check on the 35 expert-reference images, true scale:

| variant | overall | PA | FL | MT |
| --- | ---: | ---: | ---: | ---: |
| restored baseline logic | 0.2274 | 0.1498 | 0.3528 | 0.1795 |
| MT vertical-3 only | 0.2192 | 0.1498 | 0.3528 | 0.1550 |

Row-level diff versus `results/submission_local.csv`:

| column | changed rows | mean abs movement | max movement |
| --- | ---: | ---: | ---: |
| `pa_deg` | 0 | 0.0000 | 0.000 |
| `fl_mm` | 0 | 0.0000 | 0.000 |
| `mt_mm` | 285 | 0.0646 mm | 1.471 mm |

Because PA and FL were unchanged, the public loss is attributable to MT only. The score moved
0.61918 -> 0.62561, a +0.00643 overall regression. With equal target weighting, that is about
+0.0193 normalized MT error, or roughly +0.058 mm mean-absolute MT error equivalent under the 3 mm
MT tolerance.

Read: the reference-set MT improvement did not transfer. The hidden labels are closer to the old
center/perpendicular MT path, or this vertical-3 implementation does not match the host procedure
well enough to help. Either way, revert to `results/submission_local.csv` as the anchor.

## Scale Tail Is Still Separate

The tail idea is real scale work, but it is not a harmless add-on. It recovers scale for the 14 rows
left unscaled by the production router, split into:

- 10 `shape_neighbor_scale` rows.
- 4 `bottom_scale_bar_3cm` rows.

Movement versus the restored 0.619 baseline:

| file | PA rows changed | FL rows changed | MT rows changed | mean abs FL | max FL | mean abs MT | max MT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `submission_scale_tail_bar_only.csv` | 0 | 307 | 4 | 0.8082 mm | 43.286 mm | 0.1049 mm | 10.949 mm |
| `submission_scale_tail_shape_only.csv` | 0 | 307 | 10 | 0.4816 mm | 20.369 mm | 0.0851 mm | 4.686 mm |
| `submission_scale_tail.csv` | 0 | 307 | 14 | 1.2693 mm | 42.991 mm | 0.1900 mm | 10.949 mm |

The direct scale corrections are only on 4, 10, or 14 rows, but FL recentering causes 307 FL rows to
move. That makes the candidate much harder to interpret than MT vertical-3. If it scores worse, we
would not know whether the problem was the bar rows, borrowed shape scale, or FL recenter ripple.

Reviewer read:

- Do not stack any scale-tail file with the rejected MT vertical-3 candidate.
- Revert to `results/submission_local.csv` as the anchor.
- If spending another clean submission, use `results/submission_scale_tail_bar_only.csv` as a separate
  scale probe. The bar-only path has visible per-image scale evidence. Shape-neighbor scale is weaker
  and should remain later.

## Current Remaining Paths

1. Treat `submission_host_mt_vertical3_no_subpixel.csv` as rejected.
2. Keep `results/submission_local.csv` / public **0.61918** as the anchor.
3. If submitting another isolated probe, use `submission_scale_tail_bar_only.csv`.
4. Keep `submission_scale_tail_shape_only.csv` and all-tail as later/riskier probes.
5. Continue no-oracle model work through public-asset retraining/ensembling or ROI/crop cue
   detection, but do not confuse those with the immediate submission.
