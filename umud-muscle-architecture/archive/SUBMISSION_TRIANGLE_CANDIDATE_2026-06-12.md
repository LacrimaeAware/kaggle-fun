# Robust Triangle Boundary Candidate - 2026-06-12

## Status

Submission-worthy candidate, not yet submitted.

Recommended file if spending a slot:

`results/submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv`

This stacks the isolated robust-triangle geometry delta onto the current best public-LB anchor
`submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv` (0.58910).

## Why It Is Serious

The user's actual triangle idea is a piecewise upper-boundary shape:

- left 0-25%: robust deepest anchor
- middle 25-75%: robust highest anchor
- right 75-100%: robust deepest anchor

This is not the earlier straight chord. The straight chord was a misread and is weaker.

On the 35-image robust expert benchmark:

| candidate | overall | PA | FL | MT |
| --- | ---: | ---: | ---: | ---: |
| baseline true-scale CSV | 0.251 | 0.150 | 0.519 | 0.084 |
| user's exact triangle | 0.182 | 0.150 | 0.314 | 0.083 |
| user's robust triangle | **0.170** | 0.150 | **0.278** | 0.083 |

Largest local improvements:

| image | baseline score | robust score | improvement | baseline FL delta | robust FL delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| im_29_arch | 0.985 | 0.145 | 0.840 | +27.57mm | -0.46mm |
| im_27_arch | 0.440 | 0.100 | 0.340 | +13.35mm | +2.38mm |
| im_12_arch | 0.658 | 0.348 | 0.310 | +14.38mm | -4.60mm |
| im_05_arch | 0.466 | 0.269 | 0.198 | +13.66mm | +6.75mm |
| im_31_arch | 0.318 | 0.129 | 0.190 | +9.78mm | +2.13mm |

Worst remaining robust-triangle expert images:

| image | robust score | dPA | dFL | dMT |
| --- | ---: | ---: | ---: | ---: |
| im_10_arch | 0.420 | -4.36deg | +6.33mm | +0.02mm |
| im_12_arch | 0.348 | -2.41deg | -4.60mm | +0.78mm |
| im_21_arch | 0.323 | +0.26deg | +10.67mm | +0.11mm |
| im_22_arch | 0.316 | -2.46deg | +6.08mm | -0.09mm |
| im_03_arch | 0.306 | -1.08deg | +7.95mm | +0.22mm |

## Production Files

Isolated robust-triangle run:

`results/submission_robust_triangle_only.csv`

Stacked current-best candidate:

`results/submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv`

Generation:

```powershell
$env:UMUD_TOP_BOUNDARY_MODE='robust_triangle'
$env:UMUD_LOCAL_OUT='results\submission_robust_triangle_only.csv'
$env:UMUD_LOCAL_DEBUG_OUT='results\calibration_debug_robust_triangle_only.csv'
python local_infer.py
python experiments\exp37_robust_triangle_submission_stack.py
```

## Viewer

Robust-triangle expert viewer, sorted by remaining robust-triangle error:

```powershell
python benchmark_lab\review_server.py --expert-benchmark --port 8768 `
  --primary-pred-csv robust_triangle=results\benchmark_pred_robust_triangle.csv `
  --pred-csv exact_triangle=results\benchmark_pred_exact_triangle.csv
```

Open:

`http://127.0.0.1:8768/`

In this viewer, the actual robust-triangle candidate geometry is drawn live on the canvas:
magenta = robust-triangle upper boundary, yellow = lower boundary, green = robust-triangle projected
FL spans. The `old diag` layer is off by default; when enabled, its cyan spans are older projected-FL
diagnostics and should not be mistaken for the robust-triangle candidate spans.

Single-image triangle viewer for `im_29_arch`:

`http://127.0.0.1:8771/`

## Caveat

This is still a local expert-benchmark win, not a guaranteed public-LB win. The difference from the
rejected FL tricks is that this fixes a concrete boundary-geometry error and is implemented as an
isolated production flag (`UMUD_TOP_BOUNDARY_MODE=robust_triangle`). It is worth one submission slot,
but should be recorded honestly if it fails to transfer.
