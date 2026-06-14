# Submission Burn Pack - 2026-06-12

Purpose: if the daily Kaggle slots will expire anyway, spend them as controlled probes. Each CSV below
tests one axis against the protected 0.61918 baseline. These are not stacked ensemble guesses.

Update: `submission_burn_04_temporal_smooth_092.csv` scored **0.60961**, so the remaining upload
order is superseded by `SUBMISSION_BURN_AFTER_TEMPORAL_WIN_2026-06-12.md`.

Safe baseline remains `results/submission_local.csv` / downloaded `0P61918_submission_local.csv`
(public LB 0.61918).

## Recommended Burn Order

| order | file | axis | why this slot is worth information | movement vs 0.619 baseline | rough 19-label proxy |
|---:|---|---|---|---|---:|
| 1 | `results/submission_burn_04_temporal_smooth_092.csv` | temporal smoothing | Tests whether adjacent sequence frames should share a denoised median. Small broad movement, separate from geometry. | PA 111 rows mean 0.058 deg; FL 112 rows mean 0.397mm; MT 108 rows mean 0.050mm | 0.5626 |
| 2 | `results/submission_burn_05_shape_neighbor_scale_only.csv` | fallback-row scale | Clean version of the old shape-only idea: changes only the 10 stable same-shape fallback rows, with no global FL recenter ripple. | FL 10 rows mean 0.619mm, max 36.029mm; MT 10 rows mean 0.085mm, max 4.686mm | 0.5579 |
| 3 | `results/submission_burn_01_img00275_ocr_scale_only.csv` | one verified scale anomaly | Tests the single printed-ruler vs tick disagreement, isolated to IMG_00275 only. Expected LB movement is small because it is one row. | FL 1 row +26.973mm; MT 1 row +12.146mm | 0.6683 |
| 4 | `results/submission_burn_02_fl_min_extrap_top3.csv` | FL combiner | Tests the host-protocol idea: use the three most complete/minimal-extrapolation fragments instead of every fragment median. | FL 307 rows mean 5.239mm, p95 13.932mm | 0.5654 |
| 5 | `results/submission_burn_03_fl_visibility_weighted.csv` | FL combiner | Tests a smoother support-weighted alternative: larger and more visible fragments dominate without a hard top-3 cutoff. | FL 307 rows mean 5.597mm, p95 13.767mm | 0.6231 |

Baseline rough 19-label proxy is 0.5579. Treat that proxy as triage only; the labels are rough and
not official truth. It mainly says the two broad FL-combiner probes are riskier than the scale/sequence
probes.

## Existing File Not In The Main Five

`results/submission_subpixel_scale.csv` is a valid already-built precision probe, but it is so tiny
(FL mean movement 0.094mm, max 0.673mm) that it is likely below leaderboard resolution. If replacing
one of the five with an ultra-conservative low-information upload, replace burn #5 with this file.

## Generation

Post-hoc pack builder:

```powershell
python umud-muscle-architecture\experiments\exp31_submission_burn_pack.py
```

The top-3 and visibility-weighted files require CPU remeasurement from saved weights first:

```powershell
$env:UMUD_SCALE_SUBPIXEL='0'
$env:UMUD_FL_FACING='0'
$env:UMUD_FL_IDENTITY_BLEND='0'
$env:UMUD_TEMPORAL_SMOOTH='0'

$env:UMUD_FL_FRAGMENT_MODE='min_extrap_top3'
$env:UMUD_FL_FRAGMENT_TOPK='3'
$env:UMUD_LOCAL_OUT='umud-muscle-architecture\results\submission_burn_02_fl_min_extrap_top3.csv'
$env:UMUD_LOCAL_DEBUG_OUT='umud-muscle-architecture\results\calibration_debug_burn_02_fl_min_extrap_top3.csv'
python umud-muscle-architecture\local_infer.py

$env:UMUD_FL_FRAGMENT_MODE='visibility_weighted'
$env:UMUD_LOCAL_OUT='umud-muscle-architecture\results\submission_burn_03_fl_visibility_weighted.csv'
$env:UMUD_LOCAL_DEBUG_OUT='umud-muscle-architecture\results\calibration_debug_burn_03_fl_visibility_weighted.csv'
python umud-muscle-architecture\local_infer.py
```

Summary CSV written by the builder:

`results/submission_burn_pack_2026-06-12_summary.csv`

## How To Interpret Scores

- If #4 or #5 beats 0.619, the FL combiner is still a live lever; pursue support-aware/facing-per-gap
  selection rather than more scale work.
- If #1 improves, sequence exploitation is a valid low-risk postprocessor.
- If #2 improves, the remaining unscaled fallback rows matter more than expected.
- If #3 improves, keep the OCR scale partition and wire the isolated IMG_00275 fix.
- If all five regress or stay flat, the wall is probably not scale polish or simple FL aggregation;
  the next credible path is better fascicle segmentation/orientation modeling validated by the local
  human/expert/synthetic review surfaces.
