# UMUD Experiment Log

One place to see what we tried, what it scored, the exact program/flag that produced it, and how to
roll back. Newest at top. Append, do not rewrite history.

Conventions: "LB" = Kaggle public leaderboard (lower is better). "bench" = 35-expert OSF benchmark
(local, CPU, NOT an oracle - see the leakage/recenter notes below). Safe baseline file:
`Downloads/0P61918_submission_local.csv` (LB 0.61918). `results/` is gitignored.

## Submissions (what actually hit the leaderboard)

| date | method | LB | changed | how to reproduce / roll back | status |
|------|--------|----|---------| -----------------------------|--------|
| 2026-06-09 | scale router + inner-edge MT + fragment-extrapolation FL | **0.61918** | full pipeline | `Downloads/0P61918_submission_local.csv`; flags: FL_FACING=0, identity_blend=0, MT=perp_center, subpixel=0 | **BEST / baseline** |
| 2026-06-10 | facing-geometry FL (`UMUD_FL_FACING=1`) | 0.66459 | FL only | set `UMUD_FL_FACING=0` to undo | rejected (bend real, gate wrong 41% - see diagnosis) |
| 2026-06-?? | FL identity blend | 0.63905 | FL only | `FL_IDENTITY_BLEND=0` to undo | rejected |
| 2026-06-?? | MT vertical-3 | 0.62561 | MT only | reverted | rejected |
| 2026-06-?? | bar-only scale tail | 0.66711 | 4 scale rows + FL recenter ripple | reverted | rejected |

## Verified diagnostics (facts, with how each was checked)

| finding | verified how | implication |
|---------|--------------|-------------|
| **FL recenter is a crutch for bad scale.** Production multiplies every FL by `74.424/mean` (segment_then_measure.py:762), pinning the submission FL mean to 74.424 regardless of the images. | read the code; confirmed it is a single global scalar keyed to `.mean()` | per-image absolute FL cannot survive it; a giant's fascicle is shrunk to the prior. Fix = reliable per-image scale, then `FL_mm = FL_px/scale`, no mean. |
| **The mean and the scale problem are the same root.** | algebra: ratio-anchoring `FL=(FL_px/MT_px)*MT_mm` reduces to `FL_px/scale` | fixing scale deletes the mean. |
| **Bench 0.227 is NOT leakage/overfit-inflated.** 21/35 bench images are in the U-Net training set (im_07 = image_0512 at corr 1.000), but leaked images score WORSE (0.253) than novel (0.189). | content-correlation leak scan + per-image split, score_weights pipeline | 12 epochs did not memorize; bench is honest on that axis. |
| **Bench FL term IS flattered by recentering.** score_weights.py:54 rescales predicted FL to the bench TRUE mean before scoring: FL term 0.534 raw -> 0.353. Raw FL bias is +6.0mm (66.8 vs 60.8). | ran pipeline on 35, computed with/without recenter | bench FL can never reveal a wrong mean; honest bench is ~0.288 not 0.227. |
| **PRIOR is not the bench mean.** PRIOR fl=74.424 vs bench true 60.835 (off by 13.6mm > tolerance). | compared PRIOR to truth means | production ships an FL mean the bench never validated. |
| **Printed depth is OCR-readable.** "De 50 mm" / numbered ruler "...3.5cm" read off the UI with easyocr + a `0-9.cm` allowlist. Reads a depth on 23/24 sampled images; clean on the De-50mm family. | scale_ocr.py smoke test | scale can come from the printed depth, not an assumed tick interval. |

## In progress (not wired / not submitted)

| build | file | state | next |
|-------|------|-------|------|
| **OCR scale** (read printed depth/ruler -> px/mm, cross-check vs ticks) | `scale_ocr.py` | cm-ruler family DONE: ruler-number regression (>=3 collinear pts, R^2-validated) reads 3.5/4.0/4.5cm AND agrees with the tick detector within 1-2% (verified). De-50mm family: depth text reads reliably; text-confirmed via geometry (printed depth x tick-scale must place depth-zero near the image top). Full verified/text-confirmed/tick-only/flag/mean partition over 309 = the deliverable (running). | drop the FL mean on verified+text-confirmed rows; mean only on the genuine blanks |

### Benchmark validation of the scale reader (2026-06-10)

IMPORTANT framing: the 35-expert benchmark is scored with the **true** scale from the xlsx, so scale
recovery **cannot move the benchmark FL/MT score** - it is blind to it by construction. What the
benchmark CAN do is validate the *reader* against a known scale. The benchmark images turn out to be
the **same `De XX mm` device family** as the test De-50mm set, but **horizontally mirrored** (`De 65 mm`
reads in reverse; flip with `cv2.flip(im,1)` first). After flipping, the OCR reads a depth on **30/35**,
all sensible (40-65mm), and the read depth is geometrically consistent with the known true scale. So the
reader transfers to ground-truth data. The actual *improvement* from this work lives on the Kaggle test
set (the partition coverage), which the benchmark structurally cannot score.

**Scale-cost experiment** (`experiments/bench_scale_cost.py`): hide the true scale, recover it the way
we must on the test set, score FL/MT with the recovered scale vs the true scale; the gap = what scale
error costs in the real metric. Result: per-image FL/MT (no recenter) with TRUE scale = FL 0.261 / MT
0.179; this **proves scale is load-bearing** - a 2x scale error (what happened on the few benchmark
images our detectors caught) blows FL and MT each up by **+2.4 tolerances**. CAVEAT, the run is
confounded: our tick detectors are tuned for the test families and **failed on the mirrored,
lower-res benchmark** (30/35 fell to PRIOR), so the headline +1.6 overall cost overstates the test-set
cost. The clean signals: (1) scale error is catastrophic, so getting it right is the top lever; (2) the
brittle tick detectors did NOT transfer, but the OCR depth reading DID (30/35) - so OCR depth is the
robust, device-agnostic path and should become the PRIMARY scale source, not the per-family tick rules.
Honest limit: no clean "scale costs X on the test set" number exists (no test ground truth; benchmark
recovery doesn't transfer). Standing position: scale matters hugely; ours is approximately right on test
(no gross errors per MT physiology) but was unverified; the OCR work makes it verifiable + readable.
| **OCR scale reader** (`scale_ocr.py`) | DONE (2026-06-10). Full 309-image partition complete: **verified 48, text-confirmed 99, tick-only 147, flag 1, mean 14**. Where two independent reads exist (148 images), OCR and ticks agree 147/148. The one disagreement is IMG_00275: tick said 201 px/cm, printed ruler says 101 px/cm — caught and quarantined to the prior (not shipped). The 14 `mean` rows and IMG_00275 are genuine unknowns, all others are checkable. | apply OCR scale to IMG_00275 (one-image provably-correct fix) |
| **Per-gap multi-level** (assign fascicles to gaps, geometry per gap) | `experiments/per_gap_viewer.py` | REWORKED 2026-06-10 to the user's diagnosis: (1) `apo_bands` MERGES mask fragments that overlap in depth into one aponeurosis and keeps depth-separated ones distinct; (2) hard guards — reject a gap whose two apo lines cross, drop any fascicle whose superficial end isn't strictly above its deep end; (3) draw the real mask edge per-x. Prototype on ~16 images via `UMUD_PERGAP_PROTO=1`. NOT wired to production, NOT submitted. | wire facing-FL per gap (use per-gap for multi-muscle separation ONLY; FL must use `compute_facing_fl()` / minimize-extrapolation, NOT the per-gap wave trace which overshoots +24mm) |
| **FL recenter** (`segment_then_measure.py:762`) | PROVEN NO-OP on the shipped pipeline. Running `UMUD_FL_RECENTER=0` changes 0/309 rows — per-image FL already averages 74.4 before the pin step. The recenter fires only when the mean deviates from PRIOR=74.424, and it doesn't. Not a hidden confounder on the current pipeline. | no action needed unless FL method changes |

## Wanted but unbuilt / uncertain

- **Facing-FL per gap**: wire the per-gap multi-muscle separation from `per_gap_viewer.py`'s `apo_bands()` + gap formation, but compute FL using `compute_facing_fl()` (zero-bias), NOT the wave trace (which overshoots +25mm). This is the highest-value next submission target — it fixes the ~13 multi-muscle images that caused facing's LB regression without losing facing's zero-bias gain on normal images.
- **IMG_00275 scale fix**: apply OCR scale (101 px/cm) instead of the wrong tick scale (201 px/cm). One image, provably correct.
- **Fascicle segmentation**: the real bottleneck. The FL gap to DL-Track is mask quality, not geometry. Recall-bias retrain already failed. Needs better training data or architecture, not more post-processing.
- **Scale classifier** (route by device family, then OCR the exact depth). Family is learnable; exact depth must still be read (3.5 vs 4.5cm look identical otherwise).

## New diagnostic findings (2026-06-10/11)

| finding | tool | result |
|---------|------|--------|
| **FL methods side-by-side** | `bench_fl_methods.py` | straight all-frags: bias +24mm, term 2.02; **facing: bias +0.7mm, term 0.26 (zero bias)**; per-gap wave: bias +25mm, term 2.1 (same as straight — it dropped minimize-extrapolation). The overshoot is the disease; minimize-extrapolation is the cure. |
| **Fascicle bend effect** | `bench_fl_bend.py` | bend is anatomically real (FALLMUD: parabola fits apo edges 44% better than line). But the competition FL ground truth is the raters' straight-line extrapolation convention, not the true curved anatomy. Bend diverges from the scored convention and is below tolerance (marginal and not worth chasing). |
| **FALLMUD transfer** | `fallmud_fl_test.py` | Our wave FL is median 1.09x longer than the reference straight-extrapolation FL on independent NeilCronin GT masks — confirms wave overshoots, facing's minimize-extrapolation is what keeps it honest. |
| **PA geometry validated** | `compare_lines.py` + `draw_tool.py` | User hand-drew fascicles on 8 test images. Our fitted angle field is off by mean 1.8°, median 1.4°, 90th pct 3.8° — well inside the 6° PA tolerance. PA geometry is confirmed good. |
