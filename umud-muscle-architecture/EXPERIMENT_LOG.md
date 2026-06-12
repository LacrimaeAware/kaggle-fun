# UMUD Experiment Log

One place to see what we tried, what it scored, the exact program/flag that produced it, and how to
roll back. Newest at top. Append, do not rewrite history.

Conventions: "LB" = Kaggle public leaderboard (lower is better). "bench" = 35-expert OSF benchmark
(local, CPU, NOT an oracle - see the leakage/recenter notes below). Safe baseline file:
`Downloads/0P61918_submission_local.csv` (LB 0.61918). `results/` is gitignored.

## Submissions (what actually hit the leaderboard)

| date | method | LB | changed | how to reproduce / roll back | status |
|------|--------|----|---------| -----------------------------|--------|
| 2026-06-12 | current best + top-3 minimal-extrapolation FL (`submission_burn_14_temporal_subpixel_shape_fl_min_extrap_top3.csv`) | 0.62994 | burn #11 plus broad FL combiner change on 307 rows | `results/submission_burn_14_temporal_subpixel_shape_fl_min_extrap_top3.csv` | rejected; broad FL combiner harmed transfer |
| 2026-06-12 | current best + isolated IMG_00275 OCR scale (`submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv`) | **0.58910** | burn #11 plus one-row OCR scale fix | `results/submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv` | public-score neutral vs #11 |
| 2026-06-12 | temporal + subpixel + clean shape-neighbor fallback scale (`submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv`) | **0.58910** | burn #6 plus 10 stable fallback-row scale corrections | `results/submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv`; follow-ups in `SUBMISSION_BURN_AFTER_SHAPE_WIN_2026-06-12.md` | **NEW BEST** |
| 2026-06-12 | temporal smoothing + subpixel scale precision (`submission_burn_06_temporal_subpixel_scale.csv`) | **0.60936** | temporal file plus tiny gated scale precision deltas | `results/submission_burn_06_temporal_subpixel_scale.csv`; follow-ups in `SUBMISSION_BURN_AFTER_SUBPIXEL_WIN_2026-06-12.md` | superseded by shape-neighbor stack |
| 2026-06-12 | temporal smoothing at clip threshold 0.92 (`submission_burn_04_temporal_smooth_092.csv`) | **0.60961** | PA/FL/MT in 28 sequence-like clips | `results/submission_burn_04_temporal_smooth_092.csv`; follow-ups in `SUBMISSION_BURN_AFTER_TEMPORAL_WIN_2026-06-12.md` | superseded by temporal+subpixel |
| 2026-06-09 | scale router + inner-edge MT + fragment-extrapolation FL | **0.61918** | full pipeline | `Downloads/0P61918_submission_local.csv`; flags: FL_FACING=0, identity_blend=0, MT=perp_center, subpixel=0 | previous best / protected baseline |
| 2026-06-10 | facing-geometry FL (`UMUD_FL_FACING=1`) | 0.66459 | FL only | now default-off; use `UMUD_FL_FACING=1` only to reproduce/test repairs | rejected (bend real, gate wrong 41% - see diagnosis) |
| 2026-06-?? | FL identity blend | 0.63905 | FL only | `FL_IDENTITY_BLEND=0` to undo | rejected |
| 2026-06-?? | MT vertical-3 | 0.62561 | MT only | reverted | rejected |
| 2026-06-?? | bar-only scale tail | 0.66711 | 4 scale rows + FL recenter ripple | reverted | rejected |

## Verified diagnostics (facts, with how each was checked)

| finding | verified how | implication |
|---------|--------------|-------------|
| **FL recenter is a latent global mean device, but a no-op on the shipped 0.619 pipeline.** The code can multiply every FL by `74.424/mean` (segment_then_measure.py:762); on the current baseline the pre-recenter mean is already 74.4, so the multiplier is effectively 1. | read the code; confirmed with `UMUD_FL_RECENTER=0`: 0/309 rows changed | not a hidden confounder for the current baseline; any new FL method that changes the mean must report with/without this step. |
| **The mean and the scale problem are the same root.** | algebra: ratio-anchoring `FL=(FL_px/MT_px)*MT_mm` reduces to `FL_px/scale` | fixing scale deletes the mean. |
| **Bench 0.227 is NOT leakage/overfit-inflated.** 21/35 bench images are in the U-Net training set (im_07 = image_0512 at corr 1.000), but leaked images score WORSE (0.253) than novel (0.189). | content-correlation leak scan + per-image split, score_weights pipeline | 12 epochs did not memorize; bench is honest on that axis. |
| **Bench FL term IS flattered by recentering.** score_weights.py:54 rescales predicted FL to the bench TRUE mean before scoring: FL term 0.534 raw -> 0.353. Raw FL bias is +6.0mm (66.8 vs 60.8). | ran pipeline on 35, computed with/without recenter | bench FL can never reveal a wrong mean; honest bench is ~0.288 not 0.227. |
| **PRIOR is not the bench mean.** PRIOR fl=74.424 vs bench true 60.835 (off by 13.6mm > tolerance). | compared PRIOR to truth means | production ships an FL mean the bench never validated. |
| **Printed depth is OCR-readable.** "De 50 mm" / numbered ruler "...3.5cm" read off the UI with easyocr + a `0-9.cm` allowlist. Reads a depth on 23/24 sampled images; clean on the De-50mm family. | scale_ocr.py smoke test | scale can come from the printed depth, not an assumed tick interval. |

## In progress (not wired / not submitted)

| build | file | state | next |
|-------|------|-------|------|
| **Shape-neighbor stacked burn pack** | `SUBMISSION_BURN_AFTER_SHAPE_WIN_2026-06-12.md`, `experiments/exp34_after_shape_win_stack.py` | BUILT 2026-06-12 after temporal+subpixel+shape improved public LB to 0.58910. Generates `submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv` and `submission_burn_14_temporal_subpixel_shape_fl_min_extrap_top3.csv`. | submit #13 then #14 if burning the next slots; these replace older #12 and #9 because they keep the #11 gain |
| **Subpixel-stacked burn pack** | `SUBMISSION_BURN_AFTER_SUBPIXEL_WIN_2026-06-12.md`, `experiments/exp33_after_subpixel_win_stack.py` | BUILT 2026-06-12 after temporal+subpixel improved public LB to 0.60936. Generates `submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv` and `submission_burn_12_temporal_subpixel_img00275_ocr_scale.csv`. | submit #11 then #12 if burning the next slots; these replace older unstacked/less-stacked #7 and #8 |
| **Temporal-stacked burn pack** | `SUBMISSION_BURN_AFTER_TEMPORAL_WIN_2026-06-12.md`, `experiments/exp32_temporal_stack_burn_pack.py` | BUILT 2026-06-12 after temporal smoothing improved public LB to 0.60961. Generates follow-ups that keep temporal smoothing as the working baseline: temporal+subpixel, temporal+shape-neighbor, temporal+IMG_00275 OCR scale, temporal+top3 FL, and optional temporal+visibility-weighted FL. | submit the stacked files, not the original unstacked order |
| **Five-slot burn pack** | `SUBMISSION_BURN_2026-06-12.md`, `experiments/exp31_submission_burn_pack.py` | BUILT 2026-06-12. Generates/records five controlled CSV probes from the protected 0.61918 baseline: temporal smoothing, clean shape-neighbor fallback scale, isolated IMG_00275 OCR scale fix, FL top-3 minimal-extrapolation, and FL visibility-weighting. Generated CSVs live in ignored `results/submission_burn_*.csv`; summary lives in ignored `results/submission_burn_pack_2026-06-12_summary.csv`. | if daily slots would expire, submit in the documented order and record public scores back here |
| **Human benchmark lab** | `benchmark_lab/` | BUILT 2026-06-11/12. Includes `make_manifest.py`, `label_server.py`, `score_labels.py`, protocol docs, scratch rulers, draggable review overlays, and trial FL measurement. Generated seed manifests under `results/human_benchmark/`: `public_seed_manifest.csv` (24 public/FALLMUD rows) and `target_seed_manifest.csv` (24 declared human-in-loop target rows, with IMG_00275 forced first). 19/24 target rows were roughly hand-labeled in gitignored local results. Scorer has a light cv2/numpy geometry fallback, so labels can be measured locally without importing the full training stack. | use the Cintiq/review workflow to score the 0.619 baseline and repaired facing/per-gap candidates locally before the next serious submission |
| **Review surfaces** | `benchmark_lab/review_server.py`, `benchmark_lab/generate_synthetic_geometry.py` | BUILT 2026-06-12. Three local viewers exist: target human-in-loop (`--port 8767`), 35-image expert benchmark (`--expert-benchmark --port 8768`), and exact synthetic geometry cases (`--synthetic-dir results/synthetic_geometry --port 8769`). These are decision infrastructure, not submission files. | use synthetic cases as geometry unit tests, the expert viewer as a convention sanity check, and target human rows as the closest local proxy to the hidden leaderboard |
| **OCR scale** (read printed depth/ruler -> px/mm, cross-check vs ticks) | `scale_ocr.py` | DONE 2026-06-10. Full 309-image partition: verified 48, text-confirmed 99, tick-only 147, flag 1, mean 14. 147/148 cross-checks agree; IMG_00275 is the one verified 2x tick-vs-printed-ruler anomaly. | optional one-slot probe: generate an isolated IMG_00275-only OCR scale fix, inspect the one-row diff, and do not combine it with rejected tail/bar/shape changes |

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

## New diagnostic findings (2026-06-10/12)

| finding | tool | result |
|---------|------|--------|
| **Immediate submission audit** | CSV diff against `results/submission_local.csv` | No new ready 4-slot burn exists. `submission_subpixel_scale.csv` is tiny (FL mean abs delta 0.094mm, max 0.673mm); `submission_host_mt_vertical3_no_subpixel.csv`, scale-tail/bar-only, facing-FL, and identity blend are known rejected or audited worse than 0.61918. Only a clean IMG_00275-only OCR scale fix is defensible as a quick one-row probe. |
| **Target human labels and review tooling** | `benchmark_lab/label_server.py`, `benchmark_lab/review_server.py`, `benchmark_lab/score_labels.py` | 19/24 target rows have rough local hand labels in ignored `results/human_benchmark/`. The viewer can compare human masks, baseline, candidates, scale, rulers, PA/FL/MT, and scratch trial FLs. Treat these labels as triage, not official truth, but use them before another broad submission. |
| **Exact synthetic geometry benchmark** | `benchmark_lab/generate_synthetic_geometry.py`, `benchmark_lab/SYNTHETIC_ABSTRACT_BRIEF.md` | Built abstract boundary/strand cases with known exact geometry. First generated pack is useful for testing straight, steep, curved, and fan-like geometry assumptions. It is not realistic ultrasound and should not be used as leaderboard evidence by itself. |
| **Expert benchmark review mode** | `benchmark_lab/review_server.py --expert-benchmark` | Built a visual review mode for the 35-image expert set comparing robust expert consensus, our true-scale pipeline, DLTrack, SMA, scratch measurements, and diagnostic line overlays. On 2026-06-12 the local benchmark truth was made robust to clear single-rater tails without editing the source xlsx: `im_19_arch` MT drops `R7=80.13` (mean `30.03 -> 20.01`) and `im_26_arch` FL drops `R7=33.88` (mean `64.11 -> 70.16`). Full harness is now `0.1909` (PA `0.1498`, FL `0.3390`, MT `0.0840`); raw true-scale CSV is `0.251` (PA `0.150`, FL `0.519`, MT `0.084`). |
| **Benchmark error taxonomy** | `experiments/exp35_benchmark_error_taxonomy.py`, `benchmark_error_taxonomy.md` | BUILT 2026-06-12. Classifies each 35-expert image by geometry failure mode and writes local `results/benchmark_error_taxonomy.{csv,md}`. Naive wrong-way pruning does not help locally (`0.251 -> 0.252` for both signed-angle and raw-slope variants). The stronger tail/aggregation signal is projected-FL percentile: p25 improves the local raw true-scale score to `0.172` (FL `0.281`) because many expert values sit below our projected median. The user's actual triangle boundary idea, tested after the first straight-line misread, is strongest locally: robust triangle `0.170` (FL `0.278`), exact triangle `0.182` (FL `0.314`); on `im_29_arch` exact triangle gives median FL `75.81mm` vs expert `75.30mm`. Viewer now shows taxonomy tags, projected cyan fragment spans, and per-image projection-tail diagnostics. | use this as diagnostic evidence only; if pursuing, test a gated triangle/lower-percentile boundary-shape variant on target human labels/synthetic cases before any submission |
| **FL methods side-by-side** | `bench_fl_methods.py` | straight all-frags: bias +24mm, term 2.02; **facing: bias +0.7mm, term 0.26 (zero bias)**; per-gap wave: bias +25mm, term 2.1 (same as straight - it dropped minimize-extrapolation). The overshoot is the disease; minimize-extrapolation is the cure. |
| **Fascicle bend effect** | `bench_fl_bend.py` | bend is anatomically real (FALLMUD: parabola fits apo edges 44% better than line). But the competition FL ground truth is the raters' straight-line extrapolation convention, not the true curved anatomy. Bend diverges from the scored convention and is below tolerance (marginal and not worth chasing). |
| **FALLMUD transfer** | `fallmud_fl_test.py` | Our wave FL is median 1.09x longer than the reference straight-extrapolation FL on independent NeilCronin GT masks - confirms wave overshoots, facing's minimize-extrapolation is what keeps it honest. |
| **PA geometry validated** | `compare_lines.py` + `draw_tool.py` | User hand-drew fascicles on 8 test images. Our fitted angle field is off by mean 1.8°, median 1.4°, 90th pct 3.8° — well inside the 6° PA tolerance. PA geometry is confirmed good. |
