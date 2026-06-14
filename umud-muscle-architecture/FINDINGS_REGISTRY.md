# UMUD findings registry

One place for every idea, feature, and experiment, organized by concept, each tagged with what we
actually know about it. This **merges** the old front-door / plan / strategy docs and summarizes the
dated EXP journals so you never have to open them. For the chronological submission changelog see
`EXPERIMENT_LOG.md`; for current state see `docs/CURRENT_STATE.md`; for code/LB-only facts see
`VERIFIED_FACTS.md`. The dated `EXP*.md` journals and old narrative docs are moved to `archive/`.

Status is judged against code + the leaderboard, **not** against what the source doc claimed.

| tag | meaning |
|---|---|
| **[LIVE]** | current, active, or a durable idea worth pursuing |
| **[FACT]** | verified by code, the leaderboard, or a host clarification |
| **[UNTESTED]** | built or proposed, never scored on the leaderboard |
| **[REJECTED]** | tried on the leaderboard, regressed |
| **[FALSE]** | a claim the old docs asserted that code or the LB disproves |
| **[PAST]** | true once / superseded / historical context, not the current state |

---

## Scale

- **[FACT]** Per-family scale router is the one thing that ever moved the board (1.09 → 0.619). Reads actual rulers per device family; no test labels. `scale_ticks.recover_for_image`.
- **[FACT]** Scale router coverage 295/309 (95%): right_ruler_5mm 87, bottom_ticks 59, png_left_ruler 58, left_ruler_1cm 50, family_b_signature 41, none 14. (`calibration_measurement_debug.csv`)
- **[FACT]** Pixels are always square (train + test, host) — one horizontal tick scale also fixes vertical MT.
- **[FACT]** Scale is FOUR device families, not one; route by detector/signature, not by pixel shape. German Siemens scale = a faint right-edge 5mm depth ruler (the bottom "bracket" was a red herring). PNG `tick_mm=5` is correct (the feared 2x trap does not exist). IMG_00275 is the one real 2x anomaly (tick 201 vs ruler 101 px/cm), quarantined.
- **[FACT]** Displayed depth is solved: EXP63/64 OCR + deterministic guesser match all 309 reviewed depths (0 misses). But **depth ≠ scale**: `scale = depth_span_px / depth_cm`, and the trusted pixel span is the unsolved part.
- **[FACT]** Depth/text side is solved algorithmically; remaining problem is pixel-span-to-scale, not reading labels. OCR-50 stale-OCR repair fixed 32 rows.
- **[LIVE/RISK]** "Scale solved" = self-consistency, not correctness. OCR-vs-tick agreeing 147/148 cannot detect a shared mm-per-tick convention error (the dangerous double-counting one). `family_b_signature` hardcodes 134.5 px/cm at conf 1.0 for 41 imgs by fingerprinting 4 UI pixel rows; `left_ruler_1cm` pins constant 126 for 50 imgs. ~30% of test is per-image-unvalidated constant scale. `CALIBRATION_MIN_CONF=0.3` can admit known-wrong UI-text-keyed scales.
- **[LIVE]** Two-cue cross-check (114 imgs) agrees within ~0.3% after sub-pixel refinement — weakens a broad 2x-router-failure fear, but proves consistency, not correctness. Residual risk concentrated in single-cue `right_ruler_5mm` and the 14 `none` rows.
- **[UNTESTED]** Narrow 3cm scale candidates (burns #18/#19/#20/#21/#24/#25/#26/#27) for the IMG_00198–00200(+251) rows at 159.33 px/cm — built, never scored; #26==#19 and #27==#21 byte-for-byte.
- **[REJECTED]** Every **broad** scale override regressed: bar-only 3cm tail 0.66711; broad field-depth (burn #22) 0.66197; local-benchmark proxy + missing-scale (burn #28) 0.65917. Lesson: never broad-override an existing per-image scale; the field-rectangle heuristic overcounts UI height.
- **[FALSE]** "Scale is the bottleneck / a better field-span detector is the next major lever." The biggest win was a non-scale output lever (FL ×1.05). Scale recovery is largely banked.
- **[FALSE]** The 116-row field-height "disagreement set" as scale-correction candidates — EXP64 reclassified them as pixel-span (not depth) false positives.
- **[PAST]** Calibration detector originally read the right UI **text panel** not the ruler (constant 13.40 px/mm, +50% MT error at 3cm); fixed by the per-family router + `png_left_ruler` path. EXP60/61 human scale-oracle review tooling (no LB impact; production never reads the notes).
- **[PAST]** Sub-pixel scale refinement is below the metric noise floor (FL Δ ~0.094mm vs 12mm tol); off by default, cannot move the score.

## FL (the live lever)

- **[LIVE]** **FL global scale is the biggest lever and is under-tapped.** On the PA+2.5 base, FL ×1.05 → **0.52570** (current best), still climbing. Raw geometry FL mean is 91.6mm (= ~×1.23 of the pin), so likely more headroom. Bracket x1.10/1.15/1.20/1.25 (built), then bake into the pipeline.
- **[FALSE]** "FL recenter is a no-op (0/309 rows)." It is an **active ~0.81× shrink** (raw FL mean 91.6mm pinned to PRIOR 74.424), 308/309 rows by ~17mm. The "0/309" came from re-applying the pin to an already-pinned file. `segment_then_measure.py:1144-1145`.
- **[FALSE]** "FL recenter masks a +6mm overshoot, do NOT delete it" (EXP79) — **direction backwards**; the LB wants FL *longer*, so the recenter shortens it the wrong way. The +6mm "overshoot" is a benchmark artifact (the benchmark recenters FL to its own truth mean).
- **[FALSE]** "FL is unbiased on the test labels, leave it." The ~-0.46mm bias was measured on blind validators; the LB shows a large global FL undershoot.
- **[FALSE]** "FL is mask-limited / the FL gap to DL-Track is fascicle mask quality" (the GPU-pivot justification). A one-line global FL scalar improved FL with **no mask change**; recall-bias retrain made the benchmark *worse* (0.368→0.484).
- **[FALSE]** "Output-level FL levers are exhausted / FL is a post-hoc graveyard." The biggest gain of the project was a post-hoc global FL scalar.
- **[FACT]** `FL = MT/sin(PA)` identity → small PA error amplifies into FL, and FL/MT share the same scale so scale double-counts.
- **[FACT]** FL is >90% extrapolation: the visible fragment supports <10% of projected length, so FL is set by fragment **slope** + the two apo line fits + the convention, not by pixel recall (a 12px and 200px fragment at the same slope give the same FL).
- **[FACT]** FL ground truth is the raters' **straight-line low-extrapolation convention**, not anatomy. Bend is real (FALLMUD parabola fits 44% better; wave FL ~1.09× longer) but diverges from the scored convention. Do not chase curvature for FL.
- **[FACT]** With TRUE scale, benchmark FL is still ~1.19 (no better than a constant) — on the *benchmark* FL is geometry/extrapolation-limited, not scale-limited. (Note: on the hidden test the global mean was the dominant error, which the benchmark cannot see.)
- **[FACT]** PRIOR FL=74.424 (sample-submission mean) is 13.6mm — more than one tolerance — above the benchmark-true FL mean 60.835. The pin target was never validated.
- **[LIVE]** Projected-FL p25 / robust-triangle aggregation cuts benchmark FL 0.519→0.281 with no new model (benchmark-only; the recenter would erase its mean effect — re-examine now that the recenter is understood).
- **[REJECTED]** Every per-image FL *reshaping* regressed: identity blend 0.63905, top-3 minimal-extrapolation 0.62994, visibility/support-weighted 0.64511, facing FL 0.66459, robust-triangle geometry 0.60102. (These are shape changes, distinct from the global scale that worked.)
- **[PAST]** Curved-fascicle / arc-length FL as the "novel contribution" — superseded by the straight-line-convention finding. Straight-all-fragments FL overshoots +24mm on the benchmark (minimize-extrapolation was the bench cure; LB direction is the opposite, so treat bench FL direction with care).

## PA (tapped)

- **[FACT]** PA flat-shift optimum is ~+2.4 and we are on it. LB: +0=0.58910, +2=0.55075, +2.5=0.55033 (best), +3=0.55168. The shift lives only in post-run CSVs, not in `segment_then_measure.py`.
- **[FACT]** PA is solved in-distribution (benchmark term ~0.15 ≈ 0.9° MAE, below ~1.6° inter-rater SD; hand-drawn check within 1.8°). Method: TLS/PCA fragment orientation + min-6°/min-40px filters + length-weighted median.
- **[LIVE]** PA under-reads ~2° **only on the test distribution** (model mean 14.6 vs benchmark 18.3 / hand 19.9). Mechanism: the model fits individual fascicle slopes well (~1.8°); the aggregate bias is in fragment **selection** (averages in shallow/apo-parallel fragments). Partly real population shift (Rectus femoris low pennation + cerebral palsy), so a flat shift over-corrects the low-PA bulk — confirmed by +2.5 beating +3.
- **[FALSE]** "Output-level PA correction exhausted at +2" — +2.5 beat +2; optimum ~+2.4.
- **[REJECTED]** PA hinge (extra lift above 16°) 0.56681 — per-image PA reshaping overfits the 19 labels.
- **[FALSE]** Raw grayscale **texture** orientation as a PA estimator (0.62–0.64; anti-signal); contrarian "move away from texture" removes signed bias but worsens MAE. At best a QA diagnostic.
- **[FALSE]** Lower/upper/average-boundary tangent PA conventions, plain smoothing, circular means/RANSAC/endpoint axes — all worsen PA. Keep PCA + area-weighted median relative to the **deep** boundary.
- **[PAST]** Per-band fragment-count PA/MT averaging: tiny real gain (PA 0.150→0.146) but only as routing for known multi-band images, never a global average. PA conflict-gate is the only small positive boundary signal (benchmark-only). Temporal smoothing across sequence clips contributed to the old 0.58910 (now superseded).

## MT (done)

- **[FACT]** Inner-edge aponeurosis-gap MT (`UMUD_APO_INNER`, muscle-facing edges not centroids) is the banked win: benchmark MT 0.49→0.18 (~0.25mm MAE, below ~1.0mm inter-rater SD). MT is **not** recentered in code. Residual MT error is scale, not geometry.
- **[UNTESTED]** MT global-scale probes built (×0.95/×1.05) — fire one on the best FL base to confirm/rule out a hidden MT bias (same instrument-blindness that hid FL could hide a smaller MT one).
- **[REJECTED]** Vertical three-position MT (host straight-line left/mid/right): improved benchmark (0.2274→0.2192) but regressed LB 0.62561; vertical-center MT under robust triangle also regressed (proxy 0.60720). The center/perpendicular MT path stays the anchor.

## Measurement geometry

- **[FACT]** Targets: PA = angle of fascicle to deep apo; FL = superficial-to-deep distance along the fascicle (linear extrapolation past frame); MT = perpendicular apo gap at 3 width locations, averaged. Submission cols `image_id, pa_deg, fl_mm, mt_mm`. Clip bounds PA 5–45, FL 30–200, MT 10–50.
- **[FACT]** Production uses **median** fragment FL to avoid Jensen bias of mean-of-lengths. Masks are not pixel-aligned to images (different aspect ratios) — resize both to a common square size or geometry corrupts.
- **[UNTESTED]** Per-gap multi-muscle fix for the ~13 three-apo-band images (fit every band, form a gap per pair, assign fascicles, weight by fragment count, use `compute_facing_fl` per gap). Prototyped on 16 imgs, never wired.
- **[REJECTED]** Robust-triangle piecewise upper boundary: strongest benchmark geometry win (0.251→0.170) but regressed LB 0.60102. Keep as a benchmark diagnostic only. (It shortens FL ~3.4mm — opposite the LB-confirmed direction.)
- **[PAST]** EXP38–54 benchmark-only geometry/aggregation tuning (curve blends, vertical-center MT convention, story stacks). Research anchors at best; see the validation note on why they did not transfer.

## Segmentation

- **[LIVE]** Task framing: read a geometric scene (apo = bright bands, fascicles = thin diagonal streaks); the hard part is the thin low-contrast **fascicles**, which need orientation+thickness+context, not brightness alone. Durable idea: recover the **latent path geometry**, not the mask pixels (ridge/skeleton/structure-tensor/Hough are implementations).
- **[LIVE/IDEA]** Geometry-aware loss: add an orientation head with (cos 2θ, sin 2θ) encoding so learning cares about angle, not just pixel overlap. Untested on this task; durable.
- **[FACT]** Banked seg: two ResNet34 U-Nets (apo + fascicle) + TTA (benchmark 0.383→0.370). Control `seg59_02_highres_512_unet`: apo Dice 0.7945, fasc 0.2925. Segmentation transfers to independent FALLMUD data (apo IoU 0.56) — no collapse. CLAHE at inference rejected (train/test mismatch). Domain-gap retraining demoted (no real train-vs-target gap).
- **[FALSE]** The whole GPU-pivot premise ("masks are the next lever / FL is mask-limited"). Contradicted four ways in-repo and beaten by a one-line FL scalar.
- **[FALSE]** EXP72 thin-structure run (soft/dilated targets + skeleton decode) underperformed control (fasc 0.26 vs 0.29). Used the fasc-Dice wall (~0.25–0.35) to justify the pivot — but Dice is not the score.
- **[LIVE/RISK]** Segmentation experiments are uncontrolled (EXP72 changed ~9 axes; EXP77 repeats the all-axes design EXP73 condemned). If retraining, run the controlled EXP76 (one axis at a time), gate on geometry/A-proxy not Dice.
- **[UNTESTED]** EXP77 best-effort heavy notebook — **never run** (no weights/logs/result). EXP59/74/76 notebooks, masked in-domain pretraining, topology losses (clDice/Skeleton-Recall), recall-heavy inference variants — all proposed/spec, none scored.
- **[UNTESTED/PROMISING]** Classical fascicle-line extractor (CLAHE + Frangi/Sato ridge + skeleton + dominant-orientation clustering + non-crossing cleanup). Literature-validated to reach the manual noise floor with no GPU; ranked #1 next non-LB work. Prefer Radon/Hough over a bare structure tensor.

## Validation methodology (the core disease)

- **[FACT]** **The leaderboard is the only signal proven to predict the leaderboard.** Every "principled" local/benchmark win regressed (FL blend, MT vertical-3, scale-tail, facing FL, robust triangle). Gate global/calibration quantities on isolated single-variable LB probes only.
- **[FACT]** The 35-image benchmark is **blind to scale and global FL bias**: it feeds TRUE scale (`score_weights.py:42`) and recenters FL to the truth mean (`:54`). It looks ~3× better than the LB and mispredicted the LB direction 4+ times. A logic/convention sanity tool, not an oracle.
- **[FALSE]** "The 19 hand labels are a usable directional gate." They are self-measured by the same geometry engine and **missed the 0.025 FL win**; not a gate for any global/scale quantity. Keep them only for in-distribution PA sanity and per-image shape after the mean is removed.
- **[FACT]** Proxy machinery (EXP48–56) is multiple-comparisons overfitting: >2000 configs scored on the same 35 images, reporting the minimum as "headroom." The 0.170→0.131 chase is ~1.1 SE of label noise — statistically meaningless; the code itself says "intentionally overfit-prone." Every piece submitted (burns 15/16/17) regressed.
- **[FACT]** The deferral pattern: cheap decisive tests get named then skipped for expensive virtuous-looking work (Dice-vs-angle never run as designed; orientation-correctness audit deferred for 4 commits; human-row gating built then bypassed).
- **[LIVE]** The decisive offline fork (synthesis 4d): when fascicle Dice is low, is the **angle** also wrong? EXP22 (`orientation_raw_support_summary.csv`) already shows model PCA angle agrees with the raw-image orientation field to ~5° — so the angle is largely fine and the GPU pivot is hard to justify; only the low-Dice-conditioned slice is genuinely un-run.
- **[LIVE]** Methodology going forward (from external research): quantify the LB noise floor (SE ~0.5/√n) and treat sub-2×-SE moves as noise; spend the adaptivity budget like currency (Blum–Hardt Ladder); a change is robust only if it is one DOF with a mechanistic reason AND large vs noise; do not tune many params to the public LB (private-LB shakeup). For model CV use GroupKFold by subject/device, stratified on muscle/disease; adversarial-validate train vs a test-distribution proxy.
- **[FACT]** Subject/sequence leakage: test has 5-frame sequences with no IDs; random splits inflate metrics. Recover groups before any model CV.
- **[FACT]** A/B/C scale model (A = public LB, B = benchmark, C = references): "only compare A-to-A" — the right model; the failure was not believing it.
- **[LIVE]** Leader's likely edge is the disciplined error-analysis loop (look at worst predictions before tuning), not a secret model. The failure-subclass/oracle idea is a diagnostic instrument, not a leaderboard method.
- **[FACT]** Declared human-in-loop labeling of the 309 test images is allowed external data (host), not a forbidden oracle — but keep the automated and human-in-loop modes separate and declared.

## Submission results (verdicts; full chronology in EXPERIMENT_LOG.md)

- **[LIVE]** Best = **0.52570** (FL ×1.05 on PA+2.5).
- **[FACT]** Wins ladder: 1.09 (pre-router) → 0.61918 (scale router) → 0.60961 (temporal) → 0.60936 (subpixel) → 0.58910 (shape-neighbor scale) → 0.55075 (PA+2) → 0.55033 (PA+2.5) → 0.52570 (FL ×1.05).
- **[REJECTED]** burn #14 top-3 FL 0.62994; #15 robust triangle 0.60102; #16 visibility FL 0.64511; #17 vertical MT 0.60720; #22 broad field-depth scale 0.66197; #28 proxy stack 0.65917 (mislabeled "burn #15 + scale"; actually #16 FL + #17 MT proxies); FL identity blend 0.63905; facing FL 0.66459; MT vertical-3 0.62561; bar-only scale tail 0.66711.
- **[PAST]** burns #4/#6/#11/#13 (0.609→0.58910) and the 0.61918 protected baseline — all superseded by the PA/FL shifts.
- **[FACT]** Submission format: comma CSV `image_id,pa_deg,fl_mm,mt_mm` with real suffixes (IMG_00001.tif..251, IMG_00252.png..309). Semicolon, TIFF-only 251 rows, and page-style IDs all fail.

## Tooling

- **[FACT]** `local_infer.py` regenerates the full 309-row submission on CPU (~110s, reproduces Kaggle PA/MT exactly). Saved weights `seg_apo.pt`/`seg_fasc.pt`. Never spend a submission to test what CPU can check.
- **[FACT]** Pipeline switches in `segment_then_measure.py`: TTA on, fragment FL on, identity FL fallback on, scale router on (`CALIBRATION_MIN_CONF=0.3`), inner-edge MT on, temporal/facing/CLAHE off, FL recenter on (the landmine).
- **[FACT]** DL-Track facts: ships masks only (no measured PA/FL/MT table), auto-calibration reads only the right 15% (wrong for the UMUD left-ruler PNGs), headless reproduction only reaches ~0.679 because calibration is its weak link.
- **[LIVE]** Visual audit harness (overlays of apo/fascicle/MT/FL/ticks/values) — partly built (benchmark_lab review servers, synthetic-geometry unit tests). Judge changes by "did the failure factor go away," not LB wiggle.
- **[UNTESTED]** Learned scale-cue detector (exp26-29): QA/disagreement tool only (val Dice 0.16), not a router replacement. `exp23` gated pseudo-label manifest (263–273/309 pass) for future self-training. `UMUD_SCALE_OVERRIDE_CSV` opt-in hook; production never reads human notes.

## Competition facts (host-confirmed)

- **[FACT]** Annotation = two researchers, averaged: superficial+deep apo, 3 fascicles, 3 PA, 3 MT; re-evaluated when raters disagree >10mm FL / >4° PA / >1mm MT. GT PA/FL from ~3 fascicles, MT from 3 locations.
- **[FACT]** Domain shift: test adds Philips Lumify (device), Rectus femoris (muscle), and cerebral-palsy subjects — none in train. Robustness (TTA, calibration) matters.
- **[FACT]** Image/mask shape mismatch is by design (resize both to a common square). Mask sparsity is intentional (labeled only where contrast is good); a lower-third bright structure can be bone (ignore it).
- **[FACT]** Rules: novel-algorithm development required (reusing smp/U-Net/DL-Track as an improved version is allowed); external data (benchmark, public weights, test hand-labels) must be declared; whole pipeline evaluated and must be reproducible.
- **[FACT]** Compute: local AMD RX 5700 XT can't train on Py3.13 (no CUDA / no torch-directml build); Kaggle free GPU is the training path.
- **[UNTESTED]** Direct image→(PA,FL,MT) regressor (ConvNeXt/EfficientNetV2/Swin) as a complementary ensemble member — only useful once each base has real signal.
