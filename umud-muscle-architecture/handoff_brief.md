# UMUD handoff brief (for a collaborating model)

Current-state briefing so another model can get caught up and extend or cross-check this work.
Read it with the canonical docs at the end. Last substantive update: 2026-06-10.

**Read first:** `MASTER_REVIEW.md` (canonical state as of 2026-06-10/11). Then `STATE_RESET_2026-06-10.md` for pre-history. Best SUBMITTED public score is **0.61918** (#7 at time of submission). Five post-best tweaks have all regressed:

| submitted | LB | status |
|---|---:|---|
| FL identity blend | 0.63905 | rejected |
| MT vertical-3 | 0.62561 | rejected |
| bar-only scale tail | 0.66711 | rejected |
| **facing-geometry FL** (`UMUD_FL_FACING=1`) | **0.66459** | **rejected** — multi-muscle outliers + gate wrong 41% |

The facing FL candidate (consensus angle + facing-parabola apo + minimize-extrapolation) **was submitted and regressed** 0.619→0.665. The geometry is zero-bias on the 35-expert benchmark (+0.7mm bias) but fails on ~13 multi-muscle test images where 3 apo bands exist, the wrong pair is selected, and fascicles from both muscles mix into one garbage consensus.

`results/submission_local.csv` is the **0.61918 baseline** (byte-identical to `Downloads/0P61918_submission_local.csv`). The production code now defaults `UMUD_FL_FACING=0` and `UMUD_FL_IDENTITY_BLEND=0`, so a fresh run preserves the safe baseline unless a rejected probe is explicitly enabled.

## The competition

UMUD Challenge: Muscle Architecture in Ultrasound Data (Kaggle community competition, deadline
2026-11-14). For each B-mode skeletal-muscle ultrasound image, predict three numbers: pennation
angle PA (deg), fascicle length FL (mm), muscle thickness MT (mm). One row per image.

- Test set: 309 images, `IMG_00001..IMG_00251` (.tif) then `IMG_00252..IMG_00309` (.png).
- Metric (UMUD Score): tolerance-normalized MAE, tolerances PA 6 deg / FL 12 mm / MT 3 mm, equal
  weight, **lower is better**. Submission: comma CSV `image_id,pa_deg,fl_mm,mt_mm`.
- Geometry: segment the two aponeuroses (bright bands) + fascicles (thin diagonal fragments), fit
  lines, then PA = angle of fascicle to the deep aponeurosis, MT = perpendicular gap between the
  aponeuroses, FL = fascicle length between them. Identity used: **FL = MT / sin(PA)**.
- Full official facts + host forum clarifications: **`competition_reference.md`** (read it).

## Where we stand (2026-06-09)

- Best **submitted** public LB: **0.61918** (rank #7 at the time). A later FL-blend probe
  worsened to about **0.64**. Later MT vertical-3 and bar-only scale-tail probes worsened to
  **0.62561** and **0.66711**. The downloaded `0P61918_submission_local.csv` has been restored on
  disk as `results/submission_local.csv` and is byte/data-identical to that known better file.
  Older docs that say `1.09194` are pre-scale-router history.
- Leaderboard leader **0.378**; provided DL-Track benchmark **0.679**; a careful BY-HAND human
  labeling of the test set scored **0.459** on the public LB (forum, "PatrickAIForFun").
- Current default local expert-benchmark score using the full wired scoring harness
  (`experiments/score_weights.py`, true scale + TTA + inner-edge MT + fragment FL + recentered FL):
  **0.2274** (PA 0.1498, FL 0.3528, MT 0.1795) with `UMUD_FL_IDENTITY_BLEND=0`.
  The experimental blend scored **0.1873** locally but worsened the public LB from `0.61918` to
  about `0.64`; treat the blend result as a transfer failure, not a current improvement.
  The reference set is cleaner/easier than the hidden LB and should not be compared literally to
  the public score.
  `score_on_benchmark.py` is a simpler raw scorer and does not apply the same FL recentering.
- The big lever was not a secret method. DL-Track uses manual scaling; our score jump came from
  automated per-image pixels-to-mm calibration plus real per-image FL/MT measurement. Do not
  speculate about the leader's private method.

## How we iterate WITHOUT submitting (the most important tool)

The user downloaded a published **35-image expert benchmark** (OSF, gitignored at
`data/osf_arch_benchmark/...`, with `Results_benchmark_architecture_v0.1.0.xlsx`). The xlsx gives,
per image: the TRUE pixels-per-cm scale, all 7 experts' PA/FL/MT, and DL-Track's + SMA's automated
outputs. This is our **local scoreboard**:

- `benchmark_validate.py` - `load_truth()` (expert consensus, human floor, DLTrack/SMA refs) and
  `score(pred_df, truth)` (tolerance-normalized MAE). Human floor 0.307, DL-Track-with-true-scale
  0.331, SMA 0.409.
- `score_on_benchmark.py` - run the saved weights' geometry on the 35 images with TRUE scale.
- `local_infer.py` - regenerate the full 309-row submission from saved weights on CPU (~110 s with
  TTA), VERIFIED to reproduce the Kaggle PA/MT exactly. **Test changes here; never submit to test.**

Caveat: the benchmark is different devices than the Kaggle test set, so FL/MT *magnitudes* don't
transfer. The 0.619 -> 0.64 failed blend proves that FL-method rankings can also fail to transfer
when they rely on the small clean benchmark or on recentering assumptions.

## Per-target status (measured on the 35 experts, with TRUE scale)

Current default full-harness geometry: **overall 0.2274** (PA 0.1498, FL 0.3528, MT 0.1795) vs
human 0.307, DL-Track 0.331. This is strong on the cleaner benchmark when true scale is known, but
it is also a small benchmark with FL recentering, so do not transfer the number literally to the
hidden LB.

- **PA - effectively solved (0.150).** Total-least-squares (PCA) fragment orientation +
  length-weighted median + a min-6-deg filter that rejects aponeurosis-parallel fragments. Wired.
- **MT - strong with true scale (0.180).** The inner-edge aponeurosis fix (`UMUD_APO_INNER=1`) moved
  MT from a thick-band centroid problem to near-human. Remaining target-set MT error is mostly scale.
- **FL - public-transfer failure for the blend.** Pure fragment-extrapolated FL is the current
  default. Exp16 found that a 50/50 blend of fragment FL and `MT / sin(MAD-gated PA)` reduced FL
  to 0.233 locally, but the resulting public score worsened from `0.61918` to about `0.64` while
  PA and MT stayed identical. The blend remains as an experimental toggle only.

## The pipeline and its switches (`segment_then_measure.py`)

Trains apo + fascicle smp U-Nets (ResNet34, 0.5 Dice + 0.5 BCE, flips/rotations, 384 px), predicts
masks, measures geometry, writes `submission_segmentation.csv`. Runs on a Kaggle GPU via
`kaggle_segment_notebook.ipynb` (pulls the script from the public repo; no API token here). Saved
weights `results/seg_apo.pt` / `seg_fasc.pt` let everything else run on CPU.

Env switches (defaults in parens):
- `UMUD_TTA` (on) - mirror+scale test-time aug, averages sigmoids then thresholds; denoises the sparse
  masks (benchmark 0.383 -> 0.370). The only FL gain that is not a mean-fit.
- `UMUD_FRAGMENT_FL` (on) - prefer fragment-extrapolated FL when scale and a valid fragment exist.
- `UMUD_USE_IDENTITY_FL` (on) - fallback FL = MT/sin(PA), then FL values are recentered.
- `UMUD_FL_IDENTITY_BLEND` (0) - keep fragment-only FL by default. Setting `0.5` blends fragment
  FL with `MT/sin(MAD-gated PA)` before mm conversion; Exp16 local improved 0.2274 -> 0.1873, but
  the public LB worsened 0.61918 -> ~0.64, so do not use the blend as a submission default.
- `UMUD_USE_CALIBRATED_MT` (on) - MT_px / px_per_mm where scale is found.
- `UMUD_SCALE_ROUTER` (on) - the validated per-family scale router (below). `CALIBRATION_MIN_CONF` 0.3.
- `FASC_MIN_AREA` 40, `FASC_MIN_ANG` 6 - the PA-polish post-processing (wired constants).
- `UMUD_APO_INNER` (on) - fit muscle-facing inner edges of the two aponeuroses for MT.
- `UMUD_TEMPORAL_SMOOTH` (off) - median-smooth PA/FL/MT within detected sequence clips; built as a
  free side-bet, not part of the clean scale submission.
- `UMUD_FASC_POS_WEIGHT` (0) - >0 adds a pos_weight to the FASCICLE BCE to push recall (the user's
  idea to make the model draw more of each dash). Kaggle-GPU retrain only. Demoted unless a later
  correctness audit points back at segmentation.
- `UMUD_CLAHE` (0) - CLAHE contrast-normalize input in read_rgb; surfaces more fragments but MUST
  retrain BOTH models with it on (inference-only hurts, train/test mismatch). Demoted with the
  domain-gap retrain.

## Scale calibration - the leaderboard lever (NEW, the main 2026-06-09 work)

The 251 "unscaled" TIFFs ARE scalable. The host confirmed pixels are always square and bottom ticks
are 1 cm apart. The test set is **four device families by UI** (the 800x1200 size hides two devices);
each family's scale was validated by reading its actual ruler (no test labels exist):

| family | n | scale | source |
| --- | ---: | ---: | --- |
| PNG left numbered ruler | 58 | 150 px/cm | left ruler, 5 mm minor ticks (the feared 2x bug does NOT exist) |
| 644x1088 left depth ruler | 50 | 126 px/cm | left ruler 0-50 mm, 1 cm ticks |
| Telemed 800x1200 ("De 50 mm") | 49 | 134 px/cm | bottom ticks AND left 0-50 mm ruler agree |
| clean cropped/other | ~10 | varies | bottom ticks (IMG_00040 -> 78 px/cm) |
| German Siemens 800x1200 ("12L3 Quadriceps") | 87 | ~136 px/cm | faint RIGHT-edge 5 mm depth ruler; interval pinned by MT physiology + the "4.5 cm" depth label |
| Family-B signature 800x1200 | 41 | 134.5 px/cm | fixed left-margin UI signature, assigned from the validated family-B bottom-tick scale |

- `scale_ticks.py` - `recover_scale` (bottom ticks), `recover_scale_left_ruler`,
  `recover_scale_right_ruler` (German Siemens), `recover_scale_family_b_signature`, and
  `recover_for_image(gray, name)` the per-family ROUTER. Wired into `calibrate_image`.
- `subpixel_spacing.py` - harmonic-validated sub-pixel spacing estimator. Now wired as a **gated
  precision pass** for accepted bottom/right-ruler cues only (`UMUD_SCALE_SUBPIXEL=1`, default on):
  it replaces an integer/median spacing only if it agrees within 2%, and exposes residual/SE fields
  via `recover_for_image_detail()` and the calibration debug CSVs.
- Coverage from the current router: **295/309 = 95 % scaled**. Method counts from a direct run:
  right_ruler_5mm 87, bottom_ticks 59, png_left_ruler 58, left_ruler_1cm 50,
  family_b_signature 41, none 14.
- The 41 `family_b_signature` rows are the only explicit assignment path: it recognizes a fixed
  instrument UI signature and assigns the validated 134.5 px/cm scale. This is not hand-labeling,
  but it is not per-image ruler reading either; keep its method name visible in debug outputs.
- Harness: `experiments/scale_coverage.py`, `experiments/scale_qa.py` (overlays in
  `results/calibration_qa/`), `experiments/check_submission.py`. Full map: `competition_reference.md`
  sections 3, 3a, 3b.

This is the first change that targets the actual LEADERBOARD (the test set) vs the local benchmark.
The Kaggle gain is UNMEASURABLE locally (no test labels); it is submission-ready and the user decides
whether to spend a submission.

## New findings since exp30 (2026-06-10/11)

- **OCR scale partition DONE**: verified 48, text-confirmed 99, tick-only 147, flag 1, mean 14. 147/148 cross-check agreement where two reads exist. IMG_00275 caught as 2x error (tick 201 vs OCR 101 px/cm), quarantined.
- **FL recenter proven NO-OP**: `UMUD_FL_RECENTER=0` changes 0/309 rows — per-image FL already averages 74.4 before the pin. Not a hidden confounder.
- **FL methods benchmark** (`bench_fl_methods.py`): facing = zero bias (+0.7mm, term 0.26). Straight all-frags = +24mm overshoot. Per-gap wave/bend = +25mm (same overshoot — it dropped minimize-extrapolation). Facing's minimize-extrapolation is the fix; the bend/wave is NOT.
- **Fascicle bend is real but wrong convention** (`bench_fl_bend.py`, FALLMUD `fallmud_fl_test.py`): parabola fits apo edges 44% better on FALLMUD GT; wave FL is median 1.09x longer than reference straight FL. The competition scores straight-line extrapolation convention, not anatomy. Do not chase bend.
- **PA geometry validated** (`compare_lines.py`, `draw_tool.py`): user drew fascicles on 8 test images; our fitted angle is off by mean 1.8°, within the 6° PA tolerance.
- **Per-gap prototype built** (`per_gap_viewer.py`): `apo_bands()` now merges depth-overlapping fragments into one aponeurosis; hard guards reject crossing gaps and fascicles that cross an apo. Prototyped on 16 images. NOT wired to production. The wave FL overshoots — the wired version must use `compute_facing_fl()` per gap, not the wave.

## Open fronts

1. **Bound target-set scale error, do not guess it.** Use families with two independent scale cues to
   measure disagreement on the real 309 images. This sizes how much of the remaining public gap is
   scale versus measurement.
   - Started in `experiments/exp19_scale_crosscheck.py`: 114/309 images have two strict scale cues.
     Multi-cue families agree well after sub-pixel refinement (signature vs bottom ticks: 49 images,
     median 0.22%, max 0.29%; bottom vs left ruler: 29 images, 0.14%; PNG left-ruler cross-check:
     median 0.9%, max 2.5%, no >5% rows).
     This narrows remaining scale risk to the single-cue `right_ruler_5mm` family and the 14 `none`
     rows, not a broad 2x router failure.
   - Done in `experiments/exp20_subpixel_scale_refine.py`: REVIEW3's main integration request is
     started. Coverage stays 295/309; 144 accepted scale changes are all small (max 0.64% versus the
     integer router). The diagnostic sub-pixel candidate changes FL by mean 0.094 mm and MT by mean
     0.024 mm versus the restored 0.61918 baseline. Treat it as an isolated precision candidate, not
     a stacked submission.
   - Done in `experiments/exp21_scale_tail_recovery.py`: the 14 `none` rows now have a structural
     candidate, still isolated. Ten rows use stable same-shape neighbor scale (853-high family and
     small-crop family); four 800x1200 fallback rows expose a visible lower-right `3 cm` scale bar
     measured at 296 px = 98.667 px/cm. `results/submission_scale_tail.csv` changes PA by 0, MT on
     14 rows (mean abs 0.190 mm, max 10.949 mm), and FL broadly through recentering (mean abs
     1.269 mm, max 42.991 mm). Right-ruler QA flags five rows for review but shows low residual
     fractions overall (p95 0.0065). The script also writes split bar-only and shape-only candidate
     CSVs so these two recovery ideas do not have to be probed together.
   - Done in `experiments/exp22_orientation_raw_support.py`: raw-image support audit for predicted
     fragment orientation, calibrated on the 35-expert benchmark. Benchmark current PA MAE is
     0.899 deg; benchmark raw-support median disagreement q50/q95 is 5.110/7.264 deg. All 309
     target rows audit successfully; 23 are flagged for visual review. Crucially, right-ruler rows
     are 0/87 flagged and former `none` rows are 0/14 flagged, so this does not argue against the
     scale-tail work. It is a triage list, not a submission change.
   - Done in `experiments/exp23_pseudolabel_gate.py`: target-row manifest for future self-training
     or robust aggregation. It combines raw-support agreement, fragment coherence, fragment
     support/count, and scale tier. Results: 273/309 rows pass mask-level gates; 263/309 pass strict
     metric pseudo-label gates; 267/309 pass if the visible-bar scale-tail policy is allowed. This
     is not a submission CSV.
   - Done in `experiments/exp24_recenter_temporal_audit.py`: protected-baseline audit for FL
     recentering and temporal smoothing. Cached debug recentering reconstructs the protected 0.61918
     baseline almost exactly (mean abs FL delta 0.00019 mm), so the no-recenter audit is meaningful:
     removing recentering moves 308/309 rows by mean 17.147 mm. Temporal-only smoothing at 0.92 finds
     28 clips covering 140 frames and has much smaller row movement (mean normalized movement 0.020).
   - Done in `experiments/exp25_reference_error_budget_adapter.py`: reference error-budget input
     table and summary. The target scale router detects 0/35 reference rows, so this is an explicit
     oracle-scale attribution, not production-scale validation. With true scale, raw FL MAPE is
     10.365%, recentered FL MAPE is 6.833%, MT MAPE is 2.396%, and the current recentered reference
     score is 0.2274 (PA 0.1498, FL 0.3528, MT 0.1795).
   - Done in `experiments/exp26_scale_cue_pseudolabels.py`: code-generated weak labels for scale
     cues on the 309 target images. The corrected teacher policy exports only production-accepted
     router cues, plus the narrow visible-bar fallback on router-`none` rows. Results: 299/309 images
     labeled, 299 label rows total: bottom_ticks 59, right_ruler_5mm 87, left_ruler_1cm 50,
     png_left_ruler 58, family_b_signature 41, bottom_scale_bar_3cm 4. This is training prep for a
     learned cue detector, not a submission and not hand annotation.
   - Done in `experiments/exp27_external_asset_inventory.py`: local public assets are inventoried.
     The repo already has 1048 image/mask pairs for one segmentation target, 2761 for the other,
     35 public benchmark images, 309 competition target images, and one public pretrained weight
     file. External/public supervised data is real and local, not a hypothetical future download.
   - Started in `experiments/exp28_train_scale_cue_segmenter.py`: a weak-label multi-class U-Net
     harness now trains from exp26 cue masks. Smoke mode ran on CPU. A real 8-epoch CPU run with
     dilated masks and class-balanced BCE reached weak-label val Dice 0.1644. This is still training
     prep, not a production model.
   - Done in `experiments/exp29_scale_cue_model_audit.py`: learned cue-model audit against the exp26
     weak teacher. Best class-specific presence thresholds show strong weak-label agreement for
     left_ruler_ticks (F1 1.000) and right_ruler_ticks (F1 0.978), moderate bottom_tick_axis
     (F1 0.702) and ui_signature_marks (F1 0.698), and failed bottom_scale_bar (F1 0.046). Use this
     as a QA/disagreement signal only.
2. **Audit recentering/prior effects.** The full local score relies on recentering FL to a known mean;
   on the hidden target set the true mean may differ. The failed blend is proof that mean-stabilized
   or recentered local wins are not submission evidence by themselves.
3. **FL/term2 geometry.** Fragment-only FL is the current default. The 50/50 blend is now a negative
   control: it looked good locally and failed publicly. The remaining cheap work is target-family
   orientation coherence and a classical oriented-structure cross-check, not another mean-pinned CSV.
   - Done next: `exp17_blend_sensitivity.py` shows blend 0.50 is not a target-distribution outlier;
     this did **not** predict leaderboard transfer. Centering dominates: no-center mean FL jumps to
     92.4 mm.
   - Done next: `exp18_orientation_coherence.py` shows target fragments are highly coherent by family
     (coherence means ~0.992-0.998), so the extra target fragments look mostly like aligned signal,
     not random texture scatter.
4. **Temporal smoothing** (`UMUD_TEMPORAL_SMOOTH`, built, OFF by default). The test set contains
   sequence-like clips; smoothing is a modest free side-bet that cannot be scored locally.
5. **GPU retrain/domain adaptation is demoted.** `experiments/domain_gap_real.py` shows only modest
   real train-vs-target appearance shift (mean |SMD| 0.44), no test-only cluster, and no normalization
   fix. `experiments/seg_quality_test.py` exists to check mask-presence proxies; presence is not
   correctness, but the old "Lumify/domain collapse" premise is not supported.

## Constraints and values

- The artifact that matters is the thinking and writeup, not the rank. Claims proportional to
  evidence; label hypotheses; cite or drop.
- Corrected oracle policy from host discussion: human-created labels or right/wrong judgments on the
  309 test records are not automatically forbidden in this competition; the host describes that as
  external data that must be declared, with the whole code pipeline evaluated for reproducibility.
  Therefore there are two valid modes. In automated/no-oracle mode, use public assets, public
  benchmark data, and code-generated pseudo-labels only, with no hand correction. In declared
  human-in-loop mode, log the labeling protocol and every target-record judgment, save the resulting
  labels, declare them as external data, and make the final repo/notebook honest about how they were
  used. Do not mix the two modes silently.
- Public GitHub repo (github.com/LacrimaeAware/kaggle-fun): no secrets, no personal data, no
  Co-Authored-By trailer. Do not submit without explicit say-so. Local AMD 5700 XT cannot train on
  Py3.13 (no torch-directml); CPU training works but is slow; Kaggle GPU is the training path.

## Canonical docs to cross-check

- `competition_reference.md` - official rules + host clarifications + the full scale family map (3a/3b). **Most current.**
- `synthesis.md` - methods synthesis and current strategy framing.
- `experiments/README.md` - the experiment log (exp01+), with method/result/read each.
- `benchmark_findings.md` - the 35-expert benchmark analysis.
- `segment_then_measure.py` + `kaggle_segment_notebook.ipynb` - the pipeline and its Kaggle runner.
- `scale_ticks.py`, `benchmark_validate.py`, `score_on_benchmark.py`, `local_infer.py` - the calibration router and the local scoreboard.
- Older/partly-superseded: `leader_playbook.md`, `forward_plan.md`, `ranked_research_directions.md`, `codex_review.md`, `strategy_brief.md`, `plan.md`, `rundown.md`, `writeup.md`.

## Current submission recommendation

`results/submission_local.csv` IS the 0.61918 baseline (byte-identical to `Downloads/0P61918_submission_local.csv`). Safe anchor.

**Immediate next action**: build a bigger local benchmark with `benchmark_lab/`, not another blind submission. The new folder has a manifest builder, browser/Cintiq labeler, scorer, and protocol doc. Seed manifests live in `results/human_benchmark/`: `public_seed_manifest.csv` for public/FALLMUD labels and `target_seed_manifest.csv` for declared human-in-loop target labels.

**The one geometry shot left**: wire facing-FL per gap. Use `apo_bands()` + gap formation from `per_gap_viewer.py` for multi-muscle separation ONLY. Then compute FL per gap using `compute_facing_fl()` - NOT the wave trace. Do not submit it until the new human benchmark can distinguish the 0.619 baseline from the rejected 0.665 facing variant.

**Do NOT submit** based on:
- Local benchmark improvement alone (mispredicted LB direction 4 times)
- Any change that hasn't been isolated to a single dimension
- The wave/bend FL (it overshoots +25mm, same as straight; the per-gap wave dropped minimize-extrapolation)

**Rejected candidates** (all regressed from 0.61918):
- Facing FL as-is (0.66459): multi-muscle gate wrong
- FL identity blend (0.63905)
- MT vertical-3 (0.62561)
- Bar-only scale tail (0.66711)

**Do not run**: `UMUD_FL_FACING=1` unless intentionally testing a repaired facing/per-gap variant. A fresh `local_infer.py` run now defaults to the safe 0.619 fragment-FL baseline.

Canonical current-state doc: `MASTER_REVIEW.md`.
