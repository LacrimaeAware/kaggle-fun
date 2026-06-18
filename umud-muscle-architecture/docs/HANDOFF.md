# UMUD project handoff (read this first)

Self-contained state of the UMUD muscle-architecture Kaggle project, written 2026-06-15 to be read
cold by a person or a model that has no prior context. It is meant as a discussion document, not a
to-do list. Where a claim is verified it says so with a `file:line` or a leaderboard number; where it
is a hypothesis it says so. Do not upgrade a hypothesis to a fact without a check.

Companion living docs: `CURRENT_STATE.md` (terse decision-driver), `../VERIFIED_FACTS.md` (code/LB
facts only), `../FINDINGS_REGISTRY.md` (every idea by concept with a status tag), `../EXPERIMENT_LOG.md`
(chronological submission changelog), `../competition_reference.md` (host rules and device facts).

---

## 1. The task

One case is one single B-mode ultrasound image of a muscle. Predict three scalars per image:

- **PA** pennation angle in degrees: angle between the fascicle line and the **deep** aponeurosis.
- **FL** fascicle length in mm: distance between the superficial and deep aponeuroses **along the
  fascicle line**, with linear extrapolation when the fascicle runs past the image frame.
- **MT** muscle thickness in mm: perpendicular distance between the two aponeuroses, measured at 3
  width locations and averaged.

**Metric** (verified bit-for-bit in `metric.py` against the host scorer): lower is better,

```
score = (1/3) * [ MAE(PA)/6 + MAE(FL)/12 + MAE(MT)/3 ]
```

Tolerances 6 deg / 12 mm / 3 mm, equal weight per term, default per-image aggregation. A consequence
that has driven (and distorted) the whole project: a global mean shift on any one term is a
first-order lever on the score.

**Submission**: CSV with columns `image_id, pa_deg, fl_mm, mt_mm`, 309 rows. IDs are `IMG_00001.tif`
through `IMG_00251.tif` then `IMG_00252.png` through `IMG_00309.png` (the real suffixes matter).

**Prize / rules**: Kaggle "UMUD Challenge: Muscle Architecture in Ultrasound Data", host Paul
Ritsche, CHF 5000, recorded deadline 2026-11-14. Top-3 must release FAIR, open-source, reproducible
code, and the **whole pipeline is judged, not just the number**. External data (the public benchmark,
public weights, and even hand-labels on the 309 test images) is allowed if declared. So a one-off
high score from a non-reproducible CSV stack does not win; the pipeline has to regenerate it.

---

## 2. The data, and exactly what ground truth exists

This is the single most important section, because the project's central failure is a confusion about
what we can actually validate against.

Under `data/` (gitignored):

- **Training set**: 1048 aponeurosis (image, mask) pairs and 2761 fascicle (image, mask) pairs. The
  masks are the expert annotations of *where the structures are*. **There are no scalar PA/FL/MT
  labels for the training images.** The supervised target was always masks; PA/FL/MT are derived from
  the masks by geometry. This is why the system is "segment then measure".
- **Test set**: 309 images, **no labels of any kind**.
- **35-image expert benchmark** (`data/osf_arch_benchmark/.../Results_benchmark_architecture_v0.1.0.xlsx`):
  35 images with, per image, the true scale, 7 individual expert raters' PA/FL/MT (R1..R7), the
  DL-Track tool's output, and a consensus (SMA). This is the only set with both images and scalar
  truth. It ships **no masks**. Critically, these 35 images are a **different distribution** than the
  test set (training muscles VL/GM/TA, on devices that carry a ~1 cm tick convention), and they come
  with true scale.
- **Unextracted**: `GM_dynamic` and `CSA_RF` benchmark zips under
  `data/osfstorage-archive/Expert Analysed Benchmark Image Datasets/`, plus `Old/Young` annotated-image
  zips. RF (rectus femoris) is a **test-only muscle**, so the RF set is potentially high-value and has
  not been mined.
- **The user's hand-labels** from the correction UI (`benchmark_lab/correction_*`): per-image
  corrections on the actual 309 test images. This is the **only source of per-image truth on the test
  distribution.** It is small so far and grows only with the user's time.

Truth map, stated plainly:

| set | images | masks | scalar PA/FL/MT | scale | distribution |
|---|---|---|---|---|---|
| train | yes | **yes (expert)** | no | mixed | train muscles/devices |
| 35 benchmark | yes | no | **yes (7 raters)** | true | train-like, not test |
| 309 test | yes | no | no | must be recovered | the real target (RF, Lumify, cerebral palsy added) |
| UI hand-labels | subset of test | user-drawn | derived from user edits | per-image | **the test distribution** |

**Domain shift is real and documented** (`competition_reference.md` sec 7): test adds the Philips
Lumify device, rectus femoris muscle, and cerebral-palsy subjects, none of which appear in train.

---

## 3. The pipeline (how predictions are produced today)

`segment_then_measure.py` is the production pipeline; `local_infer.py` runs it over the 309 test
images on CPU in ~130 s and writes a submission, so downstream changes can be tested without a Kaggle
run. Stages:

1. **Segment**: two ResNet34 U-Nets (apo, fascicle) with test-time augmentation. Weights
   `results/seg_apo.pt`, `results/seg_fasc.pt`.
2. **Scale**: per-family router in `scale_ticks.py` reads the on-image ruler/ticks to get px/mm.
   Coverage 295/309; 14 fall back to a prior. Families: right_ruler_5mm, bottom_ticks, png_left_ruler,
   left_ruler_1cm, family_b_signature (a hardcoded instrument constant), none.
3. **Measure** (`measure()`): fit the two aponeuroses, fit fascicle fragment orientation by TLS/PCA,
   PA = length-weighted median fragment angle to the deep apo, MT = inner-edge aponeurosis gap, FL =
   per-fragment extrapolated span aggregated by `FL_FRAGMENT_MODE` (default `median`).
4. **Calibrate**: PA gets a flat additive shift; FL gets recentered (its column mean is pinned to
   `PRIOR['fl_mm']`); both PA and FL also carry global multipliers in the current best CSV.

Key flags (`segment_then_measure.py`): `USE_FL_RECENTER` ON (pins FL mean to `PRIOR['fl_mm']`),
`FL_FRAGMENT_MODE='median'` (see section 7, `min_extrap_top3` was tried and refuted), scale router ON,
inner-edge MT ON, temporal smoothing OFF.

---

## 4. The leaderboard ladder (what actually moved the public score)

Lower is better. This is the only signal that has ever predicted itself.

| score | change | nature |
|---|---|---|
| ~1.09 | pre-scale-router baseline | - |
| 0.61918 | per-family scale router + inner-edge MT + fragment FL | mechanistic, per-image |
| 0.60961 | temporal smoothing across sequence clips | aggregate variance reduction |
| 0.58910 | shape-neighbor fallback scale (+subpixel) | mechanistic, few rows |
| 0.55075 | PA +2.0 flat | global aggregate shift |
| 0.55033 | PA +2.5 flat | global aggregate shift |
| 0.52570 | FL x1.05 (on PA+2.5) | global aggregate shift |
| ~0.488 | family_b scale constant 134.5 -> 147 px/cm (FL/MT shrink on 41 rows) | **mechanistic scale fix, found by the user hand-reading ticks** |
| 0.46076 | aponeurosis band-selection fix (drop out-of-band fragments) | **mechanistic per-image fix** |
| 0.46041 | band fix + FL x1.05 (`submission_bandfix_flx105.csv`) | **current best** |

This session (2026-06-15), two confirming probes:

| score | submission | meaning |
|---|---|---|
| 0.47473 | `submission_reproduced.csv` | the median pipeline regenerated in ONE `local_infer.py` run |
| 0.49983 | `submission_minextrap.csv` | the same pipeline with `min_extrap_top3` FL: **regressed, refuted** |

Read the ladder honestly. The big mechanistic wins were the scale router, the family_b scale
constant, and the band fix. The PA and FL global shifts are aggregate band-aids: they move a column
mean and nothing else.

---

## 5. Current best and reproducibility status

- **Best submission**: `results/submission_bandfix_flx105.csv` at **0.46041**. It is a CSV stack
  (manual edits layered on the burn_13 file), not a single pipeline run.
- **Reproducible pipeline**: one `local_infer.py` run = **0.47473** (`submission_reproduced.csv`). The
  ~0.014 gap to 0.460 is leftover per-row CSV residue from the old stack (shape-neighbor fallback
  scale, subpixel, and the band fix being spliced rather than computed in-pipeline). It is not a model
  difference, and temporal smoothing was tested and does not explain it.
- So the project can regenerate ~0.475 from code today. Closing the last 0.014 to make the clean
  pipeline match 0.460 is bookkeeping, not modeling, and only matters for the reproducibility rule.

---

## 6. What the benchmark says with per-image truth, and the caveat that dominates everything

`benchmark_lab/honest_validate.py` runs the full pipeline on the 35 benchmark images with **true
scale and no FL recenter**, and scores per-image against the 7-rater consensus plus the per-term
human floor (each rater vs the mean of the others = the irreducible inter-rater noise).

Result (median FL mode, the production default):

| term | our error | human floor | reading |
|---|---|---|---|
| PA | 0.1505 | 0.2445 | below the floor: PA measurement is at the human limit here |
| FL | 0.5218 | 0.4026 | the only term with real headroom; FL over-reads by +5.8 mm |
| MT | 0.0840 | 0.0810 | at the floor: MT measurement is at the human limit here |

**The caveat that governs the whole project: the benchmark does not predict the leaderboard.** It has
mispredicted the LB direction repeatedly. The cleanest proof is from this session: `min_extrap_top3`
cut benchmark FL from 0.52 to 0.39 (below the human floor) and then **regressed the leaderboard by
0.025**. Two structural reasons: (a) the benchmark is fed true scale and is a different distribution
than the test set, so a geometry change that helps on benchmark images need not help on test images;
(b) the older `experiments/score_weights.py` made it even blinder by feeding true scale (`:42`) and
recentering predicted FL to the truth mean (`:54`), so that configuration cannot see scale or global
FL error at all.

Use the benchmark to catch a gross measurement bug. Do not use it to decide a submission. The history
of "principled local win, leaderboard regression" is long: FL identity blend, visibility-weighted FL,
robust-triangle geometry, vertical-3 MT, min_extrap, all looked good locally and all regressed.

One thing the benchmark does tell us cleanly, because MT has no fascicle and no extrapolation: with
true scale, MT sits on the human floor. So **MT error on the test set is almost entirely scale, not
anatomy.** MT is therefore a clean probe of the scale-and-aponeurosis foundation, separate from the
fascicle.

---

## 7. The core problem with the methodology so far (the reason for the reset)

For weeks the loop has been: pick a global transform (PA +k, FL xk, a scale constant), submit it,
keep it if the single public number drops. That is curve-fitting to one hidden statistic, not
measuring muscle architecture. The tell is exactly what keeps happening: wins found this way do not
have a per-image mechanism behind them, so they do not transfer, and the local instruments built to
predict them cannot, because a global probe only ever reveals the net direction of the average
residual, never per-image structure.

This has a name in the project's own notes (the "global-probe aggregation fallacy") and it has been
violated continuously. A leaderboard improvement from FL x1.05 means only "the average FL residual
was biased short by roughly that amount". It does not mean every image under-reads FL, it does not
locate which images are wrong, and it cannot distinguish a uniform bias from a few large errors with a
net tilt. The global levers (PA shift, FL scale) are now mostly spent precisely because they only ever
moved the mean, and the mean is a one-dimensional quantity.

The user has (correctly) called for scrapping this approach and rebuilding the methodology like a
scientist rather than a leaderboard-hacker.

---

## 8. The proposed real methodology (the direction from here)

Three parts, none of which has been done properly yet.

1. **A validation protocol that estimates test error before touching the leaderboard.**
   - GroupKFold on the training set by subject / device / muscle, to mirror the real domain shift, for
     any segmentation or model change. (Note: test has 5-frame sequences with no IDs; recover groups
     before any CV or random splits will leak and inflate.)
   - The user's hand-labels on actual test images as the per-image gate for the real distribution.
     These are the only test-distribution per-image truth that exists.
   - The leaderboard used a handful of times as a final confirmation, never as a search.

2. **An error decomposition into physical sources**: scale, segmentation, measurement geometry, and
   irreducible rater noise. Then fix the largest real source instead of guessing. The first concrete
   experiment needs no leaderboard slot: on a held-out train fold (so the segmenter never saw those
   images), run the measurement geometry twice, once on the **expert** masks and once on our
   **predicted** masks. The gap is the segmentation cost, per term. Combined with what the benchmark
   already shows (PA and MT are at the human limit given true scale; FL over-reads), this gives a real
   error budget.

3. **Fix the biggest real source, re-validate on CV, repeat.**

What this implies about where the score actually lives: the gap from ~0.46 to the leader's ~0.378 is
per-image structure, not a global knob. Global calibration is the last step on top of correct
geometry, it is distribution-specific (the benchmark wanted FL shorter, the test wanted it longer),
and the leaderboard is the only thing that can set it.

---

## 9. What is refuted or dead, so it is not resurrected

- **min_extrap_top3 / top-3 minimal-extrapolation FL**: regressed the LB twice (burn_14 0.62994 on the
  old base; 0.49983 this session). Looks great on the benchmark, does not transfer.
- **Per-image FL reshaping in general** (identity blend 0.639, visibility-weighted 0.645, facing FL
  0.665, robust-triangle geometry 0.601): every shape change regressed on the LB.
- **Vertical-3 MT** (0.626) and PA hinge / per-image PA reshaping (0.567): regressed.
- **Broad scale overrides** (field-depth 0.662, bar-only tail 0.667, proxy stack 0.659): regressed
  hard. Never broad-override an existing per-image scale.
- **Benchmark/proxy tuning as a submission gate**: thousands of configs ground the benchmark from
  0.17 to 0.13 (about 1.1 SE of label noise) and produced zero leaderboard gain.
- **The GPU segmentation pivot premise "FL is mask-limited"**: contradicted in-repo; a one-line FL
  scalar beat it, and recall-biased retraining made the benchmark worse. This does not mean
  segmentation is perfect; it means "retrain for better masks" was never justified by evidence and
  needs the section-8 decomposition to earn a target first.

Note the earlier doc claim that "the LB wants FL longer, so the recenter is backwards" is itself an
overclaim built on a global probe. With true scale on the benchmark, FL over-reads. The test FL
looking short is most consistent with under-recovered test scale, but that chain is unproven; treat it
as a hypothesis, not a finding.

---

## 10. Tooling reference

- `segment_then_measure.py`: the pipeline and all flags.
- `local_infer.py`: regenerate the 309-row submission on CPU (~130 s). Use this, not a Kaggle slot, to
  test downstream changes.
- `scale_ticks.py`: per-family scale router.
- `benchmark_validate.py`: honest scorer of any CSV against the 35-image consensus + human floor (no
  recenter). `--pred file.csv`.
- `benchmark_lab/honest_validate.py`: runs the full pipeline on the benchmark with true scale and no
  recenter, prints per-term error, the human floor, the signed FL bias, and the worst images. Built
  this session. Good for catching measurement bugs, not for deciding submissions.
- `experiments/score_weights.py`: the OLD benchmark scorer that feeds true scale and recenters FL.
  Structurally blind to scale and global FL error. Kept for reference; do not gate on it.
- `benchmark_lab/correction_*`: the correction UI for hand-labeling test images (the per-image
  test-truth source). Server is `correction_server.py`; UI in `correction_ui/`.
- `metric.py`: the competition metric, verified against the host scorer.
- `benchmark_lab/field_fascicle.py` + `benchmark_lab/MASKFREE_FASCICLE_NOTES.md`: an exploration of
  mask-free fascicle measurement (orientation field, blob, brightness walk, radar step). Findings: the
  classical methods bracket the truth (flat vs steep) and none beat the pipeline's mask-PCA on the
  benchmark; kept as a QA cross-check, not a submission path. See the notes file for the full record.

Weights, data, results, and any human labels are gitignored. `archive/` holds ~25 old narrative docs
and 41 dated EXP journals from before the 2026-06-14 reset; their findings are folded into
`FINDINGS_REGISTRY.md` with corrected status, so you should not need to open them.

---

## 11. Open strategic question (for the discussion this document is for)

The user wants to rebuild from their own original strategy ("identify the muscle, train networks to
recognize classes, derive the measurements from first principles"), not from more leaderboard tuning.
The concrete fork that needs their input:

- **Route by class**: classify muscle and/or device per image and run a measurement tuned per class,
  because RF, cerebral-palsy, and Lumify behave differently and one global pipeline blurs them.
- **Change the segmentation target**: change what the network is asked to find (for example an
  orientation field for fascicles, not just a binary mask), rather than post-processing the same two
  masks.
- **Rebuild measurement on the right objects**: treat the two aponeuroses and the fascicle
  orientation field as the things to get right per image, and rebuild PA/FL/MT on top of those.

The section-8 decomposition is the prerequisite for choosing among these on evidence rather than
taste.

---

## 12. How to work with this user (standing constraints)

- Plain language, no spin, no em dashes, no pep talks, no reassurance, no "great question" or "you're
  right". Numbers first. Claims proportional to evidence. Do not use "honest" as a qualifier.
- Do not call something a discovery before the leaderboard confirms it. The recurring complaint is
  exactly this overstatement.
- Always have a plan; do not end on a permission question; deliver concrete artifacts.
- This repo holds multiple competitions; commits for this project use no Co-Authored-By trailer.
- There is no per-image ground truth on the 309 test images except the user's own labels. Almost every
  per-image structural claim is currently unfalsifiable; say "unknown with current data" rather than
  inventing structure.
