# UMUD ranked research directions

> **Archive note (2026-06-09):** this was written after the `1.09194` run and is no longer the next
> work plan. The current plan is target-set scale error bounding, prior/recentering sensitivity, and
> FL/orientation geometry. Read `handoff_brief.md` and `experiments/README.md`.

Public-safe strategy note, written after the `1.09194` run. This is meant to be
copy-pasted into another model or used as the next work plan for this repo. It
does not assume that the leader's private solution is known.

## Current state

Known public leaderboard anchors:

| System | Public score | What it means |
| --- | ---: | --- |
| Current best in this repo | `1.09194` | U-Net pennation, calibrated MT on 68 images, prior FL |
| Previous best in this repo | `1.11066` | ExtraTrees PA, prior FL/MT |
| U-Net PA without calibrated MT | `1.12324` | Same segmentation PA path, all FL/MT constants |
| DLTrack 0.3.1 benchmark row | `0.67944` | Public benchmark score, not hidden labels |
| Current public leader | about `0.37766` | Leader has not published the UMUD solution |

My current read: the `1.09194` result is useful, but it is not the main game.
It says that a small slice of calibrated MT helps. It does not solve the main
problem, because most test rows still have constant FL and many still have
constant MT.

The largest likely lever is still:

```text
image -> reliable apo/fascicle geometry -> per-image px/mm scale -> real MT/FL/PA
```

The pasted geometry/path idea belongs inside this pipeline. It is not a separate
"vibes" idea. It is a way to make the fascicle side of the measurement pipeline
less brittle.

## What is public about the benchmark, and what is not

This matters because it has caused confusion.

What appears public or verifiable:

- The Kaggle leaderboard has or had a row named like `DLTrack_0.3.1_benchmark_solution.csv`
  with score `0.67944`.
- DL_Track_US is public: https://github.com/PaulRitsche/DL_Track_US
- The DL_Track_US docs describe a batch segment-and-measure tool that outputs fascicle
  length, pennation angle, and muscle thickness.
- The docs describe scaling modes: no scaling, visible bar scaling, and manual scaling.
  Manual/bar scaling also requires a known spacing such as 5, 10, 15, or 20 mm.
- The docs describe training from image/mask pairs, not from a public table of measured
  PA/FL/MT labels.
- The paper reports that a U-Net-style system detects aponeuroses and fascicle fragments,
  then computes muscle architecture parameters.

What I do not think we have yet:

- The hidden target labels.
- A per-image error breakdown for PA vs FL vs MT.
- The exact benchmark prediction CSV values, unless Kaggle or the organizers expose that
  file somewhere separately.
- The exact settings used for the Kaggle DLTrack benchmark row.
- The leader's UMUD solution.

So the correct stance is: the DLTrack benchmark is a real public reference point, and the
DLTrack code/docs/papers tell us the likely method family. But the exact benchmark file is
not the same as having the hidden labels or the leader's method.

Sources checked:

- DL_Track_US repo: https://github.com/PaulRitsche/DL_Track_US
- DL_Track_US docs: https://paulritsche.github.io/DL_Track_US/latest/
- Automated image analysis docs: https://paulritsche.github.io/DL_Track_US/latest/automated_image_analysis/
- Training docs: https://paulritsche.github.io/DL_Track_US/latest/training_your_own_networks/
- JOSS paper page: https://joss.theoj.org/papers/10.21105/joss.05206
- Original arXiv method paper: https://arxiv.org/abs/2009.04790
- OSF examples/models page linked by the docs: https://osf.io/7mjsc/

## Executive ranking

This table ranks directions by expected strategic importance, not by how easy they are.

| Rank | Direction | Expected impact | Confidence | Why it ranks here |
| ---: | --- | --- | --- | --- |
| 1 | Full segment-measure-scale pipeline, using DLTrack as a reference | Major | High | This is the path from `~1.09` toward the `0.67944` benchmark zone. It attacks FL/MT, which are still mostly constants. |
| 2 | Visual audit and measurement validation harness | Major enabler | High | Without overlays and per-stage diagnostics, every model change is blind. This is how to stop guessing. |
| 3 | Per-image scale recovery, especially for TIFFs | Major | High | PNG calibration helped, but TIFFs are `251/309` rows. Scale is required for both FL and MT. |
| 4 | Fascicle length measurement | Major | Medium | FL is still constant everywhere. Once scale exists, even imperfect FL may move more than PA tweaks. |
| 5 | Geometry-aware fascicle recovery: ridge/centerline/orientation | Medium to major | Medium | This is the user's path-following idea in practical form. It may fix PA and FL geometry, but cannot solve mm scale alone. |
| 6 | Robust segmentation training: folds, TTA, pseudo-labeling, ensembles | Medium | Medium | Likely useful after measurement validation. Risk: better Dice can still fail to improve the final metric. |
| 7 | External DLTrack data/models, if license and rules permit | Medium | Medium-low | Could improve masks/domain transfer. Risk: license ambiguity and device shift. |
| 8 | Sequence/group consistency and outlier fallback | Small to medium | Medium | Test images look grouped; smoothing can reduce bad outliers, but it needs a decent base signal. |
| 9 | Cheap CSV ablations and column recombinations | Small | High | Worth doing for diagnostics, but not a plan to close the main gap. |

## Rank 1: Full segment-measure-scale pipeline

Hypothesis:

The repo is far behind the DLTrack benchmark mostly because it does not yet have a complete
measurement pipeline. We have partial segmentation and a partial MT calibration. We do not
have broad per-image scale coverage or robust FL. The benchmark score probably comes from a
straight segment-and-measure system that returns all three targets, not from a magic direct
regressor.

Why this could make a significant difference:

- Current submissions still leave FL constant on all rows.
- MT is only calibrated on a subset, and the TIFF subset is suspicious.
- DL_Track_US is explicitly built to output FL, PA, and MT from ultrasound images.
- DLTrack's public benchmark is much better than our constant-FL/mostly-constant-MT files.
- The original method paper describes detecting aponeuroses and fascicle fragments, then
  measuring architecture parameters. That matches this competition's target shape.

Why it might fail:

- The exact benchmark settings may be different from a naive DLTrack run.
- DLTrack's pretrained models may not generalize to all UMUD devices.
- DLTrack's own auto-scaling assumptions may be wrong for some image families.
- A headless port can be tedious because the package is GUI-oriented.

First concrete experiment:

1. Build a small DLTrack reproduction lane, preferably outside the current PyTorch script:
   `umud-muscle-architecture/dltrack_reproduction.md` plus a notebook/script if needed.
2. Run DLTrack or a minimal port on a small subset first: 5 PNGs, 5 TIFFs with visible scale,
   5 TIFFs without obvious scale.
3. Save its overlay output and `Results.xlsx` equivalent.
4. Convert median per-image outputs to Kaggle columns, but do not trust it until overlays
   show sensible geometry and scale.

Success signal:

- We can produce a 309-row CSV with non-constant PA/FL/MT from a reproducible pipeline.
- Overlays show aponeurosis lines, fascicle segments, and thickness/length values that look
  plausible on a random audit sample.
- A public submission approaches the benchmark order of magnitude, even if it does not beat it.

Files likely involved:

- Read/reference: `dltrack_headless_notes.md`, `segment_then_measure.py`, `tick_calibration.py`
- New: a DLTrack reproduction script/notebook, plus a conversion script from DLTrack outputs
  to `image_id,pa_deg,fl_mm,mt_mm`

Importance:

This is the highest-ranked direction because it is the only direction that obviously attacks
all three targets in the same unit system as the leaderboard.

## Rank 2: Visual audit and measurement validation harness

Hypothesis:

The repo is currently making leaderboard moves without enough visibility into whether the
geometry is right. A visual audit loop will reveal concentrated failure modes: wrong ruler,
wrong aponeurosis band, fascicle model following aponeurosis, no fascicles, shifted masks,
bad resizing, or implausible scale.

Why this could make a significant difference:

- The leader-style playbook in `leader_playbook.md` emphasizes looking at predictions.
- DL_Track_US itself outputs a PDF with predicted structures overlaid on images; that is a
  strong hint that visual review is part of the intended workflow.
- The current score cannot tell us which target failed.
- A wrong scale detector can look numerically confident but be totally wrong; overlays catch
  this faster than leaderboard submissions.

Why it might fail:

- Visual inspection does not directly improve the score.
- If the overlay script is too manual or too slow, it may not get used.
- A few audited images can mislead unless the sample is stratified by image family and
  confidence.

First concrete experiment:

Create `visual_audit.py` or extend the current debug output to render one image per test case:

- grayscale ultrasound background
- predicted aponeurosis mask/line
- predicted fascicle mask/line/centerline
- measured MT gap
- measured FL line or curve
- detected scale ticks/ruler marks
- final PA/FL/MT values
- confidence flags and fallback reasons

Then render a stratified panel:

- 10 high-confidence PNGs
- 10 low-confidence PNGs
- 10 TIFFs with scale candidates
- 20 TIFFs with no scale candidate
- largest/smallest predicted MT and FL
- largest disagreements between ExtraTrees PA and U-Net PA

Success signal:

- We can explain most bad-looking predictions by a small list of failure factors.
- Every failed output has a machine-readable reason: no scale, unstable apo fit, unstable
  fascicle fit, out-of-range physiology, etc.
- Future changes can be evaluated by "did the failure factor go away?" instead of only "did
  the public score move?"

Files likely involved:

- `segment_then_measure.py`
- `tick_calibration.py`
- new `visual_audit.py`
- `results/visual_audit/` should stay ignored if it contains generated images

Importance:

This is rank 2 because it is the control panel for every other track. It may even deserve to
be implemented before rank 1 work, but strategically it is an enabler rather than the final
model.

## Rank 3: Per-image scale recovery, especially for TIFFs

Hypothesis:

The biggest missing feature is `px_per_mm`. PNG left-ruler calibration now looks plausible,
but it covers only 58 images. The TIFF family is most of the test set, and the current 10
calibrated TIFF rows all sharing the same `13.45 px/mm` smells like a brittle assumption.

Why this could make a significant difference:

- MT tolerance is only 3 mm, so a correct scale can matter a lot.
- FL and MT both require scale.
- PNG MT calibration produced a real leaderboard improvement while touching only part of MT.
- The DLTrack docs make scaling a first-class option. They do not treat it as optional when
  returning real-world units.

Why it might fail:

- Some TIFFs may be cropped without visible ruler information.
- Metadata may not include enough scale information.
- Sequence borrowing can propagate a bad scale across multiple images if groups are inferred
  incorrectly.
- Public leaderboard feedback cannot tell whether scale or geometry caused a change.

First concrete experiments:

1. Audit the TIFFs by family. For each image, record dimensions, border layout, any tick-like
   structures, OCR-like depth text if present, and sequence group.
2. Build scale detectors per family, not one global detector:
   - right-edge ticks
   - bottom ticks
   - text/depth readout plus known display geometry
   - metadata-derived scale if available
   - sequence borrowing if frames share acquisition settings
3. Add a calibration confidence model that punishes:
   - identical px/mm across images with different visible depth
   - scales outside plausible ranges
   - tick candidates in text panels or frame borders
   - sparse or nonparallel tick marks
4. Keep a fallback to prior MT/FL when scale is low-confidence.

Success signal:

- `px_per_mm` coverage rises beyond PNGs without producing obvious wrong-scale overlays.
- Scale values cluster by device/family/depth in a plausible way.
- Adding calibrated MT on a new family improves or at least does not hurt public score.

Files likely involved:

- `tick_calibration.py`
- `segment_then_measure.py`
- `calibration_verification_notes.md`
- maybe new `scale_family_audit.py`

Importance:

This is rank 3 because it is the difference between image geometry in pixels and leaderboard
answers in millimetres. A better model cannot infer millimetres from a cropped image unless
some scale signal or prior is supplied.

## Rank 4: Fascicle length measurement

Hypothesis:

FL remains a large untapped target. Once scale is available, even a noisy but gated FL signal
may beat a constant prior. However, FL is more fragile than MT because it depends on fascicle
orientation, intersections with aponeuroses, extrapolation, and sometimes curvature.

Why this could make a significant difference:

- FL is still constant in the current best submission.
- The DLTrack benchmark outputs non-constant FL and is far better than our current score.
- A direct line-intersection FL, `MT / sin(PA)` style FL, and median fascicle-fragment FL are
  cheap to compare once scale exists.

Why it might fail:

- Fascicle segmentation is weak and broken; Dice around `0.3` may still be usable for angle
  but unreliable for full length.
- Straight-line extrapolation is biased when fascicles curve.
- `MT / sin(PA)` is sensitive to small angle errors, especially at low PA.
- FL priors may be surprisingly hard to beat if test labels are conservative or noisy.

First concrete experiments:

1. On training masks, compute several FL estimators:
   - straight fitted fascicle line intersecting superficial and deep aponeuroses
   - median of multiple fascicle lines
   - `MT / sin(PA)` using measured MT
   - curve/spline centerline arc length where enough fascicle pixels exist
2. Compare estimators on mask-derived pseudo-labels and visual overlays.
3. In submissions, gate FL separately from MT. It is acceptable to use calibrated MT while
   keeping prior FL on rows with unstable fascicles.

Success signal:

- On train/validation masks, the chosen FL estimator is stable across small perturbations.
- Visual overlays show fascicle lines crossing the muscle belly in the right place.
- A PNG-only direct-FL ablation improves or at least gives interpretable feedback.

Files likely involved:

- `mask_geometry.py`
- `segment_then_measure.py`
- `make_postrun_variants.py`
- new FL estimator tests

Importance:

This is rank 4 because FL is likely score-moving, but it depends on scale and fascicle
geometry. It should not be tuned blindly.

## Rank 5: Geometry-aware fascicle recovery

Hypothesis:

The fascicle problem is not just "segment white pixels." It is a ridge/path/orientation
problem. Bright pixels form local structures; some structures are aponeuroses, some are
fascicles, and their geometry determines the labels. A pure Dice+BCE mask objective may not
reward the final geometry strongly enough.

This is the practical version of the pasted path idea:

```text
ultrasound brightness/probability field
-> local ridge centers
-> local orientation field
-> separate apo-like from fascicle-like structures
-> fit lines or curves
-> compute PA/FL/MT
```

Why this could make a significant difference:

- Fascicle masks are thin and broken; Dice is harsh and not always metric-aligned.
- PA and FL care about orientation and centerline more than fat pixel overlap.
- Naive brightness following will often lock onto aponeuroses unless semantic/orientation
  separation exists.
- DLTrack already exposes settings like fascicle length threshold, pennation angle ranges,
  and orientation-map style options, suggesting geometry filters are expected.

Why it might fail:

- Classical ridge following can become a complex hand-tuned system.
- Speckle noise can produce many false ridges.
- If masks are misaligned or inconsistent, geometry supervision can inherit label noise.
- It cannot solve `px_per_mm` by itself.

First concrete experiments:

1. Postprocess probability masks before changing the neural network:
   - threshold sweep
   - skeletonization
   - connected-component filtering
   - Hough/RANSAC line fitting
   - structure-tensor orientation estimation
   - reject horizontal/thick apo-like ridges for fascicle fitting
2. Create an auxiliary orientation label from fascicle masks:
   - estimate local or image-level fascicle angle from mask geometry
   - encode line orientation as `(cos 2theta, sin 2theta)` because a line has 180-degree
     symmetry
   - train a small auxiliary head only after the postprocess baseline exists
3. Score by derived PA/FL stability, not only Dice.

Success signal:

- Overlays show cleaner fascicle centerlines and fewer apo/fascicle confusions.
- PA variance decreases on obvious bad cases without collapsing every angle to the prior.
- FL line intersections become more plausible.

Files likely involved:

- `mask_geometry.py`
- `segment_then_measure.py`
- `train_segmentation.py`
- possible new `fascicle_geometry.py`

Importance:

This is a real research direction, not a distraction. I rank it below scale/FL because the
leaderboard still needs millimetres, but it is probably the most interesting route to a
novel improvement beyond a basic DLTrack reproduction.

## Rank 6: Robust segmentation training

Hypothesis:

The current U-Net run was a first attempt, not a finished segmentation system. Better folds,
augmentation, TTA, longer training, confidence handling, and possibly pseudo-labeling can
improve the masks and geometry. But this only matters if the measurement layer turns better
masks into better PA/FL/MT.

Why this could make a difference:

- The current fascicle validation quality was weak.
- Public leader-style notebooks in other competitions use folds, TTA, OOF validation, and
  visual review as standard craft.
- DLTrack's own docs warn that generalization may be limited across devices and acquisition
  settings; UMUD has multiple image families.

Why it might fail:

- Better Dice may not improve the leaderboard if the wrong target is optimized.
- More training can overfit to mask quirks or resizing artifacts.
- PA-only improvements appear smaller than scale/FL/MT improvements.

First concrete experiments:

1. Add group-aware validation if image sequences/families can be grouped.
2. Track not just Dice, but derived geometry metrics:
   - PA from validation masks
   - MT_px from validation aponeuroses
   - line-fit residuals
   - component counts
3. Add TTA only with correct inverse transform handling.
4. Try small ensembles only after one model has a validated measurement harness.
5. Consider self-training on high-confidence test masks, but keep a clean record of what was
   pseudo-labeled and why.

Success signal:

- OOF geometry improves, not just mask Dice.
- Test overlays improve on failure classes identified in the audit.
- Public score changes are consistent with the hypothesized target being improved.

Files likely involved:

- `train_segmentation.py`
- `segment_then_measure.py`
- `leader_playbook.md`

Importance:

This is important, but it should not become "train bigger U-Nets forever." It is downstream
of measurement validation.

## Rank 7: External DLTrack data/models

Hypothesis:

The DLTrack OSF bundle may provide pretrained models and image/mask training data that can
improve our apo/fascicle segmentation, especially if UMUD train masks are limited or shifted.

Why this could make a difference:

- DLTrack is built for the same measurement family.
- The docs say training data includes paired images and masks, and the method targets lower
  limb muscles including vastus lateralis.
- Pretrained masks/models might provide a stronger starting point than the current small run.

Why it might fail:

- The OSF data/model license needs direct verification before use in a public Kaggle repo.
- Device shift may make the pretrained models weak on UMUD PNGs or TIFFs.
- External mask data still does not provide measured PA/FL/MT labels.
- Combining external data without group/family awareness can make validation misleading.

First concrete experiments:

1. Verify license and Kaggle rules before committing data or weights.
2. Run the pretrained models on a small UMUD sample and render overlays.
3. If overlays are promising, use them as initialization or as an ensemble member.
4. If overlays are poor, use the data only after domain-matched fine-tuning.

Success signal:

- Pretrained or fine-tuned external models improve visual masks on UMUD without worsening
  geometry.
- The repo can document exactly what external assets were used and under which terms.

Files likely involved:

- docs only at first
- possible ignored external asset directory
- `train_segmentation.py` if fine-tuning is added

Importance:

This is promising but blocked by verification. Do not quietly commit external weights or data.

## Rank 8: Sequence/group consistency and fallback

Hypothesis:

Some test images are probably related frames or device-family groups. Real muscle architecture
should not jump wildly between adjacent frames in the same acquisition. Group smoothing and
fallback can reduce outliers.

Why this could make a difference:

- The docs and prior plan mention 5-frame sequences.
- Outlier control is a common leader-style technique.
- Calibration and segmentation failures are often obvious outliers relative to nearby frames.

Why it might fail:

- Group inference could be wrong.
- Smoothing can erase real differences if groups are not true repeated acquisitions.
- It cannot create missing FL/MT signal by itself.

First concrete experiments:

1. Infer groups from filename order, dimensions, metadata, and visual similarity.
2. For each group, compare PA/FL/MT variance and confidence flags.
3. Smooth only when confidence is low or when one frame is an obvious outlier.
4. Keep target-wise fallbacks: PA can be kept while MT falls back, or vice versa.

Success signal:

- Fallback reduces implausible extremes without collapsing valid variation.
- Public score improves on a controlled ablation.

Files likely involved:

- new `sequence_smoothing.py`
- `segment_then_measure.py` integration only after standalone testing

Importance:

This is useful finishing work, not the core unlock.

## Rank 9: Cheap CSV ablations and column recombinations

Hypothesis:

Some small score is still available by recombining the best PA source with the safest MT
calibration and testing cautious FL variants.

Why this can help:

- PA and MT columns can be recombined without rerunning a GPU model.
- The current `1.09194` run likely paid a small PA penalty by using U-Net PA instead of the
  stronger ExtraTrees PA.
- PNG-only calibrated MT avoids the suspicious 10 TIFF bottom-tick rows.

Why it is minor:

- It still leaves FL constant or mostly untrusted.
- It still touches a limited set of rows.
- The expected effect is probably in the `0.005` to `0.03` range, not a path to the leader.

Recommended order if using submissions:

1. `submission_best_pa_calibrated_mt_png_only.csv`
2. `submission_best_pa_calibrated_mt_png_direct_fl.csv`
3. Avoid trusting all-TIFF calibration until the identical `13.45 px/mm` issue is explained.

Success signal:

- It verifies whether the new MT signal combines with the stronger PA source.
- It gives a small sanity check while the bigger work proceeds.

Files involved:

- `make_postrun_variants.py`
- ignored `results/postrun_variants/`

Importance:

Do these only as probes. They are not the research direction.

## How the pasted path/ridge idea fits

The user's intuition can be stated more formally:

```text
At each local region, ask:
1. Is there a ridge-like bright structure here?
2. What is its centerline?
3. What is its orientation?
4. Is it apo-like or fascicle-like?
5. Does it form a plausible global muscle geometry?
```

This maps to known tools:

- ridge detection
- skeletonization
- random walker or path following
- structure tensor orientation
- Hough transform
- RANSAC line fitting
- graph/path tracing
- active contours
- orientation-field regression

The important caution is that aponeuroses are also bright ridges, usually stronger than
fascicles. A pure "follow brightness uphill" algorithm will often follow the aponeurosis.
So the path idea needs at least one separator:

- semantic separator: background vs aponeurosis vs fascicle
- orientation separator: horizontal/thick/long structures vs diagonal/thin fragments
- anatomical separator: only fascicle-like paths inside the muscle belly between aponeuroses
- learned separator: a segmentation model or orientation head that scores fascicle likelihood

Best first implementation:

Use the existing model probability maps, then apply classical ridge/centerline postprocessing.
Do not start by building a large new neural network. First ask whether centerline postprocess
fixes visible cases.

Best later implementation:

Add an auxiliary orientation head to the fascicle model. Encode orientation as
`(cos 2theta, sin 2theta)`, because a line has 180-degree symmetry. Optimize:

```text
loss = mask_loss + lambda * orientation_loss
```

Then measure whether derived PA/FL improves, not just whether Dice improves.

## What not to do next

- Do not chase the public leaderboard with many tiny CSV variants as the main plan.
- Do not train bigger segmentation models before building overlays and geometry metrics.
- Do not assume better Dice means better UMUD score.
- Do not assume the leader's private method is known.
- Do not commit Kaggle keys, external weights, downloaded datasets, or generated result folders.
- Do not manually label test images. Visual auditing test predictions is fine; editing test
  labels by hand is not.
- Do not treat the DLTrack benchmark as hidden truth. Treat it as a strong public reference
  method family.

## Recommended execution order

This is the order I would actually work, even though the strategic ranking above puts the
complete pipeline first.

### Phase 0: keep the submission loop alive

Run one or two cheap ablations only if useful:

1. best PA + PNG-only calibrated MT
2. best PA + PNG-only calibrated MT + gated direct FL

Expected impact: small. Useful because it tells us whether the `1.09194` run left easy signal
on the table.

### Phase 1: build the visual audit harness

Implement overlays and a failure table. This should happen before more model training. It is
the fastest way to make the next decisions less confused.

Expected impact: no direct score, but high leverage.

### Phase 2: reproduce or approximate DLTrack

Try to get a DLTrack-like full output on a subset and then on all 309 images. The goal is not
to worship DLTrack; the goal is to acquire a reference implementation and see what it does
that we do not.

Expected impact: potentially major if it produces real FL/MT.

### Phase 3: solve scale family by family

PNG left ruler is the first success. TIFFs need their own audit and scale strategy. Reject
low-confidence rows aggressively.

Expected impact: major if coverage expands safely.

### Phase 4: add FL carefully

Test FL estimators on masks and overlays. Use per-row confidence gates. FL may move score a
lot, but it is easy to make worse if fascicles are wrong.

Expected impact: major but uncertain.

### Phase 5: improve fascicle geometry

Only now invest in ridge/centerline/orientation work or auxiliary orientation heads. The
objective should be final geometry, not pretty masks.

Expected impact: medium to major if it fixes identifiable failure classes.

### Phase 6: standard craft

Folds, TTA, self-training, simple ensembling, sequence smoothing, and outlier control. These
are useful once the base pipeline is real.

Expected impact: small to medium, likely important near the benchmark/leader zone.

## Work split for multiple agents

If two agents work at the same time, avoid touching the same files.

Agent A: visual audit and failure taxonomy

- Own files: new `visual_audit.py`, generated ignored `results/visual_audit/`, doc notes.
- Avoid editing: `segment_then_measure.py` unless needed for a tiny export hook.
- Deliverable: an overlay panel and a CSV of failure factors/confidences.

Agent B: DLTrack reproduction

- Own files: new DLTrack reproduction notes/script/notebook.
- Avoid editing: current PyTorch pipeline until the DLTrack output format is understood.
- Deliverable: subset overlays, `Results.xlsx` or equivalent, and a conversion note.

Agent C: scale/TIFF audit

- Own files: `tick_calibration.py` only after reading it carefully, plus a scale audit doc.
- Avoid editing: model training code.
- Deliverable: family-level calibration table and overlay examples for each method.

Agent D: fascicle geometry

- Own files: new `fascicle_geometry.py` or postprocess prototype.
- Avoid editing: training code at first.
- Deliverable: centerline/line-fit overlays and derived PA/FL comparison on train masks.

## My ranked recommendation in one paragraph

The next serious push should not be "one more U-Net" or "one more CSV blend." The next serious
push should be a full measurement pipeline with visual diagnostics: reproduce DLTrack enough
to understand its outputs, build overlays so failures are visible, solve per-image scale
family by family, and only then improve fascicle geometry with ridge/centerline/orientation
methods. Cheap ablations are fine as probes, but the meaningful score gap lives in broad
scale coverage and non-constant FL/MT.
