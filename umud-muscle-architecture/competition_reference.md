# UMUD competition reference (official facts + host clarifications)

Canonical record of the competition rules, data facts, and host (Paul Ritsche) forum answers, so we
do not re-derive or misremember them. Sourced from the Kaggle dataset description and the discussion
threads (captured 2026-06-09). Where a fact changes what we should DO, it is flagged **ACTION**.

Current orientation note (2026-06-14): this file is the host/device-facts reference, not the living
plan. The live leaderboard state and next submission order are in `docs/CURRENT_STATE.md`. In
particular, broad scale overrides are not the current recommendation, and the live public lever is FL
global calibration.

## 1. Task and targets

- One case = one single B-mode ultrasound image. Predict three values per image.
- **PA** (deg) = angle between the fascicle line and the **deep** aponeurosis.
- **FL** (mm) = distance between superficial and deep aponeuroses **along the fascicle line**; linear
  extrapolation is used when the fascicle runs past the image frame.
- **MT** (mm) = perpendicular distance between the two aponeuroses, measured at **3 locations** across
  the muscle width and averaged.
- Test-set physiological ranges (use as clip bounds): **PA 5-45 deg, FL 30-200 mm, MT 10-50 mm**.
- Submission columns: `image_id, pa_deg, mt_mm, fl_mm`.

## 2. Rules that matter

- **"Submissions must be based on novel algorithm/model development, existing code can be re-used in an
  improved version."** Reusing libraries (smp/U-Net) and ideas from DL-Track / UltraTimTrack / DUSTrack
  is explicitly allowed *as an improved version*. Our own segment-then-measure pipeline (TLS fragment
  fitting, length-weighted PA median, TTA, MT/sin(PA) identity, tick calibration, recall-bias+CLAHE
  retrain) is our development. **We have not failed this rule.**
- **External data must be declared.** Using the published 35-image expert benchmark for local
  validation counts as external data and should be declared in the writeup. The host also clarified
  in public discussion that labeling or fine-tuning on the 309 test images is treated as external
  data, not as an automatic disqualification, provided it is declared and the overall pipeline is
  reproducible. **ACTION:** if we submit, declare every external source used, including the benchmark,
  public weights/assets, target pseudo-labels, and any human-created target labels if that path is
  chosen.
- **Public external data and pretrained models are allowed if equally accessible.** The rules allow
  external data/models unless the host specifically prohibits them, provided they are public,
  reasonably accessible, minimal/no cost for all participants, and declared/reproducible. This means
  rules-clean public UMUD/OSF benchmark data, public training data, and public pretrained reference
  weights can be used in a controlled branch. Human-created labels on the 309 test records are a
  special host-discussed case: allowed as declared external data in the host's interpretation, but
  they must not be hidden and they raise a reproducibility obligation.
- **Reproducibility:** a runnable notebook/repo must be shared; the *whole pipeline* is evaluated, not
  just the leaderboard number. A high score from a non-reproducible pipeline does not win.
- Classic CV and DL are both allowed. The provided training labels may or may not be used.

## 3. Scale / calibration

- **Pixels are ALWAYS square**, in both train and test. The host confirmed there are NO images with
  different horizontal/vertical resolution. => a single horizontal tick-mark scale also fixes the
  vertical measurement (MT). This removes the aspect-ratio worry for PA and MT.
- **Test images carry tick marks.** Most have ticks at 2/5/10 mm with numeric labels; some have only
  **bottom-edge tick marks with no labels => assume 1 cm spacing** (host, explicitly, for IMG_00036
  and IMG_00040).
- **VERIFIED locally (2026-06-09):** IMG_00036 (853x1069) and IMG_00040 (513x465) - the cropped TIFFs
  we had written off as "no ruler" - DO have faint vertical tick marks in the **last 1-2 rows** of the
  image. Our earlier scan missed them because it scanned the bottom 10% (dominated by the bright
  aponeurosis) instead of the very bottom edge. So **scale is recoverable for the TIFFs** via bottom
  ticks at 1 cm.
- The **dashed vertical line** (e.g. IMG_00040, right side) is the **tissue gain compensation** curve.
  **Ignore it** - it is not a ruler.
- Full-UI images (e.g. IMG_00001, 800x1200) have a left depth ruler + "X.X cm" label (Codex's
  png_left_ruler path handles this family).
- Some **training** images genuinely have no scale info (host). Test images appear to all carry ticks.
- **ACTION (historical, completed in large part):** build the bottom-edge tick detector family. That
  work is now wired into the router; the remaining scale problem is trusted pixel span on the hard
  cases, not rediscovering that bottom ticks exist.

### 3a. Scale is FOUR device families, not one (empirical, 2026-06-09)

Shape distribution of the 309 test images: (800,1200) x239, (644,1088) x50, (853,1069) x11, and
~8 small (513x465 etc). Crucially the families do NOT share a scale source despite sharing a size:

| family | count | scale source | detector status (exp11, scale_qa.py) |
| --- | ---: | --- | --- |
| PNG left numbered ruler | 58 | left edge numbered ruler | png_left_ruler works (conf ~1.0). This was the old "possible 2x" concern, but it was later resolved: the ruler really is 5 mm minor ticks, so 150-172 px/cm is the correct family scale. |
| Siemens 800x1200 (German UI, "12L3 Quadriceps") | ~181 | bottom ticks + "X.X cm" label | left strip is a TEXT panel, not a ruler (ungating png_left_ruler reads text-as-ruler = garbage). side/bottom detection is low-confidence (conf 0.27-0.44). Needs a clean bottom-tick reader. |
| 644x1088 (left depth ruler, "50" mm) | 50 | left edge depth ruler to 50 mm | **0 detected** - ticks are fainter than the >150 threshold. Needs a lower-threshold left-ruler reader. |
| cropped (853x1069, 513x465, ...) | ~20 | bottom ticks (1 cm) | bottom-tick detector (scale_ticks.py) works on CLEAN ones (IMG_00040 -> 78 px/cm conf 0.99, green ticks land on the real marks) but fails on faint/content-cluttered ones (IMG_00036 -> garbage 10 px/cm). |

What is SOLID: where a ruler/ticks are cleanly found, spacing detection is accurate (benchmark MAE
1.7 px/cm after the per-family mm-factor; IMG_00040 bottom ticks 78 px/cm validated by eye). What is
HARD: coverage on the last stragglers and trusting the pixel span on hard cases. No test-set scale
labels exist, so per-family readers must be QA'd visually (`results/calibration_qa/`). Tools built:
`scale_ticks.py` (bottom-tick reader), `experiments/exp11` (coverage), `experiments/scale_qa.py`
(overlays), `experiments/scale_probe.py`.

**Historical build plan:** most of this is now done. The remaining scale work is better span
validation on the hard rows and avoiding broad field-height overrides until the span is independently
trusted.

### 3b. Validated scale coverage - router built (`scale_ticks.recover_for_image`)

Resolved by READING each family's actual ruler (the only validation without test labels), then building
a per-family router. Each scale below is cross-checked, not assumed:

| family | n | scale | how validated |
| --- | ---: | ---: | --- |
| PNG left ruler | 58 | 150 px/cm | ruler reads 0->4 cm over ~595 px; 5 mm minor ticks (tick_mm=5 is correct - the feared 2x bug does NOT exist) |
| 644x1088 left ruler | 50 | 126 px/cm | left depth ruler 0->50 mm, 1 cm ticks ~126 px; all 50 identical |
| Telemed 800x1200 (English "De 50 mm", right text panel) | 49 | 134 px/cm | bottom ticks AND left 0->50 mm ruler independently agree (~134-140) |
| clean cropped/other | ~10 | bottom ticks | e.g. IMG_00040 -> 78 px/cm conf 0.99, green ticks land on real marks |
| German Siemens 800x1200 ("12L3 Quadriceps", left text panel) | 87 | ~136 px/cm | SOLVED: faint RIGHT-edge 5 mm depth ruler (not the bottom bracket). Interval pinned 3 ways: MT physiology (1cm->49mm absurd, 0.5cm->24.7mm), the "4.5 cm" depth label (ruler span ~141 vs detected 136), 4.5cm/9ticks=5mm. |
| Family-B signature 800x1200 | 41 | 134.5 px/cm | fixed left-margin UI marks identify a family whose bottom-tick scale was validated at 134.5 px/cm; this is an assigned instrument scale, not per-image ruler reading. |

**Coverage: 295/309 = 95% scaled with the current router** (was 58 PNG = 19%). Current method counts
from a direct router run: right_ruler_5mm 87, bottom_ticks 59, png_left_ruler 58, left_ruler_1cm 50,
family_b_signature 41, none 14. The German Siemens "bracket" was a red herring - the real scale is a
faint right-edge DEPTH RULER (dim gray, thr 90, x~1150), read by `recover_scale_right_ruler`
(tick_cm=0.5). NOTE the 800x1200 size hides multiple devices; route by detector/signature, not by
shape alone.

Remaining unscaled **14**: mostly cropped/awkward stragglers and a few failed ruler reads. These fall
back to the constant prior.

**Status: wired** into `segment_then_measure.calibrate_image` (`UMUD_SCALE_ROUTER`, default on).
This section explains the scale facts behind that router. The score later improved beyond the initial
`0.61918` scale-router stage; use `docs/CURRENT_STATE.md` for the live public score and submission
ordering. Tools: `scale_ticks.py`, `experiments/{scale_coverage,scale_qa,siemens_ruler,check_submission}.py`
(overlays in `results/calibration_qa/`).

## 4. Training data structure

- 2761 fascicle images + masks; 1048 aponeurosis images + masks; 309 test images.
- **Image/mask shape mismatch (host-confirmed, by design):** apo images are (800,1200,3) but apo
  masks are (864,1152) - different resolution AND aspect ratio. Host's intended handling: **resize
  BOTH image and mask to a common square SIZE** (their snippet uses 256); they align after the square
  resize (verified by the host's overlay). Our pipeline resizes both to 384 via the same albumentations
  Compose, which is consistent with this. **ACTION (cheap check):** visually confirm our resized
  image/mask pairs overlay correctly, given the differing aspect ratios.
- Masks are binarized at > 0. Apo = 2 long bands; fascicle = sparse fragments.

## 5. Annotation method (this defines the ground truth)

- **Two researchers** annotated manually; their results were **averaged** per image.
- Per image they marked: superficial apo, deep apo, **3 visible fascicles, 3 pennation angles, 3
  muscle-thickness estimates** (MT at 3 width locations).
- Re-evaluated when raters disagreed by **> 10 mm FL, > 4 deg PA, > 1 mm MT** (the inter-rater bands).
- => Ground-truth PA/FL come from ~3 fascicles/image; MT from 3 locations. Our median-of-fragments PA
  and MT/sin(PA) FL are aligned with how the labels were actually made.

## 6. Mask sparsity is intentional (answers the "why only a few fascicles" confusion)

- Host **confirmed**: aponeurosis parts and fascicle fragments were labeled only where **visibility /
  contrast was good enough**; this is intentional and varies image to image. Raters were told to
  annotate **all fascicles with clear contrast** - so the sparsity is a visibility/subjectivity
  artifact, not a bug.
- A lower third bright structure can be **bone**, intentionally NOT labeled as aponeurosis. The focus
  is **superficial muscles only**. => validates our two-band apo assumption AND the user's instinct to
  ignore the parallel "mid-line / third structure" (the exp07/09 min-6-deg fragment filter).
- Host offered to share the labeling script if fuller labels are wanted.

## 7. Devices and domain shift

- Train devices: Siemens Acuson Juniper, Telemed ArtUS EXT-1H, Telemed Echo Blaster 128, Hitachi Aloka
  Alpha-10, Philips HD11.
- Test devices: Siemens Acuson Juniper, Telemed ArtUS EXT-1H, **Philips Lumify** (Lumify is test-only).
- Muscles: train = Vastus lateralis, Gastrocnemius medialis/lateralis, Tibialis anterior; test = VL,
  GM, **Rectus femoris** (RF is test-only).
- Test includes **diseased (cerebral palsy)** subjects, ages 18-28, none overlapping with train.
- => Real domain shift in device + muscle + pathology. Robustness (TTA, calibration) matters.

## 8. Leaderboard reference points

- **Patrick (3rd place) manually annotated the test set** (~1.5 h, custom tool, fitting apo+fascicle
  models by eye) => public LB **0.45871**. A careful by-hand human baseline on the real LB is ~0.46.
- Leader ~**0.378** (prior research notes) - beats the by-hand human baseline.
- Provided DL-Track baseline ~**0.679**.
- Our **local** default full benchmark (35 expert images, true scale): **0.2274** overall
  (pa .1498, fl .3528, mt .1795) via `experiments/score_weights.py` with
  `UMUD_FL_IDENTITY_BLEND=0`. The rejected blend scored 0.1873 locally but worsened public LB
  0.61918 -> ~0.64, so do not treat local FL gains as submission proof.
- The rejected MT vertical-3 probe improved the local benchmark 0.2274 -> 0.2192 by aligning MT with
  the host's straight-line left/middle/right description, but worsened public LB **0.61918 ->
  0.62561** while PA/FL were unchanged. So the old center/perpendicular MT path remains the
  submission anchor.
- The rejected bar-only scale-tail probe worsened public LB **0.61918 -> 0.66711**. It directly
  rescaled four fallback rows and also moved FL broadly through recentering. Do not treat remaining
  tail-scale visual plausibility as submission evidence.
- Read: our measurement, given clean true scale, is strong. The remaining hidden-LB gap needs
  target-set error attribution (scale disagreement, prior/recentering sensitivity, temporal
  consistency, and FL/orientation correctness), not another guessed bottleneck.

## 9. Test-set labeling / oracle discussion and our stance

- This is not a purely theoretical concern. Public discussion/reference notes include a participant
  manually analyzing the test set and reaching **0.45871** public LB, and host/forum clarifications
  make clear that labels, manual analysis, and the final architecture estimates are distinct things.
  This matters scientifically: a human-in-the-loop oracle can be a strong diagnostic, and it explains
  why the "can we supervise this?" question keeps coming back.
- **Correction to the earlier Codex stance:** the host explicitly allowed the theoretical path of
  labeling the 309 test images and fine-tuning on them. The host's condition was that this is a use of
  external data and must be declared; the whole code pipeline is also evaluated. So the correct
  project reading is **not** "manual/oracle target labels are automatically forbidden." It is
  "manual/oracle target labels are a declared-external-data strategy with a reproducibility burden."
- This creates a tension with the broad Kaggle foundational wording about hand labeling test records.
  For this specific competition, the host's public clarification is the most relevant practical
  interpretation, but the safe version is disclosure-heavy: log the labeling protocol, save labels or
  interaction outputs, declare them as external data, and make clear how the final notebook/repo uses
  them.
- There are now two legitimate project modes:
  1. **Automated/no-oracle mode:** use public assets, benchmark data, deterministic pseudo-labels, and
     reproducible training only. This is cleaner scientifically and matches the "no hand labelling"
     leaderboard ethos.
  2. **Declared human-in-loop mode:** ask the user or another annotator to label/verify target
     records, then use those labels for validation, tuning, or fine-tuning. This is competition-
     plausible per the host, but it must be declared and preserved as external data.
- A limited "model asks human right/wrong" loop is still human-created target information. It is not
  magically different from hand labeling just because the model proposes the case. It is allowed only
  under the declared human-in-loop mode.
- Code-generated pseudo-labels remain the cleaner bridge: deterministic detectors can produce
  reproducible weak labels from target images, and they stay in automated/no-oracle mode as long as
  they are not hand-corrected.
- **Current preference unless the user chooses otherwise:** continue automated/no-oracle development
  first, because it is easier to reproduce and compare, but do not tell future collaborators that
  declared target labeling is forbidden. It is an available strategy, not the default one.

## 10. Immediate implications (ranked)

1. **Bound scale error on the real 309 target images** using two-cue ruler families. Coverage is high,
   but the remaining score gap should be measured, not attributed by assumption. Started with
   `experiments/exp19_scale_crosscheck.py`: 114 images expose two strict cues; bottom/signature and
   bottom/left pairs agree within 0.3% after sub-pixel refinement, and PNG cross-checks stay within
   2.6% max. This weakens a broad scale-router-error hypothesis but leaves single-cue/fallback rows
   to audit.
2. **Audit recentering/prior sensitivity.** The clean benchmark can use its known mean; the hidden
   target mean is unknown. The failed blend proves pinning/mean-stabilized local wins can hurt.
3. **Attack FL/orientation geometry.** FL is `MT / sin(PA)` in disguise, so small PA errors amplify.
   Use robust orientation aggregation and coherence checks before a blind retrain.
4. **Demote augmentation/self-training-for-domain** unless correctness checks point back at
   segmentation. Real train-vs-target image stats do not show a large discrete domain gap.
5. **Use supervised/public assets deliberately.** `experiments/exp27_external_asset_inventory.py`
   confirms the repo already has 1048 + 2761 public image/mask pairs, the 35-image benchmark, 309
   target images, and one public pretrained weight file. Missing assets are not the blocker.
6. **Train scale-cue recognition instead of hand-building forever.** `experiments/exp26_scale_cue_pseudolabels.py`
   exports code-generated cue masks/boxes for 299/309 target images from trusted router paths:
   bottom_ticks 59, right_ruler_5mm 87, left_ruler_1cm 50, png_left_ruler 58,
   family_b_signature 41, and bottom_scale_bar_3cm 4.
7. **Treat the first learned cue model as QA, not production.** `experiments/exp28_train_scale_cue_segmenter.py`
   and `experiments/exp29_scale_cue_model_audit.py` show the learned cue path runs and has useful
   weak-label presence signal for left/right rulers, but the full-frame thin-mask model is not good
   enough to replace the deterministic router.
8. **Declare external data** (the benchmark and any public training assets/weights) and keep the
   notebook reproducible.
