# UMUD competition reference (official facts + host clarifications)

Canonical record of the competition rules, data facts, and host (Paul Ritsche) forum answers, so we
do not re-derive or misremember them. Sourced from the Kaggle dataset description and the discussion
threads (captured 2026-06-09). Where a fact changes what we should DO, it is flagged **ACTION**.

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
  validation counts as external data and should be declared in the writeup. We do **not** label the
  309 test images. **ACTION:** if we submit, declare "published UMUD expert benchmark used for local
  validation only; test set not labeled."
- **Reproducibility:** a runnable notebook/repo must be shared; the *whole pipeline* is evaluated, not
  just the leaderboard number. A high score from a non-reproducible pipeline does not win.
- Classic CV and DL are both allowed. The provided training labels may or may not be used.

## 3. Scale / calibration  (the #1 leaderboard lever)

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
- **ACTION (highest value):** build a bottom-edge tick detector - read the last ~2 rows, find the
  periodic bright marks, assume 1 cm spacing, derive px/cm. This is the real fix for the 251 TIFFs and
  the actual leaderboard bottleneck, larger than any FL/segmentation refinement.

### 3a. Scale is FOUR device families, not one (empirical, 2026-06-09)

Shape distribution of the 309 test images: (800,1200) x239, (644,1088) x50, (853,1069) x11, and
~8 small (513x465 etc). Crucially the families do NOT share a scale source despite sharing a size:

| family | count | scale source | detector status (exp11, scale_qa.py) |
| --- | ---: | --- | --- |
| PNG left numbered ruler | 58 | left edge numbered ruler | png_left_ruler works (conf ~1.0); gives 150-172 px/cm BUT this may be **2x too high** - tick_mm is assumed 5 mm, and if the ruler ticks are 1 cm the true scale is ~75-85. The benchmark validation showed exactly this 2x (implied x0.5 -> MAE 1.7 px/cm). **OPEN: confirm the PNG ruler tick interval; the current 1.09 submission's PNG MT may be 2x off.** |
| Siemens 800x1200 (German UI, "12L3 Quadriceps") | ~181 | bottom ticks + "X.X cm" label | left strip is a TEXT panel, not a ruler (ungating png_left_ruler reads text-as-ruler = garbage). side/bottom detection is low-confidence (conf 0.27-0.44). Needs a clean bottom-tick reader. |
| 644x1088 (left depth ruler, "50" mm) | 50 | left edge depth ruler to 50 mm | **0 detected** - ticks are fainter than the >150 threshold. Needs a lower-threshold left-ruler reader. |
| cropped (853x1069, 513x465, ...) | ~20 | bottom ticks (1 cm) | bottom-tick detector (scale_ticks.py) works on CLEAN ones (IMG_00040 -> 78 px/cm conf 0.99, green ticks land on the real marks) but fails on faint/content-cluttered ones (IMG_00036 -> garbage 10 px/cm). |

What is SOLID: where a ruler/ticks are cleanly found, spacing detection is accurate (benchmark MAE
1.7 px/cm after the per-family mm-factor; IMG_00040 bottom ticks 78 px/cm validated by eye). What is
HARD: coverage (each family needs its own reader) and pinning the mm-per-tick factor per family
(the 2x trap). No test-set scale labels exist, so per-family readers must be QA'd visually
(results/calibration_qa/). Tools built: scale_ticks.py (bottom-tick reader), experiments/exp11
(coverage), experiments/scale_qa.py (overlays), experiments/scale_probe.py.

**Per-family build plan:** (1) clean bottom-tick reader for Siemens-800x1200 + cropped (~200 imgs,
the bulk) - host says 1 cm so tick_mm=10; (2) lower-threshold left-ruler reader for the 644 family
(50 imgs, depth to 50 mm); (3) re-confirm the PNG ruler interval and fix the possible 2x; (4) gate
each by visual QA overlay before wiring into the submission - a 2x-off scale would tank MT/FL.

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
The latest handoff/context public score after the scale work is **0.619**. Submission-ready changes
are locally auditable but hidden-test gain is not locally decomposable (no test labels). Tools:
scale_ticks.py, experiments/{scale_coverage,scale_qa,siemens_ruler,check_submission}.py (overlays in
results/calibration_qa/).

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
- Read: our measurement, given clean true scale, is strong. The remaining hidden-LB gap needs
  target-set error attribution (scale disagreement, prior/recentering sensitivity, temporal
  consistency, and FL/orientation correctness), not another guessed bottleneck.

## 9. Test-set leakage discussion and our stance

- The 309 test images and the DL-Track manual-analysis method are both public, so a person can manually
  label the test set and use those labels as a **proxy target** for tuning / local validation. Host:
  technically possible, cannot be policed, but it counts as **external data (must be declared)** and
  the final pipeline must still be reproducible. Patrick and TheOneTheOnly are doing variants of this.
- Host's nuance: labels != architecture estimates - even good manual fragment labels still need a
  computational step to get PA/FL/MT, and expert-vs-expert variability remains (the benchmark folder
  shows this). So a hand-labeled proxy is imperfect.
- **Our stance (per the user):** we do NOT label the 309 test images. We validate only on the published
  35-image expert benchmark. Cleaner, declarable, and keeps the pipeline honest and reproducible.

## 10. Immediate implications (ranked)

1. **Bound scale error on the real 309 target images** using two-cue ruler families. Coverage is high,
   but the remaining score gap should be measured, not attributed by assumption.
2. **Audit recentering/prior sensitivity.** The clean benchmark can use its known mean; the hidden
   target mean is unknown. The failed blend proves pinning/mean-stabilized local wins can hurt.
3. **Attack FL/orientation geometry.** FL is `MT / sin(PA)` in disguise, so small PA errors amplify.
   Use robust orientation aggregation and coherence checks before a blind retrain.
4. **Demote augmentation/self-training-for-domain** unless correctness checks point back at
   segmentation. Real train-vs-target image stats do not show a large discrete domain gap.
5. **Declare external data** (the benchmark) and keep the notebook reproducible.
