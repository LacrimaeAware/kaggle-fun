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
- Our **local** benchmark (35 expert images, true scale): **0.368** overall (pa .164, fl .449, mt .490)
  - but that is a different image set evaluated with TRUE scale; our actual LB is gated by TIFF scale.
- Read: our measurement, given scale, already matches/beats DL-Track and approaches the by-hand human
  number. **Solving TIFF scale is what would convert that into a real LB position.**

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

1. **Bottom-tick scale recovery for the test TIFFs.** The ticks are there (verified). Square pixels
   mean one scale fixes MT too. This is the real leaderboard unlock - bigger than the FL retrain.
2. **GPU retrain (recall bias + CLAHE)** - the fascicle-mask / FL lever, orthogonal to scale, evaluated
   locally on the 35-expert board with zero submissions.
3. **Declare external data** (the benchmark) and keep the notebook reproducible.
4. Cheap sanity check: confirm resized image/mask pairs align despite the aspect-ratio mismatch.
