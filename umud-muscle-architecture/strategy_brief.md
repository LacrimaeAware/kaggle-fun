# UMUD Strategy Brief

Public-safe working brief for the UMUD Challenge: Muscle Architecture in Ultrasound Data.

Companion docs: [rundown.md](rundown.md) explains the problem in plain language; [leader_playbook.md](leader_playbook.md) distills the reusable method of the current leader (suguuuuu) and maps it onto UMUD.

## Current state

The submission loop works. A first scored submission reached public LB 1.23135, around rank 31 when checked on 2026-06-09. The file that scored was `submission_pseudo_baseline_comma_309.csv`.

The valid submission format is comma-separated CSV with columns:

```text
image_id,pa_deg,fl_mm,mt_mm
```

The valid `image_id` values are the visible test filenames with their real suffixes:

```text
IMG_00001.tif ... IMG_00251.tif
IMG_00252.png ... IMG_00309.png
```

Semicolon CSVs fail because Kaggle does not parse `image_id` as a separate column. TIFF-only submissions fail because the test set has 309 images, not 251. Page-style IDs such as `image_00001` fail.

The current first model is not a serious competition model. It uses mask-derived pseudo-labels and a CPU ExtraTrees image-feature regressor. It learns some pennation-angle signal, but it does not solve fascicle length or muscle thickness.

Held-out pseudo-label result:

```text
PA: ExtraTrees MAE 3.43 deg vs median baseline 4.95 deg
FL: worse than median
MT: worse than median
```

Leaderboard context when checked:

```text
Top public score: 0.37766
DLTrack_0.3.1 benchmark: 0.67944
Current first scored submission: 1.23135
```

Lower is better. This is not a noise-limited leaderboard. There is a large real gap to close.

## What the problem really is

This is not ordinary image regression in the usual Kaggle sense. The task is to recover geometry from ultrasound images.

The targets are:

- `pa_deg`: pennation angle, the angle between a fascicle and the deep aponeurosis.
- `fl_mm`: fascicle length, the length of a fascicle between the superficial and deep aponeuroses.
- `mt_mm`: muscle thickness, the distance between superficial and deep aponeuroses.

The public training data does not include a normal target CSV. Instead it includes segmentation supervision:

- aponeurosis images and masks: 1,048 mask rows
- fascicle images and masks: 2,761 mask rows
- test images: 309 images

So the natural pipeline is:

```text
image -> segment aponeuroses and fascicles -> fit geometry -> compute PA, FL, MT
```

Direct image-to-number regression is possible, but it throws away the structure that makes the problem interpretable.

## The standard baseline

The standard method is DL-Track-US style segment-then-measure:

1. Segment superficial and deep aponeuroses.
2. Segment fascicle fragments.
3. Fit lines or curves to those structures.
4. Compute angle, length, and thickness from intersections and distances.
5. Convert pixels to millimetres using calibration from image scale/tick marks.

The public leaderboard has a `DLTrack_0.3.1_benchmark_solution.csv` entry at 0.67944. That is the first practical target. Getting from 1.23135 to about 0.68 is probably mostly implementation, not novelty.

## Main improvement levers

### 1. Reproduce the DLTrack benchmark

Priority: highest.

Reason: DLTrack 0.3.1 scores 0.67944, far ahead of the current 1.23135. Reproducing it gives a real baseline and validates the measurement pipeline.

Tasks:

- Run or port DL-Track-US inference on the UMUD test images.
- Confirm expected preprocessing and mask resizing.
- Compute PA/FL/MT with its geometry code.
- Submit a DLTrack-style baseline.

Risk: environment friction, TensorFlow/Keras dependencies, and GPU support.

### 2. Fix calibration

Priority: highest for FL and MT.

PA is scale-free. FL and MT require pixels-to-mm. Public discussion says most test images have tick marks, and ambiguous bottom tick marks can be assumed 1 cm apart.

Tasks:

- Detect tick marks automatically.
- Estimate pixel spacing per image.
- Convert fascicle length and thickness from pixels to millimetres.
- Build fallback rules for images where tick detection fails.

Possible methods:

- Crop likely bottom/side ruler regions.
- Use thresholding plus line/tick detection.
- Use Hough lines or connected components.
- Estimate repeated tick spacing via autocorrelation or peak detection.
- Smooth calibration across 5-frame sequences.

This is likely one of the strongest score levers because FL tolerance is 12 mm and MT tolerance is 3 mm.

### 3. Improve segmentation

Priority: high.

The current pseudo-label model does not segment test images. It predicts from coarse image features. Serious progress needs aponeurosis and fascicle masks on test images.

Options:

- Use DLTrack pretrained U-Nets directly.
- Fine-tune aponeurosis and fascicle segmentation models on UMUD masks.
- Train lightweight U-Net / U-Net++ / DeepLab style models.
- Use public external segmentation data such as FALLMUD if allowed and documented.

Important details:

- Aponeurosis masks and images have shape mismatches. Public discussion confirms this is known, and preprocessing must reproduce organizer/DLTrack alignment.
- Do not blindly resize with distorted aspect ratio if the geometry is later measured. If resizing is needed for model input, map predictions back carefully.

### 4. Better geometry

Priority: high after segmentation exists.

Basic DLTrack fits straight lines. Real fascicles can curve, and partial fascicles often need extrapolation.

Ideas:

- Fit robust lines with RANSAC instead of least squares.
- Reject low-confidence or implausible fascicle fragments.
- Fit curves or splines to fascicles and integrate arc length.
- Use multiple fascicles per image and aggregate robustly.
- Estimate uncertainty; fall back to priors when geometry is unstable.

Novel angle: curved fascicle modeling is a plausible real contribution because the known pipeline is mostly straight-line geometry.

### 5. Temporal consistency

Priority: medium-high.

The test set includes 5-frame sequences. Adjacent frames should have similar PA/FL/MT.

Ideas:

- Detect sequence groups from filename order or image similarity.
- Smooth predictions within each 5-frame group.
- Use median or Hampel filtering.
- Smooth calibration as well as final targets.
- Use sequence-level consistency to reject outliers.

This is low-risk post-processing if sequence grouping is reliable.

### 6. Direct-regression ensemble

Priority: medium.

Direct image regression is weaker as a primary method but useful as a complementary model.

Possible model zoo:

- ConvNeXt-Tiny or ConvNeXt-Base
- EfficientNetV2
- Swin-Tiny
- DINOv2 / DINOv3 feature extractor if available
- Simple CNN baseline

Use derived labels from masks for training. Then ensemble with segment-then-measure predictions. The direct regressor may help on images where segmentation fails.

Kaggle craft pattern:

- Build one trustworthy local validation split first.
- Train diverse models that make different errors.
- Average or ridge-stack out-of-fold predictions.
- Avoid over-tuning public leaderboard score.

The stellar-class lesson applies: ensembling reduces variance and can help, but only after each base has real signal.

### 7. Outlier control

Priority: medium.

The metric is mean absolute error normalized by tolerances. Large failures matter.

Ideas:

- Detect failed segmentations.
- Detect impossible geometry.
- Clip to physiological ranges from the competition page:
  - PA: 5 to 45 degrees
  - FL: 30 to 200 mm
  - MT: 10 to 50 mm
- Fall back to robust priors or sequence medians when confidence is low.
- Use per-target confidence rather than all-or-nothing fallback.

## Integrity constraints

Avoid manual labeling of the 309 test images as a shortcut. There is public discussion about manual labeling, but the defensible route is an algorithmic pipeline. Use only competition data, allowed public external data, public pretrained models, and documented code.

If public external data or pretrained models are used, document them.

## Suggested next experiments

### Experiment A: DLTrack reproduction

Goal: reach or approach the 0.67944 benchmark.

Output: `submission_dltrack_reproduction.csv`

Definition of success: a scored submission below 0.9, ideally near 0.68.

### Experiment B: Tick-mark calibration detector

Goal: estimate pixels-per-mm for each test image.

Output:

- calibration table with `image_id`, `px_per_mm`, confidence
- visual QA overlays for a small sample

Definition of success: stable estimates on most test images and sensible fallbacks.

### Experiment C: Aponeurosis segmentation model

Goal: predict superficial/deep aponeurosis masks on held-out training images.

Validation:

- mask IoU/Dice
- derived MT error in pixels
- visual overlays

Definition of success: robust aponeurosis detection across image types.

### Experiment D: Fascicle segmentation model

Goal: predict fascicle fragments.

Validation:

- mask Dice is useful but not enough
- derived PA error should be the main metric
- visual overlays are required

Definition of success: PA derived from predicted fascicles beats the current pseudo-label image-regression PA.

### Experiment E: Segment-then-measure submission

Goal: replace direct-regression submission with real geometry.

Pipeline:

```text
test image -> apo mask -> fascicle mask -> geometry -> tick calibration -> CSV
```

Definition of success: public score below 0.8.

### Experiment F: Ensemble

Goal: combine segment-then-measure with direct regression.

Candidates:

- DLTrack reproduction
- fine-tuned segmentation geometry
- direct ConvNeXt regressor
- direct ExtraTrees/feature model as a weak fallback

Definition of success: improvement over best single model on local validation and public LB.

## Questions for another model

Use these prompts to get useful ideas:

1. Given ultrasound images with aponeurosis/fascicle masks, what is the most robust segment-then-measure pipeline for PA, FL, and MT?
2. How would you automatically detect ruler tick marks in heterogeneous ultrasound images and estimate pixels-per-mm?
3. How should masks with shape mismatch be aligned to original images without corrupting geometry?
4. What validation scheme would avoid leakage when test data includes 5-frame sequences but training metadata has no explicit subject IDs?
5. How would you improve DLTrack-style straight-line fascicle measurement using curves or splines?
6. What outlier detection rules should be applied before falling back to priors?
7. How would you ensemble direct regression and segment-then-measure predictions without overfitting a small public leaderboard?

## Short version

The current score proves the submission loop. The next meaningful step is not a bigger direct regressor. It is to reproduce DLTrack, solve tick-mark calibration, and make a robust segment-then-measure pipeline. The likely first big jump is from 1.23135 toward the DLTrack benchmark at 0.67944. The possible novel work is better calibration, curved fascicles, temporal smoothing, and confidence-aware fallback.
