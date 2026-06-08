# Plan: UMUD Muscle Architecture

A staged plan for the UMUD Challenge (Muscle Architecture in Ultrasound Data). Deadline 2026-11-14. The task is image regression: predict pennation angle (degrees), fascicle length (mm), and muscle thickness (mm) for each ultrasound image. Metric: UMUD Score, a tolerance-normalized mean absolute error across the three parameters, lower is better. Pretrained models and public external data are allowed. The provisional top three must release FAIR, open-source, reproducible code, which suits an open repository.

## Confirmed data and metric

- Kaggle file manifest: 7,930 files. The visible structure is `apo_imgs_v1` (1,049 files, including one `Thumbs.db`), `apo_masks_v1` (1,048 masks), `fasc_imgs_v1` (2,762 files, including one `Thumbs.db`), `fasc_masks_v1` (2,761 masks), `test_images_v2` (251 test images, IMG_00001 to IMG_00251), and `sample_submission.csv`.
- There is no target CSV in the competition files. The training supervision is segmentation-style: aponeurosis images with aponeurosis masks, and fascicle images with fascicle masks. Numeric PA/FL/MT labels must be derived geometrically from predicted or supplied masks.
- Sample submission uses semicolon-separated columns: `image_id`, `pa_deg`, `fl_mm`, `mt_mm`.
- UMUD Score tolerances from the official scorer: PA 6 degrees, FL 12 mm, MT 3 mm, equal weights. The score is normalized MAE with tiny MedAE and RMSE tie-breakers. Lower is better.
- EDA (eda.py): images are uint8 .tif, LZW-compressed (read with OpenCV; tifffile needs the imagecodecs package). Test image sizes and channels vary (for example 800x1200 RGB, 853x1069 grayscale, 644x1088, 513x465), and there are no TIFF resolution tags.
- Calibration is the central difficulty. Pennation angle is scale-free and recoverable. Fascicle length and thickness in mm need a per-image pixel-to-mm scale that the files do not provide; it would have to come from a scale bar in the image or external metadata. This is the main open problem and the likely contribution opening.
- Masks are not pixel-aligned to their images (for example an 800x1200 image with an 864x1152 aponeurosis mask, a different aspect ratio), so masks must be registered or resized with care, and uniform resizing distorts angles when the aspect ratio differs.

## Prior art and method

The organizers' own open-source tool, DL-Track-US (Ritsche, Seynnes, Cronin; JOSS 2023; https://github.com/PaulRitsche/DL_Track_US), defines the standard approach, and it matters here because Paul Ritsche is also the UMUD organizer.

- Method: two U-Net segmentation networks (VGG16 encoder, ImageNet-pretrained), one for the aponeuroses and one for the fascicles, producing binary masks. The superficial and deep aponeuroses are the upper and lower contours, each fit with a line. Fascicle fragments are fit with lines, extrapolated, and intersected with the aponeurosis lines. Fascicle length is the distance between the two intersections; pennation angle is the slope difference between the fascicle and the deep aponeurosis; muscle thickness is the perpendicular distance between the aponeurosis lines. Pixels are converted to millimetres by a calibration factor (scale bar).
- The tool is Apache-2.0 and ships pretrained models and training data. The competition allows pretrained models and public external data, so reproducing or fine-tuning DL-Track-US is a legitimate and likely strong starting point, not a shortcut to avoid.
- FALLMUD (https://www.kaggle.com/datasets/angeliqueloesch/fallmud) is a public dataset with fascicle and aponeurosis segmentation masks, usable as external training data.
- The pretrained models were trained on vastus lateralis, gastrocnemius medialis, tibialis anterior, and soleus across four devices, and the authors note they may fail on other muscles or devices. The likely real work is adapting to the test set's muscles and devices (domain shift), plus leakage-aware validation.
- Error context for the metric: automated-versus-manual agreement is typically within about 1 to 2 degrees for pennation, 2 to 5 mm for fascicle length, and under 1 mm for thickness; typical magnitudes are fascicle length 54 to 69 mm, pennation 12 to 20 degrees, thickness 13 to 17 mm. The official tolerances are 6 degrees, 12 mm, and 3 mm.
- Validation evidence: subject and video-frame leakage materially inflate scores (improper splitting inflates metrics by 5 to 30 percent in analogous consecutive-frame imaging). Split by subject or sequence, never randomly across frames.

## Constraints and dependencies

- Serious deep-learning training needs a GPU. Data understanding, metric reconstruction, a format baseline, and the validation design can start on CPU.
- The current public files do not expose subject or sequence identifiers in a CSV. Group-aware validation may require recovering groups from source metadata, filename order, UMUD/OSF metadata, or downloaded image metadata.
- Full training uses image/mask data rather than a compact table. The first code path should keep data and results gitignored, but keep scripts reproducible from the Kaggle files.

## Stages

0. Data and metric. Done for the first pass: file structure, sample submission schema, 309 test image IDs, and UMUD Score tolerances are known. `baseline_constant.py` creates a correctly shaped dry-run submission from the cached manifest.
1. Baselines. (a) Format baseline: constant PA/FL/MT values over all test images, mainly to validate the submission pipeline. (b) Classical mask-to-measure baseline: derive PA/FL/MT from supplied masks to establish geometry code and plausible target distributions. (c) Direct CNN regression baseline only after numeric labels are derived or sourced.
2. Validation. Subject- or sequence-grouped K-fold. If local and leaderboard scores diverge, fix the split before anything else.
3. The domain-principled approach. Use or fine-tune DL-Track-US: run the pretrained aponeurosis and fascicle U-Nets, adapting to the test muscles and devices, then compute PA, FL, and MT geometrically. FALLMUD provides external segmentation-mask training data for fine-tuning. Compare against the direct-regression baseline. This is where domain structure, not a bigger model, can help.
4. Refinement. Ultrasound-appropriate augmentation, temporal stability across the 5-frame sequences, and a small ensemble. Keep the repository FAIR and reproducible from the start (required for prize eligibility and consistent with the project's standards).

## Improvements and novelty

Grounded in the prior-art research. The standard pipeline (DL-Track-US) has acknowledged limitations, which are the openings.

- Fascicle curvature. DL-Track-US fits straight lines and extrapolates; its authors state they are working on curvature handling. Real fascicles curve, especially at higher activation, so straight-line fascicle length is biased for curved geometries. Fitting curves (polynomial or spline) to the fascicle segments and integrating arc length is a concrete, organizer-acknowledged open problem and the most genuinely novel angle.
- Calibration (pixels to millimetres). Pennation angle is scale-free, but fascicle length and thickness need a pixel-to-mm scale. If the test images carry no scale bar, this must be inferred (image metadata, a learned estimate, or a global calibration from the training distribution). Handling it well is likely a major score differentiator and is the first thing to resolve from the data.
- Temporal consistency. The test set includes 5-frame sequences. Predicting per frame and enforcing consistency across a sequence (smoothing or a sequence model) should cut variance, and the data is structured to reward it.
- Direct regression as a complementary model. The supervision is masks, but numeric PA/FL/MT can be derived from the masks geometrically, which then allows training a direct image-to-(PA, FL, MT) regressor. It makes different errors from the segment-then-measure pipeline, so an ensemble of the two may beat either.
- Outlier control. Related tools discard several percent of predictions as unreliable. The metric is mean-absolute-error based, so detecting low-confidence images (faint or out-of-frame aponeuroses) and falling back to the prior on those reduces the large errors that dominate the mean.
- Domain adaptation. The pretrained segmentation was trained on four muscles and four devices; the test may differ. Augmentation for device and appearance shift, or self-training on the test images, addresses the gap.

Interpretability is a side benefit: segment-then-measure lets a predicted aponeurosis and fascicle be drawn on the image and checked by eye, which suits a defensible, reproducible submission.

## Why this competition

Unlike a near-solved tabular leaderboard, this is a fresh open benchmark with a structured task (segment-then-measure), a long runway, and room for domain insight rather than only ensembling craft. It is a vehicle for learning the image-competition skills (fine-tuning, augmentation, leakage-aware cross-validation, segmentation) on a problem with medical meaning.
