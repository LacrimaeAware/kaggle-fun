# UMUD Challenge: Muscle Architecture in Ultrasound Data

Kaggle community competition. https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data
Deadline 2026-11-14. Prizes total CHF 5000. The provisional top three must release FAIR, open-source, reproducible code to be eligible.

## Question

Predict three muscle-architecture parameters for each skeletal-muscle ultrasound image: pennation angle (PA, degrees), fascicle length (FL, millimetres), and muscle thickness (MT, millimetres). One row is one ultrasound image. The metric is the UMUD Score: normalized MAE with tolerances PA 6 degrees, FL 12 mm, and MT 3 mm, plus small deterministic tie-breakers. Lower is better.

## Method

First pass: create constant and model-based 309-row CSV artifacts. `mask_geometry.py` derives mask-based pseudo-labels for PA, fascicle length in pixels, and muscle thickness in pixels. `train_pseudo_baseline.py` trains an ExtraTrees image-feature regressor against those pseudo-labels. Planned next: segment aponeuroses and fascicles directly, recover tick-mark calibration, then compute PA/FL/MT geometrically.

## Result

| Approach | Local UMUD Score | Public LB |
| --- | --- | --- |
| Constant sample-submission mean, 309 rows, comma CSV | N/A, hidden labels | Not scored |
| ExtraTrees image features on mask-derived pseudo-labels | Pseudo-label PA MAE 3.43 deg; FL/MT worse than median | 1.23135 |
| Model pennation angle, prior FL 74.424 / MT 18.628 | replaces inflated model FL (mean ~100 mm) with the prior | 1.11066 |

## Caveat

The first model uses mask-derived pseudo-labels, not the hidden manual PA/FL/MT labels. Its local numbers measure agreement with extracted mask geometry, not leaderboard quality. Serious segmentation training benefits from a CUDA GPU but the current CPU pipeline is enough to build and test the measurement logic.

Submission mechanism: the competition asks for CSV. The valid IDs are the visible test filenames with their true suffixes: 251 `.tif` rows and 58 `.png` rows. Semicolon CSVs fail column parsing, 251-row TIFF-only CSVs are short, page-style IDs mismatch, and all-`.tif` IDs mismatch.

## Lesson

The task is not a normal image-regression CSV problem. The public training supervision is mask-based: 1,048 aponeurosis masks and 2,761 fascicle masks, with 309 test images. The first CPU model learns pennation angle signal from images, but FL/MT need segmentation plus tick-mark calibration. The main lever is segment-then-measure, not direct tabular target fitting.
