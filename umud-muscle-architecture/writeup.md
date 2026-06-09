# UMUD Challenge: Muscle Architecture in Ultrasound Data

Kaggle community competition. https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data
Deadline 2026-11-14. Prizes total CHF 5000. The provisional top three must release FAIR, open-source, reproducible code to be eligible.

## Question

Predict three muscle-architecture parameters for each skeletal-muscle ultrasound image: pennation angle (PA, degrees), fascicle length (FL, millimetres), and muscle thickness (MT, millimetres). One row is one ultrasound image. The metric is the UMUD Score: normalized MAE with tolerances PA 6 degrees, FL 12 mm, and MT 3 mm, plus small deterministic tie-breakers. Lower is better.

## Method

First pass: create constant and model-based 309-row CSV artifacts. `mask_geometry.py` derives mask-based pseudo-labels for PA, fascicle length in pixels, and muscle thickness in pixels. `train_pseudo_baseline.py` trains an ExtraTrees image-feature regressor against those pseudo-labels. Second pass (done): `segment_then_measure.py` trains aponeurosis and fascicle U-Nets (smp, ResNet34, Dice+BCE) on a Kaggle GPU, predicts masks on the 309 test images, fits lines, and computes pennation angle geometrically; fascicle length and thickness stay at the prior because pixels-to-millimetre calibration is not built yet. Planned next: recover tick-mark calibration so FL and MT stop being constants.

## Result

| Approach | Local UMUD Score | Public LB |
| --- | --- | --- |
| Constant sample-submission mean, 309 rows, comma CSV | N/A, hidden labels | Not scored |
| ExtraTrees image features on mask-derived pseudo-labels | Pseudo-label PA MAE 3.43 deg; FL/MT worse than median | 1.23135 |
| Model pennation angle, prior FL 74.424 / MT 18.628 | replaces inflated model FL (mean ~100 mm) with the prior | 1.11066 |
| Segmentation U-Net pennation + prior FL/MT | PA from predicted-mask geometry: mean 13.65, std 3.74 deg | 1.12324 |
| U-Net pennation + calibrated MT (68 imgs), prior FL | MT from apo gap times ruler scale; PNG MT median 20.7 mm | 1.09194 |

## Caveat

The first model uses mask-derived pseudo-labels, not the hidden manual PA/FL/MT labels. Its local numbers measure agreement with extracted mask geometry, not leaderboard quality. Serious segmentation training benefits from a CUDA GPU but the current CPU pipeline is enough to build and test the measurement logic.

Submission mechanism: the competition asks for CSV. The valid IDs are the visible test filenames with their true suffixes: 251 `.tif` rows and 58 `.png` rows. Semicolon CSVs fail column parsing, 251-row TIFF-only CSVs are short, page-style IDs mismatch, and all-`.tif` IDs mismatch.

## Lesson

The task is not a normal image-regression CSV problem. The public training supervision is mask-based: 1,048 aponeurosis masks and 2,761 fascicle masks, with 309 test images. The first CPU model learns pennation angle signal from images, but FL/MT need segmentation plus tick-mark calibration. The main lever is segment-then-measure, not direct tabular target fitting.

Update after the segmentation run: swapping the regressor's pennation for segmentation-geometry pennation moved the score the wrong way (1.11066 to 1.12324). The only column that changed was pennation; fascicle length and muscle thickness were identical constants in both files. A working hypothesis is that the segmentation models were undertrained (fascicle Dice around 0.25), so their per-image angles are more spread out (std 3.74 deg) and the tolerance-normalized MAE penalizes confident wrong angles more than the regressor's mean-reverting ones. What would falsify it: a better-trained segmentation (higher Dice, test-time augmentation, more epochs) that scores below 1.11066. Either way, pennation is the smallest of the three levers and looks near its floor; fascicle length and muscle thickness are still flat constants and account for most of the gap to the 0.679 benchmark, so pixels-to-millimetre calibration is the most likely place the leverage sits.
