# UMUD Challenge: Muscle Architecture in Ultrasound Data

Kaggle community competition. https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data
Deadline 2026-11-14. Prizes total CHF 5000. The provisional top three must release FAIR, open-source, reproducible code to be eligible.

## Question

Predict three muscle-architecture parameters for each skeletal-muscle ultrasound image: pennation angle (PA, degrees), fascicle length (FL, millimetres), and muscle thickness (MT, millimetres). One row is one ultrasound image. The metric is the UMUD Score: normalized MAE with tolerances PA 6 degrees, FL 12 mm, and MT 3 mm, plus small deterministic tie-breakers. Lower is better.

## Method

First pass: create a valid constant submission from the real test image IDs using `baseline_constant.py`. Planned next: derive PA/FL/MT from the supplied aponeurosis and fascicle masks, then use or fine-tune DL-Track-US style segmentation models and compute the three measurements geometrically. Pretrained models and public external data are allowed.

## Result

| Approach | Local UMUD Score | Public LB |
| --- | --- | --- |
| Constant sample-submission mean, 251 rows | N/A, hidden labels | Submitted (score pending) |

## Caveat

The first baseline only validates the output pipeline. It is not evidence of model quality. Local scoring requires held-out labels derived from masks or source metadata. Serious segmentation training depends on a GPU. Calibration is the main blocker: test images vary in size with no scale metadata, so fascicle length and thickness in mm cannot be produced without recovering a per-image pixel-to-mm scale; pennation angle is scale-free and is the tractable parameter.

## Lesson

The task is not a normal image-regression CSV problem. The public training supervision is mask-based: 1,048 aponeurosis masks and 2,761 fascicle masks, with 251 hidden-label test images. The main lever is segment-then-measure, not direct tabular target fitting.
