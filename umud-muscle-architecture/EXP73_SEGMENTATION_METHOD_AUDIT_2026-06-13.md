# EXP73 Segmentation Method Audit

Date: 2026-06-13

Purpose: slow down and audit the segmentation strategy before generating yet another notebook. The
goal is to separate facts, implementation issues, literature-supported methods, and the next
controlled experiment.

## Immediate Recommendation

Stop `seg72_01_soft5_tversky_640_unetpp` if it is still running and bundle/download whatever logs and
debug outputs exist.

Reason: the partial EXP72 log shows it is slower and worse than the conservative EXP59 control:

| run | pipeline | apo best Dice | fasc/thin-line best Dice | note |
|---|---|---:|---:|---|
| `seg59_02_highres_512_unet` | `2026-06-13.02` | `0.7945` | `0.2925` | finished in `33.2 min`; conservative binary target |
| `seg72_01_soft5_tversky_640_unetpp` | `2026-06-13.03` | `0.7873` | `0.2594` by fasc epoch 20 | much slower; soft target + skeleton-dilate decoder |

This does not prove segmentation cannot improve. It does prove that the first EXP72 formulation is
not currently earning its GPU time.

## Core Audit Finding

EXP72 was directionally aimed at a real problem, but it did **not** faithfully implement the strongest
known thin-structure methods. It implemented a local approximation:

- soft/dilated target masks,
- threshold sweep,
- hard morphological skeletonization after thresholding,
- heavy augmentation,
- larger image sizes and U-Net++.

The literature-backed thin-structure methods are more specific:

- clDice introduces a centerline-aware metric/loss using skeleton/mask overlap and soft skeletonization
  during training, not only a post-processing skeleton step.
- Boundary loss addresses severe class imbalance by optimizing a contour/distance-based objective
  that complements regional Dice/Cross-Entropy.
- Skeleton Recall Loss uses a tubed ground-truth skeleton and soft recall against predicted
  probabilities, avoiding expensive differentiable skeletons while still training against a
  connectivity-aware target.
- Newer centerline-boundary losses explicitly note that centerline-only losses can miss geometric
  detail and need boundary/geometric terms too.

So EXP72 should be treated as a failed approximation, not as a failed test of the whole class of
thin-structure segmentation methods.

Sources:

- clDice: https://arxiv.org/abs/2003.07311
- Boundary loss: https://proceedings.mlr.press/v102/kervadec19a.html
- Skeleton Recall Loss: https://arxiv.org/html/2404.03010v1
- Centerline Boundary Dice: https://arxiv.org/abs/2407.01517

## Implementation / Method Issues Found

### 1. Too many knobs changed at once

`seg72_01` changed architecture, resolution, loss, augmentation, target mode, thresholding, and
postprocessing together. When it underperforms, the result is hard to diagnose.

This is not merely hindsight. It means the run cannot answer the basic question: did soft targets
help, did skeleton decoding hurt, did heavy augmentation hurt, or did U-Net++/640 simply train worse?

### 2. The pipeline cannot cleanly isolate apo and fasc training

`segment_then_measure.py` uses global settings for both tasks:

- `UMUD_MODEL_ARCH`
- `UMUD_IMG_SIZE`
- `UMUD_LOSS_MODE`
- `UMUD_AUG_LEVEL`
- `UMUD_AUTO_THRESHOLD`

But the two tasks are different. The apo task is a broader band/edge segmentation problem; the fasc
task is the sparse thin-line problem. The code now supports target modes separately, but not
architecture/loss/augmentation/epoch isolation by target. That is a real experimental-design
limitation.

Next code should allow either:

- reuse/freeze/load a known-good apo model and retrain only fasc, or
- target-specific env vars like `UMUD_APO_*` and `UMUD_FASC_*`.

### 3. Post-hoc skeleton decoding is not the same as a topology-aware loss

EXP72 validates by:

1. predicting a probability map,
2. thresholding it,
3. skeletonizing it,
4. optionally re-dilating it,
5. scoring binary Dice against the original mask.

That can easily delete useful probability mass. The thin-structure literature supports skeleton or
centerline information inside the loss/metric, not necessarily this hard post-process. This is the
most likely specific method flaw in `seg72_01`.

### 4. Soft target training is evaluated only through a hard binary decoder

For `soft5`, the model is trained against a blurred target. Validation still scores against the
original binary target, after hard thresholding and skeletonization. This is a valid possible design,
but it is incomplete because we did not also log:

- threshold-only Dice,
- skeleton Dice,
- skeleton-dilate Dice,
- raw probability overlap / average precision,
- line-component geometry quality.

Therefore the current validation number cannot tell whether the probability map improved but the
decoder ruined it.

### 5. We still lack a validation metric aligned with the leaderboard measurement

The model is optimized/selected by pixel Dice. The submission score is geometry after connected
components, line fitting, calibration, FL aggregation, and temporal/scale post-processing. A mask can
have similar Dice but different component topology and downstream measurements.

For segmentation work, every candidate should log at least:

- validation Dice,
- component count distribution,
- accepted fragment count,
- geometry success rate,
- PA/FL/MT output distribution,
- expert-benchmark score with the candidate weights where feasible,
- side-by-side debug masks for fixed target images.

## What This Says About The Project Wall

The project is not pure coin-flipping. Some improvements were real:

- temporal smoothing and scale routing transferred publicly;
- subpixel/shape-neighbor scale stacked into the current best `0.58910`;
- scale/text/depth tooling is now much better understood.

But the 35-image expert benchmark over-rewarded geometry changes that did not transfer. This created
a false sense that robust triangle, visibility FL, vertical MT, and other local fixes were closer to
submission-ready than they were.

The current public evidence says:

- narrow sequence/scale fixes can transfer;
- broad geometry proxies have not transferred;
- broad field-depth scale override failed;
- conservative segmentation has not yet obviously improved thin-line validation;
- EXP72's first heavy approximation underperforms the conservative control.

## Corrected Next Experiment

Do not generate another all-in-one heavy notebook. The next notebook should be a controlled thin-line
ablation, and it should not retrain/re-risk the apo side.

### Step 1 - Instrument before training more

Add or run a scoring script that can compare, on the same validation split/checkpoint:

- threshold-only vs skeleton vs skeleton-dilate decoding,
- multiple thresholds,
- component/fragment counts,
- downstream geometry outputs.

This should be done before another overnight run if possible.

### Step 2 - Freeze the baseline shape

Use the EXP59 control settings as the base:

- `IMG_SIZE=512`
- `MODEL_ARCH=unet`
- `LOSS_MODE=dice_bce`
- `AUG_LEVEL=light`
- no skeleton postprocess
- only threshold sweep as the first isolated change.

### Step 3 - Test one axis at a time

Suggested controlled matrix:

1. baseline binary target + threshold sweep only;
2. baseline + `soft5`, no skeleton;
3. baseline + `dilate3`, no skeleton;
4. baseline + boundary/distance loss if implemented;
5. baseline + skeleton-recall-style loss if implemented;
6. only after those, test skeleton decoding as a separate decoder ablation.

### Step 4 - Implement literature-backed loss only after instrumentation

The highest-priority method to implement is not another post-hoc skeleton decoder. It is a
connectivity-aware training loss:

- skeleton-recall-style loss on a tubed ground-truth skeleton, or
- soft-clDice/clDice-inspired loss if compute is acceptable,
- optionally combined with Dice/BCE and a distance/boundary term.

The key is: the skeleton/centerline signal belongs in training/evaluation, not only after thresholding.

## Current Go / No-Go

- `seg59_02_highres_512_unet`: keep as a control artifact.
- `seg72_01_soft5_tversky_640_unetpp`: stop/bundle; not worth continuing unless it suddenly exceeds
  the `0.2925` fasc Dice control.
- Remaining EXP72 runs: hold. They inherit the same too-many-knobs design problem.
- Next generated notebook: should be EXP74, a controlled thin-line ablation, after instrumentation is
  added or explicitly planned.
