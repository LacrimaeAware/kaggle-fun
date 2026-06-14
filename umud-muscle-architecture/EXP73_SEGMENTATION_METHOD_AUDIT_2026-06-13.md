# EXP73 Segmentation Method Audit

Date: 2026-06-13

Purpose: do the slower audit that should have happened before treating EXP72 as "the" heavy
segmentation answer. This note separates facts, implementation details, known method gaps, and the
next controlled experiment.

## Straight Answer

EXP72 does **not** prove that segmentation is capped.

EXP72 does prove that `seg72_01_soft5_tversky_640_unetpp` is not currently earning GPU time, and that
the EXP72 matrix is not a clean enough experiment to diagnose the thin-line problem.

If `seg72_01` is still running, stop it or let it finish only if you specifically want the partial
artifact. Bundle/download logs, configs, debug masks, and weights. Do not continue the remaining EXP72
runs blindly.

The next notebook should be EXP74: instrumentation plus controlled one-axis ablations. It should not
be another "bigger everything" overnight run.

## Observed Run Facts

| run | pipeline | apo best Dice | fasc/thin-line best Dice | runtime / status | read |
|---|---|---:|---:|---|---|
| `seg59_02_highres_512_unet` | `2026-06-13.02` | `0.7945` | `0.2925` | finished in `33.2 min` | current conservative control |
| `seg72_01_soft5_tversky_640_unetpp` | `2026-06-13.03` | `0.7873` | `0.2594` by fasc epoch 20 | much slower | worse than control in the pasted log |

This is enough to reject continuing `seg72_01` as-is. It is not enough to reject segmentation work.

## Code Audit Facts

These are direct facts from `segment_then_measure.py` and the EXP72 notebook.

1. Most training knobs are global for both targets.

   `UMUD_IMG_SIZE`, `UMUD_MODEL_ARCH`, `UMUD_LOSS_MODE`, `UMUD_AUG_LEVEL`, `UMUD_AUTO_THRESHOLD`, and
   `UMUD_THRESHOLD_SWEEP` are shared by apo and fasc training. Only the target mask mode is cleanly
   target-specific through `UMUD_APO_TARGET_MODE` and `UMUD_FASC_TARGET_MODE`.

   That matters because apo is a broad boundary-band task while fasc is the sparse thin-line task. A
   heavier setting can hurt apo while trying to help fasc, and EXP72 retrains both at once.

2. Soft/dilated targets are training targets, not topology losses.

   `prepare_loss_mask()` converts the augmented binary mask into `soft5`, `soft7`, `dilate3`,
   `dilate_soft5`, etc. for training. This is a reasonable target transform, but it is not clDice,
   boundary loss, skeleton recall, or a distance-transform loss.

3. EXP72 skeletonization is post-hoc and non-differentiable.

   `binarize_prob()` thresholds the probability map, then applies OpenCV morphological skeletonization
   and optional dilation. That hard decoder is used during validation and inference. It does not teach
   the model connectivity during backpropagation.

4. The current validation score is one hard-decoder Dice number.

   In `train_segmenter()`, validation computes binary Dice after the selected decoder. It does not log:

   - threshold-only Dice,
   - skeleton Dice,
   - skeleton-dilate Dice,
   - probability-map quality,
   - connected component count,
   - accepted fragment count after measurement filters,
   - PA/FL/MT downstream measurement distributions,
   - expert-benchmark score with the candidate weights.

   Therefore a worse `fasc val_dice` could mean the probability map is worse, or the hard decoder is
   worse, or the metric is simply misaligned with the downstream line-fitting task.

5. EXP72 changed too many axes at once.

   `seg72_01` changed all of these together:

   - architecture: U-Net -> U-Net++,
   - image size: 512-ish control -> 640,
   - batch size,
   - loss: Dice/BCE -> BCE/Tversky,
   - augmentation: light -> heavy,
   - target: binary -> soft5,
   - thresholding: fixed -> auto sweep,
   - decoder: threshold -> skeleton-dilate,
   - fragment area gate: 40 -> 14.

   A negative result cannot answer which part failed.

6. Debug outputs still miss the most useful diagnostics.

   `UMUD_SAVE_PRED_DEBUG` saves hard masks for the first N test images. It does not save probability
   maps or validation-set decoder sweeps. For thin structures, a probability map can be useful even
   when a hard threshold looks bad.

## Literature Check

Primary sources support the idea that thin-structure segmentation often needs losses or metrics that
care about topology, boundaries, or centerlines. They do **not** say that a hard skeleton postprocess
after thresholding is the same thing.

- clDice defines a centerline Dice metric and soft-clDice differentiable loss for tubular structures:
  https://arxiv.org/abs/2003.07311
- Boundary loss was designed for highly unbalanced segmentation and complements region losses with a
  contour/distance objective:
  https://proceedings.mlr.press/v102/kervadec19a.html
- Skeleton Recall Loss targets connectivity for thin tubular structures without expensive
  differentiable skeletons:
  https://arxiv.org/abs/2404.03010
- Centerline Boundary Dice argues that centerline/topology terms need geometric/boundary detail too:
  https://arxiv.org/abs/2407.01517

Conclusion: EXP72 was a directionally sensible approximation, but not a faithful test of the strongest
thin-structure method family.

## What Was Actually Wrong With EXP72

This is not "we tried improving the old method and hindsight says old was good." The design mistake
inside EXP72 is narrower and more actionable:

1. It mixed method axes, so it cannot identify a cause.
2. It used post-hoc skeletonization instead of training/evaluating with a topology-aware objective.
3. It optimized one hard pixel Dice number instead of the measurement pipeline's component/line
   behavior.
4. It re-risked apo while trying to solve fasc.
5. It did not preserve enough debug state to tell whether the model or the decoder failed.

Those are fixable methodology problems.

## Hypotheses To Test Next

| hypothesis | why plausible | how to falsify quickly |
|---|---|---|
| Hard skeleton-dilate decoding hurt `seg72_01` | it can delete or reshape thin probability mass before Dice and measurement | run the same checkpoint through threshold-only, skeleton, and skeleton-dilate decoders |
| Heavy augmentation hurt sparse fragments | rotations/noise/blur can distort tiny partial lines more than broad apo bands | compare light-vs-heavy with all other knobs fixed |
| Soft targets helped probabilities but not hard Dice | blurred targets reward near misses, but fixed binary Dice may not reveal that | save probability maps and score threshold sweeps / AP-like overlap |
| Apo retraining confounded the run | apo was already near 0.79-0.80 and EXP72 made it worse | reuse/freeze the known-good apo model and retrain/test fasc only |
| Pixel Dice is not selecting leaderboard-good masks | component topology and line fits matter more than raw pixel overlap | log component counts, accepted fragments, PA/FL/MT distributions, expert score |

## Immediate Go / No-Go

- Keep `seg59_02_highres_512_unet` as the control artifact.
- Stop/bundle `seg72_01` unless it has already finished and produced outputs worth inspecting.
- Hold `seg72_02`, `seg72_03`, and `seg72_04`; they inherit the same confounded design.
- Do not generate another overnight notebook until EXP74 instrumentation exists.

## Corrected Next Experiment

Build EXP74 as a controlled ablation, not a heavier matrix. The full plan is in
`EXP74_CONTROLLED_SEGMENTATION_ABLATION_PLAN_2026-06-13.md`.

Minimum EXP74 requirements:

1. Use EXP59 settings as the control: `512`, U-Net, ResNet34, Dice/BCE, light augmentation, no skeleton
   decoder, default area/angle filters.
2. Add target-specific training/reuse so a known-good apo model can be loaded while fasc is tested.
3. Save validation probability maps or enough summary stats to run decoder sweeps offline.
4. Score every candidate by:
   - pixel Dice by decoder and threshold,
   - component count,
   - accepted fragment count,
   - geometry success rate,
   - PA/FL/MT distribution,
   - expert-benchmark score where weights are available.
5. Test one change at a time:
   - baseline + threshold sweep only,
   - baseline + `soft5`, no skeleton,
   - baseline + `dilate3`, no skeleton,
   - decoder-only skeleton/skeleton-dilate on the same checkpoint,
   - only after instrumentation, a true topology/boundary loss.

## Practical Answer For The Current Kaggle Run

If EXP72 is the run still sitting on Kaggle, I would not spend more night-hours on it. The partial
trajectory is worse than the control and the experiment is too confounded to teach us much. Bundle it
so the logs and masks are not lost.

If EXP59 is still running, it is acceptable to let it finish because it is the conservative control
matrix and produces useful comparison artifacts. But do not interpret "more epochs with similar Dice"
as the ceiling until EXP74 checks decoder/probability/geometry diagnostics.

## Privacy / Public Repo Note

This audit contains no private labels, no Kaggle secrets, no local human review annotations, and no
generated images. Keep `results/`, `data/`, OCR caches, trained weights, and target human notes
ignored unless deliberately sanitized.
