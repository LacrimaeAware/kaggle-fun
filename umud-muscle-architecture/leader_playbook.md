# Leader playbook: learning from suguuuuu (sugupoko)

The current UMUD public-leaderboard leader is suguuuuu (Kaggle handle sugupoko) at 0.37766, marked "no hand labelling", well ahead of the DL-Track benchmark at 0.67944. He is a Kaggle competitions grandmaster with first-place finishes in imaging and signal competitions.

Honesty note: he has not published his UMUD solution. The notebooks of his that are public are from other competitions (a Pokemon match-prediction baseline, a prostate-epithelium segmentation baseline, his HMS and other solutions). So nothing below is his UMUD code. It is his reusable method, read from his public notebooks and write-ups, mapped onto UMUD.

## His public code shows a reusable segmentation recipe

From his SMP U-Net baseline (written for a prostate-epithelium segmentation challenge, but the recipe is general):

- Library: segmentation_models_pytorch (smp). Model: U-Net with a ResNet34 encoder, ImageNet-pretrained, one output channel, sigmoid.
- Loss: a Dice plus BCE combination.
- Augmentation: albumentations, horizontal and vertical flips, 90-degree rotations, ImageNet normalization.
- Optimizer AdamW, CosineAnnealingLR, threshold 0.5 at inference, RLE output.

This is the template for UMUD's aponeurosis and fascicle segmentation: swap the prostate data for the UMUD apo/fasc image-and-mask pairs and keep the smp U-Net plus Dice-BCE plus flips-and-rotations recipe. It is a known-good, low-friction starting point on a CUDA GPU, and runnable (slower) via DirectML or CPU.

From his neural-net baselines, the general training craft: 5-fold StratifiedKFold, out-of-fold predictions, averaging the per-fold test predictions, mixed precision, early stopping on the validation metric, cosine learning rate, and saving the best-on-validation state per fold.

## His grandmaster write-ups add the higher-level playbook

From his 1st, 3rd, and 4th-place solutions (HMS brain activity, RSNA lumbar spine, PlantCLEF, pig posture):

- Multi-stage segment-then-measure. RSNA: detect keypoints, crop, then classify. UMUD has the same shape: segment aponeuroses and fascicles, then measure geometry. Build the stages separately and validate each one.
- Look at the data and the predictions before tuning. He renders the worst-loss validation cases and reads them. For UMUD, overlay the predicted aponeurosis and fascicle lines on the image and check by eye. That is how he found a colour shift and a flip bug in his pig solution.
- Domain-shift augmentation. The single biggest gain in his pig solution was a custom desaturation augmentation matched to the test images' colour. UMUD has device and appearance shift between training and test; augment for it.
- Self-training on the test set. His PlantCLEF solution domain-adapted on test-tile pseudo-labels, FixMatch-style, with hard labels and a high confidence threshold. For UMUD, pseudo-label confident test segmentations and fine-tune toward the test domain.
- Test-time augmentation with label-aware remapping. Flips and rotations at inference, averaging logits, watching for symmetry that needs a remap (a missed vertical-flip remap cost him a large drop until fixed).
- Ensemble by the cheapest diversity knob first. He reaches for the same model with a different hyperparameter or seed before a different architecture, and uses equal-weight averaging rather than out-of-fold-tuned weights, which cost him leaderboard score more than once.
- Anchor on the honest validation signal. When out-of-fold and leaderboard scores disagree near the top, he anchors on the leaderboard for ensemble decisions and keeps a strong baseline visible.
- Outlier and quality control. His muscle cross-section tool discards a few percent of unreliable predictions. For UMUD, detect failed segmentations and fall back to priors.

## Mapped to concrete UMUD actions

1. Train aponeurosis and fascicle segmentation: smp U-Net (ResNet34, ImageNet), Dice-BCE, flips and rotations, 5-fold, mixed precision. Validate by Dice and, more importantly, by the PA and MT error derived from the predicted masks.
2. Run inference on the test images, derive geometry (the existing mask_geometry.py), calibrate with the tick marks, and submit. First target: pass the DL-Track 0.679 benchmark.
3. Add test-time augmentation (flips and rotations) on the segmentation, with any needed symmetry handling.
4. Self-train: pseudo-label confident test segmentations, fine-tune toward the test domain.
5. Ensemble the segment-then-measure pipeline with a direct regressor, and, if time allows, a second segmentation backbone.
6. Outlier control: clip to physiological ranges and fall back to sequence medians where geometry is unstable.

The honest summary: the leader's edge is not a secret method. It is disciplined segment-then-measure with a standard segmentation backbone, validation by looking at predictions, domain-shift handling, self-training, and clean ensembling. All of it is documented and reproducible, which suits this repository's standards.

## 2026-06 update: calibration is the most likely single edge

A web check could not reach his actual UMUD notebook or the competition discussion (Kaggle was behind a browser check), so this stays a hypothesis. What was verifiable points hard at calibration:

- DL-Track-US (the 0.679 benchmark tool) uses a manual scaling tool: a human enters the scale or clicks a known distance. UMUD requires a fully automated prediction over 309 images, so the benchmark run almost certainly applied one fixed/assumed scale, which is wrong for images at different depths and inflates fascicle length and muscle thickness error.
- DL-Track's published accuracy is roughly 5 mm fascicle length, under 1 mm thickness, under 1.5 degrees angle. A rough decomposition of the leader's 0.378 lands near those figures, which is only reachable with a correct per-image scale.

So the leading hypothesis is that his edge is automated per-image pixels-to-millimetre calibration (tick-mark detection), not pennation. Pennation is scale-free and is the smallest of the three levers. The whole gap from a constants-only score (~1.11) to the benchmark (~0.68) is the value of real per-image fascicle length and thickness. We already segment the aponeuroses, so muscle thickness in pixels is measurable now and waits only on scale. The concrete next step is a tick-mark detector returning pixels-per-millimetre per image, validated against the "bottom ticks about 1 cm apart" assumption on real images.
