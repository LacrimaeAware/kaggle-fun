# EXP75 External Ultrasound and Kaggle Method Review

Date: 2026-06-14

Status: research review and next-action plan.

## Straight Answer

Before this pass, the project had not done enough broad external research. EXP73 looked at thin-line
segmentation losses, but it did not adequately cover muscle-ultrasound measurement pipelines, scale
asset detection, Kaggle segmentation practice, pseudo-labeling discipline, or self-supervised
ultrasound pretraining.

This review changes the next plan. Do not simply run EXP72 longer. The next serious work should be:

1. build an independent classical ultrasound line-extraction harness;
2. upgrade EXP74 into a Kaggle-grade controlled segmentation and diagnostics notebook;
3. test in-domain self-supervised pretraining before another blind long GPU run;
4. treat scale as an auxiliary detection/imputation problem with confidence and OOF checks, not a
   broad field-height override.

## What The External Sources Say

### 1. Muscle-ultrasound measurement is not just segmentation

DL_Track uses U-Net-style networks to detect aponeuroses and multiple fascicle fragments, then
computes muscle thickness, pennation angle, and fascicle length from those detections.

Source: [DL_Track paper, arXiv 2009.04790](https://arxiv.org/abs/2009.04790)

Implication for UMUD:

- Better masks matter, but the measurement layer after the masks still matters.
- Our viewer and benchmark work should keep showing the actual measured lines, accepted fragments,
  rejected fragments, and final aggregation, not just mask Dice.

### 2. The non-crossing and dominant-orientation story has external support

An Attention U-Net muscle-ultrasound pipeline describes this sequence:

- isolate the inner muscle section;
- apply CLAHE;
- use an elongated/ridge-like structure filter;
- cluster/threshold the likely fascicle pixels;
- remove tiny structures;
- skeletonize and fit line segments;
- connect collinear segments;
- choose the dominant orientation cluster by total segment length;
- extend lines to the muscle boundaries;
- if two extended lines intersect inside the muscle, remove the one whose orientation differs most
  from the median line orientation.

Source: [Automatic Extraction of Muscle Parameters with Attention UNet in Ultrasonography](https://www.mdpi.com/1424-8220/22/14/5230)

Implication for UMUD:

- The user's "coherent story" intuition is not random. It resembles published classical extraction
  logic.
- Our earlier wave/non-crossing implementation was too unstable to trust, but the concept should not
  be discarded.
- The next implementation should be a separate harness with inspectable steps, not a hidden
  postprocess bolted into `segment_then_measure.py`.

### 3. Small PA errors can create large FL errors

A dynamic ultrasound tracking paper reports that a 1.4 degree change in initialized pennation angle
changed fascicle length by several millimeters and substantially changed downstream error.

Source: [Automated Method for Tracking Human Muscle Architecture on Ultrasound Scans during Dynamic Tasks](https://www.mdpi.com/1424-8220/22/17/6498)

Implication for UMUD:

- The user's concern was valid: FL is not independent from PA and boundary geometry.
- Benchmark tables that say "FL is wrong" are incomplete. We need to label whether FL is wrong
  because of scale, PA, upper boundary, lower boundary, fragment support, multi-band routing, or mask
  quality.

### 4. In-domain ultrasound pretraining is a real candidate

Masked pretraining of U-Net on ultrasound images reports strong gains in low-label ultrasound
segmentation settings by first training the network to reconstruct masked image content, then
fine-tuning for segmentation.

Source: [Masked pretraining of U-Net for ultrasound image segmentation](https://www.nature.com/articles/s41598-025-11688-2)

Implication for UMUD:

- A better overnight notebook should not only be "more epochs."
- We can pretrain on all available unlabeled ultrasound-looking frames, then fine-tune on the provided
  masks.
- This is especially attractive for fascicle masks, where validation Dice is weak and labels are thin.

### 5. Augmentation should be controlled, not maximal

A 2025 ultrasound augmentation review found that ordinary image augmentations can help ultrasound,
but segmentation gains are task-sensitive and performance can decline after adding too many
augmentations.

Source: [Revisiting Data Augmentation for Ultrasound Images](https://arxiv.org/html/2501.13193v1)

Implication for UMUD:

- EXP72 being "heavier" did not make it more scientific.
- EXP74 should test augmentation families one at a time and record mask diagnostics plus downstream
  geometry, not only validation Dice.

### 6. Kaggle practice says to add diagnostics, folds, pseudo-label discipline, and auxiliary models

Kaggle segmentation writeups repeatedly mention CLAHE/preprocessing, class-balanced sampling,
loss combinations, threshold tuning, overlap tiles, TTA, pseudo-labeling, folds, and ensembles.

Sources:

- [Image Segmentation Tips and Tricks from 39 Kaggle Competitions](https://dev.to/jakubczakon/image-segmentation-tips-and-tricks-from-39-kaggle-competitions-l97)
- [NVIDIA Kaggle Grandmaster stacking playbook](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/)
- [Compete to Win pseudo-labeling paper](https://arxiv.org/abs/2304.07519)

Implication for UMUD:

- A single train/val split and one output CSV is not enough.
- For segmentation, we need OOF predictions, fold variance, threshold sweeps, TTA deltas, component
  counts, and geometry summaries.
- For scale/text/ruler detection, we should train auxiliary detectors or classifiers where possible
  and use confidence/consensus. Do not broad-override scale from a single brittle heuristic.
- If using pseudo labels, use high-confidence consensus only. Do not train on every pseudo mask from
  one model.

## What This Means For Our Current Failure Modes

### Segmentation

EXP59 remains the conservative control. EXP72 is not a sufficient test of thin-structure methods
because it changed too many knobs and used hard skeleton decoding after thresholding rather than
training a topology-aware model.

Next:

- build EXP74 with probability maps, decoder sweeps, component counts, accepted-fragment counts, and
  geometry summaries;
- add target-specific apo/fasc training so fasc experiments cannot damage apo quality;
- after instrumentation, test clDice, boundary/distance loss, skeleton-recall-style loss, and masked
  pretraining;
- add folds or at least repeated validation splits before trusting a new mask model.

### Classical line extraction

This is the most important new action item from the review.

Build a harness that does not depend on a new neural network:

1. take the image and current aponeurosis/muscle-band region;
2. apply CLAHE inside the band;
3. run ridge/line filters such as Sato, Frangi, structure tensor, or oriented kernels;
4. threshold/cluster candidate fascicle pixels;
5. skeletonize;
6. fit line segments;
7. connect collinear segments;
8. cluster orientations by total segment support;
9. extend candidate lines to the boundaries;
10. resolve intersections inside the band by penalizing the line that disagrees most with the
    dominant local orientation;
11. output PA/FL as an independent candidate, not as an immediate public submission.

This directly tests whether our learned fascicle masks are missing a recoverable texture signal.

### Scale

EXP64 solved displayed-depth reading well enough for this stage. The failed burn #22 showed that
displayed depth does not equal trusted px/mm without a reliable pixel span.

Next:

- train or fit separate detectors for ultrasound field, ruler/tick span, text/depth, and device family;
- score them against the reviewed 309-row scale manifest;
- use confidence and consensus before changing public predictions;
- do not revive broad field-height overrides until span detection is separately validated.

### Geometry/story stack

The local expert benchmark remains useful for understanding conventions, but public submissions have
rejected broad proxy stacks.

Next:

- keep robust triangle, visibility/support FL, vertical MT, and class routing as diagnostic viewer
  models;
- do not submit another broad geometry stack until exact production-owned features are wired and
  inspected against target labels or a stronger validation set;
- use the classical line-extraction harness as a new independent geometry source rather than only
  reweighting existing fragments.

## Ranked Next Work

1. **Classical fascicle-line extractor harness.**
   Fast local experiment, strongly connected to published ultrasound measurement logic, and directly
   tests the coherent-story idea.

2. **EXP74 controlled segmentation diagnostics.**
   Before another long GPU run, make the run explain itself: probability maps, threshold sweeps,
   components, accepted fragments, and measurement distributions.

3. **Masked in-domain ultrasound pretraining.**
   Stronger than "more epochs" because it uses all unlabeled ultrasound-looking images to adapt the
   encoder/decoder before supervised fine-tuning.

4. **Kaggle-grade folds/OOF/TTA/threshold protocol.**
   Needed to stop confusing one lucky split or one bad notebook with real progress.

5. **Scale auxiliary model stack.**
   Important, but only after span detection is validated. The depth text alone is not enough.

## What Not To Do

- Do not run EXP72 longer as if it were a clean test.
- Do not submit another broad scale override from displayed depth plus field height.
- Do not promote local benchmark geometry stacks without exact production wiring.
- Do not train on all pseudo labels from one model.
- Do not add every augmentation at once and call it a serious heavy run.

## Immediate Implementation Checklist

- Add the classical extractor as `experiments/exp75_classical_fascicle_extractor.py`.
- Make it write a bundle under `results/exp75_classical_fascicle_extractor/`:
  - per-image candidate lines;
  - accepted/rejected line reasons;
  - intersection diagnostics;
  - PA/FL/MT predictions if boundaries are available;
  - preview overlays for viewer v2.
- Update EXP74 notebook generation to include OOF-style diagnostics and decoder sweeps before any
  long training matrix.
- Add a future `seg76_masked_pretrain` notebook spec instead of another "more epochs" notebook.
- Add scale auxiliary detector rows to the feature database, but do not treat them as solved.
