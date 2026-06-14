# EXP78 Scale Review and Recall Segmentation State

Date: 2026-06-14

Status: current synthesis after the full scale-review pass and the EXP77 segmentation pivot.

## Straight Answer

The recent scale work was real and should not be forgotten, but it fixed **displayed field depth**,
not the whole scale problem.

The user manually reviewed the full 309-image test set in the scale-review UI. That review exposed
systematic OCR/family mistakes and gave us a checked audit target. After repairs, the algorithm-only
depth guesser matched the reviewed depth labels on all 309 rows without using the notes file as input.

That means:

- displayed depth is now audited and algorithmically recoverable;
- human notes are an audit set, not required by default production inference;
- `px/cm` still needs a trusted span from ticks, ruler spacing, or a correctly detected ultrasound
  field rectangle;
- broad field-depth scale submissions failed badly, so depth must not be treated as scale by itself.

## What The User Manually Confirmed

The user reviewed every test image's proposed displayed depth in the local scale-review UI. Important
corrections/observations from that pass:

- many images marked uncertain by the first detector did have readable depth text;
- no-surrounding-overlay cropped image families should default to the 50 mm family unless stronger
  evidence overrides it;
- `IMG_00040.tif` belongs to that no-surrounding-overlay 50 mm family;
- yellow-arrow/tick families often use half-centimeter ticks, with smaller tick spacing on some
  3 cm-depth images;
- `IMG_00234.tif` showed the stale-OCR failure pattern: raw OCR could say `50 mm` while the proposed
  corrected depth should be 35 mm;
- human shorthand like `3.5`, `5.5`, and `40 mm` must normalize to `35`, `55`, and `40` mm.

The repaired depth guesser was then rerun without reading the human notes as predictors. It matched
the reviewed depth labels on all 309 images.

Algorithm-only source breakdown from EXP63/EXP64:

| source | rows |
|---|---:|
| OCR depth text | 202 |
| field-height/scale depth guess | 50 |
| tick-scale family repair | 37 |
| cropped/no-overlay family | 20 |
| total | 309 |

## What Scale Submissions Taught Us

Public leaderboard scale-related reads:

| burn | score | meaning |
|---:|---:|---|
| #11 | 0.58910 | current public best; temporal + subpixel + clean shape-neighbor scale fallback |
| #13 | 0.58910 | isolated `IMG_00275` OCR scale correction tied #11 |
| #22 | 0.66197 | broad field-depth scale override failed hard |
| #28 | 0.65917 | local-benchmark proxy plus missing-scale patch failed hard |

The important distinction:

- **Narrow/gated scale routing helped.**
- **Broad depth-to-scale overrides hurt.**

So the next scale problem is not "read depth"; it is "derive a trustworthy pixel span for that
depth." The needed span may come from tick spacing, ruler labels, or field-rectangle detection, but
it must be validated independently before being applied broadly.

## Current Production Boundary

Default production inference should not read `oracle_notes.json`.

Human-reviewed scale information should enter production only through an explicit opt-in override
CSV, for a controlled probe. Otherwise, the production path should use deterministic OCR/family/tick
logic.

The current standing rule:

- use EXP64/EXP63 depth logic as the audited displayed-depth source;
- do not apply field-depth scale corrections unless the pixel span is independently trusted;
- do not repeat broad field-depth submissions like #22.

## Segmentation State After The Scale Wall

The current active notebook is EXP77:

`kaggle_seg77_best_effort_heavy_auto.ipynb`

Main run:

`seg77_01_best_unetpp640_dilate_soft5_cldice`

EXP77 is not just chasing validation Dice. It is trying to produce more useful geometry by changing
mask formulation:

- soft/dilated fascicle targets;
- CLDice-style topology loss;
- U-Net++ / high-resolution variants;
- threshold sweep;
- debug mask export.

But the user's diagnosis is important: Dice-optimized thresholds can be too conservative for the
final PA/FL/MT task. The model may need to "guess more" and let the geometry layer reject bad
fragments.

Therefore the next segmentation follow-up after a checkpoint exists should include **inference-only
recall-heavy variants**, without retraining:

- lower fascicle thresholds such as `0.12`, `0.16`, `0.20`, `0.24`;
- lower `FASC_MIN_AREA` such as `6`, `10`, `14`;
- compare `threshold` versus `skeleton_dilate`;
- inspect accepted fragment counts, PA/FL/MT distributions, calibration debug, and `pred_debug_*`
  masks.

This directly tests the user's "guess more, then filter geometrically" idea.

## EXP72 Downloaded Partial Run

The user saved `umud_seg72_thin_structure_outputs.zip` from Downloads. It was extracted locally under
ignored results:

`results/seg72_downloaded_partial_2026-06-13/`

It contained weights and logs, but no submission CSV, no calibration debug CSV, and no `pred_debug_*`
masks because it was bundled while training was still running.

Audit from the log:

| target | best Dice | epoch | control best | delta |
|---|---:|---:|---:|---:|
| apo | 0.7873 | 43 | 0.7945 | -0.0072 |
| fasc | 0.2606 | 30 | 0.2925 | -0.0319 |

Conclusion: EXP72 weights are real and distinct, but the partial run is worse than the EXP59 control
by validation Dice and has no generated masks to visually compare. Keep it as a negative artifact, not
as a candidate branch.

## Next Actions

1. Let EXP77 finish far enough to produce weights, summary, debug masks, and submission CSVs.
2. Inspect `seg77_best_effort_summary.csv`, logs, calibration debug, and `pred_debug_*`.
3. If EXP77 produces a usable checkpoint, generate recall-heavy inference-only variants before
   deciding that the segmentation model failed.
4. If segmentation still fails, build the EXP75 classical fascicle-line extractor and/or a trusted
   scale-span detector.
5. Keep the full 309-row human depth review as an audit reference, not a hidden manual production
   dependency.
