# EXP65 Scale Solver Submission Candidates

## Purpose

Use the EXP64 depth/text audit to create conservative public-best scale probes
without broad-applying the 116 EXP61 field-height disagreements.

The current public-best anchor remains:

```text
results/submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv
public LB: 0.58910
```

## Current Scale Interpretation

Displayed depth is now solved algorithmically over the full test set:

- EXP64 direct OCR depth: `237/309`
- EXP64 OCR plus deterministic fallbacks: `309/309`
- misses versus human review: `0`

But displayed depth is only half of scale:

```text
scale_px_per_cm = depth_span_px / depth_cm
```

The unresolved part is `depth_span_px`: the pixel span that corresponds to the
displayed depth. For rows with clear text plus a clear scan/ruler span, this is
usable. For many of the 116 EXP61 disagreements, the rough field detector is
probably counting UI height, so broad override remains unsafe.

## Candidate Files

### Burn 18: existing 3-row 3 cm scale probe

```text
results/submission_burn_18_oracle_scale_198_200_direct.csv
```

Rows changed from public best:

- `IMG_00198.tif`
- `IMG_00199.tif`
- `IMG_00200.tif`

Scale assumption:

```text
478 px / 30 mm = 159.333 px/cm
```

This is the cleanest current scale-only probe.

### Burn 19: 4-row 3 cm scale probe

Script:

```text
experiments/exp65_scale_solver_submission_candidates.py
```

Output:

```text
results/submission_burn_19_public_best_plus_3cm_scale_198_200_251.csv
```

Rows changed from public best:

- `IMG_00198.tif`
- `IMG_00199.tif`
- `IMG_00200.tif`
- `IMG_00251.tif`

`IMG_00251.tif` is the new row made explicit by EXP64 as the same 3 cm
text/ruler family. It is plausible, but it is riskier than burn 18 because its
MT moves by `-5.434 mm`, larger than one MT tolerance.

## Recommended Use

Submit burn 18 first.

If burn 18 improves or ties the current best, submit burn 19 to test whether
adding `IMG_00251.tif` helps.

If burn 18 worsens, do not submit burn 19 in the same direction. That would
mean the 3 cm pixel-span correction is not transferring cleanly, or the public
subset is not rewarding it.

Do not resubmit robust triangle as the "best benchmark" candidate: it already
public-tested at `0.60102`, worse than `0.58910`. The remaining benchmark-best
class route is not production-wired yet, so it is not a real CSV candidate.
