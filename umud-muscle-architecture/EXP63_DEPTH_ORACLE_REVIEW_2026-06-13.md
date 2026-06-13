# EXP63 Depth Oracle Review

## Purpose

The scale review UI was too scale-math-heavy. The next pass is depth-first:

```text
What field depth do we think this image is?
```

The user can mark each proposed depth as correct, wrong, or unclear, then move
through the image set with keyboard shortcuts.

## UI Changes

Viewer:

```text
http://127.0.0.1:8774/scale-review/
```

Keyboard:

- `Q`: mark proposed depth correct
- `W`: mark proposed depth wrong
- `E`: mark unclear
- `A`: previous image
- `D`: next image

The top strip now shows proposed depth first. Scale remains visible only as
evidence.

## Manifest Changes

`experiments/exp60_scale_oracle_review_pack.py` now enriches the manifest with:

- `depth_guess_mm`
- `depth_guess_source`
- `depth_guess_note`
- `submitted_scale_state`

Guess priority:

1. existing human oracle note,
2. parsed depth text,
3. known scale but unknown depth,
4. unknown / needs oracle.

## Current Meaning

This review is for the full 309-image test set. A correct depth does not by
itself guarantee a correct scale; the scale still needs a valid field height or
tick spacing. But depth is the user-reviewable fact, so it is the right first
question.
