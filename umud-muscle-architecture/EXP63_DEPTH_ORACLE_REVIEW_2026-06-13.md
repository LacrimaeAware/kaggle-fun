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

The default pack is now `Remaining unreviewed`, because the starter pack is
only a sampled subset. Use the dropdown to switch to `All 309` or the old
`Starter pack`.

Keyboard:

- `Q`: mark proposed depth correct, save, and advance
- `W`: mark proposed depth wrong
- `E`: mark unclear
- `A`: previous image
- `D`: next image

Wrong and unclear intentionally do not auto-advance, because those cases usually
need a corrected depth or a comment.

The top strip now shows proposed depth first. Scale remains visible only as
evidence.

The review UI now treats that proposed depth as the single answer to confirm.
Raw OCR depth, px/cm, tick values, and other detector evidence are collapsed
under debug evidence so stale OCR text does not compete with a corrected
human-or-family depth proposal. For example, `IMG_00234.tif` can have raw OCR
depth `50 mm` while the review answer is `35 mm` because a human oracle note
overrides the stale OCR parse.

## Manifest Changes

`experiments/exp60_scale_oracle_review_pack.py` now enriches the manifest with:

- `depth_guess_mm`
- `depth_guess_source`
- `depth_guess_note`
- `submitted_scale_state`

Guess priority:

1. existing human oracle note,
2. parsed depth text,
3. cropped/no-surrounding-overlay family prior: 50 mm depth,
4. field-height from scale, snapped to a normal depth,
5. known-scale common-depth prior: 50 mm,
6. global common-depth prior: 50 mm.

There should be no blank depth proposals in the review manifest. The
algorithm must always put a number in front of the user because the review is
testing the guesser we would have at submission time.

The cropped/no-surrounding-overlay prior is keyed from the image-family
signature rather than one-off IDs: `1069x853` full-field crops and `~464x513`
full-field crops are proposed as 50 mm depth unless OCR or a human note has
already supplied a stronger value. This catches cases like `IMG_00040.tif`.

For tick-only rows without text, the next guess is `image_height / px_per_cm`,
converted to mm and snapped to `{30, 35, 40, 45, 50, 60, 65}` when the raw value
is close. This catches the `1088x644` rows with `126 px/cm` as about `51.1 mm`,
so they are proposed as `50 mm`.

## Current Meaning

This review is for the full 309-image test set. A correct depth does not by
itself guarantee a correct scale; the scale still needs a valid field height or
tick spacing. But depth is the user-reviewable fact, so it is the right first
question.
