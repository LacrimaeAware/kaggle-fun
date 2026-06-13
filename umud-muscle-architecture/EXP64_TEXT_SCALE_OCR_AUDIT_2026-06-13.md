# EXP64 Text Scale OCR Audit

## Purpose

Answer the scale-review question without relying on the human review notes as
the predictor:

```text
Can the code independently read or infer the displayed field depth?
```

EXP63 already showed that a deterministic depth guesser matched the full review.
EXP64 makes the text side explicit by installing/running OCR locally and reading
multiple UI regions instead of a single full-frame pass.

## Implementation

Script:

```text
experiments/exp64_text_scale_ocr_audit.py
```

Local OCR environment:

```text
.venv_ocr
```

The venv is ignored by git. It contains `easyocr`, `torch`, `opencv-python-headless`,
and `pandas`. On Kaggle this should be reproducible by installing `easyocr` when
needed, since the existing code already assumes optional package installs.

The script:

1. OCRs targeted UI crops (`top_right`, `middle_right`, `bottom_full`,
   `left_full`) at 3x resolution.
2. Caches raw OCR tokens under ignored `results/exp64_text_scale_ocr/tokens/`.
3. Extracts depth labels such as `De 50 mm`, `3 cm`, `3.5 cm`, `5.0 cm`, and
   OCR variants like `35 cm` for `3.5 cm`.
4. Fuses OCR with deterministic non-human fallbacks:
   - edge numeric ruler labels, e.g. top `0` plus bottom `50`;
   - cropped/no-overlay 50 mm image family;
   - 1200x800 tick-scale family repair when OCR drops the half-centimeter digit.
5. Compares to the human review only after prediction, as an audit.

## Result

Full 309-image run:

- OCR text depth found: `237/309`
- fused algorithmic depth found: `309/309`
- reviewed rows available: `309/309`
- misses versus review: `0`

Fused source breakdown:

- `ocr_text`: `233`
- `numeric left ruler label 50 mm`: `50`
- `cropped/no-overlay 50 mm family`: `20`
- `tick-family 3.5 cm`: `3`
- `tick-family 4.0 cm`: `2`
- `tick-family 5.5 cm`: `1`

Important interpretation: the code still does not OCR every row directly. It
does, however, now explain every reviewed depth using a non-human algorithmic
cue. The human notes are evaluation labels, not predictor inputs.

## Scale Meaning

Depth alone still does not equal `px/cm`.

```text
scale_px_per_cm = field_depth_px / field_depth_cm
```

EXP64 solves the displayed-depth side. The remaining scale problem is deciding
which pixel span corresponds to that displayed depth:

- edge ruler tick spacing,
- numeric ruler span,
- true scan-field rectangle,
- or a device/family-specific field geometry.

The 116 EXP61 disagreements are therefore not depth disagreements. They are
pixel-span disagreements. Many are likely false positives from using the full
image/UI height instead of the actual depth span.

## Next

Build a scale solver that produces one row per image with:

- displayed depth and source,
- candidate pixel spans from ticks, numeric ruler labels, and field rectangle,
- agreement/disagreement between candidates,
- final `scale_px_per_cm`,
- confidence tier and reason.

Do not submit a broad 116-row field-height override until that solver separates
real scale corrections from field-rectangle false positives.
