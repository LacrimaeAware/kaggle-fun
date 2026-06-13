# EXP61 Oracle Scale Patch

## Purpose

The scale review found real failures in the scale partition:

- Some visible `cm` labels were misread or missed.
- Some rows were marked `mean` even though the UI has readable depth/ruler cues.
- Some no-text rows have tick patterns that need family-specific interpretation.

This experiment converts human scale notes into an audit table. It does not
blindly submit corrections. If the human note provides only field depth, the
script estimates `px/cm` from:

```text
ultrasound_field_height_px / visible_depth_mm * 10
```

and compares that against the existing tick/router scale.

## Files

- Script: `experiments/exp61_oracle_scale_patch.py`
- Inputs:
  - `results/scale_oracle_review/oracle_notes.json`
  - `results/scale_partition.csv`
- Outputs:
  - `results/scale_oracle_review/oracle_scale_patch_audit.csv`
  - `results/scale_oracle_review/oracle_scale_overrides.csv`

Outputs are ignored because they contain local human review notes.

## Current First-Pass Findings

The first user pass added observations for 25 rows:

- `IMG_00004`, `IMG_00236`: visible field depth is `4.5 cm`, not `50 mm`.
- `IMG_00234`: visible field depth is `3.5 cm`, not `50 mm`.
- `IMG_00228`, `IMG_00244`: visible field depth is `4 cm`, not `50 mm`.
- `IMG_00057`, `IMG_00065`, `IMG_00114`, `IMG_00123`, `IMG_00131`, `IMG_00140`: visible `50 mm` depth text was missed.
- `IMG_00198`, `IMG_00199`, `IMG_00200`: readable `3 cm` text/ruler rows should not be fallback/mean.
- `IMG_00036`, `IMG_00038`, `IMG_00039`, `IMG_00040`, `IMG_00043`, `IMG_00044`, `IMG_00045`, `IMG_00047`, `IMG_00050`, `IMG_00051`, `IMG_00053`: no obvious depth text, but visible tick patterns/major ticks remain usable evidence.

Running `experiments/exp61_oracle_scale_patch.py` on those notes produced:

- `11` rows where the corrected depth label confirms the existing tick scale.
- `3` real scale candidates: `IMG_00198`, `IMG_00199`, `IMG_00200` at about `159.33 px/cm`.
- `11` no-text rows where no depth-derived scale can be computed yet.

This is an important distinction: the OCR label can be wrong while the numeric
tick scale is still right. The first-pass actionable numeric correction is
therefore concentrated in `IMG_00198-00200`, not all 25 reviewed rows.

## OCR Plan

Training a general OCR model is probably overkill before exhausting cheaper
options. The immediate route is:

1. Improve parsing so split labels like `4` + `cm`, `3,5 cm`, and tiny
   lower-corner labels are handled. First parser patch is now in `scale_ocr.py`.
2. Use field-height geometry when a depth label is known.
3. Use tick/major-tick patterns as cross-checks.
4. Only then consider a small supervised digit/depth classifier trained on
   cropped UI labels if the parser still misses many labels.

## Production Hook

`segment_then_measure.py` now supports:

```powershell
$env:UMUD_SCALE_OVERRIDE_CSV='results/scale_oracle_review/oracle_scale_overrides.csv'
```

When set, `calibrate_image()` uses the explicit `image_id -> px/cm` rows before
the automatic scale router. The default behavior is unchanged unless the env var
is provided.

## Submission Rule

Do not create a public scale candidate until the audit separates:

- depth-label corrections that only improve confidence,
- real `px/cm` corrections,
- no-text tick-family guesses.
