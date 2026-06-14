# EXP58 - Test Scale Status

Date: 2026-06-13

Purpose: answer precisely how much of the 309-image test scale is verified versus inferred. This
uses `results/scale_partition.csv`; it does not use hidden labels.

Harness: `experiments/exp58_scale_status.py`

Ignored outputs:

- `results/scale_status_summary.csv`
- `results/scale_status_unverified_rows.csv`

## Result

| category | tiers | rows | percent |
|---|---|---:|---:|
| independently confirmed | `verified + text-confirmed` | 147 | 47.57% |
| detector scale available | `verified + text-confirmed + tick-only` | 294 | 95.15% |
| unresolved or fallback | `flag + mean` | 15 | 4.85% |

Detailed tier counts:

| tier | rows | percent |
|---|---:|---:|
| verified | 48 | 15.53% |
| text-confirmed | 99 | 32.04% |
| tick-only | 147 | 47.57% |
| mean | 14 | 4.53% |
| flag | 1 | 0.32% |

## Interpretation

The honest wording is:

- Scale is available for 294/309 rows through the current detectors.
- Scale is independently confirmed for 147/309 rows.
- The tick-only 147 rows are plausible and mostly cross-checked by family logic, but they are not
  hidden-label ground truth.
- The remaining 15 rows are the actual scale uncertainty tail.

This explains the public results: scale improvements helped when they repaired clear fallback/router
issues, but broad visual tail fixes were risky and regressed.

## Next

Do not spend more broad scale-tail submissions unless a row has independent evidence. Longer-term,
train a scale-cue / ultrasound-field detector, but the larger near-term lever is better mask
segmentation.
