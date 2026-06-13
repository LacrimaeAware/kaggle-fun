# EXP51 - PA Gate Search

Date: 2026-06-12

Purpose: test whether weighted PA variants are useful only on specific geometry classes or diagnostic
thresholds. EXP50 showed weighted PA is worse globally, so this is a diagnostic gate search rather
than a production rule.

Baseline: PA median under the robust-triangle/vertical-MT frame (`overall 0.166104`, PA `0.149963`).

Harness: `experiments/exp51_pa_gate_search.py`

Ignored output bundle: `results/exp51_pa_gate_search/`

## Best Local Signals

| gated PA variant | gate | n | overall | PA | PA delta |
|---|---|---:|---:|---:|---:|
| `raw_wtrim10_area_us_visible_frac` | `boundaries_not_parallel` | 10 | 0.164 | **0.145** | -0.005 |
| `raw_wtrim10_area_us_visible_frac` | `expert_FL_below_projected_median` | 22 | 0.164 | 0.145 | -0.005 |
| `raw_wtrim10_area_us_visible_frac` | `n_items >= q75` | 10 | 0.165 | 0.146 | -0.004 |
| `raw_median` | `multi_band_risk` | 9 | 0.165 | 0.146 | -0.004 |

## Interpretation

There is a small PA opening, but it is class-specific and likely overfit-prone. The useful clue is
that PA support weighting helps most when boundaries are nonparallel or the image has many fragments.
That supports the user's class-aware/story-stack direction, but it is not enough evidence for a
production gate by itself.

## Next

Inspect the top gated cases in viewer v2. If the same visual story appears in rough target labels,
turn the gate into an explicit candidate; otherwise keep it as a diagnostic clue only.
