# EXP50 - Story Weight Grid

Date: 2026-06-12

Purpose: test the user's weighted-mean diagnosis directly. The earlier viewer work made clear that
support and story consistency cannot matter if the final reducer always collapses back to a plain
median. This sweep tests weighted means, weighted trimmed means, and trajectory-residual weights over
the 35-image expert benchmark.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv` (`0.170` overall).

Harness: `experiments/exp50_story_weight_grid.py`

Ignored output bundle: `results/exp50_story_weight_grid/`

## What Was Tested

- Support weights: area, ultrasound-field fraction, visible fraction, and crossing downweight.
- Trajectory-residual weights: fit a local direction story from fragment slope vs x-position, then
  downweight fragments whose direction disagrees with that local story.
- Weighted trimmed reducers: same as weighted mean, but trims the weighted tails first.
- Same-story reducers: PA and FL use the exact same weights, so one fragment story drives both terms.

## Best Results

| variant | overall | PA | FL | MT | read |
|---|---:|---:|---:|---:|---|
| `PA_median__FL_raw_wtrim10_area_us_rawlocal3_sigma7__MT_vertical` | **0.144** | 0.150 | **0.210** | 0.070 | best benchmark combo |
| `same_wmean_area_us_rawlocal7_sigma4` | 0.149 | 0.158 | 0.220 | 0.070 | best coherent same-story reducer |
| EXP49 `PA_median__FL_wmean_area_us_frac__MT_vertical` | 0.149 | 0.150 | 0.227 | 0.070 | simpler area x US-field FL weight |
| EXP48 story stack | 0.151 | 0.140 | 0.242 | 0.070 | previous class-aware anchor |
| robust triangle anchor | 0.170 | 0.150 | 0.278 | 0.083 | baseline |

## Interpretation

The support-weighting idea is real on the expert benchmark. The best gain comes from FL:
`0.278 -> 0.210` against the robust-triangle anchor. MT stays at the already-good vertical-center
convention. PA remains the hard part: the benchmark still prefers the plain PA median (`0.150`) over
weighted PA variants (`~0.153-0.160`).

That means the current evidence says:

- using ultrasound-field on-screen support for FL is useful;
- using the same support weights for PA is conceptually clean, but not yet locally better;
- the missing PA feature is probably not support alone, but a better local curve/trajectory model.

## Viewer Wiring

The v2 expert viewer now includes two EXP50 models:

- `Story weight grid best`: the benchmark-best pragmatic combo.
- `Story weight same-story`: the cleaner shared-weight PA+FL reducer.

The projected-line inspect readout now exposes raw/corrected ultrasound-field fraction and
area x US-field weights, so the support-weighted FL behavior can be checked visually.

## Next

Do not treat this as public-submission proven. The public board has already punished several FL
combiner changes. Use this as the next local research anchor and inspect whether the weighted lines
look visually sane on the expert viewer and on the rough human-labeled target rows before promoting.
