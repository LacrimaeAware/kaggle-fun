# EXP53 - Median / Weighted Reducer Blends

Date: 2026-06-12

Purpose: test the user's point that median and weighted mean do not have to be mutually exclusive.
This sweep blends the robust median anchor with weighted PA/FL reducers from EXP50 and EXP52.

Harness: `experiments/exp53_median_weight_blends.py`

Ignored output bundle: `results/exp53_median_weight_blends/`

## Best Results

| variant | overall | PA | FL | MT | read |
|---|---:|---:|---:|---:|---|
| `PA_blend25_exp52...wtrim10_sat16_rawus_rawmid__FL_blend85_exp50...raw_wtrim10_area_us_rawlocal3_sigma7__MT_vertical` | **0.143** | **0.149** | 0.211 | 0.070 | best local benchmark score so far |
| EXP50 best | 0.144 | 0.150 | **0.210** | 0.070 | nearly tied, simpler |
| EXP52 best | 0.147 | 0.150 | 0.222 | 0.070 | saturating support only |
| robust triangle anchor | 0.170 | 0.150 | 0.278 | 0.083 | baseline |

## Interpretation

The best local result comes from:

- PA: keep the median mostly intact, but blend 25% toward a saturating support/position weighted PA.
- FL: blend 85% toward the EXP50 raw-span weighted-trimmed support reducer.
- MT: keep vertical-center MT.

This validates the mechanism that weighted information can help without fully replacing the median.
The gain over EXP50 is tiny (`0.14334` vs `0.14354`), so the safer scientific statement is "median
plus weighted support contains a little PA signal," not "this is a public-board lock."

## Next

Use this as the current local research best. Before a public burn, compare it against rough
human-labeled target rows and inspect whether the weighted support lines look visually sane.
