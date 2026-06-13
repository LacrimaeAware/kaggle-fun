# EXP52 - Saturating Support And Position Grid

Date: 2026-06-12

Purpose: test the idea that visible fragment length should be a saturating weight: tiny fragments
count little, moderate fragments gain trust quickly, and very long fragments do not receive unlimited
extra authority. Also tests where the fragment lies along its projected full span.

Baseline anchor: `results/benchmark_pred_robust_triangle.csv` (`0.170` overall).

Harness: `experiments/exp52_saturating_support_position_grid.py`

Ignored output bundle: `results/exp52_saturating_support_position_grid/`

## Best Results

| variant | overall | PA | FL | MT | read |
|---|---:|---:|---:|---:|---|
| `PA_median__FL_raw_wtrim10_sat12_none_rawmid__MT_vertical` | **0.147** | 0.150 | 0.222 | 0.070 | best EXP52 combo |
| `same_wmean_sat12_rawus` | 0.153 | 0.161 | 0.227 | 0.070 | best same-story saturating reducer |
| EXP50 best | 0.144 | 0.150 | 0.210 | 0.070 | still better |
| robust triangle anchor | 0.170 | 0.150 | 0.278 | 0.083 | baseline |

## Interpretation

The saturating-length idea is sane and improves over the robust anchor, but it does not beat EXP50's
trajectory-residual weighted trim. The important negative result is PA: saturating support/position
weights do not beat PA median globally, though EXP53 later shows a small partial-blend gain.

## Next

Keep saturating length as a principled candidate family, especially for future target-label scoring.
Do not promote this exact EXP52 best over EXP50/EXP53 on benchmark evidence alone.
