# Exp38 Curve/Correction Ablation - 2026-06-12

Status: local expert-benchmark test only. Not wired to production, not submitted.

## Goal

Test a reduced set of boundary/correction ideas without full combinatorics:

- robust triangle as the known anchor;
- smooth curve fit to the bottom/muscle-facing edge of the upper apo mask;
- partial blends between robust triangle and smooth curve;
- rotate-only local non-crossing correction;
- support/correction-weighted aggregation;
- center MT vs mean-gap/area-style MT.

The harness deliberately matches production's two-largest-apo-band convention for this round. A
future per-gap harness should change that deliberately, not accidentally.

## Command

```powershell
python experiments\exp38_curve_correction_ablation.py
```

Generated ignored bundle:

`results/exp38_curve_correction_ablation/`

Important files:

- `summary.csv`
- one prediction CSV per variant
- `geometry_bundle.json` with per-image boundaries and projected spans for viewer reconstruction
- `pull_summary.json`

## Pull Direction

Hidden straight-line reference only, not a candidate:

- robust triangle moves FL by **-3.40 mm** on average versus the old straight boundary.
- robust triangle lowers FL on **34/35** rows and raises FL on **1/35** row.

Interpretation: robust triangle mostly works locally by shortening an overprojecting straight-boundary
measurement. That is useful, but it is also exactly the thing to watch for overcorrection.

## Results

Lower is better. Signed FL is candidate minus expert consensus.

| variant | overall | PA | FL | MT | signed FL | FL better/worse vs robust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blend25 curve + rotate | **0.155** | 0.155 | **0.238** | 0.073 | +1.00 mm | 17 / 18 |
| blend50 curve median | 0.160 | 0.150 | 0.261 | **0.068** | -0.82 mm | 19 / 16 |
| blend25 curve median | 0.160 | 0.150 | 0.258 | 0.073 | +1.31 mm | 18 / 17 |
| robust triangle median | 0.170 | **0.150** | 0.278 | 0.083 | +2.39 mm | 0 / 0 |
| robust triangle rotate | 0.171 | 0.154 | 0.275 | 0.083 | +2.16 mm | 9 / 9 |
| robust triangle area MT | 0.177 | 0.150 | 0.278 | 0.103 | +2.39 mm | 0 / 0 |
| smooth curve median | 0.273 | 0.150 | 0.598 | 0.072 | -6.53 mm | 10 / 25 |
| smooth curve area MT | 0.273 | 0.150 | 0.598 | 0.072 | -6.53 mm | 10 / 25 |
| smooth curve rotate | 0.286 | 0.157 | 0.627 | 0.072 | -6.92 mm | 9 / 26 |
| smooth curve rotate support | 0.353 | 0.173 | 0.813 | 0.072 | -9.39 mm | 7 / 28 |

## Read

- Pure smooth curve is too aggressive. It follows the bottom edge but pulls FL too low and worsens many
  rows.
- Partial curve blend is the useful region. A 25-50% blend improves local FL and MT without the full
  smooth-curve collapse.
- Rotate-only non-crossing correction is small. On robust triangle alone it barely moves the aggregate:
  FL improves but PA worsens slightly. On 25% curve blend it gives the best local overall.
- Area/mean-gap MT is not a free win. It worsens robust triangle MT and is only better for the curve
  variants because their center MT already changed.
- Best local variant is `blend25_curve_rotate`, but it improves FL on 17 rows and worsens 18 rows versus
  robust. The gain is real in the average metric, not universal row-wise dominance.

## Next

Do not submit from this script directly. If pursuing, wire `blend25_curve_rotate` as an explicit
production flag and generate a stacked candidate the same way robust triangle was generated. Before
spending a slot, add the variant geometry to the expert viewer so the boundary/spans are inspectable
per image.
