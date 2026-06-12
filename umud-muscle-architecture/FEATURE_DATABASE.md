# UMUD Feature Database

Long-term tracking ledger started 2026-06-12.

Purpose: stop treating ideas as one-off guesses. Every feature should have a named baseline, a benchmark delta when available, a public leaderboard delta when submitted, and enough semantic notes that future work can tell whether it was a real mechanism or an accidental compensating error.

Companion machine-readable file: `FEATURE_DATABASE.csv`

## Rules For New Rows

- Always name the comparison baseline. A delta without a baseline is invalid.
- Record benchmark and public deltas separately. The 35-image expert benchmark and the Kaggle public leaderboard are different oracles.
- Record what changed: PA, FL, MT, scale, temporal smoothing, boundary geometry, fragment aggregation, training/data, or tooling.
- Keep untested ideas in the same ledger as tested ideas. The status column should say whether it is untested, bench-tested, public-tested, rejected, candidate, or infrastructure.
- Prefer clear feature names over short cryptic names. Example: `PA conflict-gated local angle replacement` is better than `pa_gate`.
- If a feature worsens a score, keep it. Bad deltas are information, especially for PA and MT.

## Current Working Baselines

| baseline id | score | meaning | file / evidence |
|---|---:|---|---|
| `public_061918` | 0.61918 | protected older public baseline | `Downloads/0P61918_submission_local.csv` |
| `public_058910` | 0.58910 | current public best | `results/submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv`; tied by burn #13 |
| `bench_raw_true_scale_0251` | 0.251 | raw true-scale expert-benchmark geometry before robust triangle | `SUBMISSION_TRIANGLE_CANDIDATE_2026-06-12.md` |
| `bench_robust_triangle_0170` | 0.170 | robust-triangle expert-benchmark anchor | `results/benchmark_pred_robust_triangle.csv` |

## Feature Ledger

Negative deltas are good. Positive deltas are bad.

| id | feature | status | axis | benchmark delta | public delta | reference baseline | notes |
|---|---|---|---|---:|---:|---|---|
| `F001` | Protected fragment/scale baseline | baseline | all | n/a | n/a | n/a | Public score 0.61918. Keep as a recovery anchor. |
| `F002` | Temporal smoothing across sequence-like clips | public-tested / accepted | PA+FL+MT temporal | n/a | -0.00957 | public 0.61918 -> 0.60961 | First public win after wall. Suggests sequence/neighbor consistency is real. |
| `F003` | Subpixel scale precision stacked on temporal smoothing | public-tested / accepted | scale | n/a | -0.00025 | public 0.60961 -> 0.60936 | Tiny but positive; likely real scale precision, not a main lever alone. |
| `F004` | Clean shape-neighbor fallback scale stacked on temporal+subpixel | public-tested / accepted | scale routing | n/a | -0.02026 | public 0.60936 -> 0.58910 | Largest recent public win. Strong evidence that scale/fallback routing remains load-bearing. |
| `F005` | Isolated IMG_00275 OCR scale correction on current best | public-tested / neutral | scale one-row fix | n/a | +0.00000 | public 0.58910 -> 0.58910 | Structurally justified and public-neutral. Keep as final candidate tie-breaker. |
| `F006` | Broad top-3 minimal-extrapolation FL combiner on current best | public-tested / rejected | FL aggregation | n/a | +0.04084 | public 0.58910 -> 0.62994 | Local intuition did not transfer; broad FL combiner harmed real board. |
| `F007` | Robust upper-boundary triangle from deepest-left / highest-middle / deepest-right anchors | bench-tested / candidate | upper boundary -> FL+MT | -0.081 | pending | bench 0.251 -> 0.170; public ref 0.58910 pending | Strongest local geometry candidate. Mostly shortens previous FL overshoot. Needs public test before promotion. |
| `F008` | 25% smooth upper-boundary curve blend plus rotate-only local correction | bench-tested / not wired | upper boundary + fragment angle correction | -0.015 | n/a | bench robust 0.170 -> 0.155 | Best local exp38 score, but partly arbitrary and row-wise mixed. Needs viewer wiring before any public slot. |
| `F009` | PA conflict-gated local angle replacement only when neighbor angle disagrees clearly | bench-tested / small add-on | PA | -0.002 | n/a | bench robust 0.170 -> 0.168 | Small PA win: PA term 0.150 -> 0.144. This is not generic smoothing. It only replaces obvious local conflicts. |
| `F010` | Lower-boundary tangent PA from smooth lower boundary | bench-tested / rejected | PA lower boundary | +0.021 | n/a | bench robust 0.170 -> 0.191 | Worsened PA. Lower-boundary tangent is not supported by current expert benchmark. |
| `F011` | Lower-boundary tangent PA from quartile polyline lower boundary | bench-tested / rejected | PA lower boundary | +0.013 | n/a | bench robust 0.170 -> 0.183 | Less bad than smooth tangent but still worse. |
| `F012` | Plain local PA smoothing toward local median angle | bench-tested / rejected | PA | +0.001 to +0.003 | n/a | bench robust 0.170 -> 0.172/0.173 | Confirms arbitrary smoothing is the wrong framing. |
| `F013` | Smooth upper-boundary curve with no blend | bench-tested / rejected | upper boundary -> FL+MT | +0.103 | n/a | bench robust 0.170 -> 0.273 | Overcorrects downward. Useful as evidence that "curve" cannot just follow the edge literally. |
| `F014` | Area/mean-gap MT instead of center MT | bench-tested / rejected for robust anchor | MT | +0.007 | n/a | bench robust 0.170 -> 0.177 | Not universally better. It only looked good in some curve variants. |
| `F015` | Facing-geometry FL | public-tested / rejected | FL | locally strong in older bench | +0.04541 | public 0.61918 -> 0.66459 | Public failure despite local promise. Likely multi-band/per-band misrouting. Keep concept only if repaired with band separation. |
| `F016` | Identity FL blend | public-tested / rejected | FL | misleading local win | +0.01987 | public 0.61918 -> 0.63905 | Benchmark-specific; do not revive without a new structural reason. |
| `F017` | Vertical three-position MT | public-tested / rejected | MT | n/a | +0.00643 | public 0.61918 -> 0.62561 | Host-style MT did not transfer as implemented. |
| `F018` | Bar-only scale tail correction | public-tested / rejected | scale tail | n/a | +0.04793 | public 0.61918 -> 0.66711 | Bad transfer. Do not use broad tail scale patch. |
| `F019` | On-screen vs off-screen projection support weighting using real image-region bounds | bench-tested / candidate variant | FL aggregation / support | -0.011 | n/a | bench robust 0.170 -> 0.159 | Gentle strict scan-region linear weighting helps FL-only. Harsh visible+scan squared weighting is rejected. |
| `F020` | Lower robust boundary shape for MT/PA symmetry | bench-tested / mixed-rejected for MT | lower boundary -> MT/PA | +0.009 MT-only; -0.003 FL-only | n/a | bench robust 0.170 -> 0.179 MT-only; 0.167 FL-only | Lower quartile median polyline helps FL slightly but worsens MT. Do not use as MT replacement yet. |
| `F021` | Per-band geometry separation before FL measurement | bench-tested / rejected naive averaging | multi-band routing | +0.004 average; +0.001 largest-gap | n/a | bench robust 0.170 -> 0.174 / 0.171 | Naive per-band averaging improves PA/MT on multi-band rows but worsens FL. Revisit only as targeted routing/filtering. |
| `F022` | Synthetic geometry benchmark pack | infrastructure | benchmark tooling | n/a | n/a | n/a | Useful for unit-testing geometry logic, not direct public evidence. |
| `F023` | Human-in-loop target benchmark labels | infrastructure | benchmark tooling | n/a | n/a | n/a | 19 rough target rows exist. Needs careful quality control and feature scoring protocol. |
| `F024` | Scale-cue segmentation / learned scale asset detector | idea / untested | scale ML | n/a | n/a | n/a | Long-term ML path: learn ticks/text/rulers/image region instead of hand-coded brittle routers. |

## Current Read

The current best public improvement stack is not a single magic geometry fix. It is:

1. temporal smoothing;
2. subpixel scale precision;
3. clean shape-neighbor scale fallback;
4. optionally the public-neutral isolated OCR correction.

The most promising unsubmitted geometry idea is robust upper-boundary triangle. The most promising PA-specific idea is conflict-gated local angle replacement, but its gain is small. The most important untested support idea is real on-screen/off-screen projection weighting; that has not been tested yet and should not be confused with local PA smoothing.

## Next Tests To Add To This Database

1. Stack and inspect `strict_scan_region_linear_support_weighted_FL_only` with robust triangle. It is locally useful but hurts existing FL undershoots.
2. Revisit per-band only as targeted routing/filtering for known multi-band failure families; naive averaging is rejected.
3. Revisit lower-boundary shape only if a viewer shows an MT-specific failure that the benchmark aggregate is hiding.
4. Public test of robust triangle if spending a slot, recorded as a delta from `public_058910`.
