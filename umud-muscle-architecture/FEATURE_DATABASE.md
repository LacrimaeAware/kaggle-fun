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
| `F025` | Per-band fragment-count PA/MT only while keeping FL fixed | bench-tested / small add-on | PA+MT | -0.0015 | n/a | bench robust 0.170 -> 0.169 | Confirms exp41's PA/MT signal is real but tiny; simple band average is rejected. |
| `F026` | Vertical center MT gap under robust-triangle geometry | bench-tested / small candidate | MT | -0.004 | n/a | bench robust 0.170 -> 0.166 | MT term improves `0.083 -> 0.070`; three-position and mean-width MT are rejected. |
| `F027` | PA referenced to upper-boundary or average-boundary tangent | bench-tested / rejected | PA | +0.061 to +0.153 | n/a | bench robust 0.170 -> 0.231-0.323 | Strong negative evidence: keep PA relative to lower/deep boundary convention. |
| `F028` | Best local benchmark stack: scan-region FL + PA conflict/per-band + vertical-center MT | bench-tested / research anchor | PA+FL+MT | -0.017 | n/a | bench robust 0.170 -> 0.153 | Useful local anchor for viewer inspection and production wiring; not a public-transfer claim. |
| `F029` | Alternative PA line fitters and orientation aggregators | bench-tested / rejected | PA | +0.000 to +0.010 | n/a | bench robust 0.170 -> 0.170-0.181 | PCA+area median is hard to beat; circular means, endpoint axes, RANSAC, and x/y fields worsen PA. |
| `F030` | Raw grayscale texture orientation for PA | bench-tested / rejected | PA | +0.026 to +0.164 | n/a | bench robust 0.170 -> 0.197-0.334 | Texture orientation is not aligned with the scored angle convention; blending toward it fails. |
| `F031` | Contrarian move away from raw texture orientation | bench-tested / rejected | PA | +0.003 to +0.115 | n/a | bench robust 0.170 -> 0.173-0.285 | Small move fixes signed bias but worsens MAE. Raw texture residual is not a correction direction. |
| `F032` | Geometry-class story gating for PA/FL/MT | bench-tested / research anchor | PA+FL+MT | -0.019 | n/a | bench robust 0.170 -> 0.151 | Per-band PA on multi-band rows, conflict PA elsewhere, scan-region FL on low-support rows, vertical-center MT. Promising but overfit risk. |
| `F033` | Weighted trimmed FL using ultrasound-field support and local trajectory residual | bench-tested / research anchor | FL aggregation support | -0.026 | n/a | bench robust 0.170 -> 0.144 | Best EXP50 combo keeps PA median, uses raw-span weighted-trimmed FL, and vertical-center MT. FL improves `0.278 -> 0.210`; public transfer unproven. |
| `F034` | Same-story weighted PA+FL reducer using shared support and trajectory residual weights | bench-tested / mixed research | PA+FL aggregation | -0.021 | n/a | bench robust 0.170 -> 0.149 | Conceptually cleaner than F033 because PA and FL share the same fragment weights, but PA worsens `0.150 -> 0.158`; useful for diagnosis, not yet the best candidate. |
| `F035` | Class-gated weighted PA only on nonparallel/high-support-risk rows | bench-tested / diagnostic | PA class gate | -0.002 | n/a | PA median frame 0.166 -> 0.164 | EXP51 improves PA `0.150 -> 0.145` when a raw weighted-trim PA is applied only to `boundaries_not_parallel` rows. Overfit risk is high. |
| `F036` | Saturating visible-fragment support and projected-position weighting | bench-tested / mixed research | PA+FL support | -0.023 | n/a | bench robust 0.170 -> 0.147 | EXP52 validates the mechanism as sane, but it does not beat EXP50. Best use is future target-label scoring, not current promotion. |
| `F037` | Median anchor blended with weighted support reducers | bench-tested / current local research best | PA+FL support blend | -0.027 | n/a | bench robust 0.170 -> 0.143 | EXP53 best: PA median blended 25% toward saturating support/position PA; FL median blended 85% toward EXP50 weighted-trim FL; tiny gain over EXP50. |
| `F038` | Allowed-only class and term route over local story models | bench-tested / benchmark-best research route | class-aware PA+FL+MT routing | -0.039 | n/a | bench robust 0.170 -> 0.131; bench median-weight 0.143 -> 0.131 | EXP55/EXP56: excludes `DLTrack`, `SMA`, and `our_pipeline_true_scale`. Full route is benchmark-best but not production-wired; prefix-5 captures most gain. |
| `F039` | Production split stack from current public best | public-tested / rejected | public burn planning | n/a | +0.01192 / +0.05601 / +0.01810 | public 0.58910 anchor -> #15 0.60102, #16 0.64511, #17 0.60720 | EXP57 proxy stacks all regressed. These were production-delta proxies, not the full EXP55 route; do not promote them as defaults. |
| `F040` | Test scale status tiers | diagnostic / tracking | scale verification | n/a | n/a | `scale_partition.csv` | EXP58: 147 independently confirmed, 294 detector-scaled, 15 unresolved/fallback. Scale is not hidden-label ground truth; tick-only rows remain plausible but not proven. |
| `F041` | Configurable high-resolution segmentation retraining | infrastructure / next candidate | training/data | pending | pending | current U-Net 384 baseline | EXP59 adds image size, architecture, encoder, loss, augmentation, threshold, batch-size, and weight-tag knobs. First serious candidate is `seg59_02_highres_512_unet`. |
| `F042` | Human-oracle scale review pack | infrastructure / next candidate | scale verification | n/a | pending | EXP58 scale partition | EXP60 creates a full 309-row scale manifest and 41-row starter pack for the user to verify confident, tick-only, and fallback scale guesses. Corrections save locally under ignored `results/scale_oracle_review/`. |
| `F043` | Human-oracle scale override CSV | infrastructure / candidate | scale correction | n/a | pending | EXP60/EXP63 reviewed notes | EXP61 now parses depth labels consistently and supports opt-in `UMUD_SCALE_OVERRIDE_CSV`. Full-depth audit finds 193 rows confirming existing scale and 116 field-depth scale candidates; many candidates likely reflect field-rectangle overcounting UI height, so broad override is not submission-safe. |
| `F044` | Isolated oracle scale candidate for IMG_00198-00200 | generated / submission candidate | scale correction | n/a | pending | public 0.58910 | EXP62 changes only three rows from public-best using field-depth scale; FL rises +10.7 to +16.9 mm and MT shifts -0.3 to -2.2 mm. |
| `F045` | Depth-first oracle review UI | infrastructure | scale/depth verification | n/a | n/a | EXP60 scale manifest | EXP63 makes the review workflow ask only for field depth first, with `Q/W/E` status and `A/D` navigation. This separates the human-readable fact from later px/cm computation. |
| `F046` | Tick-scale family repair for stale OCR-50 depth | implemented / reviewed | scale/depth parsing | n/a | pending | EXP63 reviewed notes | The 32 wrong reviewed depths all came from stale OCR accepting `50 mm`. New 1200x800 tick-scale family rules map 110.7->55mm, 135.4->45mm, 152.3->40mm, 159.5->35mm, 174.0->70mm. Algorithm-only depth audit now matches the full 309-row human review 309/309 without using notes as predictor input. |
| `F047` | Multi-region text scale OCR audit | implemented / reviewed | scale/depth OCR | n/a | pending | EXP63 reviewed notes | EXP64 runs EasyOCR over targeted UI crops and caches tokens. Direct OCR finds displayed depth on 237/309; OCR plus deterministic fallbacks covers 309/309 with zero misses versus review. This fixes the "OCR not installed / OCR only full-frame" gap. |
| `F048` | Conservative 3 cm scale-span probe | generated / submission candidate | scale correction | n/a | pending | public 0.58910 | EXP65 creates burn #19: current public best plus `IMG_00198-00200` and `IMG_00251` using the 3 cm OCR/ruler span (`478px / 30mm = 159.333 px/cm`). Existing burn #18 tests only `IMG_00198-00200`. | submit #18 first; use #19 only if #18 improves or ties |
| `F049` | Robust triangle retested with conservative 3 cm scale repair | generated / submission candidate | upper boundary + scale correction | bench robust triangle 0.170; public old robust 0.60102 | pending | burn #15 robust triangle | EXP66 creates burn #20 by changing `IMG_00198-00200` and burn #21 by also changing `IMG_00251`, recomputing FL/MT from robust debug pixels at `159.333 px/cm`. | submit #20 if using a repaired-scale benchmark-geometry slot; hold #21 unless isolating `IMG_00251` |
| `F050` | Guarded field-depth scale probe from algorithmic depth plus scan-field height | public-tested / rejected | broad scale correction | n/a | +0.07287 | public 0.58910 | EXP67 creates burn #22 from current public best, changing 114 rows where EXP64 depth and EXP61 field height imply a plausible 80-180 px/cm scale and old-vs-new disagreement is moderate. Public score `0.66197`. | reject broad field-height override; improve span detector before revisiting |
| `F051` | Robust triangle plus broad field-depth scale probe | generated / diagnostic | upper boundary + broad scale correction | public old robust 0.60102 | pending | burn #15 robust triangle | EXP68 creates burn #23 by applying EXP67's failed broad scale adjustment to robust triangle. Changes 114 rows and mostly lowers FL/MT. | diagnostic only; submit only to test interaction, not because it is expected to improve |

## Current Read

The current best public improvement stack is not a single magic geometry fix. It is:

1. temporal smoothing;
2. subpixel scale precision;
3. clean shape-neighbor scale fallback;
4. optionally the public-neutral isolated OCR correction.

The most promising unsubmitted geometry idea is robust upper-boundary triangle. The most promising PA-specific idea is conflict-gated local angle replacement, but its gain is small. Real ultrasound-field on-screen/off-screen projection weighting is now tested and locally useful for FL, especially with weighted trimming and local trajectory-residual weights. It does not solve PA by itself.

## Next Tests To Add To This Database

1. Inspect EXP53 `Median/weight blend best` plus EXP50 `Story weight grid best` in viewer v2, especially lines with low ultrasound-field support.
2. Build a real local curve/trajectory PA model; support weighting and partial blends only barely improve PA.
3. Score EXP50/EXP53 candidates against the rough target human-label set after quality checks, before any public promotion.
4. Inspect and production-wire vertical-center MT and strict/US-field weighted FL as explicit flags before any public test.
5. Revisit per-band only as a PA/MT add-on or targeted routing/filtering; naive FL averaging is rejected.
6. Public test of robust triangle or EXP50 only with a clearly named delta from `public_058910`, not as an assumed improvement.
7. Production-wire the EXP50/EXP53 weighted reducers and EXP55 class gates before claiming the `0.131264` route as a real submission candidate.
8. Run the EXP59 GPU segmentation matrix before spending more submissions on broad geometry proxies.
9. Use EXP63 depth guesses as the audited algorithmic depth source: 309/309 now match human review without notes as predictor input.
10. Promote EXP64 depth/text inference into the next scale solver: OCR/fallback depth is solved, but `px/cm` still needs a trusted pixel span.
11. Submit EXP65's conservative 3 cm scale sequence as a controlled probe: burn #18 first, then #19 only if #18 does not regress.
12. If retesting robust triangle, use EXP66 burn #20 rather than the old burn #15 file, because #20 includes the confirmed 3 cm scale-span repair.
13. Do not use EXP67 broad field-depth scale as a default: public score `0.66197` rejects the current span heuristic.
14. If testing the same scale move on robust triangle anyway, use EXP68 burn #23 and treat it as diagnostic only.
15. Improve/validate visible-field rectangle and ruler-span detection before accepting EXP61 field-depth scale candidates as production defaults; current full pass still depends on a heuristic rectangle.
16. Treat any `UMUD_SCALE_OVERRIDE_CSV` submission as an explicit human-reviewed scale probe, not as the default production path.
