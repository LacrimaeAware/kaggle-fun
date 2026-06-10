# UMUD forward plan after 1.09194

> **Archive note (2026-06-09):** this plan is historical. The current repo state is the `0.61918`
> restored baseline with 295/309 scale reads; the later FL blend regressed to ~0.64 and is off by
> default. Domain-gap retraining remains demoted.
> Read `handoff_brief.md`, `synthesis.md`, and `experiments/README.md` before using this file.

Public-safe plan after the confidence-gated MT run scored `1.09194`.

## What the score means

The score improved:

```text
previous best:                 1.11066
U-Net PA + prior FL/MT:         1.12324
U-Net PA + calibrated MT rows:  1.09194
```

Hypothesis: calibrated MT is real signal, but the global gain is small because only
`68/309` rows changed MT and all FL values stayed constant.

Evidence for:

- The cleanest comparison is `1.12324 -> 1.09194`, because both use the U-Net PA path.
- The generated debug file changed MT on all `58/58` PNGs and only `10/251` TIFFs.
- PNG MT median is about `20.7 mm`, which is physiologically plausible and not a random
  collapse to the prior.

Evidence against / uncertainty:

- The public leaderboard score does not reveal per-target errors.
- The 10 TIFF bottom-tick rows may still be wrong-scale. They all share `13.45 px/mm` and
  MT values around `26-29 mm`, so they should be treated as suspicious until verified.
- U-Net PA appears weaker than the older ExtraTrees PA, masking part of the MT gain.

## Immediate no-GPU ablations

Use `make_postrun_variants.py` with the downloaded `calibration_measurement_debug.csv`.
It generated four CSVs under `results/postrun_variants/`:

- `submission_best_pa_calibrated_mt_png_only.csv`
- `submission_best_pa_calibrated_mt_all.csv`
- `submission_best_pa_calibrated_mt_png_direct_fl.csv`
- `submission_best_pa_calibrated_mt_all_direct_fl.csv`

Recommended order if spending submissions:

1. `submission_best_pa_calibrated_mt_png_only.csv`
   - Tests stronger old PA plus the safest calibrated MT rows.
   - Drops the suspicious 10 TIFF bottom-tick rows.
   - Expected magnitude: small, likely around the `1.08-1.10` zone if the assumptions hold.
2. `submission_best_pa_calibrated_mt_png_direct_fl.csv`
   - Tests whether direct FL from the same reliable PNG scales helps.
   - Higher risk, because fascicle geometry is noisier than aponeurosis thickness.

The `all` variants are useful for ablation but risk carrying the suspicious TIFF scale rows.

## Bigger work, not just leaderboard cleanup

The leader gap is not closed by the PNG MT patch. To move from `~1.09` toward the DLTrack
benchmark and beyond, the work has to stop being "small column recombination" and become a
real measurement pipeline.

Priority 1: DLTrack reproduction / headless port

- Goal: reproduce or approximate the public `0.67944` benchmark.
- Why: it gives a full PA/FL/MT measurement baseline and exposes the scale assumptions.
- What to learn: preprocessing, segmentation masks, scale handling, post-processing, and
  whether its code can batch-process the 309 images without GUI/manual clicks.

Priority 2: scale coverage for TIFFs

- The current scale detector helps PNGs and a tiny TIFF subset only.
- For cropped TIFFs, likely options are sequence borrowing, metadata/depth extraction,
  visual ruler recovery if any hidden border remains, or DLTrack-style assumptions.
- This is one of the main missing pieces because TIFFs are `251/309` rows.

Priority 3: FL measurement

- Once scale is trusted, test direct line-intersection FL and `MT / sin(PA)` style FL.
- Use confidence gates; FL is more fragile than MT.
- The no-GPU PNG direct-FL variant is the first cheap probe.

Priority 4: PA source selection

- Current evidence suggests ExtraTrees PA is better on the public score than U-Net PA.
- Do not assume better segmentation Dice means better score; validate derived PA error and
  public ablations separately.

Priority 5: better segmentation only after measurement validation

- Fold/TTA U-Nets, self-training, and model ensembles make sense after FL/MT measurement is
  functioning.
- Bigger networks alone will not solve missing pixels-to-mm scale.

## Plain recommendation

Do the cheap ablation first because it costs no GPU:

```text
best PA + PNG-only calibrated MT
```

Then stop chasing tiny column swaps and put serious effort into DLTrack/headless scale and
FL measurement. That is the likely route to a significant difference.
