# Next Label Pack

Created: 2026-06-11

## Purpose

Stop using the leaderboard as the only judge. This pack should answer whether the 0.619 baseline and
the rejected 0.665 facing-FL variant differ for reasons we can see and measure locally.

## Files

- Public/external manifest:
  `results/human_benchmark/public_seed_manifest.csv`
- Declared human-in-loop target manifest:
  `results/human_benchmark/target_seed_manifest.csv`
- Target labels save folder:
  `results/human_benchmark/target_labels/`
- Public labels save folder:
  `results/human_benchmark/public_labels/`

## First Target Rows

The target manifest starts with `IMG_00275`, the known scale anomaly where ticks and printed ruler
disagree by about 2x. After that it samples target images randomly. The next refinement should replace
some random rows with known hard cases from per-gap/facing failure diagnostics.

## Labeling Goal

Minimum useful pack:

- 8 to 12 target rows with clean aponeurosis + fascicle masks;
- at least 4 rows that look like multi-gap or multi-muscle failures;
- at least 4 public/FALLMUD rows to verify that the label convention and scorer behave sensibly
  against existing public masks.

Draw only visible structures. Do not extrapolate off-screen or across missing regions. For `apo`,
drawing the gap-facing boundary line is acceptable; it does not need to be a perfect full-band fill.
Multiple strokes are okay, but avoid connecting the upper and lower boundaries into one blob. Use
`dot line` for straight fragments, `brush` for filled boundary bands, and `curve chain` only when a
visible structure is genuinely curved. Leave scale/PA/FL/MT boxes blank unless you intentionally
measured them elsewhere; the scorer derives those from the masks. Use the top-bar zoom controls when
the frame is too small.

Decision threshold:

- The pack is useful once `score_labels.py` can measure human masks on at least 8 target rows.
- A new geometry submission is only worth considering if it improves those rows without breaking the
  public/FALLMUD sanity rows.

## Commands

Start target labeling:

```powershell
python umud-muscle-architecture\benchmark_lab\label_server.py `
  --manifest umud-muscle-architecture\results\human_benchmark\target_seed_manifest.csv `
  --out-dir umud-muscle-architecture\results\human_benchmark\target_labels `
  --port 8765
```

If `8765` is already occupied, use `8766` or another free port.

Start public/FALLMUD labeling:

```powershell
python umud-muscle-architecture\benchmark_lab\label_server.py `
  --manifest umud-muscle-architecture\results\human_benchmark\public_seed_manifest.csv `
  --out-dir umud-muscle-architecture\results\human_benchmark\public_labels `
  --port 8766
```

Score target labels:

```powershell
python umud-muscle-architecture\benchmark_lab\score_labels.py `
  --manifest umud-muscle-architecture\results\human_benchmark\target_seed_manifest.csv `
  --labels-dir umud-muscle-architecture\results\human_benchmark\target_labels `
  --out umud-muscle-architecture\results\human_benchmark\target_scores.csv
```

Score public labels:

```powershell
python umud-muscle-architecture\benchmark_lab\score_labels.py `
  --manifest umud-muscle-architecture\results\human_benchmark\public_seed_manifest.csv `
  --labels-dir umud-muscle-architecture\results\human_benchmark\public_labels `
  --out umud-muscle-architecture\results\human_benchmark\public_scores.csv
```
