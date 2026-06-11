# Synthetic Abstract Geometry Brief

This note is intentionally domain-neutral. Treat every image as an abstract measurement problem:
two boundary curves enclose a region, and thin internal strands run between the boundaries.

## Goal

Build controlled benchmarks where the correct answer is known by construction. This lets us test
measurement rules without relying on uncertain visual interpretation.

Each generated case has:

- an upper boundary curve;
- a lower boundary curve;
- several internal strands, either straight or curved;
- a known pixel scale;
- exact target values for angle, strand length, and gap thickness;
- masks for the boundary curves and strands;
- a score table comparing the current straight-line measurement rule against the exact target.

## Why This Helps

The current measurement rule often fits a straight line to a visible strand fragment and extends that
line until it intersects the two boundary curves. This can work when strands are straight and the
boundaries are simple. It can fail when:

- the strand bends;
- only a short fragment is visible;
- several strands fan toward a common region;
- the boundary is curved but represented as one straight line;
- mixed strand families are combined into one aggregate.

Synthetic cases separate these factors one at a time.

## Generated Files

Run:

```powershell
python umud-muscle-architecture\benchmark_lab\generate_synthetic_geometry.py `
  --n 32 `
  --out-dir umud-muscle-architecture\results\synthetic_geometry
```

Outputs:

- `truth.csv`: exact target measurements for every generated image.
- `measure_light_scores.csv`: predictions from the current straight-line scorer.
- `summary_by_family.csv`: grouped errors by synthetic family.
- `images/`: rendered grayscale images.
- `labels/`: boundary and strand masks.
- `manifest.csv`: viewer-compatible manifest.

Open the viewer:

```powershell
python umud-muscle-architecture\benchmark_lab\review_server.py `
  --synthetic-dir umud-muscle-architecture\results\synthetic_geometry `
  --port 8769
```

## Current Synthetic Families

- `straight_low_curvature`: mostly straight strands, simple boundaries.
- `straight_steeper`: steeper strands, simple boundaries.
- `mild_curved_strands`: small strand curvature.
- `strong_curved_strands`: large strand curvature.
- `curved_boundaries`: boundary curves are not well represented by one straight line.
- `fan_like`: strands bend or fan toward a shared region.
- `partial_low_support`: only short visible pieces are shown.
- `mixed_angles`: strands from different angle families appear in one image.

## First Readout

In the first 32-case generated pack, the straight/steeper families are the easiest for the current
straight-line scorer. Strongly curved and fan-like cases are much harder, especially for strand
length. That is the expected result and makes this pack useful for testing curve-aware alternatives.

## Next Questions

1. Can a curve-aware strand measurement reduce length error on `strong_curved_strands` without
   hurting straight cases?
2. Can boundary curves be estimated robustly enough to reduce the `curved_boundaries` error?
3. Can the scorer detect mixed angle families and avoid aggregating incompatible strands?
4. Can low-support cases be flagged instead of confidently measured?
