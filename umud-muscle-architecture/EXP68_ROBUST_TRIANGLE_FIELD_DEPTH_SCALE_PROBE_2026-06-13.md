# EXP68 Robust Triangle Field-Depth Scale Probe

## Purpose

Apply the same EXP67 broad field-depth scale adjustment to the best actual
309-row benchmark-driven production candidate: robust triangle.

This exists because EXP55/EXP56's `0.131` benchmark route is not production
wired. The closest real CSV candidate is:

```text
results/submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv
public LB: 0.60102
```

## Candidate

Script:

```text
experiments/exp68_robust_triangle_field_depth_scale_probe.py
```

Output:

```text
results/submission_burn_23_robust_triangle_field_depth_scale_probe.csv
```

Summary:

```text
results/submission_burn_23_robust_triangle_field_depth_scale_probe_summary.csv
```

## What It Does

- Starts from burn #15 robust triangle.
- Reuses EXP67's guarded field-depth scale proposals.
- Changes only FL/MT on `114` rows.
- Leaves PA unchanged.
- For rows with old scale, rescales existing robust FL/MT by
  `old_scale / new_scale`.
- For rows without old scale, recomputes FL/MT from robust debug pixels.

## Movement

Compared with burn #15:

| target | changed rows | mean abs delta | max abs delta | mean signed delta |
|---|---:|---:|---:|---:|
| PA | 0 | 0.000 | 0.000 | 0.000 |
| FL | 114 | 4.822 mm | 26.236 mm | -4.301 mm |
| MT | 114 | 1.304 mm | 5.377 mm | -1.214 mm |

Largest mechanism:

```text
90 / 114 changed rows: 50 mm depth and field_h_px = 800 -> 160 px/cm
```

## Audit After Burn #22

Burn #22 applied this same broad scale idea to the public-best baseline and
scored:

```text
0.66197
```

That is a strong negative public result. It means the broad field-depth span
heuristic is probably overcorrecting many rows. EXP68 is therefore diagnostic,
not recommended as a likely improvement.

## Recommendation

Do not promote this scale adjustment as a default.

Only submit burn #23 if intentionally testing whether robust-triangle geometry
interacts differently with the same failed broad scale move. Otherwise, the
more useful next work is to train or improve the scan-field/ruler-span detector
before touching broad scale again.
