# EXP62 Oracle Scale Submission Candidate

## Purpose

Create a narrow public candidate from the EXP61 scale audit. This is not a broad
OCR/scale rewrite. It starts from the current public-best CSV and changes only
the rows with explicit field-depth scale candidates.

## Candidate

Script:

```powershell
python experiments\exp62_oracle_scale_submission_candidate.py
```

Output:

```text
results/submission_burn_18_oracle_scale_198_200_direct.csv
```

Changed rows:

| image_id | scale px/cm | FL delta | MT delta |
|---|---:|---:|---:|
| `IMG_00198.tif` | 159.33 | +15.81 mm | -0.31 mm |
| `IMG_00199.tif` | 159.33 | +10.72 mm | -2.20 mm |
| `IMG_00200.tif` | 159.33 | +16.93 mm | -0.36 mm |

PA is unchanged. The candidate changes only 3/309 rows.

## Interpretation

This is the cleanest scale-only public probe from the first oracle pass. It is
much safer than broad tail/mean scale edits because it is tied to visible
`3 cm` depth/ruler evidence and explicit field-height geometry.

It is still not guaranteed to improve: if the prior/temporal/global FL pin was
accidentally compensating for these rows, increasing FL could hurt. But this is
the correct isolated way to test the newly found scale evidence.
