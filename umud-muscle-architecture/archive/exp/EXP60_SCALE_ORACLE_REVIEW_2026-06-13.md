# EXP60 Scale Oracle Review

## Purpose

The scale reader currently assigns a scale to every test image, but not every
assignment has the same evidence. This experiment turns that into a human
review workflow instead of treating the detector as ground truth.

The user can act as the oracle for a small number of cases first, then expand
to all 309 if needed.

## What Exists

- Builder: `experiments/exp60_scale_oracle_review_pack.py`
- Server: `benchmark_lab/scale_review_server.py`
- Viewer: `benchmark_lab/scale_review_v1/`
- Generated local-only files:
  - `results/scale_oracle_review/manifest.csv`
  - `results/scale_oracle_review/start_pack.csv`
  - `results/scale_oracle_review/oracle_notes.json`

The generated review files live under `results/`, so they are ignored and do
not expose private human notes in the public repo.

## Current Scale Groups

From the current `results/scale_partition.csv`:

- `confidence_check`: verified or text-confirmed rows. These are not the main
  work, but the starter pack includes a few so the human can check whether the
  machine is actually right when it claims confidence.
- `tick_only_oracle`: tick detector gives a concrete scale, but OCR/text logic
  did not independently confirm it.
- `urgent_oracle`: fallback or flagged rows. These are the highest-value human
  checks.

Important clarification: `tick-only` does not mean there is no visible ruler or
number. It means the current code did not get an independent OCR/ruler
confirmation strong enough to promote the row to verified/text-confirmed.

## How To Run

Build or refresh the review pack:

```powershell
python experiments\exp60_scale_oracle_review_pack.py
```

Start the local viewer:

```powershell
python benchmark_lab\scale_review_server.py --port 8773 --pack start
```

Open:

```text
http://127.0.0.1:8773/scale-review/
```

The viewer can switch between `Starter pack` and `All 309`. It autosaves
status, corrected scale, visible ticks/depth, and comments.

## Notebook Clarification

For segmentation retraining, the practical Kaggle path is notebook-shaped, not
just a one-line command. The command-line flags added in EXP59 are still useful
because the Kaggle notebook can set the same environment variables before
running `segment_then_measure.py`. RunPod/local shell can use the one-liner
directly, but Kaggle should use a sequence of cells:

1. install/check dependencies,
2. set `UMUD_*` configuration variables,
3. run training/inference,
4. verify outputs,
5. save submission and tagged weights.

## Next Use

1. Review the starter pack first.
2. If confident rows are mostly correct, focus on `urgent_oracle` and a sample
   of `tick_only_oracle`.
3. Feed confirmed corrections back into the scale partition as explicit rows,
   not as a broad mean/tail assumption.
4. Only after that, decide whether a scale-only public submission is worth
   another slot.
