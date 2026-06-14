# EXP79 PA aggregation bias investigation (2026-06-14)

Status: a real, evidence-backed lever was found, then red-teamed down to "real but smaller and more
uncertain than it first looked." This doc records the finding, the disconfirming evidence, and the
disciplined next steps. It supersedes the part of `METHODOLOGY_AUDIT_2026-06-13.md` that called PA
"solved": PA is solved *in-distribution* and runs low *on the test distribution*.

## The question this answers

Where does the public 0.589 error actually live, on the real test set (not the 35-image benchmark
which is fed true scale and recentered, so it cannot see the public gap)? Run with local artifacts
only (no GPU/torch): `calibration_measurement_debug.csv` (309 live rows), `benchmark_pred_truescale.csv`
+ expert truth, and `results/human_benchmark/target_human_vs_submission.csv` (19 rough hand labels on
actual test images). Scripts: `experiments/diag_state_audit_2026_06_14.py`,
`experiments/diag_pa_bias_2026_06_14.py` (numbers independently re-derived and confirmed).

## What the data shows (verified)

Per-term A-proxy of the live best submission vs the 19 hand labels (tol PA6/FL12/MT3):

| term | norm MAE | bias | read |
|---|---:|---:|---|
| MT | 0.376 | ~0 | **best term. The scale/manual-depth work paid off.** |
| FL | 0.656 | -0.46mm | high spread, near-zero bias (the FL recenter is holding the mean) |
| PA | 0.642 | model -2.9deg (median) | worst, and it is a one-sided bias: 18 of 19 labels say true PA > predicted |

- Overall A-proxy 0.558 vs actual public 0.589: the 19 rough labels predict the board within ~0.03.
  They are a usable directional gate, not "no validation signal."
- On the **expert benchmark** (in-distribution): PA bias -0.32deg, truth~0.99*pred. PA is genuinely
  solved there. The under-prediction appears **only** on the domain-shifted test set.
- Model PA mean across all 309 = **14.64deg**; expert-truth mean 18.34; hand-label mean 19.86.

## The red-team (why this is NOT a clean +3deg win)

Three independent checks (one numerical re-derivation, two adversarial) found:

1. **The benchmark-mean argument is population-confounded.** The benchmark muscles are
   gastrocnemius/soleus/VL (high pennation). The test set adds **Rectus femoris (low pennation,
   ~5-14deg)** and **cerebral palsy** subjects. Both push the *true* test PA mean down, so
   model 14.64 < benchmark 18.34 is partly correct calibration, not pure bias. Comparing a test mean
   to a different-population benchmark mean is the same category error this project keeps making.
2. **The hand-label "human" PA is self-measured, not independent.** Per `target_scores.csv`,
   `human_pa_deg = pa_deg_measured` via the same `light_cv2_numpy` geometry engine on the user's
   hand-clicked masks. So 18/19 one-sidedness can come from ONE steepening habit in the clicks, not 19
   independent confirmations. MT being unbiased while PA is biased supports this (a clicking habit
   distorts angles but not the vertical thickness).
3. **The effect is concentrated, and the instrument is wrong.** human/sub PA ratio ~1.20 (multiplicative,
   not additive); the 19 labeled images are a higher-PA subset (submission mean 16.3 vs full-set 14.6);
   the test median PA is 14 with 30% below 12.5. A blanket constant +3 over-corrects the low-PA bulk.

## The reconciliation (the actual mechanism)

`compare_lines.py` (the per-fascicle check) found the model's slope field is within **~1.8deg** of the
user's drawn fascicles. So the model fits *individual* fascicle angles well; it is the **fragment
selection / aggregation** that runs low in aggregate on harder images (it averages in shallower /
apo-parallel fragments, or misses steep faint ones), plus the hand labels are likely somewhat steep.
The real aggregate bias is therefore somewhere between ~0 and ~3deg, likely ~2deg, and it lives in the
higher-PA images, not the low-PA Rectus-femoris bulk.

This matters because the fix is **fragment selection**, which connects directly to the EXP78
recall-heavy idea ("guess more, then filter geometrically") — but the filter should favor the steep,
clean fascicles, not just add pixels.

## Plan (in order)

0. **No-slot verification first.** On 8-10 test images, overlay the model's accepted fascicle fragments
   against the visible steep fascicles (use `compare_fasc_viewer.py` / `per_gap_viewer.py`). Question:
   does the model's *accepted-fragment set* skew shallower than the clear fascicles a human would pick?
   If yes, the aggregate bias is real and the lever is fragment selection. If the model's fragments
   already match the steep fascicles, the +3 is mostly a hand-label habit and we drop the shift.
1. **One isolated probe, conservative.** If spending a slot, submit `submission_pa_shift_p20.csv`
   (PA + 2.0deg, clip [5,45], FL/MT untouched). +2.0 is inside the overlap of all evidence (1.8deg
   per-fascicle, ~2.9deg aggregate-but-confounded). Decision rule: improves the LB -> the bias is real,
   escalate (de-shrinkage or aggregation fix); regresses -> the labels were steep-biased, abandon the
   shift. Either outcome is high-information and costs one slot. Do NOT fire +3/+4.5/linear blind.
2. **seg77 recall-heavy follow-up (already running).** When weights land, run the inference-only
   recall-heavy variants (lower fascicle threshold/min-area) AND check whether the resulting accepted
   fragments raise the aggregate PA toward the hand labels. That is the root-cause test of the
   aggregation bias, and the real justification for the segmentation work.
3. **FL stays.** FL is high-variance but unbiased on the test labels; the FL recenter is currently
   masking a real ~+6mm fragment-FL overshoot (benchmark FL bias +5.79mm un-recentered), so do NOT
   just delete it. Fix FL at the source (steeper aggregation lowers the overshoot) before touching it.

## Validation discipline (the actual disease, restated)

Gate every probe on the 19 test-distribution labels AND a visual check, never on the 35-image
benchmark. Treat the 19 labels as directional (they are self-measured). One slot, one isolated
hypothesis, with a written decision rule before submitting.

## Artifacts created
- `experiments/diag_state_audit_2026_06_14.py`, `experiments/diag_pa_bias_2026_06_14.py` (diagnostics)
- `experiments/make_pa_calibration_probes_2026_06_14.py` (probe generator)
- `results/submission_pa_shift_p15.csv` / `p20` / `p25` / `p30`, `submission_pa_shift_p45.csv`,
  `submission_pa_linear.csv` (PA-only isolated probes; p20 is the recommended first slot)
