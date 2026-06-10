# Current Alignment After Cross-Model Review

Date: 2026-06-10

**Current front door:** `STATE_RESET_2026-06-10.md`. It records the post-0.619 submission failures:
FL blend 0.63905, MT vertical-3 0.62561, and bar-only scale-tail 0.66711. The best anchor remains
`results/submission_local.csv` at 0.61918.

This note reconciles the pasted cross-model review with the work now in the repo. It is meant to
answer: what are we doing next, what is stale, and what should not be confused for a submission
candidate?

## What The Pasted Review Got Right

- The failed FL blend is the durable warning. A local win that depends on reference-set recentering
  or mean behavior is not submission evidence.
- Scale and orientation have been audited much harder than before. Broad scale failure and broad
  orientation collapse are no longer the best explanations for the remaining score gap.
- The production FL recentering step is a real red flag. `exp24_recenter_temporal_audit.py` showed
  that removing it moves 308/309 rows by mean 17.147 mm, so every FL candidate must report whether
  its effect is direct or recenter ripple.
- Public/free/equally accessible external data and models are rules-clean if declared and
  reproducible. Host discussion also treats human-created labels on the 309 test records as external
  data that can be used if declared and made reproducible enough for the final pipeline audit.
- The user cares about the score first. A method can be intellectually satisfying and still not be
  the next score-moving bet.

## Oracle / Creator-Discussion Correction

The earlier alignment note still underweighted the creator/forum context. The actual situation is not
"oracle forbidden"; it is "oracle equals declared external data":

- Human-in-the-loop target analysis is a powerful diagnostic. Public notes include a participant
  manually analyzing the test records and scoring around 0.459, so an oracle-style process can
  absolutely expose what the automated system is missing.
- The host/forum context also makes clear that masks, manual analysis, and final measurements are
  not identical. A human label still needs a reproducible measurement pipeline, and expert variation
  remains.
- The host explicitly says labeling or fine-tuning on the 309 test images is a use of external data
  that needs to be declared. That means an active-learning loop, user right/wrong checks, or full
  manual target annotation are not automatically disallowed here; they are human-in-loop external
  data strategies.
- The written Kaggle rule still explains why disclosure matters. Do not silently tune on human target
  judgments and then present the result as a purely automated run.

Operational stance: default to the automated/no-oracle track until the user explicitly chooses a
declared human-in-loop track. If we do use human judgments on target records, log every query/answer,
save the labels/protocol, declare them as external data, and keep the final pipeline reproducible.
Code-generated target pseudo-labels remain in the automated track if they are not hand-corrected.

## What Is Stale Or Superseded

- "Run the error budget" is done. `exp25_reference_error_budget_adapter.py` built the adapter. With
  oracle scale on the 35-image reference set, recentered FL remains the largest local term
  (PA 0.1498, FL 0.3528, MT 0.1795).
- "Run the recenter/no-recenter audit" is done. That is `exp24`.
- "Run the orientation correctness audit" is done. `exp22_orientation_raw_support.py` compares
  predicted fragment orientation against independent raw-image structure tensor orientation. It is
  not just internal coherence.
- "Inventory external/public assets" is done. `exp27` confirms 1048 + 2761 public image/mask pairs,
  the 35-image benchmark, 309 target images, and one public weight file are already local.
- "Try a learned scale-cue detector" is partly done. `exp26` exports weak labels, `exp28` trains a
  cue model, and `exp29` audits it.

## Cue Learning: Current Verdict

The user's ML instinct was right: if we keep hand-building every ruler/tick reader forever, we are
underusing ML. But the first learned scale-cue model is not a score candidate.

Evidence for it:

- It learned useful presence signal against the weak teacher for some cue classes:
  left-ruler F1 1.000, right-ruler F1 0.978, bottom-tick F1 0.702, UI-signature F1 0.698.
- It gives a second, independently trained signal for QA/disagreement against the deterministic
  router.

Evidence against it:

- The mask model is weak as a segmenter: the improved run reached only 0.1644 weak-label val Dice.
- The lower-bar class has only four positives and failed as a learned class (F1 0.046).
- It learns from the deterministic teacher, so it mostly inherits teacher blind spots. It is
  currently a robustness/QA tool, not a router replacement.

Decision: keep exp26-29, but do not spend a submission on them and do not make them production scale
logic yet. If this path continues, the next version should be ROI/crop-based rather than full-frame
thin-mask segmentation.

## Jensen Bias Status

The pasted review's Jensen warning is valid for a mean-of-fragment-lengths combiner. Production does
not currently do that: `segment_then_measure.py` uses median fragment FL (`fl_fragment_px =
median(fls)`) and keeps the rejected identity blend off by default. So Jensen bias is a caution
against future combiners, not the next obvious production bug.

## Score-First Next Direction

The score-first path is no longer more scale polishing. The current ranking is:

1. Preserve the downloaded 0.61918 baseline as the comparison anchor.
2. Reject the isolated host-protocol MT candidate:
   `results/submission_host_mt_vertical3_no_subpixel.csv`.
   - It improved the 35-reference score 0.2274 -> 0.2192 locally, but the public LB worsened
     0.61918 -> **0.62561**.
   - PA/FL changed on 0 rows, so the regression is attributable to MT only. The hidden labels are
     closer to the old center/perpendicular MT path, or our vertical-3 approximation does not match
     the target labeling well enough to help.
3. Do not submit the FL low-extrapolation top-3 candidate. It was a good structural hypothesis but
   worsened the local FL term (0.3528 -> 0.3668).
4. Do not stack scale-tail into the rejected MT candidate. The tail files are real scale probes, but
   they move FL on 307 rows through recentering. The bar-only split was submitted and worsened public
   LB 0.61918 -> **0.66711**, so tail is rejected as the next-probe path.
5. Read `NEXT_SUBMISSION_REVIEW.md` before asking another model to verify the current submission.
6. Stop isolated CSV probes until there is either a verified bug fix, a declared human-in-loop
   validation/tuning path, or a substantial model branch.
7. Use exp29 only as a disagreement audit for the router, not as production logic.
8. Move the main score effort to measurement/model quality:
   - controlled public-asset retraining or fold/seed ensembling,
   - denser/cleaner structure supervision on public training images,
   - conservative self-training only through exp23-style gates,
   - temporal-only as a small isolated probe, not a presumed margin-closer.
9. Keep all FL changes isolated from recentering effects and report direct row movement.

## What We Should Not Do

- Do not ask the user to mark target images right/wrong and then use those marks while pretending the
  run is no-oracle. If we choose that route, call it declared human-in-loop external data and document
  it.
- Do not submit MT vertical-3 again.
- Do not submit bar-only, shape-only, or all-tail scale-tail again without new evidence.
- Do not submit cue-model output.
- Do not submit FL low-extrapolation top-3.
- Do not submit another reference-mean or local-FL win without structural evidence.
- Do not treat "more ML" as automatically score-moving. The ML path must target the measured
  bottleneck, not a term that is already mostly audited.
