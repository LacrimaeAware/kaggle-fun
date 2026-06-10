# Current Alignment After Cross-Model Review

Date: 2026-06-10

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
  reproducible. Human labels or human predictions on validation/test records are not.
- The user cares about the score first. A method can be intellectually satisfying and still not be
  the next score-moving bet.

## Oracle / Creator-Discussion Correction

The earlier alignment note underweighted the creator/forum context. The actual situation is more
nuanced than "oracle bad":

- Human-in-the-loop target analysis is a powerful diagnostic. Public notes include a participant
  manually analyzing the test records and scoring around 0.459, so an oracle-style process can
  absolutely expose what the automated system is missing.
- The host/forum context also makes clear that masks, manual analysis, and final measurements are
  not identical. A human label still needs a reproducible measurement pipeline, and expert variation
  remains.
- But for our own submission path, the written Kaggle rule still matters: submissions may not
  incorporate information from hand labeling or human prediction of validation/test records.

Operational stance: the user may look at overlays to understand the method, but we should not turn
their per-image right/wrong judgments into training labels, row filters, corrections, thresholds, or
submitted values for the 309 target records. If a human judgment changes the target-row output path,
we treat it as unsafe. If public/external training records are labeled, or code generates target
pseudo-labels reproducibly, that remains in-bounds if declared.

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
2. Use exp29 only as a disagreement audit for the router, not as production logic.
3. Move the main score effort to measurement/model quality:
   - controlled public-asset retraining or fold/seed ensembling,
   - denser/cleaner structure supervision on public training images,
   - conservative self-training only through exp23-style gates,
   - temporal-only as a small isolated probe, not a presumed margin-closer.
4. Keep all FL changes isolated from recentering effects and report direct row movement.

## What We Should Not Do

- Do not ask the user to mark target images right/wrong and use those marks. That is human
  prediction/hand labeling of target records.
- Do not submit cue-model output.
- Do not submit another reference-mean or local-FL win without structural evidence.
- Do not treat "more ML" as automatically score-moving. The ML path must target the measured
  bottleneck, not a term that is already mostly audited.
