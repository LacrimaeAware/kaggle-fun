# Claude audit — the other model's recent scale work

Scope: commits `10bf610` (scale cross-check), `2213725` (subpixel wiring), `8f39b92` (scale-tail
recovery). I read the committed code and independently checked the claims against the real artifacts.
I did not edit any production files.

## One-line verdict

The work is **careful, honest, and baseline-safe** — but it is three commits deep into the
**already-solved scale lever**, its score impact is **negligible-to-marginal**, and the one
decision-relevant test it keeps naming (orientation correctness) keeps being deferred. The pattern you
noticed — "it says it'll go that route, then doesn't" — is real and visible in the commit sequence.

## Claims I verified that hold up (credit where due)

- **Baseline is genuinely preserved.** `results/submission_local.csv` is **byte-for-byte equal** to
  `0P61918_submission_local.csv` (0 of 309 rows changed on any term). All three commits kept the
  shipped file at the 0.61918 baseline. This is the most important safety property and it's honest.
- **The subpixel pass is off-by-default.** Commit `2213725` edited production `scale_ticks.py` and
  `segment_then_measure.py`, but the submission is unchanged — so the refinement is wired as an
  available/diagnostic path, not enabled. No silent change to the shipped output. Honest.
- **The tail candidates are genuinely isolated** (I initially mis-flagged this — see below). The
  `bar_only` candidate meaningfully changes exactly **4 rows**; the "307 rows changed" I first saw is
  sub-0.3 mm FL **recenter ripple** (the FL mean-pin re-scales all rows when any row moves). The
  model's "isolated candidate" framing was correct.
- **Risk tiering is reasonable** (bar > shape > stacked), and it correctly refused to stack changes
  or auto-submit.

### My own corrected error (for the record)
My first pass flagged "candidates change 307/309 rows with FL deltas to 43 mm" as a red flag. That was
wrong: I was counting femtometer-level recenter ripple. Drilling in showed only **4 rows** move
meaningfully. Checking beat asserting — worth noting because the same discipline is what this whole
project keeps needing.

## What is overstated or carries real risk

1. **`bar_only` is not "low risk."** `recover_bottom_scale_bar_3cm` (exp21) detects a bright
   horizontal run of 250–330 px in the lower-right and **divides its length by 3.0 — a hard,
   unvalidated "3 cm" assumption.** There is no OCR of the bar label and no cross-check; the
   function's own docstring warns "do not generalize this detector without OCR or family QA." The 4
   recovered rows jump from the fallback MT 18.63 to **MT 21–30 mm**, with three of the four at
   **26–30 mm** (the high end of the whole dataset). If that bar is actually 4 cm, those four rows are
   inflated ~33%. This is the **same class of assumption that burned the Telemed-1 cm-vs-German-5 mm
   call earlier** — reading a bar and assuming its physical length. It should be visually/OCR-confirmed
   before any leaderboard probe, and rated medium, not "strongest, low risk."

2. **The subpixel integration cannot move the score.** Its own candidate delta vs baseline is
   **FL 0.094 mm mean / MT 0.024 mm mean** — against tolerances of **12 mm and 3 mm**. That's ~0.8% of
   the FL tolerance. Even if shipped (it isn't), it would change nothing the metric can see. The
   REVIEW3 complaint it was answering ("validated tools aren't in production") is therefore still not
   resolved — and resolving it wouldn't matter, because the effect is below the noise floor.

3. **The tail recovery's ceiling is tiny by construction.** It touches 4 rows (`bar_only`) to 14 rows
   (`all_tail`) out of 309. Even if every recovered scale were perfect, the maximum possible score
   movement is small — and `shape_only` borrows scale from same-canvas neighbors (assignment, not a
   per-image read), which is the higher-risk family-assignment pattern again.

## The pattern (your actual question)

Three commits in a row, all scale: cross-check → subpixel → tail. Scale is the **solved** lever — on
clean data the measurement is already leader-class; the remaining gap is term2 geometry plus whatever
we can't see. Each of these commits ends by naming the orientation-correctness audit as "the next real
step," and then the next commit is more scale. That is the deferral you flagged. It is not random
thrashing — every piece is gated and baseline-safe — but it is **diminishing returns on a solved
problem while the one decision-relevant test goes unrun.**

## Is it progress or busywork?

Mostly careful busywork, with one marginal-but-real item: the 4 `bar_only` rows, **if** the 3 cm
assumption holds. Nothing in these three commits can move the leaderboard by more than a rounding
error except possibly those 4 rows, and those are gated on an unverified assumption that
systematically produces high MT. So: not wasted (the cross-check did weaken the "scale is secretly 2×
wrong everywhere" fear, which has real diagnostic value), but not on the critical path either.

## What I would actually do next

The **orientation-correctness audit** — the test that's been "next" for four turns and is the only
cheap, offline, decision-relevant thing left:

- Over the inter-aponeurosis region, measure fascicle orientation with an **independent classical
  estimator** (structure tensor or Radon) and compare it, per image, to the network's PA.
- On the 35-image reference (where we have truth) this calibrates the classical estimator; on the 309
  targets it tests whether the network's orientation is **coherent or confidently wrong** — the
  "presence vs correctness" gap the mask-quality check could not see.
- This is the gate for the bigger bets: if orientation is already correct, then segmentation is fine
  and self-training / denser labels won't help (the gap is term2 geometry + scale, both near ceiling);
  if it's subtly wrong, that both explains the gap to the leader and justifies a GPU run. Either way
  it converts "should we spend a GPU run / a submission" from a guess into an evidence-based call.

Until that audit is run, more scale-tail polishing is optimizing a term that's already at its ceiling.
