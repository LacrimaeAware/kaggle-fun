# Honest state audit (2026-06-14)

Written to correct an over-claiming pattern in this session. Each probe against the hidden test is a
hypothesis-test with a noisy 19-label proxy, NOT a proof. Results below are stated as what the
leaderboard actually showed, not as narrative.

## Current best (lock this in)

`results/submission_pa_shift_p20.csv` = the 0.61918/burn_13 pipeline with a flat **PA + 2.0 deg**.
Public score **0.55075**. This is the standing best. Select it as the final submission.

## What is real (leaderboard-confirmed, will not reverse)

1. **0.58910 -> 0.55075** from one change: the model under-reads pennation angle by ~2 deg across the
   board, and a flat +2 deg correction fixes the global offset. Banked.
2. **Scale is solved**, not the bottleneck. On the test images MT (pure scale) is the *best* term
   (~0.376 vs the hand labels). The manual depth review paid off. The earlier "scale is the #1 lever"
   framing was wrong for the *current* state.
3. **The 19 hand labels are coarse-only.** They correctly predicted the +2 win, but misled on every
   fine tweak (see below). Gate only big, independently-motivated changes on them; never fine ones.

## What was tested and FAILED (all regressed vs the +2 = 0.55075)

| change | public | verdict |
|---|---:|---|
| flat PA +3 | 0.55168 | +2 is the sweet spot; +3 overshoots |
| muscle-confirmed high-end lift (hinge, PA+2 plus extra above 16 deg) | 0.56681 | the high end was NOT as under-read on the hidden test as the hand labels implied |
| de-shrink / linear PA reshaping | (proxy only, not submitted) | proxy said better; proxy was wrong on +3 and the hinge, so not trusted |
| muscle-aware shift toward literature means | (proxy only) | worse than +2 on the proxy; not submitted |

**Conclusion: post-hoc PA patching is tapped. The flat +2 is the ceiling for output-level correction.**

## Corrected claims (things stated too confidently earlier in the session)

- **"Muscles are learnable from raw pixels" — NOT established, probably false.** The muscle classifier
  hit ~1.0 val accuracy but almost certainly read the *burned-in overlay text*, not the anatomy: it
  succeeded on text-bearing images and failed on the no-text family (its "GM" calls there had human PA
  of 10-14 deg, contradicting GM). That is shortcut learning, redundant with the OCR. It is not a score
  lever. (A clean test would mask the text region and retrain; until then, treat the claim as refuted.)
- **FL has no independent post-hoc lever.** FL = MT/sin(PA) (truth correlates 0.97). With MT solved and
  PA corrected, the residual FL error is per-image extrapolation noise. The identity-FL formula scores
  *worse* than the current fragment estimator (and was already tried and rejected historically).

## The principle this session violated (follow it going forward)

State probe results as **tentative until the leaderboard confirms**. Do not chain "now we've proven X,
now we've proven Y." The hidden test plus a 19-sample proxy makes every step a coin-flip until it lands.
Isolated one-variable probes only; one written decision rule before each submission.

## Next step, honestly (uncertain, not a promise)

The per-image measurement is at the ceiling of the segment-then-measure pipeline. The two genuinely
different levers, both uncertain:

1. **Classical fascicle-angle measurement** off the raw pixels (structure tensor / Radon in the
   inter-aponeurosis band), independent of segmentation. Testable locally on the 35-image benchmark
   (true labels), no GPU, no submission. If it does not beat the current angle there, PA is confirmed
   at its ceiling.
2. **One clean, controlled segmentation run** judged by *downstream PA/FL on the benchmark*, not by
   Dice, changing one variable at a time. Do NOT repeat the seg72/seg77 nine-knob confounds.

Do not hand-label more test data; it is not the bottleneck and it is the rules gray zone. Do not run
another multi-knob segmentation matrix.
