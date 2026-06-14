# UMUD Methodology Audit (2026-06-13)

> **Correction (2026-06-14):** Section 1's claim that "PA is solved" holds only *in-distribution*.
> A later diagnostic (`EXP79_PA_AGGREGATION_BIAS_INVESTIGATION_2026-06-14.md`) found the model's PA
> runs ~2deg low *on the test distribution* due to fragment aggregation, and that MT (scale) is
> actually the best term on the real test images. Read EXP79 alongside this file.

A logic-and-process audit, not a code review. Scope: the whole folder (121 py, 76 md, 6 notebooks).
Built from a direct read of the problem definition, the metric, the core thought-process docs
(`synthesis.md`, `MASTER_REVIEW.md`, `competition_reference.md`), the production engine
(`segment_then_measure.py`), and a forensic pass over the experiment log, the proxy machinery, the
scale stack, and the segmentation work. Every number below is quoted from the repo or the metric.

This file is meant to **replace** the scattered self-audits as the decision layer, not become doc #77
that gets admired and not used. See the last section on doc hygiene.

---

## 0. One-paragraph verdict

You do not have a knowledge problem. Your own docs state the correct understanding repeatedly:
`MASTER_REVIEW.md` section 0 nails the validation trap, `synthesis.md` section 4d names the single
decisive experiment, section 6 of MASTER nails the FL convention insight. You have an **enactment
problem**. The understanding gets written down and then violated, because the loop that chooses what
to do next is steered by a validation signal you have *proven* cannot predict the score. The result
is many weeks of disciplined-looking motion that mathematically cannot move the leaderboard, because
it optimizes a coordinate the leaderboard does not measure. The fix is not more methods. It is to
stop optimizing the broken signal, run the one cheap experiment that settles the strategy, and gate
every future submission on a signal that actually transfers.

---

## 1. What this problem actually is, stated correctly

This is the part the docs circle but never pin down in one place, so here it is.

**The metric is normalized mean absolute error.** From `metric.py`: the secondary (median) and
tertiary (RMSE) terms are weighted 1e-6 and 1e-9, so they are tie-breakers only. The score is, to
five decimals:

```
score = (1/3) * [ MAE(PA)/6  +  MAE(FL)/12  +  MAE(MT)/3 ]
```

Tolerances PA=6 deg, FL=12 mm, MT=3 mm are the normalizers. Lower is better.

**Two of the three targets are already at the floor.** On the clean 35-image benchmark with true
scale your terms are PA 0.1498, FL 0.3528, MT 0.1795. Convert out of normalized units:

| target | your MAE | tolerance | human/label noise (inter-rater SD) |
|---|---:|---:|---:|
| PA | ~0.90 deg | 6 deg | ~1.6 deg |
| MT | ~0.54 mm | 3 mm | ~1.0 mm |
| FL | ~4.2 mm | 12 mm | ~5.4 mm |

PA and MT are already **below the inter-rater noise of the labels themselves**. They cannot be
meaningfully improved by any method, because you are already inside the disagreement band of the
humans who made the ground truth. `MASTER_REVIEW.md` section 7 confirms PA is fit within 1.8 deg by
hand-drawn check. So this is **not** a "predict three numbers" problem. It is an **FL problem and a
scale problem**. Every hour spent on a PA proxy or an MT proxy is spent on a term that is already at
ceiling. (You submitted a vertical-MT proxy and a visibility-weighted-FL proxy anyway; both
regressed.)

**FL has two couplings that dominate everything else:**

1. `FL = MT / sin(PA)` is a geometric identity, so a small PA error amplifies into FL.
2. FL and MT are both produced by dividing pixels by the same px-per-mm scale, so **a scale error
   double-counts**: it hits FL and MT in the same direction, and the metric sums them. This is why
   `competition_reference.md` calls scale "the #1 leaderboard lever." It is correct, and it follows
   directly from the metric. Scale is the single highest-leverage variable in the whole problem.

**Your FL estimator is more than 90% extrapolation.** `benchmark_error_taxonomy.md` records that the
visible fascicle fragment often supports less than 10% of the projected full length. The FL number is
therefore determined by fragment **slope**, the two aponeurosis line fits, and the **extrapolation
convention**. It is essentially **not** determined by how many fascicle pixels you segment. A 12-pixel
fragment and a 200-pixel fragment at the same slope give the same FL. Keep this fact; it kills a whole
research direction in section 4.

**The FL ground truth is a human convention, not anatomy.** `MASTER_REVIEW.md` section 6: the raters
measured a straight-line, low-extrapolation span between aponeuroses. The real fascicle bends (you
confirmed this on FALLMUD, parabola fits 44% better), but the *scored* quantity is the straight-line
convention. This is one of your best findings. You then repeatedly proposed anatomy-faithful
curvature that diverges from the convention, and it kept failing the metric, exactly as the insight
predicts.

**Correctly stated:** win FL by nailing the rater's straight-line low-extrapolation convention and
the px-to-mm scale correctness. Everything else is at or near the noise floor. If a planned piece of
work is not moving FL-via-convention or scale-correctness, it almost certainly cannot move the score.

---

## 2. The central error: you are optimizing a compass you have proven is broken

There are three score scales and you documented them correctly (`MASTER_REVIEW.md` section 0):

- **A** = public leaderboard (the only number that counts). Scale must be recovered, FL bias is
  exposed. Your best is 0.589.
- **B** = the 35-image expert benchmark. Fed **true scale** and **FL recentered to the true mean**.
  By construction it is **blind to the #1 lever (scale)** and **hides FL bias**. You wrote: "A sanity
  tool, NOT an oracle (it has mispredicted the LB direction 4 times)."
- **C** = other people's reference scores.

You wrote the rule yourself: "only compare A to A ... The project's single recurring mistake was
treating a B-scale win as a reason to spend an A submission." That sentence is dated 2026-06-10.

Then the record shows exactly that mistake, repeatedly, **after** the rule was written:

- Experiments EXP38 through EXP56 ground the B score from 0.170 down to **0.131** through story
  weights, saturating support, class routes, and term routes. EXP53 is literally labelled "current
  local research best"; EXP54 crowns an "overall winner." This is a leaderboard-of-B framing for a
  metric you classified as a non-oracle.
- That B headroom is a mirage. B is at 0.131 while A is stuck at 0.589. The 2.6x gap between them
  **is** the scale-plus-FL-bias gap that B is built to hide. Lowering B does not close it.
- You then spent real submissions translating B-wins onto the board. Every one regressed:
  robust triangle 0.601, visibility-weighted FL 0.645, vertical MT 0.607, field-depth scale 0.662,
  local-benchmark proxy stack 0.659. Out of roughly 18 public submissions in the whole project,
  exactly **four** ever improved a standing best (the scale router, temporal smoothing, subpixel
  scale, shape-neighbor scale). Since the last real win you are **0 for 5**.

This is the streetlight effect in its pure form. The keys (the score) are in the dark (scale and FL
on the hidden set, which B cannot see). You keep searching under the lamp (B, where movement is
visible) because that is where you can see your hands move. Lowering B feels like progress and is not.

**The biggest single logic error in the project: using B, or visual plausibility, as a submission
gate, after proving B does not transfer.**

---

## 3. The actual root cause: you have no transfer-valid validation set

Section 2 is the symptom. This is the disease. You have **no local signal that predicts A**. With no
such signal, every submission is a blind probe, which is why you are 0-for-5 on blind probes. B is
not it (proven). Visual plausibility is not it (the candidates "looked right on the canvas" and
regressed). Internal scale cross-cue agreement is not it (section 7).

The one asset that *is* on the A distribution is the 19-of-24 rough human labels on the **actual test
images** (`benchmark_lab/`, `results/human_benchmark/`). `MASTER_REVIEW.md` section 0 calls this "the
first local oracle on the actual test distribution, the thing that fixes our core problem," and then
the very next sessions keep spending slots without gating on it. You built the right instrument and
left it on the bench.

**Until you have a validation signal that predicts A, you should not spend a single further
submission.** This is not a methods question. It is the precondition for any methods question to be
answerable.

---

## 4. The segmentation pivot rests on a premise your own evidence contradicts

The current "active direction" is GPU segmentation retraining (EXP59/72/76/77). The stated
justification is "FL is mask-limited, the FL gap to DL-Track is fascicle mask quality"
(`MASTER_REVIEW.md` sections 7 and 9.4). That premise is contradicted, inside your own repo, four
ways:

1. `benchmark_findings.md`: with **true scale**, FL is still ~1.19, "no better than a constant even
   with perfect scale ... a geometry problem, not a scale problem." Geometry, not pixels.
2. `MASTER_REVIEW.md` section 4: the recall-bias segmentation retrain made the benchmark **worse**,
   0.368 to 0.484. Verdict in your own words: "more fragments = noise; **can't fix FL by drawing more
   pixels**." This is the opposite of "FL is mask-limited."
3. PA is already accurate to 1.8 deg. Better fascicle masks mostly sharpen PA and fragment count.
   PA is at ceiling, so that gain is wasted.
4. The FL estimator is >90% extrapolation (section 1), so it consumes fragment **slope**, not
   fragment **pixel recall**. Higher Dice barely propagates to FL.

So "FL is mask-limited" and "FL is geometry-limited, more pixels make it worse" both live in your
docs, written days apart, with no experiment adjudicating between them, and the **GPU strategy is
staked on the one the evidence rejects**.

Worse, the decisive experiment that would settle it is already designed and costs zero submissions.
`synthesis.md` section 4d: *"when fascicle Dice is low, is the measured fascicle ANGLE also wrong?"*
You wrote: "That single diagnostic settles 4c ... it costs no Kaggle submission." It has **never been
run.** This is the cheap decisive test being deferred in favor of the expensive virtuous-looking one,
which is the same deferral pattern `CLAUDE_AUDIT.md` already caught with the orientation audit ("next
for four turns," never run).

The experiments are also uncontrolled. EXP72 changed roughly nine axes at once (architecture, size,
batch, loss, augmentation, target, threshold, decoder, area gate); EXP73 correctly diagnosed that a
negative result from it is causally uninterpretable. Then **EXP77, the currently recommended overnight
run, repeats the same all-axes-at-once design**, four days after EXP73 dissected exactly that mistake.
The controlled notebook (EXP76) exists and was demoted to "only if EXP77 leaves a question." You
diagnosed the disease and re-prescribed it.

---

## 5. The proxy machinery is multiple-comparisons overfitting on 35 noisy images

The story-weight, saturating-support, class-route, and term-route experiments (exp48 to exp56) are
not models. They are free parameters grid-searched to minimize the 35-image B score. Conservatively
**over 2,000 distinct configurations** are scored against the same 35 images, and the minimum is
reported as the result. The code even says so: exp51 "this is intentionally overfit-prone," exp55
"this intentionally overfits the 35-image expert benchmark." The problem is that the resulting minima
are then quoted as "headroom."

The statistics make the gains meaningless. Inter-rater label noise gives a standard error of the
B-mean over 35 images of about **0.035**. The entire celebrated chase, 0.170 to 0.131, is **0.039,
about 1.1 SE**. The route-search gain 0.143 to 0.131 is about **0.34 SE**. The leave-one-out
"load-bearing" deltas (0.0012) are about **0.03 SE**. You are reporting the maximum order statistic
of a noisy search and calling it improvement. With 2,000 draws you are guaranteed to find a
configuration that far below the mean by luck alone.

Then the stacking compounds it. Submissions are built as deltas-on-deltas, and at least one is
mislabeled: burn #28 is described as "burn #15 plus scale" but its FL is actually burn #16's
visibility-weighted proxy and its MT is burn #17's vertical proxy, that is, **two already-rejected
proxies relabelled and resubmitted** (it scored 0.659). You cannot run a clean experiment if you do
not know what is in the file you are submitting.

---

## 6. The FL recenter is an illegitimate construct you cannot even agree about

`segment_then_measure.py:1144` pins the entire FL column's mean to the prior constant 74.424 by
multiplicative rescale, default **ON**:

```python
if USE_FL_RECENTER and (USE_FRAGMENT_FL or USE_IDENTITY_FL) and sub["fl_mm"].mean() > 0:
    sub["fl_mm"] = (sub["fl_mm"] * (PRIOR["fl_mm"] / sub["fl_mm"].mean())).clip(...).round(3)
```

`synthesis.md` section 2 calls mean-matching a "non-negotiable" thing **not** to treat as evidence.
The construct is exactly that, left on as a default. The code comment is candid that it "masks the FL
geometry's ~+6mm overshoot" and is "a leaderboard bet, not a free win." So the shipped FL is a biased
measurement cosmetically corrected to a target mean, not an honest per-image measurement. Any future
FL method with a different mean gets silently dragged back to 74.424, erasing real signal. It is a
dormant landmine even when it is currently a no-op.

And you do not agree with yourself about whether it is a no-op. `MASTER_REVIEW.md` section 5: turning
it off "changes **0/309 rows**." `synthesis.md` section 6 (exp24): turning it off "moves **308/309
rows by mean 17.147 mm**." Both describe the same flag on the shipped pipeline. They cannot both be
true. When your own audits disagree by 308 rows about what one default does, the documentation has
stopped tracking the code. Reconcile this before anything else in the FL path is trusted.

Note also: the prior FL=74.424 is the sample-submission mean, and the benchmark-true FL mean is
60.835. They differ by **13.6 mm, more than the 12 mm tolerance**. So the constant you pin to is
itself more than one tolerance away from the only ground truth you can see.

---

## 7. "Scale is solved" confuses self-consistency with correctness

`MASTER_REVIEW.md` section 5 declares scale solved because independent cues agree (OCR vs ticks
147/148). Agreement of two cues proves **consistency**, not **correctness**. If the mm-per-tick
assumption is wrong, two readers of the same ruler agree and are both wrong together. That is exactly
the "2x trap" you keep worrying about (`competition_reference.md` section 3a) and the IMG_00275 silent
2x error you caught. Consistency cannot detect a shared scale convention error, which is the dangerous
one because it double-counts into FL and MT.

Two specific overfit-to-the-visible-set hazards in the scale router:

- `scale_ticks.recover_scale_family_b_signature` assigns a **fixed 134.5 px/cm at confidence 1.0** by
  recognizing four bright UI pixels at hardcoded rows (73, 82, 293, 302). This is fingerprinting UI
  furniture to a memorized scale, tuned to the 41 visible test images. A hidden image from a slightly
  different UI build, or a different device that happens to match the signature, gets a confident
  wrong scale.
- `CALIBRATION_MIN_CONF=0.3` admits scales that `calibration_verification_notes.md` shows can be
  known-wrong: the detector keys off depth-independent UI text rows, producing up to +50% MT error on
  shallow scans, and "confidence does not protect against this."

Scale is the highest-leverage lever in the problem (section 1) and it is resting on assumptions you
flagged as dangerous yourself. "Solved" is too strong. "Self-consistent on the visible set, correctness
unaudited on the hidden set" is accurate.

---

## 8. The deferral pattern: motion as a substitute for the decisive test

Across the project, the cheap decisive experiment is named and then skipped in favor of expensive
work that looks like progress:

- The Dice-vs-angle diagnostic (section 4): designed, "costs no submission," never run.
- The orientation-correctness audit (`CLAUDE_AUDIT.md`): "the next real step" for four straight
  commits, never run, while three more scale commits shipped.
- Gating submissions on the 19 human test-rows (section 3): built, then bypassed.

The common shape: when there is a choice between a boring decisive test and an interesting
indecisive one, the interesting one wins. This is the single habit most worth breaking, because every
one of these tests would have redirected weeks of work.

---

## 9. What is actually good (do not throw this away)

This audit is harsh because you asked for harsh. The project is not incompetent; it is misdirected.
Real strengths, keep them:

- **The scale router is the one thing that ever moved the board** (1.09 to 0.619). The instinct that
  scale is the lever was correct and well executed at the detection level.
- **The A/B/C distinction** (`MASTER_REVIEW.md` section 0) is exactly the right mental model. The
  failure is not believing it.
- **The synthesis 4d diagnostic design** is the right experiment. It just needs running.
- **The FL-is-a-convention insight** (section 6 of MASTER) is genuinely sharp and most people would
  miss it.
- **Baseline discipline** is real: a protected 0.619 file, off-by-default rejected probes, row-by-row
  diffs before submitting. That hygiene is why a bad probe never silently corrupted the live score.

The pattern in this list: your **diagnostic** work is strong and your **selection** work is broken.
You find the truth and then choose actions against a signal that ignores it.

---

## 10. What I would do, in order (a forcing function, not a menu)

1. **Freeze submissions.** You are 0-for-5 since the last real win. Do not spend another slot until
   step 3 exists. Slots are the scarcest resource and they are being fed to a broken compass.
2. **Run the Dice-vs-angle diagnostic this week, on CPU, no submission.** On the 35-set, where you
   have truth, plot per-image fascicle Dice against per-image PA error and against per-image FL error.
   If the scatter is flat (it almost certainly is for FL), the entire GPU segmentation pivot is dead
   on arrival and you have saved every overnight run. You designed this experiment; just run it.
3. **Build one transfer-valid gate before any further submission.** Finish the human labels on the
   actual test images (24 to as many as you can stand), or hold out a labeled slice. Score every
   future candidate against it. Never again spend a slot on a change whose only evidence is B.
4. **Put all FL effort on convention plus scale-correctness, not new masks.** The straight-line,
   low-extrapolation aggregation already moves benchmark FL from 0.519 to 0.281 with **no new model**
   (`MASTER_REVIEW.md` section 1, the projected-FL p25 / robust-triangle result). The leverage is in
   the aggregation convention and in auditing px-per-tick **correctness** (not just cross-cue
   consistency) on the hidden families, especially the family_b_signature and conf-0.3 rows.
5. **Delete `USE_FL_RECENTER` or make it loud and explicit.** First reconcile the 0-rows vs 308-rows
   contradiction so you know what your own pipeline does. A global mean-pin should never be a silent
   default.
6. **Only retrain segmentation if step 2 shows the angle is genuinely wrong on low-Dice fascicles.**
   If you do, run the controlled notebook (EXP76), one axis at a time, freeze apo while testing fasc,
   and gate on geometry or the A-proxy, never on Dice.

---

## 11. A note on the documentation itself

There are 76 markdown files here and they contradict each other on the two most load-bearing
questions (FL mask-limited vs geometry-limited; recenter no-op vs moves-every-row). The doc layer has
become a place where understanding is **deposited** rather than **acted on**. Writing a sharper
synthesis has, more than once, substituted for running the experiment the synthesis recommends.

Concrete hygiene: collapse to a handful of living docs (one current-state, one verified-facts, one
experiment-ledger), date every claim, and when two docs disagree, resolve it immediately rather than
adding a third that explains the disagreement. This audit included: if it becomes doc #77 that gets
read and admired while step 2 still does not run, it has failed at the same thing everything else
here failed at.
