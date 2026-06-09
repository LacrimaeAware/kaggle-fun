# UMUD synthesis: goals, intuitions, problems, and the plan

This is the single canonical document for the UMUD work as of the `1.09194` run. It folds in
the scattered planning notes (Codex's `codex_review.md`, `forward_plan.md`,
`ranked_research_directions.md`) and the conversation that produced them. It is written to keep
the original intuitions intact rather than reduce them to standard method names. The document
map at the end says what every other file is for.

> **Status note (2026-06-09):** the framing and intuitions below are current and intact, but the
> "Where we are" numbers (section 2) reflect the older `1.09194` run. For the CURRENT state - PA
> effectively solved (0.164), the local 35-expert scoreboard we now iterate on, TTA wired, and the
> validated scale router that scales 167/309 test images - read **`handoff_brief.md`** and
> **`competition_reference.md`** first. The intuitions here (scene-reading, path geometry, the
> fascicle bottleneck) all held up; the scale work confirmed the per-image-scale intuition.

A note on voice: the sections marked "(your framing)" are deliberately stated as the intuition,
with tool names kept secondary, because the recurring failure in earlier summaries was
flattening a real idea ("recover the latent path geometry") into a tool ("just use Hough").

---

## 1. What this task actually is (your framing)

It is not "predict three numbers from an image." It is closer to **reading a geometric scene**
out of a noisy ultrasound picture, and the three numbers fall out of that scene.

The scene has two kinds of objects:

- **Aponeuroses**: the thick, bright, roughly horizontal boundary bands. Easy to see.
- **Fascicles**: the thin, broken, diagonal fiber-like streaks between the aponeuroses. Hard.

The targets are geometric consequences of those objects:

- `PA` = the angle between a fascicle and the deep aponeurosis.
- `FL` = the length of a fascicle between the two aponeuroses.
- `MT` = the distance between the two aponeuroses.

A key realization from the conversation: the obvious bright lines are mostly **aponeuroses**, not
the thing that's hard. The hard part is the thin, broken, low-contrast fascicles. So the
bottleneck is fascicle geometry, not the bright bands.

The public training supervision is masks, not a label table, so the natural shape is:

```text
image -> segment aponeuroses + fascicles -> fit geometry -> compute PA/FL/MT -> convert px to mm
```

---

## 2. Where we are

| System | Public score | Meaning |
| --- | ---: | --- |
| Current best | **1.09194** | U-Net pennation, calibrated MT on 68 images, FL constant |
| Previous best | 1.11066 | ExtraTrees pennation, FL/MT constant |
| U-Net PA, no calibrated MT | 1.12324 | same PA path, all FL/MT constant |
| DL-Track 0.3.1 benchmark | 0.67944 | public benchmark output, not hidden labels |
| Public leader (sugupoko) | 0.37766 | UMUD solution not published |

What is built: a two-U-Net segment-then-measure pipeline (`segment_then_measure.py`), a
standalone ruler/scale detector (`tick_calibration.py`), pennation from geometry, and calibrated
muscle thickness on the 58 PNGs plus 10 (suspicious) TIFFs. Apo segmentation reaches roughly
0.6-0.8 Dice; fascicle segmentation is weak (~0.15-0.3 Dice).

What is **not** built: fascicle length is a flat constant on all 309 rows, and thickness is a
flat constant on 251 of 309. So we are measuring about one of the three targets per image.

---

## 3. The core problem, stated plainly

We are far from the benchmark **because we have not built the benchmark's method**, not because
of a secret. The benchmark measures all three numbers in millimetres on every image; we measure
pennation (roughly) and hard-code the other two on almost every row. The gap from 1.09 to 0.68 is
mostly the unbuilt two-thirds.

Two clarifications that caused confusion and should stay settled:

- **The benchmark is public; the leader is not.** DL-Track-US is open source (its code, the two
  Keras U-Nets, the geometry, the scaling logic - all read directly). Its models are on OSF. So
  the 0.67944 row is the output of a public tool, and reproducing it is real engineering, not
  theorizing. What is private is the **leader's** 0.37766 method.
- **The public method has the same hard part we do: scale.** DL-Track gets pixels-to-millimetres
  by asking a human to click or type a known distance. There is no automatic scale in it. So even
  the public tool cannot be run fully automatically over 309 images without *you* solving
  calibration - which is exactly the unsolved piece. "The method is public" carries a giant
  asterisk: "now solve the scale problem yourself."

So knowing the recipe is not the same as having cooked it. We have cooked about a third.

---

## 3b. Evidenced update from the OSF expert benchmark (see benchmark_findings.md)

The downloaded expert benchmark (35 images with true scale, 7 experts, and DL-Track's/SMA's own
outputs) turned some speculation into measurement and corrected an overclaim. Scored the same way as
the competition, on those 35: the human floor (one expert vs the rest) is ~0.31, DL-Track's real
measurement is ~0.33 (human-level), and our constant-prior baseline is ~0.92. So the supported claim
is that replacing constants with real per-image measurement is worth ~0.59 *within this set*.
Correction: do NOT compare the Kaggle 0.68 benchmark with the 0.33 here and call the gap "scale" -
they are different evaluations (different images, devices, hidden labels), and an earlier draft wrongly
did this. Scale is a *necessary part* of real measurement (per-image scale varies 74-126 px/cm, and a
fixed-scale upper bound is ~0.6), but its standalone value to our Kaggle score is unmeasured - the one
real scale fix we ran moved 0.03. The ceiling (~0.3) is human reproducibility, which the leader (0.378)
sits near. Caveat: benchmark images are different devices than the Kaggle test set, so their tick
convention (1 cm) does NOT transfer - the Kaggle PNG family is ~5 mm, verified against its depth text.
This set is also our local scoreboard (`benchmark_validate.py`: real measured PA/FL/MT, CPU, instant),
which is what makes the oracle/visual loop possible without fast training.

## 4. The intuitions driving the "clever" path (your framing)

These are the ideas to protect from flattening. Tool names appear only as possible
implementations, never as the idea itself.

### 4a. Brightness is a field, not a pile of independent pixels

A human does not label pixels one at a time. You see contrast, clusters, thick bright bands,
thin diagonal streaks, and a rough direction, and *then* idealize those into geometric forms.
Your mental algorithm was roughly: look at bright pixels, ask what nearby bright pixels they
connect to, follow paths of high brightness, find the stable repeated paths, separate the
thick near-horizontal paths from the thin diagonal ones, and fit idealized geometry to those
paths.

The pieces that matter and are easy to lose:

- brightness is a *field* with density; spread-out white dots read as gray; clustering matters.
- the *center* of a bright cluster matters more than every raw bright pixel - we may want
  centerlines, not fat masks.
- the local decision is *directional*: from a pixel, which connected path does it belong to?
- but naive "follow brightness uphill" gets pulled into the **aponeurosis**, because the
  aponeurosis is brighter and thicker. So fascicle-versus-aponeurosis is not brightness alone;
  it is brightness *plus orientation plus thickness plus context*.

The deeper idea: **recover the latent path geometry, not the mask pixels.** Ridge detection,
skeletonization, structure tensors, RANSAC, Hough, graph tracing are only names for how. They
are not the idea.

### 4b. Put the target's geometry into the representation and the loss (the clock lesson)

Your clock project taught a transferable lesson: encode the *geometry of the target* in the
representation. The clock used `(sin, cos)` because time is circular - 11:59 and 12:00 are close
in reality but far apart as the scalars 719 and 0 - and recovered time with `atan2`.

UMUD is not identical (pennation does not wrap a full circle), but line orientation has 180-degree
symmetry, so a line at theta and theta+180 is the same unoriented line, which suggests an
encoding like `(cos 2theta, sin 2theta)`.

The real point is not "replace arctan with sin." It is: **do not only compute the angle after the
mask; make the learning process itself care about the angle and the geometry.** Concretely, that
becomes an optional auxiliary objective:

```text
encoder
 |- mask head:        predicts fascicle pixels (what we do now)
 |- orientation head: predicts fascicle orientation, e.g. (cos 2theta, sin 2theta)
loss = mask_loss + lambda * orientation_loss
```

### 4c. The honest tension: does the network learn the geometry on its own, or not?

This is genuinely open, and you hold it as open. A neural net can learn local contrast,
orientation, edges, and texture, so maybe a plain mask model already encodes enough geometry.
But it is trained on Dice+BCE, which asks "did the predicted mask overlap the target mask?",
while the competition asks "did the final geometry come out right?" Those can diverge:

```text
pixel objective  !=  geometry objective
```

Your lean is "we probably need more geometry," while explicitly allowing "maybe the net learns
enough." This is not a thing to argue about; it is a thing to **measure** (see 4d).

### 4d. Masks are intermediate supervision, not truth; Dice is a diagnostic, not the score

We train against masks, but masks are not the hidden truth, and they can be imperfect:
misaligned with the image, resized badly, too thick or too thin. Dice is only mask overlap; the
leaderboard scores the final PA/FL/MT. On thin fascicles Dice is brutal - a one or two pixel
offset on a thin line tanks it - so a low fascicle Dice does not by itself mean failure.

So the decisive question is not "is fascicle Dice low?" It is:

```text
when fascicle Dice is low, is the measured fascicle ANGLE also wrong?
```

That single diagnostic settles 4c. If the angle is right while Dice is low, the network already
recovers the geometry and Dice is misleading - then stop chasing fascicle masks and spend the
effort on scale and FL. If the angle is also wrong, the geometry is genuinely failing - then the
geometry-aware ideas in 4a/4b are warranted. This is the fork the whole "more geometry?" question
turns on, and it costs no Kaggle submission.

### 4e. The oracle / failure-subclass idea (from structured-transform-discovery)

This is the newest idea and the one most at risk of being flattened into "active learning." Stated
in its own terms:

The model is not only wrong image-by-image. It is wrong along **latent subclasses / factors**. A
plain validation loss only knows the *class* (fascicle vs not); it does not see that the errors
concentrate in a subclass like "thin + faint + diagonal + sitting right next to a bright
aponeurosis." Your structured-transform work is precisely about discovering within-class factors
(it recovers slant, thickness, width as named, visualizable axes from class labels alone, no
factor labels).

The oracle move: a human looks at discovered failure clusters and *names the factor* - "this is
the thin-faint-near-apo failure" - and that factor-level signal guides the next step (oversample
that subclass, add augmentation that simulates faint fascicles, weight the loss there, change the
threshold or postprocess for that subclass). The point is to push the model along a *named factor*
faster than a scalar validation loss could, and it is **not** cheating - it is factor-level
guidance, not editing test labels. It connects to the pre-training thought: if the model reliably
gets a certain kind of feature wrong, you can nudge it there instead of waiting for the loss to.

Your own repo's honest caveat has to ride along with this, because it is the discipline that keeps
the idea sound. The structured-transform conclusion is verbatim: the explicit factor work is **"a
diagnostic and debiasing instrument, not a classification-accuracy method,"** and a plain
discriminative baseline matched or beat the whole metric family; the durable value was
interpretability, shortcut detection, and a label-light way to find and down-weight a nuisance.
The repo even flags the exact open thread this is: whether discovered within-class factors reduce
human labeling effort - turning "name all factors" into "verify this proposed factor" with fewer
oracle queries - is untested.

Mapped onto UMUD without flattening: use failure-subclass discovery as the **instrument** that
tells us where the model breaks and guides targeted fixes; do not expect the factor method itself
to win the leaderboard; keep the plain baseline (U-Net + geometry + calibration) visible as the
thing that actually scores.

---

## 5. What the leader's playbook actually told us (honest)

`leader_playbook.md` does **not** contain the leader's UMUD solution - it is unpublished. The
playbook is his reusable style, read off his notebooks from *other* competitions and mapped onto
UMUD. So your confusion is legitimate: we have his public notebooks and still cannot see his UMUD
moves, because notebooks show the recipe, not the thousand dataset-specific debugging decisions.

His surface recipe is not exotic and we already copied its first layer: smp U-Net, ResNet34,
Dice+BCE, augmentation, AdamW, cosine LR. The likely difference is not one magic model; it is the
full **error-analysis loop** run with discipline: train, look at the worst predictions as
overlays, find a systematic bug (alignment, a flip, a color shift, a bad threshold), fix it, then
folds, TTA, self-training on confident test cases, simple ensembling, outlier fallback, repeat.
The single most cited habit in his writeups is *looking at predictions before tuning*. That is
the practical gap: we have mostly been staring at a `val_dice` number instead of looking at where
and how the model fails. It is unglamorous, and in imaging competitions it can be the whole gap.

What we have not verified and should not assume: that calibration specifically is his edge (that
was an earlier over-claim - we have no source on his actual method), nor that he used external
labeled data, temporal smoothing, or anything else in particular.

---

## 6. The plan

Two tracks run in parallel. Track A is the standard measurement pipeline and is the most likely
near-term score mover. Track B is where your intuitions and a possible novel contribution live;
it is *instrumented* by the same validation harness Track A needs first.

### The harness comes before both tracks

Build the visual + numeric validation harness first, because it is the control panel for
everything and it settles the 4d fork. Render, per image: the grayscale background, the predicted
apo and fascicle masks, the fitted lines, the measured MT gap and FL line, the detected scale
ticks, and the final PA/FL/MT with confidence/fallback reasons. Then, on held-out training masks,
report derived **PA error in degrees** and FL/MT in pixels by comparing geometry-from-predicted-
masks against geometry-from-true-masks. This is the first time we get a local number that tracks
the real targets instead of Dice. The UMUD organizers also publish an expert-analyzed benchmark
(35 vastus/gastroc/tibialis/soleus images measured by seven experts, on the platform's Benchmarks
page) - that is a small but real set of *measured* values to validate against; grab it if the
download works.

### Track A: make the measurement real

1. **Validation harness** (above) - settles "do we need more geometry," catches scale bugs by eye.
2. **Scale, family by family.** PNG left-ruler works; the 251 TIFFs are the gap, and the 10
   calibrated TIFFs sharing `13.45 px/mm` are suspect. Build per-family detectors (right ticks,
   bottom ticks, depth-text readout, metadata, sequence borrowing) and a confidence model that
   punishes a constant scale across different depths and ticks found inside text panels.
3. **Fascicle length.** Once scale is trusted, test direct line-intersection FL versus
   `MT / sin(PA)` versus curved arc length on the held-out masks, and gate FL per row.
4. **DL-Track reproduction** as a *reference*, not a worship target: run the public tool headless
   on a subset to see what its masks, geometry, and scale choices do, then on all 309.
5. **Craft** once the base is real: folds, TTA (with correct inverse transforms), self-training on
   confident test masks, simple ensembling, sequence smoothing, outlier fallback.

### Track B: the geometry and factor ideas

1. **Postprocess before re-architecting.** On the existing probability maps, try threshold sweeps,
   skeletonization/centerlines, component filtering, RANSAC/Hough line fits, structure-tensor
   orientation, and rejecting thick near-horizontal (apo-like) ridges when fitting fascicles. Ask
   whether centerline postprocessing fixes the visible failures before building anything new.
2. **Orientation head** (4b): derive `(cos 2theta, sin 2theta)` orientation labels from fascicle
   masks, add the auxiliary head, and judge it by **derived PA error, not Dice**.
3. **Failure-subclass discovery + oracle** (4e): cluster the harness's failures into interpretable
   subclasses, have the human verify/name them, and feed that back into augmentation, sampling,
   loss weighting, or per-subclass postprocessing - as a diagnostic instrument, with the plain
   baseline kept visible.

### What not to do

Do not make tiny CSV column swaps the main plan (they move hundredths; fine only as cheap probes).
Do not train bigger U-Nets before the harness exists. Do not assume better Dice means a better
score. Do not assume the leader's method is known. Do not commit keys, external weights/datasets,
or generated result folders. Do not hand-label test images.

### One cheap probe worth spending a submission on

`best PA + PNG-only calibrated MT` (the `submission_best_pa_calibrated_mt_png_only.csv` variant):
it pairs the stronger ExtraTrees pennation with the safe PNG MT and drops the 10 suspicious TIFFs,
and should land near ~1.08. It tells us whether the MT signal stacks with the better PA. Then stop
polishing variants.

---

## 7. Concrete next implementation order

```text
1. Validation harness: overlays + held-out PA-in-degrees / FL-MT-in-pixels error.
   -> answers: when fascicle Dice is low, is the angle also wrong?
2. (one probe) best PA + PNG-only calibrated MT submission.
3. Scale for TIFFs, per family, behind a depth-aware confidence model.
4. Fascicle length estimators on held-out masks, gated per row.
5. DL-Track headless reproduction on a subset, then all 309, as a reference.
6. Geometry-aware fascicle work: centerline postprocess, then orientation head.
7. Failure-subclass discovery + oracle guidance (Track B instrument).
8. Craft: folds, TTA, self-training, ensembling, sequence smoothing, outlier fallback.
```

The thesis in one sentence: this is a geometry-reading problem where masks are only intermediate
supervision and Dice is misaligned with the metric; the near-term score lives in per-image scale
and non-constant FL/MT, and the interesting, possibly-novel route is to make the model care about
fascicle path geometry and to discover and correct its failures at the subclass level - with the
plain baseline always kept visible as the thing that actually scores.

---

## 8. Document map

Canonical:

- **`synthesis.md`** (this file) - the entry point: goals, intuitions, problems, plan.

Reference (still accurate, more detail on one topic each):

- `writeup.md` - competition writeup and the score table.
- `strategy_brief.md` - the improvement levers and compute/GPU status.
- `dltrack_headless_notes.md` - verified DL-Track source facts (scale math, headless path, models/labels).
- `calibration_verification_notes.md` - the tick-detector bug and the PNG left-ruler fix.
- `leader_playbook.md` - the leader's inferred style (explicitly not his UMUD code).
- `handoff_brief.md` - short status briefing for a collaborating model.
- `plan.md`, `rundown.md` - early plan and plain-language explainer.

Folded into this synthesis (kept for their per-item detail, but this file is the entry point):

- `codex_review.md` - Codex's hypothesis map and priority experiments.
- `forward_plan.md` - Codex's post-1.09194 plan.
- `ranked_research_directions.md` - Codex's ranked directions with per-rank experiments and an agent work-split.
