# UMUD synthesis: goals, intuitions, problems, and the plan

This is the single canonical document for the UMUD work as of the `0.61918` baseline and the failed
`~0.64` blend probe. It folds in
the scattered planning notes (Codex's `codex_review.md`, `forward_plan.md`,
`ranked_research_directions.md`) and the conversation that produced them. It is written to keep
the original intuitions intact rather than reduce them to standard method names. The document
map at the end says what every other file is for.

> **Status note (2026-06-09):** the project has moved materially since the older `1.09194` text.
> Current best submitted public LB is **0.61918**. A later 50/50 FL blend worsened to about **0.64**
> while leaving PA and MT identical, so the blend is rejected as a submission default. The current
> scale router reads **295/309** target images; the default full 35-expert local harness is back to
> **0.2274** with fragment-only FL (`UMUD_FL_IDENTITY_BLEND=0`). The domain-gap/augmentation-first
> hypothesis was tested on real train-vs-target frames and demoted. Read **`handoff_brief.md`** first
> for the tight operational state. The intuitions here (scene-reading, path geometry, the fascicle
> bottleneck) still matter.

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

| System | Score | Meaning |
| --- | ---: | --- |
| Current best submitted public LB | **0.61918** | restored safe baseline; scale router + real FL/MT moved us past DL-Track benchmark |
| Rejected blend probe | ~0.64 | changed FL only; PA/MT identical to 0.61918 file |
| Provided DL-Track benchmark | 0.679 | public reference pipeline on the hidden test |
| Public leader (sugopoko/sugupoko) | 0.378 | UMUD solution not published |
| By-hand human public baseline | 0.459 | third-party manual test-set analysis, declared external/manual |
| Our default 35-expert local harness | **0.2274** | true scale, clean benchmark, recentered fragment-only FL; not directly comparable to hidden LB |
| Rejected blend local harness | 0.1873 | looked better locally but regressed public LB, proving this is not a reliable FL submission oracle |

What is built: a two-U-Net segment-then-measure pipeline (`segment_then_measure.py`), TTA,
inner-edge MT, fragment-extrapolated FL, temporal smoothing toggle, and a per-family scale router
that currently reads **295/309** target images. Current clean-data local terms are PA 0.1498,
FL 0.3528, MT 0.1795 via `experiments/score_weights.py`. The rejected blend local terms were
PA 0.1498, FL 0.2326, MT 0.1795.

What is **not** solved: the hidden target gap to the leader remains. The older hypothesis was
"segmentation domain gap"; real-data checks now weaken that. The remaining gap is more likely a
mix of target-set scale error, prior/mean mismatch, and FL/orientation geometry. The failed blend
made one rule non-negotiable: do not treat global mean matching or benchmark recentering as hidden
test evidence.

---

## 3. The core problem, stated plainly

The old core problem was that we were far from the benchmark **because we had not built the
benchmark's method**: PA was measured, FL/MT were mostly constants. That is no longer the current
state. We now measure all three targets on most rows and beat the provided DL-Track benchmark on
the public LB.

The current core problem is narrower and harder:

```text
target score 0.619  ->  leader 0.378
```

No test labels exist, so the next work is not to declare a new bottleneck from vibes. It is to
bound error sources with label-free checks: two-cue scale disagreement, prior/recentering
sensitivity, sequence consistency, and classical-vs-network orientation agreement.

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

So knowing the recipe is not the same as having cooked it. We have cooked the standard pipeline;
the remaining work is disciplined error attribution and small, testable improvements.

---

## 3b. Evidenced update from the OSF expert benchmark (see benchmark_findings.md)

The downloaded expert benchmark (35 images with true scale, 7 experts, and DL-Track's/SMA's own
outputs) turned some speculation into measurement and corrected an overclaim. Scored the same way as
the competition, on those 35: the human floor (one expert vs the rest) is ~0.31, DL-Track's real
measurement is ~0.33 (human-level), and our constant-prior baseline is ~0.92. So the supported claim
is that replacing constants with real per-image measurement is worth ~0.59 *within this set*.
Correction: do NOT compare the Kaggle 0.68 benchmark with the 0.33 here and call the gap "scale" -
they are different evaluations (different images, devices, hidden labels), and an earlier draft wrongly
did this. Scale is a *necessary part* of real measurement, and current target coverage is high
(295/309), but its standalone value to the current Kaggle score is still something to measure, not
assume. The ceiling (~0.3) is human reproducibility, which the leader (0.378) sits near. Caveat:
benchmark images are different devices than the Kaggle test set, so their tick convention does NOT
transfer. This set is also our local scoreboard (`benchmark_validate.py` plus the fuller
`experiments/score_weights.py` harness), which makes the oracle/visual loop possible without fast
training.

Latest local score distinction:

- `experiments/score_weights.py`: **0.2274** with the current default `UMUD_FL_IDENTITY_BLEND=0`,
  and FL recentered to the benchmark mean.
- `UMUD_FL_IDENTITY_BLEND=0.5`: **0.1873** locally, but publicly regressed `0.61918 -> ~0.64`.
  This is now recorded as a failed transfer test, not a current improvement.
- `score_on_benchmark.py`: **0.288** raw/simple scoring; useful sanity check, not the headline
  current local number.

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

Two tracks still run in parallel, but their priorities changed. Track A is now error attribution
on the real target distribution. Track B is FL/orientation geometry, where your path-geometry
intuition still has the most room.

### Current correction: domain-gap retraining is demoted

`experiments/domain_gap_real.py` tested the train-vs-target appearance gap on real frames:
mean |SMD| = 0.44, no feature crosses |SMD| > 1 globally, test frames mix into train clusters,
and CLAHE/z-score/min-max normalization does not close a hidden global-intensity gap. The older
"Lumify/unseen-device segmentation collapse" hypothesis is not supported by this check.

`experiments/seg_quality_test.py` is the companion mask-presence probe. Its outputs should be read
carefully: presence is not correctness. The unresolved question is whether extra target fragments
are coherent signal or texture noise, not whether the segmenter simply collapses.

### Track A: bound the public-gap causes

1. **Scale bound on the target set.** Use two-cue families to measure disagreement between
   independent rulers. `experiments/exp19_scale_crosscheck.py` started this: 114/309 images have
   two strict cues, and no pair class has >5% disagreement after sub-pixel refinement. The broad
   2x-router-failure hypothesis is weakened; remaining scale risk is concentrated in the single-cue
   `right_ruler_5mm` family and the 14 `none` rows. `exp21_scale_tail_recovery.py` now turns those
   14 rows into an isolated scale-tail candidate: 10 stable shape-neighbor recoveries and four
   visible `3 cm` scale-bar recoveries, while right-ruler QA shows low residual fractions and five
   review rows. The script writes all-tail plus split shape-only and bar-only candidate CSVs.
2. **Prior/recentering audit.** The full clean score benefits from recentering FL to a known
   benchmark mean. The failed blend proves that this kind of local FL win can point in the wrong
   direction publicly. Mean matching is now an audit risk, not evidence.
3. **Temporal smoothing side-bet.** Sequence-like clips exist. `UMUD_TEMPORAL_SMOOTH` is built and
   off by default; it is a modest variance reducer that needs a leaderboard probe to value.

### Track B: FL/orientation geometry

1. **Robust FL combiner.** `experiments/term2_geometry.py` shows why FL amplifies PA error and why
   mean per-fragment FL is biased. `experiments/exp16_fl_combiner.py` found a 50/50 blend that
   beats fragment-only on the 35-expert harness (0.2274 -> 0.1873), but the public LB worsened
   0.61918 -> ~0.64. The blend remains available for experiments, but the default is fragment-only.
2. **Orientation coherence.** Extra fragments on target frames can be signal or texture. Coherence
   around a robust orientation center is a label-free signal/noise test. `exp18_orientation_coherence.py`
   found high target coherence by family (~0.992-0.998 means), so extra target fragments look mostly
   like aligned signal rather than random texture.
3. **Raw-support orientation cross-check.** `experiments/exp22_orientation_raw_support.py` now checks
   whether predicted fragments align with local raw-image line orientation. It does not find broad
   target collapse: 23/309 rows flag for review; right-ruler and former `none` rows flag 0 times.
   This is a pseudo-label gate candidate, not a replacement PA estimator.

### What not to do

Do not spend the next GPU run on augmentation just because synthetic probes suggested a domain
gap. Do not assume mask-presence means orientation correctness. Do not treat the clean local
score as hidden-test truth. Do not hand-label test images. Do not hide family-signature scale
assignments; keep their method names visible.

---

## 7. Concrete next implementation order

```text
1. Keep `results/submission_local.csv` restored to the downloaded `0.61918` baseline and use it as
   the comparison anchor.
2. Compare every candidate row-by-row against that baseline before any submission.
3. Keep sub-pixel scale refinement isolated: `results/submission_subpixel_scale.csv` is a tiny
   precision candidate, not a stacked submission.
4. Review the exp21 scale-tail candidate (`results/submission_scale_tail.csv`) and its overlays; do
   not stack it with sub-pixel or temporal smoothing for a probe.
5. Review exp22 raw-support overlays and use them as gates for any future self-training/ensembling.
6. Only then reconsider fold/seed ensembling, conservative self-training, external DL-Track data, or
   dense classical pseudo-labels.
```

The thesis in one sentence: this is a geometry-reading problem where masks are only intermediate
supervision and Dice is misaligned with the metric; the remaining score lives in measured
target-set error attribution plus FL/orientation geometry, not in another blind augmentation pass.

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
