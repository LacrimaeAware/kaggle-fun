# UMUD State Reset - 2026-06-10

Read this first. It supersedes the scattered "next candidate" notes written before the latest
leaderboard results.

## WHERE THINGS STAND (2026-06-10, AFTER the facing submission)

The facing-geometry FL candidate (described at the bottom of this section) **WAS submitted and
REGRESSED**: public LB **0.61918 -> 0.66459**. The safe baseline is therefore still the 0.61918 file,
preserved at `Downloads/0P61918_submission_local.csv` (results/ is gitignored). Do NOT resubmit facing
as-is. The rest of this section is the diagnosis of *why* it regressed, from independent ground truth.

### Why facing regressed - diagnosed on independent ground truth (FALLMUD)

The user downloaded FALLMUD (`data/dropoff/FALLMUD/`), two independent annotated ultrasound
collections, to test our geometry against masks we never trained on. **NeilCronin** (309 imgs, same
format as the competition: horizontal apos + sparse fascicle dashes) is directly usable.
**RyanCunningham** (504 imgs, dense fibre-orientation field, rotated/vertical apo convention) is a
different format and is set aside for now.

Independent findings on NeilCronin GT masks:
- **Segmentation transfers.** Our apo/fascicle U-Nets on images we never trained on: apo IoU 0.56
  (training 0.69), fascicle coverage 52% (training 49%). Modest drop; the segmentation half holds up.
- **The apo bend is REAL.** A parabola fits the GT aponeurosis edges **44% better** than a line
  (residual 1.73px -> 0.96px, over 156 apo edges). The user's bend intuition is confirmed on untouched
  data - facing was catching a real phenomenon, not noise-fitting the 35-benchmark.
- **But facing's gate is wrong 41% of the time.** Facing only keeps the parabola if it bows *toward*
  the muscle (`A_sup>=0, A_deep<=0`). On independent apos that condition holds only **59%** of the
  time; the other 41% the apo genuinely bends the opposite way and facing throws the bend away and
  forces a straight line.
- **And the FL effect sits under the tolerance floor, with harmful outliers.** Median FL change from
  facing is **3.6%** (~2.7mm, well under the 12mm FL tolerance), but the signed mean is **-6.6%**: a
  few images get large erroneous shortenings (mean << median = outlier skew). Production recenters FL
  to the prior mean, absorbing the systematic part - so what reaches the leaderboard is the outlier
  *spread*, and it transfers as harm.

Synthesis: **the bend is real; facing is a real phenomenon caught with a flawed rule.** The
"bow-toward-muscle-only" gate is half-wrong, and the gain on FL is below the tolerance floor while the
outliers poke above it. That fully explains why facing improved the 35-benchmark yet sank the
leaderboard - and it matches the standing lesson that the 35-image set is a sanity tool, not an oracle.

### The per-gap multi-level fix (BUILT, NOT wired, NOT submitted)

Separately, the user diagnosed a "two levels" bug on multi-muscle test images: 3 apo bands, band-
selection picks the wrong pair, and fascicles from BOTH muscles get mixed into one consensus ->
orthogonal/garbage extrapolations (worst on the ~13 extreme images that drove most of the facing
regression). Prototype: `experiments/per_gap_prototype.py`; interactive viewer
`experiments/per_gap_viewer.py`. It fits every apo band, forms a gap per consecutive pair, assigns
each fascicle to its gap, runs the geometry PER GAP, then fragment-count-weighted averages the gaps.
This is the honest next lever (it fixes a *verified* bug, satisfying the submission policy below). It
is NOT wired into production and NOT submitted. Note it inherits the facing geometry, so the gate fix
above should land first.

The facing FL code is still in `segment_then_measure.py` behind `UMUD_FL_FACING` (default on); set
`UMUD_FL_FACING=0` to reproduce the 0.61918 FL path.

## Prior baseline + the rejected probes (history)

Best submitted, and the current safe baseline: `0P61918_submission_local.csv` (preserved in Downloads)
-> public LB **0.61918** (fragment-only FL, perp/center MT, scale router, no sub-pixel). The facing
candidate regressed, so this is what to fall back to. Everything else submitted was worse and must NOT
be resubmitted:

| submitted file | public score | changed surface | decision |
| --- | ---: | --- | --- |
| facing-geometry FL (`UMUD_FL_FACING=1`) | 0.66459 | FL only | rejected (bend real, gate half-wrong - see top) |
| `submission_local.csv` with FL identity blend | 0.63905 | FL only | rejected |
| `submission_host_mt_vertical3_no_subpixel.csv` | 0.62561 | MT only | rejected |
| `submission_scale_tail_bar_only.csv` | 0.66711 | 4 scale rows + FL recenter ripple | rejected |

## What Actually Worked

The real gain came from the scale-router submission, not from later measurement tinkering:

- Per-family scale recovery reached 295/309 target images.
- The submission used fragment-only FL, old center/perpendicular MT, and the validated scale router.
- It beat the provided public reference pipeline and reached **0.61918**.

That remains the anchor.

## What Failed And What It Means

### 1. FL blend: local win, public loss

The 35-image benchmark liked blending fragment FL with the MT/sin(PA) identity. Public LB worsened
0.61918 -> 0.63905. PA and MT were unchanged, so this was a clean FL-only failure.

Lesson: the 35-image reference set is not a valid oracle for FL-method changes, especially when
recentered means are involved.

### 2. MT vertical-3: local win, public loss

The host discussion suggested MT was measured as straight lines at left/middle/right. Implementing a
vertical-3 approximation improved the 35-image benchmark 0.2274 -> 0.2192, but public LB worsened
0.61918 -> 0.62561. PA and FL were unchanged, so this was a clean MT-only failure.

Lesson: "protocol-aligned" is not enough. Either the approximation did not match the labeler/tool
implementation, or the hidden distribution rewards the old MT measurement path.

### 3. Bar-only scale tail: intuition warning confirmed

The bar-only scale-tail candidate touched four direct fallback rows, but FL recentering moved most
rows slightly. Public LB worsened badly: 0.61918 -> 0.66711.

Lesson: the remaining fallback-scale/tail ideas are not safe leaderboard probes. A visible-looking
scale cue is not enough when the baseline prior may be less wrong than a bad scale recovery.

## The Wall

The current wall is real enough to stop blind CSV probing.

The project is no longer failing because we have not tried enough small post-processing ideas. The
evidence says the opposite: after the scale-router win, small "principled" isolated probes have all
overfit our reasoning or local proxy and moved the public score down.

The remaining gap likely mixes:

- subjective labeler/tool conventions rather than physical ground truth,
- segmentation/measurement quality in the core geometry,
- hidden distribution shift not captured by the 35-image benchmark,
- FL recentering/mean sensitivity,
- and possibly a few still-wrong scale rows, but not broad scale-router collapse.

## Submission Policy From Here

No more submissions from:

- local benchmark improvement alone,
- public host-protocol interpretation alone,
- visual plausibility of a scale cue alone,
- tiny row-count probes that trigger global FL recentering,
- stacked changes.

Submit only if one of these is true:

1. A declared human-in-loop/active-learning path is intentionally chosen, logged, and disclosed.
2. A substantial model branch is trained or ensembled from public/declarable data and produces a
   candidate with clear audit evidence.
3. A candidate fixes an actual verified bug in the 0.619 pipeline, not a speculative convention.

## Practical Next Directions

1. Treat `Downloads/0P61918_submission_local.csv` as the safe baseline (0.61918). `results/submission_local.csv` has been regenerated back to the 0.61918 baseline and is byte-identical to the Downloads file.
2. Stop treating `results/submission_scale_tail_bar_only.csv`, shape-only, all-tail, MT vertical-3,
   and FL blend as live candidates.
3. Decide explicitly whether to use a declared human-in-loop target-labeling path. The host's public
   clarification makes that a legitimate declared-external-data strategy, but it changes the project
   mode and must be documented honestly.
4. If staying no-oracle, move to real model work: public-asset retraining, fold/seed ensembling,
   better structure supervision, or ROI/crop cue models used for QA rather than direct submission.
5. Keep the 35-image reference set for debugging only. It has now failed to predict multiple
   leaderboard changes in the correct direction.

## File Map

- `Downloads/0P61918_submission_local.csv`: best known public score, **0.61918** (the safe baseline).
- `results/submission_local.csv`: regenerated to the 0.61918 baseline (byte-identical to the Downloads file). Safe to use as-is.
- `data/dropoff/FALLMUD/`: independent annotated ultrasound sets (NeilCronin usable, RyanCunningham different format) used for the bend diagnosis above.
- `NEXT_SUBMISSION_REVIEW.md`: post-submission audit notes; now records the rejected MT and bar-only
  tail probes.
- `current_alignment.md`: active strategy alignment.
- `handoff_brief.md`: domain-aware handoff for another model.
- `experiments/README.md`: experiment ledger.
- `competition_reference.md`: rules, host facts, and leaderboard facts.
- `synthesis.md`: longer narrative synthesis; useful, but this reset is the current front door.
