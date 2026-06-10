# UMUD State Reset - 2026-06-10

Read this first. It supersedes the scattered "next candidate" notes written before the latest
leaderboard results.

## Bottom Line

Best known submission remains:

`results/submission_local.csv` -> public LB **0.61918**

Everything submitted after that has been worse:

| submitted file | public score | changed surface | decision |
| --- | ---: | --- | --- |
| `submission_local.csv` with FL identity blend | 0.63905 | FL only | rejected |
| `submission_host_mt_vertical3_no_subpixel.csv` | 0.62561 | MT only | rejected |
| `submission_scale_tail_bar_only.csv` | 0.66711 | 4 direct scale rows plus FL recenter ripple | rejected |

Do not submit any of those rejected variants again. For Kaggle auto-selection, select the **0.61918**
`submission_local.csv`.

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

1. Preserve `results/submission_local.csv` as the safe baseline.
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

- `results/submission_local.csv`: best known public score, **0.61918**.
- `NEXT_SUBMISSION_REVIEW.md`: post-submission audit notes; now records the rejected MT and bar-only
  tail probes.
- `current_alignment.md`: active strategy alignment.
- `handoff_brief.md`: domain-aware handoff for another model.
- `experiments/README.md`: experiment ledger.
- `competition_reference.md`: rules, host facts, and leaderboard facts.
- `synthesis.md`: longer narrative synthesis; useful, but this reset is the current front door.
