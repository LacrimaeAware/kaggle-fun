# UMUD current state (canonical, undated)

Terse decision-driver. For the full self-contained narrative read `HANDOFF.md`. Rule: a claim may
drive a decision only if it carries a code `file:line` or an exact leaderboard number. Everything else
is a hypothesis.

## Best score

**Confirmed public best: `0.46041`** = aponeurosis band-fix + FL x1.05 (`results/submission_bandfix_flx105.csv`).
This is a CSV stack. The reproducible one-run pipeline (`local_infer.py`, median FL) is `0.47473`
(`submission_reproduced.csv`); the ~0.014 gap is old per-row CSV residue, not a model difference.

Older docs that say 0.52570 or 0.55075 are stale; they predate the family_b scale fix and the band fix.

## Leaderboard ledger (the only proven oracle)

| change (isolated) | public |
|---|---:|
| scale router + inner-edge MT + fragment FL | 0.61918 |
| temporal smoothing | 0.60961 |
| shape-neighbor fallback scale | 0.58910 |
| PA +2.0 / +2.5 / +3.0 flat | 0.55075 / 0.55033 / 0.55168 |
| FL x1.05 on PA+2.5 | 0.52570 |
| family_b scale 134.5 -> 147 px/cm (41 rows) | ~0.488 |
| aponeurosis band fix | 0.46076 |
| band fix + FL x1.05 | **0.46041** |
| median pipeline, one run (`submission_reproduced.csv`) | 0.47473 |
| min_extrap_top3 FL (`submission_minextrap.csv`) | 0.49983 (REFUTED) |
| MT x0.95 on best (`submission_mtx095_on_best.csv`) | 0.53395 (MT bracketed, x1.0 optimal) |
| FL x1.10 on best (`submission_flx110_on_best.csv`) | 0.48369 (FL bracketed, ~x1.05 optimal) |

**All three global levers are now bracketed two-sided** (PA +2.4 via +2/+2.5/+3; MT x1.0 via x0.95 wall;
FL x1.05 via x1.00/x1.05/x1.10), so **0.46041 is the confirmed floor of global tuning** — no global probe
left with a rationale. The mechanistic wins (scale router, family_b constant, band fix) are real and
per-image; the PA/FL/MT global shifts only move a column mean and are spent.

## Per-term truth (code + LB + benchmark grounded)

- **Benchmark with true scale, no recenter** (`benchmark_lab/honest_validate.py`): PA 0.1505 (below the
  0.2445 human floor), MT 0.0840 (at the 0.0810 floor), FL 0.5218 (floor 0.4026, over-reads +5.8 mm).
  So given correct scale, PA and MT measurement are at the human limit and FL over-reads. **This does
  NOT predict the LB** (min_extrap proved it again: benchmark 0.39, LB 0.49983).
- **MT is a clean scale probe**: it has no fascicle and no extrapolation and is at the human floor with
  true scale, so test MT error is almost entirely scale, not anatomy.
- **PA and FL are not "tapped".** Only their global means were nudged on the LB. Per-image PA and FL
  error is unmeasured on the test set (no test truth except the UI hand-labels).
- **Scale**: 295/309 rows get a recovered px/mm; 14 fall back. Per-family detection is solid;
  absolute per-image correctness is unvalidated except where the user hand-read ticks (family_b).

## Validation discipline

1. The leaderboard is the only valid gate for global/calibration quantities (scale level, FL/PA
   shift), and it is distribution-specific. Use it a few times to confirm, never to search.
2. The 35-image benchmark does NOT predict this LB, even un-blinded. Use it to catch a measurement
   bug, not to decide a submission. `score_weights.py` is doubly blind (feeds true scale `:42`,
   recenters FL `:54`); `honest_validate.py` removes the recenter but is still the wrong distribution.
3. The only per-image TEST truth is the user's corrections on the 309 images (the correction UI).
4. For model/segmentation changes: GroupKFold by subject/device/muscle (recover sequence groups
   first). Never gate model changes on the benchmark or on visual plausibility.

## Plan (methodology reset, 2026-06-15)

Stop leaderboard-hacking global multipliers. Build a real loop (see `HANDOFF.md` sec 8):

1. Error decomposition, no LB slot: run `measure()` on expert train masks vs predicted train masks on
   a held-out fold; the gap is the per-term segmentation cost. Pairs with the benchmark result (PA/MT
   at floor, FL over-reads) to give an error budget.
2. Build the test-distribution gate from the UI hand-labels.
3. Choose the strategic direction with the user (route-by-class vs new segmentation target vs rebuild
   measurement on apo + orientation objects), then improve the largest real source and validate on CV.

Best submission to keep for final selection: `results/submission_bandfix_flx105.csv` (0.46041).
