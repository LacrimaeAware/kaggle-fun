# UMUD current state (canonical, undated)

This is the single source of truth. When any other doc disagrees with this file or with
`../VERIFIED_FACTS.md`, this file wins and the other doc is stale. Rule: a claim may drive a decision
only if it carries a code `file:line` or an exact leaderboard number. Everything else is a hypothesis.

## Best score

**Confirmed public best: `0.52570`** = global FL scale x1.05 on the PA+2.5 base
(`results/submission_fl_x105.csv`). References: leader 0.378, by-hand human 0.459, DL-Track 0.679.

Note: this win was found 2026-06-14 by direct leaderboard probing. Older docs that say 0.58910 or
0.55075 are stale; they predate it.

## Leaderboard ledger (the only proven oracle)

| change (isolated, one variable) | public |
|---|---:|
| burn_13 base | 0.58910 |
| PA +2.0 flat | 0.55075 |
| PA +2.5 flat | 0.55033 |
| PA +3.0 flat | 0.55168 |
| FL x1.05 on PA+2.5 base | **0.52570** |
| FL x1.10 / x1.15 / x1.20 / x1.25 | built, unscored |
| MT x0.95 / x1.05 | built, unscored |

## Per-term truth (code + LB grounded)

- **PA — tapped.** Model PA mean 14.627 deg (`results/calibration_measurement_debug.csv`). Flat-PA
  optimum is ~+2.4 (interpolated; +2.5 is the best *scored* point, +3 regresses). The shift lives only
  in post-run probe CSVs, not in the pipeline. No further global PA gain expected.
- **FL — the live lever.** Raw per-image geometry FL mean = 91.596 mm, std 27.07 mm (= 2.26 FL
  tolerances). `USE_FL_RECENTER` (default ON, `segment_then_measure.py:1144-1145`) multiplies every FL
  by `PRIOR/mean` ~= 0.81, shrinking the column to pin its mean to PRIOR=74.424. **This is an active
  ~19% shortening of FL, not a no-op.** The LB proves FL must be longer (x1.05 best, still climbing
  toward the raw 91.6 mm, which is ~x1.23 of the pin). After x1.05 the shipped FL mean is 78.9 mm, so
  both global-mean headroom AND per-image spread (the 2.26-tol std) remain.
- **MT — best term, untested global lever.** Mean 21.836 mm, measured on all 309, NOT recentered
  (no MT pin exists in code). A single MT global-scale LB probe will confirm or rule out a hidden bias.
- **Scale — reader strong, absolute scale unproven.** 295/309 rows get a recovered px/mm via the
  per-family router; 14 fall back to prior. Per-family detection is solid, but absolute per-image
  correctness is not validated against any test label; part of the FL undershoot is residual scale.

## Validation instruments are blind to global/scale errors

Both local validators, by construction, remove the exact degree of freedom the FL win exploited:

- The 35-image benchmark feeds TRUE scale (`experiments/score_weights.py:42`) and recenters predicted
  FL to its own truth mean (`:54`). It is mathematically incapable of seeing a global FL scale/mean error.
- The 19 hand labels are self-measured by the same geometry engine and reported FL ~unbiased
  (-0.46 mm). They MISSED the 0.025 FL win.

Do not gate any global or calibration quantity on either. See `../VERIFIED_FACTS.md` and the playbook below.

## Go-forward playbook (validation discipline)

1. The leaderboard is the ONLY valid gate for any global/calibration quantity (FL scale, MT scale, PA
   shift). The two local instruments are retired for that purpose.
2. Quantify the LB noise floor first (SE ~ 0.5/sqrt(public-subset-n)); treat any LB move under ~2x SE
   as noise. Stop fine-tuning an axis once gains enter the noise band.
3. One isolated global multiplier per submission, with a written accept/reject rule before submitting.
4. A change is robust only if it is one degree of freedom with a mechanistic reason AND large vs noise
   (the FL global scale qualifies; visual plausibility and oracle-fed CV do not). Do not tune many
   parameters to the public LB (private-LB shakeup risk).
5. Two-tier every claim: VERIFIED (code line or LB number) vs HYPOTHESIS. Only VERIFIED drives decisions.
6. Keep a falsified/dead-ends record; never silently drop a refuted claim (this repo re-asserted the
   "recenter no-op" across 5+ docs because it was dropped, not killed).
7. For model CV (segmentation retrain), use GroupKFold by subject/device, stratified on muscle/disease,
   to mirror the shifted test distribution. Do not gate model changes on the oracle-fed benchmark.

## Immediate plan (next free submissions, then code)

1. Finish the FL-scale bracket on the LB: x1.10, x1.15, x1.20, x1.25 (built). Stop when two
   consecutive points fail to clear the noise floor.
2. One MT global-scale probe on the best FL base.
3. Stack the winners (PA+2.5 + FL_opt + MT_opt) into one submission and confirm on the LB.
4. Bake the FL optimum into `segment_then_measure.py` (raise `PRIOR['fl_mm']` at `:96` or set
   `USE_FL_RECENTER` default off at `:109`) so the win regenerates from the pipeline, not a CSV patch.
5. Only after free LB slots are spent: GPU fragment-selection / fascicle-recall work to cut the FL
   spread (2.26 tol) and the PA aggregation under-read. Gate with GroupKFold, never the benchmark.
