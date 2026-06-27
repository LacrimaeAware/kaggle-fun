# Selector V3 powered A/B — NEUTRAL at full power (2026-06-27)

Model B. Ran the powered/sequential A/B the NEEDS_N500 verdict called for, on the repaired non-inert V3
artifact, with the transplant logging fix in place. **Verdict: B_V3_POWERED_NEUTRAL. Promotion:
DO_NOT_SUBMIT / PARK_V3.** Selector default off, gameplay unchanged, 11/11 test suites pass.

## The answer
The primary question was "is the repaired V3 actually better on deployed+mirror when powered?" **No.** At
full power the combined effect is +2.6pp and not distinguishable from zero.

| metric | off | V3 | delta | p | 95% CI |
|---|---|---|---|---|---|
| **deployed+mirror combined (n=1000)** | 47.5% | 50.1% | **+2.6pp** | **0.263** | [-1.8, +7.0]pp |
| deployed (n=500) | 44.8% | 46.0% | +1.2pp | 0.751 | [-5.0, +7.4] |
| mirror (n=500) | 50.2% | 54.2% | +4.0pp | 0.229 | [-2.2, +10.2] |
| denpa92 sentinel (n=200) | 87.0% | 93.5% | +6.5pp | 0.042 | [+0.7, +12.3] |

denpa92 is the only individually significant cell, but its Holm-adjusted p is 0.168 (fails correction across
the 4 tests) and it is a sentinel, not a promotion cell. The study's minimum detectable effect at n=1000/arm
is ~6.3pp; the observed +2.6pp is below it, so this is "underpowered to confirm a small edge," not "proven
zero." Either way a sub-MDE self-play number cannot justify promotion, and local self-play does not predict
the ladder.

## The headline lesson: early-stopping bias, fully illustrated
The combined-primary effect shrank monotonically as power rose:

```
n=20 smoke   +15.0pp
Stage A      +11.7pp   (n=120 combined)
Stage A+B    + 7.5pp   (n=400, p=0.040)  <- crossed p<0.05
Stage A+B+C  + 2.6pp   (n=1000, p=0.263) <- full sample, not significant
```

The A+B interim look was "significant." The full sample was not. With no alpha-spending plan, stopping at
the first significant look would have been a false positive. This is exactly why the N500 run was worth
doing rather than promoting on the n=20 smoke or the n=400 look. The Stage C deployed cell even went -3pp
(non-significant, p=0.51); Cochran's Q on the deployed trend (+16.7 -> +3.6 -> -3.0) = 4.20, p=0.123 =
consistent with noise around a true ~zero effect, not a real regression.

## Two worries from the smoke, resolved
1. **denpa92 -15pp "regression" was noise.** At n=200 it is +6.5pp. No regression anywhere at power.
2. **"Overrides anti-correlate with winning" was mostly a game-length confound.** Override RATE is nearly
   flat (win 0.269 vs loss 0.282); the count gap (9.7 vs 10.1) tracks game length (34.7 vs 35.6 decisions;
   corr ~0.91). A small residual rate difference survives (Mann-Whitney p=0.006), but any within-arm
   override-vs-result split is observational and confounded (overrides and outcome are both downstream of
   board state). The clean causal number is the randomized between-arm delta, +2.6pp ITT, which is ~zero.

## Safety (genuine and strong)
0 errors, **0 executed terminal overrides**, 11,599 applied overrides all 100% table-backed, 3,522
blocked-terminal/veto fallbacks (the safety layer firing correctly), across 1,700 games. V3 is safe; it is
just not better.

## Adversarial review
Independent reviewer recomputed every number from the raw files and AGREED with B_NEUTRAL. Confirmed pooling
is correct (summed W/L, not averaged percentages), Newcombe CI == Wald at this n, the deployed trend is
noise (Cochran Q), denpa92 fails all multiple-comparison corrections, MDE ~6.3pp. It caught one false
sub-claim of mine ("override rates ~equal" -- they are not, MW p=0.006) which I replaced with the
non-causality framing above. No headline number changed.

## What this means for the program
This confirms the diagnosis from the V3 design review: the live runtime uses a state-BLIND action-type
support table (T(a), keyed FAMILY||COMPACT_SEMANTIC_KEY), so it intervenes constantly (11.6k overrides)
without a state-specific reason and produces a safe-but-neutral policy. The powered run is the clean
evidence that T(a) is not enough. The right next lever is state-conditioned T(s,a) -- Model A's offline V4
prototype (running in the Codex worktree). PARK V3 as the safe baseline; judge V4 against it offline first.

## Artifacts
`data/generated/starmie_selector_v3_powered_ab/`: VERDICT.json, diagnostic_report.json (full pooled stats +
interim looks + Holm + MDE + length-confound + top lookup keys), baseline_manifest.json, preflight_report.json,
stage_{A,B,C}_summary.json, stage_{A,B,C}_changed_decisions.jsonl, stage_{A,B,C}_game_summary.jsonl,
review_examples.html. New tools (reuse the smoke engine + logging verbatim; only change opponent subsetting /
per-opponent N / staged output): `tools/selector_v3_powered_ab_v1.py`, `tools/selector_v3_powered_diag_v1.py`.
Note: this is the authorized N500 run, so it lifts the smoke-era 50-game cap; the live agent and selector
thresholds/table are untouched and default off.
