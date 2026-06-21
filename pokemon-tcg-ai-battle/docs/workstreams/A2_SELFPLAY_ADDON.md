# A2 add-on — Teacher V1 stability on production self-play states

**Status: final.** Authorized A2-only add-on. No A3 / Teacher V2 / new experiment.

Purpose: test whether Teacher V1's instability on replay states also appears on the states the production
`agent_search` agent actually visits, and compare. Same tool (`tools/audit_teacher_stability.py`), same
`audit_decision` + aggregation, same 16 cross-seed + 16 same-seed protocol, `n_determ=8` — directly
comparable.

## Headline

The instability **reproduces on the agent's own visited states**, and is slightly worse there: 40.2%
stable vs 50.5% on replays. The engine-dominated decomposition **holds on both distributions** (the
determinization draw explains ≈0 of the instability in replays and only ~+0.03 on self-play). So Teacher
V1's noisiness is not an artifact of the replay distribution — the production agent genuinely faces a
roughly-coin-flip teacher on about half of its own non-forced decisions, and that noise is engine rollout
RNG, not the hidden-world sampling.

## Artifacts

- Replay audit: `pokemon-tcg-ai-battle/data/manifests/teacher_v1_stability_full.jsonl` (+ `_summary.json`),
  1094 decisions, frozen `replays_20260618` snapshot (hash-verified), deck-stratified.
- Self-play audit: `pokemon-tcg-ai-battle/data/manifests/teacher_v1_stability_selfplay.jsonl`
  (+ `_summary.json`), 999 decisions, `agent_search` self-play (DENPA92), 26 games, ≤40 decisions/game.

Field schema identical in both (shared `audit_decision`): stability_class, confidence_weight,
avg_soft_policy, avg_advantage, mean_acceptable_set_size, mean_within_value_variance (+
across_seed_modal_value_std), mean_completed_determinizations, mean_top_two_margin.

## Comparison

| metric | replay (n=1094) | self-play (n=999) | read |
|---|---|---|---|
| stable frac (top action ≥90% of seeds) | 0.505 | 0.402 | self-play less stable (−0.10) |
| near-tie frac | 0.212 | 0.259 | more near-ties on-policy |
| unstable frac | 0.282 | 0.338 | more unstable on-policy (+0.06) |
| mean cross-seed top-action stability | 0.783 | 0.748 | slightly less stable (−0.035) |
| mean engine-only (same-seed) stability | 0.772 | 0.774 | ~equal |
| **determinization extra instability** | −0.011 | +0.026 | ≈0 in both; world-draw explains almost none |
| mean top-two margin (hand-eval scale) | 2722 | 20409 | self-play mean inflated by terminal-reaching rollouts (±1e6 leaves); compare by distribution, not mean |
| mean acceptable-set size | 3.10 | 2.61 | self-play more decisive on average |

Stability by action type — replay: ability **0.678** (low), evolve 0.846, attack 0.849 (high). Self-play:
evolve **0.642** (low), select-card 0.835, attack 0.843 (high). Attack is stable in both; evolve flips
from most-stable (replay) to least-stable (self-play); per-type splits are thin, do not over-read.

## What the comparison establishes

1. **Not a replay artifact.** Teacher V1 instability is present, and marginally larger, on states the
   production agent actually reaches. ~60% of its own non-forced decisions are unstable or near-ties.
2. **Engine-rollout-dominated holds on-policy.** Fixing the determinization (same seed) leaves stability
   essentially unchanged on both distributions (replay −0.011, self-play +0.026). The instability is the
   engine's internal coin/shuffle RNG in the rollout, robust across distributions.
3. The self-play acceptable set is smaller (2.61 vs 3.10) and margins more extreme, i.e. the agent's own
   states are more bimodal: some very decisive, many genuinely ambiguous — but still ~40% stable overall.

## Caveats

- Self-play states share one deck (DENPA92) and one policy, so less diverse than the cross-deck replay set;
  per-game capping spread them over 26 trajectories but they remain on-policy/correlated.
- The self-play mean top-two margin is skewed by decisions whose rollouts reach terminal wins (±1e6 hand-
  eval leaves). Treat margins by distribution, not the mean; the stability fractions are the robust read.
- Directional reads, not tight CIs. The determinization-vs-engine decomposition is the robust quantity.
- Forced decisions (lethal/KO floor) are excluded by design; this is the non-forced strategic subset.

## Interpretation constraint (held)

In-bounds: "Teacher V1 has high instability/noise, on both replay and on-policy states, mostly not
explained by determinization/world-sampling in these audits." Out-of-bounds and NOT claimed: "opponent
belief is refuted," "belief cannot help strength," "A3.3 is proven best." Belief-as-strength remains a
separate, untested question requiring an opponent-sensitive leaf/search.

## Proposed A3 next step (no implementation; awaiting signoff)

The self-play corroboration tightens the rationale but proves nothing about a fix. Proposed first Teacher
V2 candidate, when authorized: **A3.3 variance-triggered selective computation** — adaptive determinization
budget concentrated on the high-variance / near-tie / small-margin decisions (now confirmed to be ~half of
the agent's own decisions), at equal total compute, paired against **A3.1 uniform higher-N** as the equal-
budget baseline, gated against V1 on the A4 criteria (lower repeated-query instability via this harness,
lower counterfactual regret, outcome calibration, equal-budget head-to-head). World-sampling levers (shared
worlds A3.2, belief-prior sampling A3.5) stay deprioritized **for stability only**; selective depth (A3.6)
and belief-as-strength remain separate strength hypotheses, not refuted.

Branch A stops here and waits for review/signoff.
