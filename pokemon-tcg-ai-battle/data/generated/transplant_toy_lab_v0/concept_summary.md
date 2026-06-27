# Feature-Delta Transplant Toy Lab V0 — Concept Summary

**Verdict: B_AXIS_DELTA_DIRECTIONAL_ONLY.** Axis-conditioned feature-delta similarity helps, but the dominant
lever is **family-conditioned retrieval**, not per-axis weighting. Concept test only — synthetic data, no gameplay,
no model training, no real-replay implementation.

## Setup (transparent + honest)
A toy of 12 axes and 11 action roles across 4 families. The ground-truth value reads ONLY each role's **decisive
axis subset**, and those subsets **differ by family** (e.g. `attach_main` -> {energy_shortfall, target_role};
`attack_KO` -> {safe_dev, prize_liability, effect_nullification}; `heal` -> {opponent_threat}). Four axes
(`hidden_risk_proxy`, `noise_a/b/c`) are pure confounders that never affect value. 1500 memory rows + 500 held-out
queries; observed outcomes carry Gaussian noise. This structure **mirrors the real domain** (the audit and Model
A's family projections both say decisive axes are family-specific), but it is an ASSUMPTION, not a measured fact.

## Results (held-out value MSE / bad-transplant rate; bad = nearest neighbor has opposite-sign true value)
| method | MSE | bad% | note |
|---|---|---|---|
| M0 global scalar (cross-family, learned weights) | 0.78 | 43.4 | the naive baseline |
| M0b same-family + UNIFORM weights (control) | 0.43 | 28.4 | isolates family-filtering |
| M1 family hand-mask | 0.46 | 25.6 | imperfect mask (omits effect_nullification on ATTACK) |
| M2 axis-conditioned (structure-derived weights) | 0.38 | 22.0 | |
| M4 learned per-family axis mask | 0.38 | 22.8 | |
| M5 M4 + support/abstain | 0.30 | 18.5 | 33% abstention |

Decisive-axis detection recall (M3 leave-one-axis / learned-weight ranking vs the true decisive set): **0.62**.

## What the toy PROVED
1. **Family-conditioning is the first-order win.** Restricting transplant to same-family neighbors cut MSE **44.8%**
   (0.78 -> 0.43) with no axis weighting at all. This is the single biggest lever.
2. **Per-axis conditioning is a real but modest refinement.** On top of family-filtering, learned/structure-derived
   axis weights cut MSE a further **12%** (0.43 -> 0.38) and bad-transplants **~20%** (28.4 -> 22.8). Useful, not huge.
3. **A wrong hand-mask is no better than no mask.** M1 (0.46) ~= the uniform control (0.43); the deliberately
   imperfect ATTACK mask (missing the decisive `effect_nullification`) erased the benefit. **Use LEARNED axis
   weights, not hand-picked masks.**
4. **Support/abstain meaningfully reduces bad transplants** (22.8% -> 18.5%) at a 33% coverage cost — a real
   safety lever, exactly the kind the V2 selector needed.

## What the toy did NOT prove
- That real replay data actually HAS this family-dependent-decisive-axis structure (assumed by construction).
- The real-world MAGNITUDES. Real consequence targets are noisier and more confounded than the toy's clean
  `true_value`; the 12% axis-conditioning gain could shrink toward zero under real noise.
- Anything about gameplay win rate. This is retrieval/value-estimation quality, not policy quality.

## What transfers to Model A's real transplant layer
The toy SUPPORTS the design Model A already chose (family projections + per-family similarity + support/abstain),
and refines the priorities:
- **Primary:** retrieve within action family (or finer, per semantic role). This is where most of the value is.
- **Secondary:** weight axes by LEARNED importance per family — worth doing, but a refinement, not a breakthrough;
  do not over-engineer elaborate per-axis schemes expecting large gains.
- **Keep the support/abstain gate** — it is the cheapest way to cut bad transplants, and it directly addresses the
  out-of-distribution risk.
- Hand-tuned axis masks are risky (a single missed decisive axis = no gain); prefer data-derived weights.

## Risks that remain
- **Distribution shift** (the recurring lesson): the transplant memory is expert/replay states; the selector acts
  on our-agent states. The toy assumes query and memory share structure; reality may not. Abstain-on-OOD is essential.
- **Confounded consequence targets**: real "what happened after" mixes opponent variance and later own decisions;
  the toy's clean value overstates separability.
- **Family granularity**: the toy's gain came mostly from family-level conditioning; if real decisive axes vary by
  ROLE within a family (likely), per-role conditioning may matter more than per-axis weighting.

## Bottom line
The user's feature-delta intuition is **directionally correct but second-order**: conditioning the analogy on the
action's family (and abstaining when unsupported) is what matters most; per-axis delta weighting adds a modest,
learned-only refinement. Model A should keep its family-conditioned + support-gated design, prefer learned axis
weights over hand masks, and not expect per-axis conditioning alone to rescue the selector.
