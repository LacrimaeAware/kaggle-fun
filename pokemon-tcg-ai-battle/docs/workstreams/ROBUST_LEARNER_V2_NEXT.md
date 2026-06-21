# Robust Learner V2 Next-Step Memo

Branch: `exp/robust-learner-v2`

Current package commit: `3af3662`

Status: waiting for joint review with Model A's final A2 production-self-play teacher-stability audit. This memo is synthesis only; it does not authorize or implement B2.

## B1 Diagnostic Conclusions

### B1.1 Representation Ceiling

On stable/high-agreement Teacher V1 labels, the compressed root alone is not enough to identify the teacher-preferred action reliably, but root plus action identity nearly memorizes the stable labels.

Summary:

- Root-only expected top-1: `0.530 / 0.586 / 0.490` for train/val/test.
- Root plus action expected top-1: `0.958 / 0.979 / 0.979`.
- Root plus semantic key matches root plus action in the sampled stable subsets.

Conclusion: the first blocker is not simply that all useful information is missing from the frozen representation. B2 should preserve action identity/semantic key information explicitly and should test any learned representation against the stable-label ceiling before adding more moving parts.

### B1.2 Teacher-Label Noise

Teacher V1 top-1 labels are noisy under repeated queries, so hard argmax labels are not safe as the only training target.

Summary:

- Stable labelled decisions: `64 / 64 / 64` for train/val/test.
- Unstable labelled decisions: `162 / 189 / 106`.
- Not applicable: `1 / 0 / 1`.
- Action types with notable instability include `SELECT_CARD`, `ATTACH`, `ABILITY`, and `END`; `END` was unstable in the sampled B1.2 replay labels.

Conclusion: B2 should prefer soft policy targets, acceptable-action sets, normalized advantage/regret, and confidence weights over one hard top-1 label. Stable labels are appropriate for ceiling/overfit tests; unstable labels should be down-weighted or represented as uncertainty.

### B1.3 On-Policy Shift

The 100-game ranker diagnostic supports compounding on-policy errors / covariate-shift-like failure in the old ranker.

Summary:

- `rank` vs `heuristic`, 100 games: `18-82`, 0 draws/errors.
- Student/ranker visited decisions: `2980`.
- Teacher-applicable decisions: `2939`; not-applicable decisions: `41`.
- Acceptable-set agreement: `0.688`; hard top-1 agreement: `0.439`.
- Mean regret: `30694.7`; high-regret decisions at `>=1000`: `335`.
- Before first unacceptable action: acceptable agreement `0.965`, high-regret `5`.
- First unacceptable action: acceptable agreement `0.103`, high-regret `21`.
- After first unacceptable action: acceptable agreement `0.634`, high-regret `308`.

Conclusion: the old ranker performs much better before its first unacceptable action than at or after that point. B2 should be evaluated on student-visited states, not only replay states, and should track before/after first unacceptable action as a core diagnostic.

## Implications For B2 Design

If B2 is authorized after joint A/B review, the design should likely include:

- action-conditioned representation tests before any broad training run;
- training targets that include soft policy, acceptable sets, advantage/regret, and confidence weights;
- label filters or weights for unstable Teacher V1 decisions;
- explicit evaluation on on-policy student/ranker states;
- per-action-type reporting, especially for `ATTACH`, `END`, `ABILITY`, `EVOLVE`, and `PLAY`;
- a small overfit/ceiling gate on stable labels before larger data collection or DAgger-like loops.

This does not mean DAgger is guaranteed to fix the failure. It means any proposed B2 plan should directly address the observed mismatch between replay/stable-label behavior and old-ranker visited-state behavior.

## Proposed Next Gated Step

After Model A's A2 self-play audit is complete, review the A2 and B1 packages together and choose one of:

1. Authorize a B2 design document only, with no training yet.
2. Authorize a narrow B2 prototype focused on stable/soft teacher targets and on-policy evaluation.
3. Hold B2 and first resolve Teacher V1/Teacher V2 reliability questions from A2/A3 planning.

Preferred next action for Branch B is option 1: write a B2 design spec that names targets, filters, metrics, and gates before implementation.

## Risks And Caveats

- Teacher V1 instability remains real; B1.3 is not a proof that every disagreement is covariate shift.
- The distance-to-reference metric is a rough frozen-train L1 check, not a learned distribution-distance model.
- B1.3 covers `rank` vs `heuristic` at the recorded settings; broader opponents or decks would need separate authorization.
- High regret sometimes appears under stable teacher labels, which argues against dismissing B1.3 as label noise, but it does not make Teacher V1 infallible.
- Terminal win/loss remains auxiliary; action-level agreement/regret is the primary diagnostic target.

## Inputs Needed From Model A A2

Before Branch B should start B2, we need Model A's final A2 self-play package to answer:

- whether Teacher V1 instability on production `agent_search` self-play states resembles replay-state instability;
- stability rates by action type on self-play states;
- margins, value variance, completed determinizations, confidence weights, soft policy, advantage, and acceptable-action-set fields for self-play states;
- whether instability appears concentrated in the same action types Branch B saw in replay/ranker-state audits;
- caveats on using Teacher V1 outputs as B2 targets before Teacher V2 or selective computation exists.

No B2 implementation should start until those A2 results are reviewed with this B1 package.
