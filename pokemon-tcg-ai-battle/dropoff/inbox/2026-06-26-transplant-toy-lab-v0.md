# Feature-Delta Transplant Toy Lab V0 (2026-06-26)

Model B. Synthetic concept test of the feature-delta transplant idea. No real data, no gameplay, no training.
**Verdict: B_AXIS_DELTA_DIRECTIONAL_ONLY.**

## Question
Does axis-conditioned feature-delta similarity beat one global similarity score for deciding whether an
action analogy transfers?

## Answer (honest, after fixing a confound)
The first run looked like a clean win (A), but it conflated two effects. I added a same-family-uniform control to
isolate axis-conditioning from family-filtering. The decomposition:
- **Family-conditioning is the dominant lever: -44.8% MSE** (global cross-family 0.78 -> same-family-uniform 0.43).
- **Per-axis conditioning adds a real but modest -12% MSE / -20% bad-transplants** on top (0.43 -> 0.38).
- **Support/abstain** cuts bad-transplants further (22.8% -> 18.5%) at 33% abstention.
- **A wrong hand-mask ~= no mask** (M1 0.46 ~= control 0.43): use LEARNED axis weights, not hand-picked.

So the user's feature-delta intuition is directionally right but **second-order**: conditioning the analogy on the
action family (+ abstaining when unsupported) is what matters most; per-axis delta weighting is a learned-only
refinement, not a breakthrough.

## What it proved / didn't
PROVED (in a toy with family-dependent decisive axes + confounders): family-conditioning >> per-axis; learned >
hand mask; abstain cuts bad transplants. NOT PROVED: that real replay data has this structure (assumed), the real
magnitudes (toy targets are cleaner than reality), or anything about win rate.

## Implication for Model A's real transplant layer
Keep the family-conditioned + support-gated design (already chosen). Prefer learned per-family axis weights over
hand masks. Don't over-engineer per-axis schemes expecting large gains. Abstain-on-OOD is essential (the recurring
distribution-shift risk: memory = expert states, selector acts on our-agent states).

## Artifacts
`data/generated/transplant_toy_lab_v0/`: toy_ground_truth_rules.json, toy_replay_memory.jsonl,
similarity_method_comparison.json, feature_removal_similarity_report.json, review_examples.{html,jsonl},
concept_summary.md. Tool: `tools/transplant_toy_lab_v0.py` (reproducible, seed=7).
