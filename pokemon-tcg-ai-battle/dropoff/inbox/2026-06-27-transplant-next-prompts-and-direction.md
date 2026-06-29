# Transplant — Reconciled Direction + Next Prompts

Date: 2026-06-27
Status: prompts staged for the user to fire. Nothing launched. Pure research lane.

## Audit-of-the-audit reconciliation (where the deep-research report stands after the second model's critique)

ACCEPT (both models agree):
- Retrieval core has strong prior-art overlap (NEC/MFEC/KBRL family).
- Support/OOD abstention is standard and necessary, not a new fix.
- K_Δ is dangerous ONLY under a causal total-effect estimand.
- Estimand must be committed before more building.
- Current evidence does not justify promoting any transplant implementation.

DOWNGRADE (the report overstated; the second model is right):
- "core is not novel" -> "not novel at the retrieval-skeleton level"; the legal-option structure + support/OOD abstention + fail-closed integration is a real engineering object.
- "Δ is most likely to hurt" -> "Δ is risky and unproven; faulty only for a causal total-effect estimand; under predictive/residual estimand it is a plausible feature needing a clean benchmark."
- "more data alone does not help" -> "more data improves COVERAGE/support; it does not by itself remove OOD/confounding." (And the report's 'unlearning' concern came from ITERATED Bellman backups, which a non-iterative kNN transplant does not even do.)
- "must beat TD3+BC" -> "TD3+BC/IQL/CQL are comparator FAMILIES; adapting them to the legal-option CABT interface is nontrivial; a direction, not an immediate gate."

REJECT:
- That V3/V5 negatives disprove the context+delta idea. They are FORM-NEG (scoped to that exact formulation/target/eval), not a verdict on the family.

UNDER-WEIGHTED BY THE REPORT (the second model's best point):
- The failures so far may be REPRESENTATION failures, not transplant-theory failures. If K_s does not include deck/zone/prize/belief/lethal-clock state, it cannot approximate "winningness", and the deck-out example cannot work. Test representation richness as a first-class variable.

## Estimand commitment (the hinge, now resolved)

The object is CASE-BASED VALUE PREDICTION with a LOCAL RESIDUAL (advantage) target:
  U(s,a,Δa)   = E[Y | S≈s, A≈a, Δ≈Δa]
  V_local(s)  = E[Y | S≈s]
  A_hat       = U(s,a,Δa) − V_local(s)
NOT causal total-effect estimation. Δ is a predictive descriptor, not a control. This defuses the causal-bias version of the K_Δ critique. Two real concerns remain even under this estimand:
  (1) predictive quality: is Δa (predicted via one-step apply) informative or just noise?
  (2) redundancy: if K_s/K_Δ use the same features the heuristic already uses, transplant just re-learns the heuristic and adds nothing (this is the likely cause of the +2.6pp neutral V3). Transplant must beat the heuristic's OWN implicit value, not merely predict outcomes.

## "If transplant were dead, what then?" — honest applied option map (ordered by my estimate of detectable payoff)

1. SHARPEN THE RULER FIRST. Nearly every result reads "underpowered, p≈0.26, neutral". If a real 3-5pp gain is undetectable at feasible N, no method can be iterated. Paired seat-swap, common random numbers, variance reduction, stronger local-meta opponents, larger cheap-sim N. The bottleneck may be the evaluation, not the method.
2. SEARCH QUALITY/BUDGET. Your own finding: uncapped N=32 sampling beat deployed (first win). The engine supports forward simulation (H001). Better search with the strong heuristic as the LEAF evaluator does not need a learned policy to beat the heuristic; it amplifies it. Highest-probability applied win.
3. OFFLINE RL DONE RIGHT (not naive RL). "I tried RL and it did nothing" = naive/online RL, which offline-RL theory predicts will fail on a fixed expert corpus (OOD overestimation). IQL never evaluates OOD actions; it is the single most appropriate UNTRIED tool for "learn from expert logs." BC-from-top-pilots measured by agreement-rate is the cheapest baseline, and you already have the diagnostic ("we attack too early / develop-before-attack").
4. TRANSPLANT AS RESIDUAL TIEBREAK. Only where support is high and the heuristic is uncertain. This is what the C3 family-limited selector was groping toward.

Honest meta: against a strong hand-tuned heuristic (770.9 LB, beats all learned attempts) with noisy evaluation, the realistic gain is small and hard to measure. The durable research output may be a rigorous BOUNDARY MAP (what cannot be beaten and why), which is exactly what your past projects produced and what you have said you value. That is not failure; it is the finding.

The transplant theory benchmark below is cheap, settles your strongest intuition, and is worth running regardless of the applied bets.

---

## PROMPT — MODEL A (recommended next step; pure research, no gameplay)

```
CURRENT TASK:
CONTEXT-DELTA TRANSPLANT — SYNTHETIC GROUND-TRUTH BENCHMARK V0 (THEORY LANE)

Worktree:
C:\Users\EcceNihilum\.codex\worktrees\0557\pokemon-ai-agent

Purpose:
Settle, on SYNTHETIC data with KNOWN ground truth, whether context+delta retrieval has any
signal advantage over context-only and action-only baselines, whether delta-conditioning
helps or hurts, and whether past failures were REPRESENTATION failures rather than
transplant-theory failures. Pure offline research. No gameplay, no live agent, no selector
export, no real replay needed for the core result (real-data is a later, separate step).

ESTIMAND (committed):
Case-based VALUE PREDICTION with a LOCAL RESIDUAL (advantage) target, NOT causal total effect.
  U(s,a,Δa)=E[Y|S≈s,A≈a,Δ≈Δa];  V_local(s)=E[Y|S≈s];  A_hat=U−V_local.
No causal identification is claimed. Δ is a predictive descriptor, not a control.

Do not: run gameplay; touch the live agent; export a selector; modify Model B repo/runtime;
tune on held-out test; claim causal effects.

Generators (known ground truth):
  G1 PREDICTIVE:        Y=f(s)+g(s,a,Δ)+ε
  G2 LOCAL-RESIDUAL:    Y=V(s)+A(s,a,Δ)+ε
  G3 GOOD-EXCEPT-LOCAL: g(s,a)>0 except where s_risk<ε   ("draw good except near deck-out")
  G4 SPURIOUS-COOCCUR:  two actions co-occur in a cluster but have different functional Δ
  G5 CAUSAL-WARNING:    a graph where Δ is a mediator (confirm total-effect bias exists; confirm estimand-C avoids it)
  G6 HIDDEN-CONFOUND:   unobserved h drives both a and Y (measure degradation)

Estimators (run on every generator):
  M0 ACTION_ONLY T(a); M1 CONTEXT_ONLY K_s; M2 DELTA_ONLY K_Δ; M3 CONTEXT+DELTA K_s·K_Δ;
  M4 CONTEXT+DELTA RESIDUAL (committed estimand); M5 SUPPORT_GATED (n_eff/contradiction/OOD abstain);
  optional M6 LEARNED_METRIC (similarity learned from consequences).

Representation-sensitivity (possibly the dominant factor):
  Re-run every estimator under feature sets of increasing richness:
    R0 shallow action label; R1 +coarse context; R2 +the axis needed to express the local-bad region.
  Test the hypothesis: past failures are REPRESENTATION failures (K_s does not define a meaningful
  local family), not transplant-theory failures.

Controls: leave-one-decision-out and leave-one-episode-out retrieval; report same-episode NN rate;
Δa computed before outcome (no leakage); report n_eff/support coverage.

Metrics: outcome MSE/corr (G1); advantage-recovery error vs known A (G2,G3); best-action rank/AUC
where ground-truth-best is known; delta-vs-no-delta gap (does K_Δ ever beat M1/M4-without-Δ, under
which R?); measured bias under G5/G6; abstention calibration.

Artifacts: data/generated/context_delta_transplant_synth_v0/
  estimand_definition.md, generators.md, method_comparison.json, representation_sensitivity.json,
  delta_value_report.md, causal_warning_report.md, review_examples.html, closeout.json

Verdict:
  A. DELTA_ADDS_SIGNAL_UNDER_GOOD_REPRESENTATION
  B. TRANSPLANT_WORKS_BUT_DELTA_NEUTRAL (value is from context; Δ adds nothing)
  C. REPRESENTATION_BOUND (fails until features rich enough; failure was representation, not theory)
  D. NO_SIGNAL_EVEN_WITH_GROUND_TRUTH (kills the idea cleanly)
  E. PIPELINE_INVALID
Stop after report + verdict. No real-data run, no export.
```

## PROMPT — MODEL B (optional; only if you want the infra cleaned; no transplant work)

```
CURRENT TASK:
PROJECT STABILIZATION / SUBMISSION-PATH INSULATION V0

Repository:
C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\pokemon-tcg-ai-battle  (live tree)

Purpose:
Protect the stable, default-off infrastructure and the submission path while the transplant
theory lane runs. Inventory only; no new gameplay idea, no A/B, no selector enable, no transplant.

Do not: run A/B; change gameplay; enable selector; tune heuristics; merge automatically; build submission.

Audit + report:
1. Stable default-off infra (selector_trace logging, repaired transplant-field logging,
   compact_semantic_action_key bridge, turn_context_v0, consequence feasibility probe,
   packer parity harness, smoke harness, full-test harness fix).
2. Experimental artifacts NOT to promote (V1/V2-C3/V3 selectors, T(a) table, V4/V5 artifacts).
3. Confirm current default behavior: selector off, gameplay unchanged, submission path insulated.
4. The dedicated repo problem: pokemon-ai-agent main is a hello-world stub and the real work is
   UNCOMMITTED in .codex/worktrees/0557. Recommend: commit-the-worktree vs keep-in-kaggle-fun,
   and preserve gitignored data/generated before any worktree removal.
Write: data/generated/project_stabilization_v0/current_branch_inventory.md,
       merge_readiness_report.json, dropoff/inbox/2026-06-27-project-stabilization-v0.md
Verdict: A STABLE_INFRA_READY / B NEEDS_CLEANUP / C PIPELINE_DIRTY. Audit only.
```

## Not written as prompts yet (the applied alternatives) — available on request
IQL/BC-from-experts screen; evaluation-sharpening lab; search-budget/leaf-value lab. Say the word and I draft these.
```
