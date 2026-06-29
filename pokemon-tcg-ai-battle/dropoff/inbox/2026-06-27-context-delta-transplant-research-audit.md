# Context-Delta Replay Transplant — Literature Audit (deep research, cited)

Date: 2026-06-27
Mode: pure research, ~90% theory/prior-art. Adversarial audit, not a build task.
Method: deep-research harness (5 search angles, 24 primary sources fetched, 103 claims extracted, 25 adversarially verified with 3-vote refute, 20 confirmed / 5 killed). All load-bearing findings rest on primary peer-reviewed sources.

---

## Verdict (lead)

The transplant idea is **sound-but-not-novel in its core, and the one distinctive piece is the one most likely to hurt.**

- The retrieval-aggregation engine `û = Σ w_i y_i / Σ w_i` is a normalized kernel-weighted nearest-neighbor estimator (Nadaraya-Watson form). It is **literally the readout of Neural Episodic Control**, the skeleton of Model-Free Episodic Control, and a descendant of Ormoneit & Sen's Kernel-Based RL (2002). This is ~20 years of named prior art. The core is not new.
- The setting (large logged expert corpus, no counterfactual simulation, naive RL not winning) is **correctly diagnosed** as an offline-RL / counterfactual-query problem whose canonical hazard is OOD-action overestimation. Support-gating + abstention is a **legitimate but standard** member of the support-constraint family (BCQ, CQL), and it must beat trivially simple baselines (TD3+BC).
- The genuinely distinctive component, conditioning the retrieval weight on the post-action delta `K_Δ`, is **the part most likely to be faulty.** `Δ_a = φ(s_after_a) − φ(s)` is a post-treatment / mediator variable; weighting on it induces **overcontrol (post-treatment) bias** when the target is the action's total value, even under randomization. Your own AUC result (delta 0.5887 < no-delta 0.5938) is consistent with `K_Δ` adding bias/error rather than signal.

Bottom line: **conditionally valid and worth benchmarking** as an interpretable, abstaining, low-data retrieval estimator, but not novel, and the Δ-conditioning you find "deep" is precisely where the theory predicts trouble, **unless** you commit to value-prediction (not causal-effect) as the estimand and can show Δ is a near-sufficient statistic for the action's consequences.

---

## 1. Landscape — what this already is

| Family | Key work | Relation to the transplant |
|---|---|---|
| Kernel-based RL | Ormoneit & Sen 2002 | Canonical ancestor: local-averaging normalized kernel estimate of the value/Bellman operator from logged data. Your `û` is this. |
| Model-Free Episodic Control | Blundell et al. 2016 | Non-parametric k-NN value estimator averaging stored returns over an embedding. Same retrieval-aggregation skeleton. Stores the **max** return (risk-seeking in stochastic settings); you use a weighted mean, milder but same family. |
| Neural Episodic Control | Pritzel et al. 2017 | Readout `o = Σ_i w_i v_i`, `w_i = k(h,h_i)/Σ_j k(h,h_j)` is **algebraically identical** to your `û`. Its real edge is data efficiency (54.6% vs DQN 15.7% at 10M frames) but it is **overtaken by 40M frames** (Prioritized Replay 89.0% vs NEC 83.3%). Lesson: the advantage is low-data robustness, not generalization. |
| Retrieval-augmented RL | R2A, Goyal et al. 2022 | The field **learns the retrieval metric end-to-end** (neural retrieval + information bottleneck). Your fixed hand-built product kernel `K_op·K_ctx·K_Δ` is the weakest design choice relative to the state of the art. |
| Offline RL / OOD-action | Levine et al. 2020 (survey); BCQ Fujimoto 2019; CQL; IQL; TD3+BC Fujimoto & Gu 2021 | Frames your setting exactly: "making and answering counterfactual queries." Core failure = action distribution shift → overestimation of OOD actions ("extrapolation error"). Support-gating/abstention = the BCQ-style support constraint. **TD3+BC** (TD3 + a behavior-cloning term + state normalization, one extra hyperparameter) matches BCQ/CQL/BEAR at half the runtime — the simple baseline to beat. |
| Non-parametric value-estimation theory | Shah & Xie 2018; Zhao & Lai 2024 | Sample complexity `Õ(1/ε^(d+3))` with **matching minimax lower bound `Ω̃(1/ε^(d+2))`**. A lower bound holds against any algorithm: no kernel escapes the curse of dimensionality for free. |
| Good/bad controls (causal) | Cinelli, Forney & Pearl 2024; Montgomery, Nyhan & Torres 2018 (AJPS) | The skeptical core. Conditioning on a mediator/post-treatment variable blocks part of the effect and biases the estimate "even when treatment is randomly assigned." This is what `K_Δ` does. |

Convergent reading: the transplant is a **recombination** of episodic control (the engine) + offline-RL support constraints (the gate) + a causally-fraught novel key (the delta). Only the delta key is distinctive, and it is the riskiest piece.

---

## 2. Soundness audit

Assumptions a retrieval/matching estimator needs: consistency, overlap/positivity (candidate actions must have close analogs), ignorability / no unmeasured confounders, SUTVA-equivalent, support under the behavior policy.

- **(a) Policy confounding.** Logged expert actions are not randomly assigned; skilled pilots pick actions based on hidden evaluations. Matching on observable context `K_ctx` only de-confounds to the extent context captures what drove the choice. **Hidden pilot skill/style is a plausible unmeasured confounder** correlated with both the action chosen and the outcome. Leave-one-agent / leave-one-deck-out retrieval *tests for* leakage but does not *remove* the confounder. Mitigations: partial. Status: conditional.
- **(b) Δ is post-treatment — the crux.** `Δ_a` is caused by the action. Weighting retrieval by Δ-similarity conditions on a mediator. Pearl (overcontrol bias) and Montgomery et al. (post-treatment conditioning "ruins" effect estimates even under randomization) are confirmed (3-0). This is **target-dependent**: conditioning on the mediator is valid only if the estimand is the *controlled direct effect* with the mediator fixed, not the *total effect*. Mitigations proposed (residualization, support-gating, leakage control) **do not address this**. Status: faulty if the estimand is the action's total causal value; defensible if the estimand is value prediction (see §3).
- **(c) Curse of dimensionality.** `Ω̃(1/ε^(d+2))` is a theorem, not a tunable. BUT `d` is the **intrinsic** dimension of the (context, Δ) space under smoothness, not raw feature count — a well-chosen low-dim embedding can make it tolerable. With ~4,500 replays, the real question is whether per-query effective support `n_eff` is large enough in most cells. Status: binding constraint; survivable only with aggressive dimension control.
- **(d) Same-trajectory leakage.** Real and correctly flagged; the nearest neighbor is often the same game's continuation. Leave-one-decision/-episode retrieval is the right mitigation. Status: adequately handled if enforced.
- **(e) Credit assignment / reward sparsity.** Final win/loss is distal and noisy. Residualizing against a local baseline `V̂(s)` (advantage `A ≈ V(s_{i+1}) − V(s_i)`) is the standard variance-reduction move and is appropriate. Status: adequate.

---

## 3. The four framings, in plain terms, then matched

- **(i) Local treatment-effect estimation.** "What is the causal effect of taking action a versus not, for situations like this?" Treats action as a treatment, state as covariates, outcome as the response. **Prescribes** matching on *pre-treatment* context. **Warns** hard against conditioning on the post-treatment Δ, and demands overlap + ignorability. → This framing is where the method is **faulty rather than fitting**.
- **(ii) Case-based / non-parametric value estimation.** "Find past situations like this one and average what happened." No causal claim, just prediction. This is NEC/MFEC/KBRL. → **This is what the object actually is.** Best primary fit.
- **(iii) Offline RL.** "Learn a good policy from a fixed batch without exploring." The support-gating/abstention is a conservative offline-RL move. → Secondary fit; explains the safety machinery.
- **(iv) Metric learning.** "Learn the similarity function so that close = behaves-similarly." → What the field does (R2A, bisimulation, successor features) and what your fixed kernel **neglects**. The gap, not the fit.

The collision: your strongest intuition (Δ-similarity) lives in framing (i)'s danger zone while the engine lives in framing (ii). **The estimand you choose decides whether Δ-conditioning is clever or biased.** You have not committed to one, and the verdict flips on it.

---

## 4. Does it fix offline RL?

Partially, and in a standard way. Support-gating + abstention is a principled defense against OOD-action overestimation — it is the BCQ support-constraint idea. It **buys** interpretability, explicit abstention, and low-data robustness. It **costs** a ceiling on how far it can ever recommend beyond logged expert behavior (Levine: staying close to the behavior policy "may come at a substantial cost in final performance"), plus the curse of dimensionality and Δ-prediction error. It does **not** dissolve the underlying counterfactual-inference difficulty; it relocates it into the kernel design and the Δ estimate. Note (confirmed): more corpus data alone does not fix offline RL — the value has to come from the conservatism/gating, not from corpus size. That validates "naive RL has not won" but warns that bulk replay volume is not the lever.

---

## 5. Is there an obvious better choice?

No silver bullet (naive RL not winning is real), but there are **named comparators you must benchmark against and have not**:

- **TD3+BC** — the trivially simple offline-RL baseline; if the transplant cannot beat it, the complexity is unjustified.
- **IQL / CQL** — conservative offline-RL value estimators that handle OOD actions without explicit retrieval.
- **Learned-metric retrieval (R2A-style), or a bisimulation / successor-feature embedding** — if retrieval is the bet, learn the metric instead of hand-fixing it. The literature says this is where the performance is, and it is a candidate explanation for your delta-weighting underperformance (the fixed kernel, not the Δ idea per se).

The transplant is a **reasonable, under-explored point** in the design space *if* its selling point is interpretability + abstention + low-data robustness for a legal-option decision system. It is not obviously better on raw performance.

---

## 6. Novel vs recombination

- Retrieval-aggregation core: **not novel** (NEC/MFEC/KBRL).
- Support-gating/abstention: **not novel** (BCQ family); one instance of the support-constraint principle (do not overclaim it as the same as all support methods — that overgeneralization was refuted).
- Product kernel with an explicit operator/family term `K_op` for legal-option decision systems: **packaging**, a sensible engineering choice, not a new method.
- **Context + action-delta / effect-similarity retrieval (`K_Δ`): the only genuinely distinctive piece** — and it is the causally fraught one. Its novelty and its risk are the same object.

---

## 7. What would falsify it / what would be strong evidence

Domain-independent toy benchmarks with known ground-truth advantage:

1. **Generally-good-except-locally-bad action.** `g(s,a) > 0` except where `s_risk < ε`. Can context+delta detect the local bad region that action-type `T(a)` misses?
2. **Spurious co-occurrence vs functional delta.** Two actions co-occur in the same "deck" cluster but have different effects. Does correlation-similarity fail where delta-similarity succeeds?
3. **Same-context / different-delta** and **same-delta / different-context.** Isolate whether Δ carries independent signal.
4. **Hidden-confounding stress test.** Inject an unobserved `h` driving both action and outcome; measure degradation.
5. **Decisive-axis recovery under feature-subset removal.** Ground-truth decisive axes known; does the axis-sensitivity diagnostic recover them?

Strong evidence FOR the idea = on a synthetic environment where ground-truth advantage is known, **`K_Δ` beats the no-Δ retrieval baseline**, and that gain survives leave-one-episode/agent retrieval. You have not shown this; your one real-data test points the other way (small n).

**Do NOT cite these as support (refuted 0-3 / 1-2 in verification):** a convergence guarantee for NN-regression from a single logged path for this one-step form; KBRL-style consistency "without causal assumptions" for this estimator; the claim that R2A already establishes "condition on a retrieved corpus" as the identical thesis; the claim that all support-constraint methods "differ only in closeness." The estimator's consistency cannot be assumed.

---

## The refined question (maximum-efficacy path)

The entire promise reduces to one empirical, answerable question:

> Is there a feature map φ such that the delta `Δ = φ(s') − φ(s)` is a near-sufficient statistic for the action's consequences AND adds retrieval signal beyond context + action-type, under leakage-controlled retrieval?

- If **yes**: the idea has legs as *value prediction* (not causal effect) with abstention, in the low-data regime — a defensible, interpretable niche.
- If **no**: it collapses to known episodic control plus a fragile, possibly-biasing extra key, and the right move is learned-metric retrieval or a standard conservative offline-RL baseline.

Before any of that: **commit to the estimand** (causal effect vs value prediction). That single decision determines whether `K_Δ` is the deep insight you feel it is or the bias the literature predicts.

---

## Open questions carried forward

1. Estimand: causal effect of the action (then `K_Δ` is overcontrol-biased) or value/usefulness prediction (then `K_Δ` is defensible but makes no causal claim, and current evidence says it did not help)?
2. Does `K_ctx` actually de-confound logged expert actions, or does pilot skill/style remain an unmeasured confounder?
3. Intrinsic dimensionality of the (context, Δ) space after embedding — is `n_eff` adequate at ~4,500 replays?
4. Would a learned metric beat the fixed kernel, and is that (not the Δ idea itself) the real cause of the delta-weighting underperformance?
5. The minimal synthetic benchmark where ground-truth advantage is known: does `K_Δ` ever beat no-Δ retrieval at all?

## Sources (primary)

KBRL (Ormoneit & Sen 2002); MFEC (Blundell et al. 2016); NEC (Pritzel et al. 2017); R2A (Goyal et al. 2022); Q-learning with Nearest Neighbors (Shah & Xie 2018), optimal-rate reaffirmation (Zhao & Lai 2024); Offline RL survey (Levine et al. 2020); BCQ (Fujimoto et al. 2019); TD3+BC (Fujimoto & Gu 2021); CQL; IQL; A Crash Course in Good and Bad Controls (Cinelli, Forney & Pearl 2024); Post-treatment conditioning (Montgomery, Nyhan & Torres 2018, AJPS).
