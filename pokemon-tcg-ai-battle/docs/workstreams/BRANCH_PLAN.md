# Two-branch plan (Expert Iteration: planner + learner -> student-guided search)

Provenance: authored by the other model, pasted by the user 2026-06-18, saved here as the canonical plan.
Structure: planning creates stronger targets (Branch A), learning generalizes them (Branch B), then the
learned policy guides stronger planning (integration). References: ExIt (arXiv 1705.08439), DAgger
(1011.0686), DART (1703.09327), Successor Features (1606.05312).

## Mandatory shared preflight (do BEFORE branching; commit as SPLIT_BASE)
1. Finish the Praxel-vs-DENPA92 deck A/B. [DONE: Praxel deck 0.300 under our search -> keep DENPA92.]
2. Select ONE production deck for both branches.
3. Freeze the current agent_search as BASELINE_V1: N_DETERM=8, aggro continuation, current time budget, selected deck.
4. Record exact git commit, deck hash, config, baseline results.
5. Create ONE frozen semantic state/action schema.
6. Create ONE teacher-query interface returning, per legal action: semantic action key, mean search value,
   value variance, completed-determinization count, score margin from best, soft action probs / normalized
   advantages, chosen action.
7. Verify semantically-equivalent actions canonicalize correctly.
8. Verify train-time and live-time state/action encodings are identical.
9. Commit as SPLIT_BASE.
10. Create both branches from exactly SPLIT_BASE: `exp/search-teacher-v2`, `exp/dagger-embeddings-v2`.
Neither branch merges to main until both have final reports.

## Branch A — Stronger Planner and Teacher (exp/search-teacher-v2)
Mission: the strongest live search under the time limit, AND a substantially stronger OFFLINE teacher
(allowed to be slower than live) that labels data for Branch B. Not another aggro-vs-setup sweep.

DO NOT: train/modify the neural ranker; build card embeddings; run DAgger; add many card-specific hand
heuristics; modify the frozen schema; change deck AND search methodology in one experiment; declare a
result from <200 games; merge to main; call an experiment positive from prediction fit; call opponent
belief refuted using a search structurally insensitive to opponent hidden cards.

- A1 Baseline reproduction: reproduce BASELINE_V1; verify identical deck/budget/seeds/seat-swaps/opponent
  pool; log ACTUAL completed determinizations per decision (not just configured N); abort if not reproducible.
- A2 Teacher stability audit: >=500 non-forced decisions (replay + baseline-self-play + failed-ranker
  states); query the teacher >=16x per decision with different determinization seeds; report top-action
  stability, value variance, top-2 margin, acceptable-action set, disagreement by decision type; do NOT use
  one hard argmax when unstable; export soft targets + mean advantages.
- A3 Build offline Teacher_V2 (may be slow): test N_DETERM 16/32/64; shared hidden-world samples across
  sibling actions; selective deeper search on low-margin/high-uncertainty decisions; multiple continuation
  policies (aggro, stochastic/legal-random, replay-frequency, mixtures); robust aggregation (mean, LCB,
  pessimistic quantile); longer opponent-response horizon on a bounded subset. Teacher_V2 returns mean
  value, uncertainty, advantage, soft policy target, semantic equivalence class.
- A4 Improve LIVE search under the real budget via ADAPTIVE allocation: normal budget on clear decisions;
  spend extra determinizations only when top-2 margin small / hidden-world variance high / candidate
  policies disagree; selective depth only on uncertain decisions; keep forced/legal rules outside search;
  compare every candidate vs BASELINE_V1 under the exact same wall-clock.
- A5 Validation: 40 games = screening only; promotion >=200 (seat swaps, multiple seeds); final >=400 with
  Wilson interval, no excess timeout/error, improvement across representative decks, exact hashes recorded.
- Deliverables: agent/search_teacher_v2.py; tools/query_teacher_v2.py; teacher-label data manifest;
  docs/workstreams/SEARCH_TEACHER_V2.md; table of ALL experiments incl negatives; one frozen online-search
  candidate; one frozen offline Teacher_V2; a machine-readable teacher API for Branch B.
- Success: EITHER online search beats BASELINE_V1 under equal budget, OR Teacher_V2 is materially more
  stable/deeper and yields higher-quality counterfactual labels for Branch B.

## Branch B — Robust Learned Policy, Embeddings, Affordances (exp/dagger-embeddings-v2)
Mission: can a learned system understand full state, learn action consequences, recover from its own
mistakes, generalize across decks/cards, and become a search prior. DAgger because supervised imitation
trains on one distribution while the deployed policy induces another; DART-style augmentation must be
generated LEGALLY and usually RELABELLED (Pokemon state changes are not label-preserving like image rotations).

DO NOT: modify the frozen search baseline / teacher API; use the old 47-feature vector as the sole state;
train only on winner replays; treat one teacher argmax as truth when variance is high; randomly perturb
numeric features and keep the old label; declare success from offline top-1 alone; deploy the student as
production before diagnostics; merge to main; add architecture complexity before representation+target
diagnostics pass.

- B1 Why the old student failed (before retraining): (1) representation ceiling -- overfit a big model on a
  small subset with semantic action scoring; if it cannot nearly memorize low-entropy teacher decisions, the
  representation/labels are insufficient, fix before DAgger. (2) teacher label stability via the API (entropy,
  variance, top-2 margin; soft targets for unstable; down-weight ties). (3) on-policy shift -- let the ranker
  play >=100 games, query the teacher on every visited decision, report agreement/regret before vs after the
  first disagreement, by turn, by distance from replay distribution, on high-margin decisions. Do not claim
  covariate shift without this measurement.
- B2 Full state/action encoder: exact entities (active, every bench, hand, discard, prize/deck counts,
  statuses, attached energy, damage, legal targets); action rep (type, acting card/entity, target, effects,
  resource consumed, equivalence class); public history (short recent sequence); card rep (designed features
  + learned embedding + unconstrained residual). Permutation-invariant set/entity encoder; not just card-ID lookup.
- B3 Targets: main = within-decision advantage, teacher soft policy, teacher uncertainty. Auxiliary
  successor-affordance = future legal-option expansion, attack-unlock prob, expected prize/KO, survival
  through opp response, energy/resource continuity, future hand/draw/tutor availability, deckout risk.
  Exact simulator consequences are inputs/exact labels, not approximated. Keep a latent residual path.
- B4 Safe augmentation: label-preserving only (legal-option order permutations w/ transformed labels,
  identical-copy permutations, perspective canonicalization, rule-confirmed equivalent target permutations);
  everything else needs teacher relabelling. Generate reachable recovery states through the ENGINE (one
  plausible non-teacher action then continue; resample hidden world; omit/reorder a setup action; sample
  states after high-regret decisions). Never alter state arbitrarily in feature space; never assume the
  original action stays optimal after a state change.
- B5 Three-round DAgger/DART: R0 clean replay + Teacher_V2 labels; R1 student-generated games labelled by
  Teacher_V2 on visited states, retrain; R2 add controlled legal perturbations/recovery states; R3 only if R2
  improved on-policy regret. Report each round: semantic agreement, action regret, high-margin + high-crit
  accuracy, teacher-query distribution shift, arena, error/timeout. Stop after 3 if no improvement.
- B6 Ablations: full; no embedding; no affordance heads; no history; no DAgger data; old compressed state;
  one-hot vs learned embedding. Held-out-DECK test (same-deck fit is not generalization evidence).
- B7 Output (not a standalone replacement): policy prior over legal actions, predicted advantage,
  uncertainty, successor-affordance vector, semantic action embeddings.
- Deliverables: agent/student_v2.py, agent/student_prior_v2.py, tools/dagger_collect.py,
  tools/dart_recovery_states.py, tools/train_student_v2.py, dataset manifest w/ hashes+source breakdown,
  docs/workstreams/ROBUST_LEARNER_V2.md, round-by-round results, all ablations, frozen Student_V2.
- Success: at least one of -- on-policy teacher regret declines over rounds; Student_V2 improves held-out-deck
  action ranking; improves equal-budget search as a prior; affordance supervision improves regret or search
  strength. Offline affordance accuracy alone is not success.

## Model allocation, ownership, integration
- 3 models: A implements Branch A, B implements Branch B, C is an adversarial auditor (does NOT rewrite
  architecture; catches no-op experiments, dead/missing features, action-equivalence errors, train/inference
  mismatch, leakage, mixed-player labels, unmatched budgets, unstable teacher labels, false causal claims,
  results before the gate).
- Shared read-only after split: agent/teacher_api.py, agent/state_action_schema.py, docs/workstreams/SPLIT_BASE.md.
  Branch A owns search_teacher_v2.py + tools/search_*/query_teacher_v2.py + SEARCH_TEACHER_V2.md. Branch B owns
  student_v2.py + student_prior_v2.py + tools/dagger_*/dart_*/train_student_v2.py + ROBUST_LEARNER_V2.md.
  Neither edits agent/main.py, the production package, or the other branch's files (only at integration).
- Integration (exp/student-guided-search-v1, after both freeze): test BASELINE_V1; Branch A strongest live
  search; Branch B student alone (diagnostic); search + student candidate ordering; search + student
  continuation policy; search + uncertainty-based budget allocation. Keep SEARCH as final authority (student
  proposes/prioritizes, search evaluates/decides). Do not prune permanently until teacher-best is in the
  student's top-k with very high recall. Same deck/budget/seat-swaps/seeds/opponents, >=400 games for final.
  Succeeds only if student-guided search beats the strongest standalone search under the same budget.

## Recommendation (other model): finish the deck A/B (done), make the SPLIT_BASE preflight commit, then split.
Branch A = safety/performance lane; Branch B = the representation-and-recovery lane.
