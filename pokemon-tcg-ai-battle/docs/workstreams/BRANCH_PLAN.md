# Pokémon TCG AI — Parallel Research Branch Plan v2

**Date:** 2026-06-18  
**Fresh project checkpoint:** remote `main` reported at commit `5498dc9`; current proven live agent is `agent_search` with `N_DETERM=8` on DENPA92; the learned ranker has offline search-imitation signal but fails in arena play.

## Decision

Keep the two-branch structure, but revise it as follows:

- **Branch A: Planner / Teacher V2** develops a stronger *offline* teacher and, secondarily, a stronger live search agent.
- **Branch B: Robust Learner V2** diagnoses the learned-policy failure, builds a full state/action representation, and runs bounded DAgger/DART-style recovery training with embeddings and successor-affordance auxiliaries.
- **A third model is an adversarial auditor**, not a second unsupervised coder on either branch.
- Both branches start from one shared, frozen preflight commit.
- Branch B may begin with Teacher V1 immediately; it does not wait for Branch A. Once Teacher V2 is frozen, Branch B performs one final relabel/retrain pass.
- Final integration is **student-guided search**. The student proposes/prioritizes; search remains the authority until equal-budget evidence proves otherwise.

This is not a commitment to “search only.” It is an Expert Iteration program: planning creates stronger targets, learning generalizes them, and the learned model guides a stronger next planner.

---

# Evaluation of the recent pushback

## Accurate and adopted

### 1. The teacher does not have to be the live agent

Correct. A slow offline teacher can use substantially more computation than the competition permits. Expert Iteration explicitly separates planning from generalization, then uses the learned policy to improve subsequent search.

**Plan change:** Branch A now treats the slow offline teacher as its primary deliverable, not merely a by-product of live-search tuning.

### 2. A fixed current-search teacher is not the theoretical ceiling of the integrated system

Correct. A student trained only to clone the current live search is normally bounded by that teacher, apart from approximation/generalization effects. But a student trained on a stronger offline planner, then used to guide a stronger new planner, can participate in a policy-improvement loop.

**Plan change:** the final acceptance test is not “student beats Teacher V2 alone.” It is:

> **student-guided live search beats the strongest standalone live search under the same wall-clock budget.**

### 3. DAgger is the principled response to sequential distribution shift

Correct. DAgger trains on observations induced by the learner’s own policy rather than assuming replay/expert positions are representative of deployment.

**Plan change:** DAgger is no longer treated as a low-upside search-compression exercise. It is the core robustness mechanism in Branch B.

### 4. Legally perturbed recovery states are a valid extension

Correct, with one critical restriction: Pokémon state perturbations are generally **not label-preserving**. They must be produced through the engine and relabelled by the teacher. Safe invariances such as option-order permutations may retain transformed labels.

**Plan change:** Branch B includes a DART-style recovery-state phase, but forbids arbitrary numeric feature perturbations.

### 5. The shared schema/API preflight is the highest-leverage immediate task

Correct. The project has repeatedly lost time to silent schema, option-equivalence, feature-activation, and train/serve mismatches.

**Plan change:** neither branch may begin research experiments until shared golden fixtures and parity tests pass.

## Accurate in direction, but overstated

### 6. “The learner can use outcomes and therefore exceed the teacher”

Possible, but not automatic. Terminal outcomes contain information outside a hand evaluator, but they are delayed, noisy, policy-dependent, and not counterfactual. Winner actions are not necessarily optimal actions.

**Plan rule:** terminal outcome is an auxiliary critic/validation target. The primary action target remains repeated counterfactual action advantage from search. Outcome loss may be added only through an ablation.

### 7. “A few DAgger rounds will take tens of minutes”

Unsupported as a general estimate. Cost depends on the number of visited decisions and Teacher V2 query latency.

**Plan rule:** benchmark teacher throughput first. Run a fixed pilot, report labels/second and total projected cost, then proceed. No model may promise a duration before measuring it.

### 8. “More determinizations means a stronger teacher”

Plausible, but not guaranteed. More samples can reduce Monte Carlo noise, while determinization can retain strategy-fusion and nonlocality problems in imperfect-information games.

**Plan rule:** teacher strength is established through stability, counterfactual regret, and head-to-head evidence—not configured `N` alone.

## Rejected or modified

### 9. Continuous replay fetching should directly feed ongoing experiments

Rejected. A continuously mutating dataset destroys reproducibility.

**Replacement:** maintain:

- an append-only rolling raw corpus;
- immutable dated snapshots with hashes/manifests;
- fixed train/validation/test splits per experiment;
- chronology-, player-, and deck-held-out evaluation.

### 10. Ask the user to choose the production deck before work starts

Unnecessary.

**Decision:** freeze DENPA92 for algorithmic branch comparisons because it is the validated baseline. Finish any already-running deck A/B and record it separately, but do not let deck selection block the split. A deck may replace DENPA92 only through a separate, predeclared promotion gate.

---

# Shared preflight — mandatory before splitting

Create a single commit named conceptually `SPLIT_BASE_V2`. Both branches must be created from its exact SHA.

## P0. Freeze the production baseline

Record:

- commit SHA;
- deck list and hash: DENPA92;
- `agent_search` configuration;
- `N_DETERM=8`;
- continuation policy;
- per-decision and match-time budget logic;
- fallback policy;
- baseline opponents, seeds, seat alternation, win rate, CI, errors, and timeouts.

Do not infer the baseline from mutable code later. Save a machine-readable config.

## P1. Freeze immutable replay snapshots

Create:

```text
data/manifests/replays_YYYYMMDD_HHMM.json
data/splits/replays_YYYYMMDD_HHMM_split.json
```

The manifest must include:

- file hashes;
- player/submission identity;
- deck hash;
- episode date;
- result;
- parser version;
- skipped-file reasons.

The rolling downloader may continue appending raw files, but no running experiment may silently consume new files.

## P2. Create the semantic state/action schema

One shared module must define:

- exact card/entity identity;
- active, bench, hand, discard, prizes, deck counts;
- attached energy, damage, status, evolution stack;
- legal action type;
- acting entity/card;
- target entity;
- resource consumed;
- semantic equivalence key;
- recent public action/event history;
- deck/archetype identity where allowed.

Orderless zones must be represented as sets/multisets, not arbitrary positional vectors.

## P3. Create golden fixtures

Select at least 100 real decisions covering all major action types.

For every fixture, save:

- raw observation;
- legal options;
- semantic action keys;
- card IDs;
- equivalence classes;
- exact root encoding;
- live inference encoding;
- expected forced/non-forced classification.

Required tests:

- trainer encoding equals live encoding exactly;
- all PLAY actions resolve card identity;
- semantically equivalent options collapse correctly;
- strategically distinct actions do not collapse;
- option permutations transform labels correctly;
- no feature is unexpectedly constant/dead;
- teacher queries do not mutate the root state.

No branch work starts until all tests pass.

## P4. Create Teacher API V1

The API must return per decision:

```text
semantic_action_key
mean_value
value_variance
completed_determinizations
top_two_margin
normalized_advantage
soft_policy_target
acceptable_action_set
forced_action_flag
teacher_seed/config hash
```

It must support repeated queries of the same root with controlled seeds.

## P5. Establish data/evaluation partitions

Required held-outs:

- held-out games;
- held-out players/submissions;
- held-out deck/archetype;
- chronological future slice;
- learner-generated on-policy states.

Never split candidate rows from the same decision across train/test.

## P6. Branch creation

From the exact preflight SHA:

```text
exp/planner-teacher-v2
exp/robust-learner-v2
```

Create auditor notes in:

```text
docs/audits/
```

Do not merge either branch into `main` during development.

---

# Branch A — Planner / Teacher V2

## Mission

Build:

1. a more reliable and stronger **offline counterfactual teacher** than the current live N=8 search; and
2. if evidence supports it, an improved live search agent under the real time budget.

The offline teacher is allowed to be slow. Its purpose is to label Branch B and provide a policy-improvement operator.

## Absolute prohibitions

Do not:

- modify Branch B learner architecture;
- train embeddings or run DAgger;
- add a large card-specific heuristic library;
- change deck and methodology in one experiment;
- call configured search depth/sample count “strength” without evidence;
- use one hard argmax when repeated teacher queries are unstable;
- call opponent belief refuted with a search architecture that cannot exploit it;
- merge to `main`.

## A1. Reproduce and instrument Teacher V1

Before improvement attempts:

- reproduce the frozen baseline;
- log actual determinizations completed per action;
- log search time by decision;
- log fallback rate;
- log candidate value variance;
- log top-two margins;
- verify shared hidden-world samples are used across sibling actions where feasible.

Abort if the frozen baseline cannot be reproduced.

## A2. Teacher stability audit

Sample at least 1,000 non-forced decisions from four sources:

- top-player replay states;
- production-search self-play;
- old ranker arena failures;
- multiple decks/archetypes.

Query Teacher V1 at least 16 times per state with different determinization seeds.

Report:

- semantic top-action stability;
- pairwise action-order stability;
- value variance;
- top-two margin;
- acceptable-action-set size;
- instability by action type, deck, and turn;
- stability before/after stronger sampling.

Output soft policies and averaged advantages. Do not force unstable decisions into one hard class.

## A3. Build Teacher V2 candidates

Test one change at a time.

### A3.1 Sampling

Test higher offline determinization counts such as 16, 32, and 64, subject to measured throughput.

### A3.2 Shared worlds

Use the same hidden-world samples for every sibling action within a root decision.

### A3.3 Selective computation

Spend extra computation only when:

- top-two margin is small;
- value variance is high;
- candidate policies disagree;
- the decision is high-impact by downstream outcome spread.

### A3.4 Continuation-policy mixture

Do not repeat a global “aggro versus setup” choice.

Construct an ensemble of:

- current rollout;
- legal stochastic rollout;
- replay-derived continuation policy;
- search/student policy when available;
- opponent-policy mixture conditioned on deck and public history.

Evaluate actions across the mixture and expose uncertainty.

### A3.5 Opponent sensitivity

Separate:

- hidden-card belief;
- opponent action policy.

Reopen belief modelling only in a search horizon where the opponent’s choices materially affect the leaf. Test a mixture of plausible opponent policies rather than pretending to know the exact submitted bot.

### A3.6 Selective depth

Deeper search is a hypothesis, not an assumption.

Apply it only to:

- small-margin decisions;
- tactical KO/survival branches;
- states where the shallow teacher is unstable;
- a bounded critical-decision subset.

## A4. Teacher V2 acceptance

Teacher V2 must outperform Teacher V1 on at least two of:

- lower repeated-query action instability;
- lower counterfactual regret under long rollouts;
- improved terminal-outcome calibration;
- stronger head-to-head play when allowed equal or greater offline compute;
- improved ability to label high-margin/high-criticality decisions consistently.

A larger `N` by itself is not acceptance.

## A5. Live search candidate

Use Teacher V2 findings to build a live planner under the exact competition budget.

Permitted mechanisms:

- adaptive determinizations;
- student/replay policy for ordering;
- uncertainty-triggered extra search;
- selective depth;
- robust continuation mixtures.

Promotion requires:

- equal wall-clock budget;
- seat swaps;
- multiple seeds;
- representative deck panel;
- at least 400 games for final promotion;
- Wilson interval and timeout/error comparison.

## Branch A deliverables

```text
agent/teacher_api_v2.py
agent/search_live_v2.py
tools/audit_teacher_stability.py
tools/query_teacher_v2.py
tools/evaluate_teacher_regret.py
docs/workstreams/PLANNER_TEACHER_V2.md
data/manifests/teacher_v2_*.json
```

Final artifacts:

- frozen Teacher V2;
- frozen live-search candidate;
- machine-readable soft-label/action-advantage dataset;
- complete negative-result table;
- measured query throughput.

---

# Branch B — Robust Learner V2

## Mission

Determine whether a learned model can become useful by:

- representing the full state and legal action;
- learning from its own induced state distribution;
- training on recovery states;
- using meaningful card/action embeddings;
- predicting temporally extended tactical events;
- and guiding search rather than immediately replacing it.

Branch B begins with Teacher API V1. It later performs one final pass with frozen Teacher V2.

## Absolute prohibitions

Do not:

- modify the search teacher;
- use the old compressed feature vector as the sole input;
- train only on winner replays;
- treat exact raw option index as the target;
- preserve labels after strategic state perturbations;
- call offline top-1, Pearson, AUC, or affordance accuracy “success”;
- deploy the student as production before on-policy diagnostics;
- add model complexity before representation/label tests pass;
- merge to `main`.

## B1. Diagnose the old failure before replacing it

### B1.1 Representation ceiling

Take a clean low-entropy subset where repeated teacher queries strongly agree.

Train a deliberately over-capacity model on a small subset.

Required questions:

- Can the current representation nearly memorize stable teacher decisions?
- How many distinct raw states collapse to the same representation?
- How much teacher-label entropy remains after conditioning on the encoded state?

Interpretation:

- poor training fit means representation/label insufficiency;
- high training fit plus poor held-out fit means generalization/data problem;
- stable held-out fit plus live collapse supports distribution shift.

Do not claim DAgger is the cause/fix before this test.

### B1.2 Teacher stability

Consume repeated Teacher API queries.

Use:

- averaged advantages;
- soft action distributions;
- acceptable-action sets;
- confidence weighting.

Exclude or down-weight near-ties.

### B1.3 Direct on-policy shift measurement

Run the old student for at least 100 games.

Query the teacher on every visited state.

Report teacher agreement/regret:

- before first student disagreement;
- after first disagreement;
- by turn;
- by distance from training data;
- by teacher margin;
- by action type;
- on catastrophic/high-regret errors.

This determines whether DAgger is actually targeting the observed failure.

## B2. Build full entity/action/history representation

### State encoder

Represent:

- active Pokémon;
- each bench entity;
- hand multiset;
- discard multiset;
- prizes/deck counts;
- attached energy;
- damage/status;
- evolution stacks;
- public Stadium;
- once-per-turn/resource-use flags;
- recent public action/event history.

Use permutation-invariant set/entity processing for unordered zones.

### Action encoder

Represent:

- action type;
- acting card/entity;
- target;
- action-specific numeric parameters;
- resource consumed;
- decoded card effects;
- semantic equivalence class.

### Card representation

Compare:

1. one-hot/card ID;
2. designed numeric/effect features;
3. learned embedding;
4. designed features + learned residual embedding.

The learned embedding must be evaluated on held-out decks/cards. Same-deck memorization is not evidence of embedding value.

## B3. Targets

### Primary targets

- within-decision action advantage;
- teacher soft policy;
- teacher uncertainty;
- acceptable-action set.

### Auxiliary outcome target

Terminal win/loss may be used as an auxiliary critic, never as the sole policy label. Run a no-outcome ablation.

### Successor-affordance targets

Predict expected future accumulation over a finite continuation horizon:

- legal-option expansion;
- draw/tutor availability;
- attack unlock probability;
- KO/prize events;
- survival through opponent reply;
- energy/resource continuity;
- irreversible resource expenditure;
- deckout/resource exhaustion;
- tactical sequence completion.

Immediate simulator-computable events should be exact inputs/labels. Learn the future accumulation, not a noisy approximation to already-known facts.

Keep an unconstrained latent residual path. Do not force the entire policy through the human concept vocabulary.

## B4. Safe augmentation and recovery states

### Label-preserving transformations

Allowed only when verified by rules/golden tests:

- legal-option permutation with transformed targets;
- identical-card-copy permutation;
- canonical player perspective;
- truly equivalent target permutations.

### Teacher-relabelled perturbations

Generate legal reachable states through the engine:

- execute one plausible non-teacher action;
- omit or reorder a legal setup action;
- sample a stochastic continuation;
- vary a plausible hidden world;
- collect states immediately after high-regret student actions;
- collect recovery states from student self-play.

Every strategic perturbation is relabelled by the teacher.

## B5. Bounded DAgger/DART loop

First benchmark Teacher API throughput.

Then run no more than three rounds without a new decision.

### Round 0

Train on:

- frozen clean replay states;
- stable Teacher V1 labels;
- safe invariance augmentation.

### Round 1

- let Student V2 generate games;
- collect all visited decisions;
- query the teacher;
- add confidence-weighted labels;
- retrain.

### Round 2

- repeat on-policy collection;
- add controlled legal recovery perturbations;
- relabel;
- retrain.

### Round 3

Run only if Round 2 improves on-policy regret or arena performance.

After each round report:

- semantic teacher agreement;
- teacher-relative regret;
- top-k recall;
- high-margin and high-criticality accuracy;
- pre/post-first-error performance;
- arena win rate;
- timeouts/errors;
- held-out-player/deck performance.

Stop after three rounds if on-policy regret and arena play do not improve.

## B6. Required ablations

Compare:

- full model;
- no learned embedding;
- one-hot ID instead of embedding;
- no public history;
- no successor-affordance heads;
- no DAgger data;
- no recovery perturbations;
- old compressed representation;
- hard labels versus soft teacher targets;
- no terminal-outcome auxiliary.

## B7. Branch B acceptance

Branch B succeeds if at least one is demonstrated:

- on-policy teacher regret declines across DAgger rounds;
- held-out-deck action ranking improves because of the embedding;
- successor-affordance supervision improves action regret;
- student top-k recall is high enough to guide search safely;
- student-guided equal-budget search beats unguided search.

Offline auxiliary prediction alone is not success.

## Branch B deliverables

```text
agent/student_v2.py
agent/student_prior_v2.py
agent/entity_encoder_v2.py
tools/audit_representation_ceiling.py
tools/measure_on_policy_shift.py
tools/dagger_collect.py
tools/dart_recovery_states.py
tools/train_student_v2.py
docs/workstreams/ROBUST_LEARNER_V2.md
data/manifests/student_v2_*.json
```

Final artifacts:

- frozen Student V2;
- policy prior;
- action advantages;
- uncertainty;
- successor-affordance vector;
- round-by-round dataset manifests/results;
- full ablation table.

---

# Third model — adversarial auditor

The third model is read-only by default.

## Mission

Prevent another false conclusion caused by a no-op, dead feature, broken equivalence key, leakage, or mismatched evaluation.

## Auditor checks

Before every result is accepted, verify:

- branch started from the correct SHA;
- data manifest is immutable;
- no train/test decision leakage;
- player/deck chronology split is correct;
- action labels are semantically canonical;
- features are non-dead and populated;
- training and live encodings are identical;
- teacher queries are repeated/stable;
- arms differ in the intended variable;
- wall-clock budgets match;
- actual determinizations completed are logged;
- sample-size and CI gates were declared before the run;
- reported result matches raw logs;
- causal language does not exceed evidence.

The auditor may add tests and review notes but may not silently redesign either branch.

---

# File ownership

## Shared, frozen after split

```text
agent/state_action_schema_v2.py
agent/teacher_api_v1.py
tests/golden_state_action_fixtures/
docs/workstreams/SPLIT_BASE_V2.md
data/manifests/
```

Changes require auditor approval and cherry-pick into both branches.

## Branch A owns

```text
agent/search_live_v2.py
agent/teacher_api_v2.py
tools/query_teacher_v2.py
tools/audit_teacher_stability.py
tools/evaluate_teacher_regret.py
docs/workstreams/PLANNER_TEACHER_V2.md
```

## Branch B owns

```text
agent/entity_encoder_v2.py
agent/student_v2.py
agent/student_prior_v2.py
tools/audit_representation_ceiling.py
tools/measure_on_policy_shift.py
tools/dagger_collect.py
tools/dart_recovery_states.py
tools/train_student_v2.py
docs/workstreams/ROBUST_LEARNER_V2.md
```

Neither branch edits production `agent/main.py` or submission packaging. Those change only on the integration branch.

---

# Deck and replay policy

## Production deck

- Freeze DENPA92 for branch comparisons.
- Finish existing deck experiments separately.
- A new deck replaces the baseline only through a separate promotion experiment.
- Branch B must still evaluate representation transfer on held-out alternative decks.

## Replay collection

Use both:

- scheduled append-only fetches;
- on-demand fetches after notable leaderboard changes.

But experiments consume only immutable snapshots. Never train from a live directory whose contents change during a run.

Recommended split strategy:

- train: earlier episodes/players/decks;
- validation: held-out games and players;
- test: chronological later slice plus held-out deck/archetype;
- separate on-policy learner-state test.

---

# Integration branch

Create only when Branch A and Branch B freeze their artifacts:

```text
exp/student-guided-search-v1
```

## Systems to compare

1. frozen production baseline;
2. Branch A strongest standalone live search;
3. Student V2 standalone — diagnostic only;
4. search with student candidate ordering;
5. search with student continuation policy;
6. search with student uncertainty-based budget allocation;
7. search with student top-k proposal but no irreversible pruning;
8. optional search with successor-affordance-informed leaf features.

## Integration safety rule

Initially:

> Student proposes and prioritizes. Search evaluates and decides.

No permanent pruning until the student’s top-k recall of Teacher V2’s acceptable action set is demonstrably high.

## Final success gate

The integrated system must beat the strongest standalone search:

- same deck;
- same opponent panel;
- same wall-clock budget;
- seat swaps;
- multiple seeds;
- at least 400 final games;
- no increased timeout/error rate;
- CI and exact artifact hashes recorded.

This is the concrete route by which embeddings/DAgger/affordances can provide gameplay advantage beyond merely cloning search.

---

# Three-model allocation

| Model | Assignment |
|---|---|
| Model 1 | Branch A lead: Planner / Teacher V2 |
| Model 2 | Branch B lead: Robust Learner V2 |
| Model 3 | Shared-preflight owner, adversarial auditor, and later integration lead |

Do not place two autonomous implementers on the same branch. That increases schema drift and merge risk. The third model should challenge and verify both branches, then own integration.

---

# Immediate execution order

1. Finish and record any already-running deck A/B, but do not let it block the split.
2. Freeze DENPA92 and current `agent_search` as the production baseline.
3. Snapshot the replay corpus and create immutable manifests/splits.
4. Implement semantic schema, Teacher API V1, and golden fixtures.
5. Pass every preflight test.
6. Commit `SPLIT_BASE_V2`.
7. Create both branches from that exact SHA.
8. Start Branch A teacher-stability audit.
9. Start Branch B representation-ceiling and on-policy-shift diagnostics.
10. Auditor reviews both before either begins large training/search sweeps.
11. Freeze Teacher V2.
12. Branch B performs one final Teacher V2 relabel/retrain pass.
13. Integrate as student-guided search.
14. Promote only on equal-budget arena evidence.

---

# Research basis

- Ross, Gordon, and Bagnell (2011), **DAgger / A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**: https://proceedings.mlr.press/v15/ross11a.html
- Laskey et al. (2017), **DART: Noise Injection for Robust Imitation Learning**: https://arxiv.org/abs/1703.09327
- Anthony, Tian, and Barber (2017), **Thinking Fast and Slow with Deep Learning and Tree Search / Expert Iteration**: https://arxiv.org/abs/1705.08439
- Kitchen and Benedetti (2018), **ExIt-OOS: Learning from Planning in Imperfect-Information Games**: https://arxiv.org/abs/1808.10120
- Brown et al. (2020), **ReBeL: Combining Deep Reinforcement Learning and Search for Imperfect-Information Games**: https://arxiv.org/abs/2007.13544
- Schmid et al. (2021/2023), **Student of Games**: https://arxiv.org/abs/2112.03178
- Barreto et al. (2016/2017), **Successor Features for Transfer in Reinforcement Learning**: https://arxiv.org/abs/1606.05312
- Project research state: https://github.com/LacrimaeAware/kaggle-fun/tree/main/pokemon-tcg-ai-battle
- Finance-research methodological precedent: https://github.com/LacrimaeAware/finance-research
- Structured-transform caution on interpretable factors versus discriminative performance: https://github.com/LacrimaeAware/structured-transform-discovery
