# Robust Learner V2

Branch: `exp/robust-learner-v2`

Base: `2f29e9378d07bd98f299deb656e61ea2b8bcaf25` (`SPLIT_BASE_V2`)

## Operating Rules

- `docs/workstreams/BRANCH_PLAN.md` is authoritative.
- Shared split-base files are frozen: `agent/state_action_schema_v2.py`, `agent/teacher_api_v1.py`,
  `tests/golden_state_action_fixtures/`, `data/manifests/`, and `data/splits/`.
- Branch B owns only the learner/diagnostic files named by the plan.
- DENPA92 remains the branch-comparison deck unless a separate promotion gate changes it.
- Experiments consume immutable snapshots and fixed splits, not the live replay corpus.
- Terminal win/loss is auxiliary only, never the primary action target.

## Start State

The Branch B worktree was created from `SPLIT_BASE_V2`. The split-base golden tests pass:

```text
SPLIT_BASE_V2 golden tests | 130 fixtures | schema split_base_v2.0
ALL PASS  (8/8)
```

## B1.1 Representation-Ceiling Audit

Initial tool: `tools/audit_representation_ceiling.py`.

Purpose:

- query Teacher API V1 repeatedly on frozen-snapshot replay decisions;
- keep only low-entropy/stable teacher labels;
- measure how often the compressed root encoding collapses distinct raw states;
- estimate teacher-label entropy after conditioning on that encoding;
- report an exact memorizer ceiling for increasingly rich representations.

This is a diagnostic, not a student architecture. It is meant to answer whether representation/label
insufficiency appears before we spend compute on DAgger or larger models.

### B1.1 Train/Validation/Test Samples

Local setup: the Branch B worktree has an ignored junction from `data/external` to the main checkout's
gitignored `data/external`, because Teacher V1 needs the engine wrapper there. The audit still verifies
each replay file hash against `data/manifests/replays_20260618.json`.

Command shape:

```text
python tools/audit_representation_ceiling.py --replay-dir C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\pokemon-tcg-ai-battle\data\external\replays --partition <train|val|test> --max-decisions 48 --repeats 3 --n-determ 4 --time-budget 5.0 --min-agreement 1.0 --output docs\workstreams\robust_learner_v2_b1_1_<partition>.json --include-decisions
```

Artifacts:

- `docs/workstreams/robust_learner_v2_b1_1_train.json`
- `docs/workstreams/robust_learner_v2_b1_1_val.json`
- `docs/workstreams/robust_learner_v2_b1_1_test.json`

Summary:

| Partition | Scanned | Labelled | Stable | Unstable | Not applicable | Agreement histogram | Root entropy weighted | Root+action expected top-1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| train | 152 | 151 | 48 | 103 | 1 | 0.333:35, 0.667:68, 1.000:48 | 0.0991 | 0.958 |
| val | 152 | 152 | 48 | 104 | 0 | 0.333:38, 0.667:66, 1.000:48 | 0.0752 | 0.979 |
| test | 139 | 138 | 48 | 90 | 1 | 0.333:37, 0.667:53, 1.000:48 | 0.0574 | 0.979 |

Memorizer ceiling:

```text
train root_only 0.530 -> root_plus_type 0.747 -> root_plus_action 0.958
val   root_only 0.586 -> root_plus_type 0.849 -> root_plus_action 0.979
test  root_only 0.490 -> root_plus_type 0.750 -> root_plus_action 0.979
```

Interpretation: "unstable" here means repeated Teacher V1 queries returned different top semantic actions
for the same root decision. That is expected from the split-base reproducibility note: determinization
sampling, native engine rollout RNG, and near-tie actions all contribute. It does not imply the split base
must be remade. It does imply B1.2 teacher-stability/confidence weighting is not optional before training.

The small stable subset does not show compressed-root collision as the primary blocker. Action identity is
the clear discriminative ingredient: root-only memorization is weak, type helps, and root+action nearly
memorizes stable labels. Next: run the B1.2 teacher-stability audit over the unstable cases and confidence
signals instead of treating top-1 labels as hard truth.

## B1.2 Teacher-Stability Readout

Same tool, expanded output schema (`audit_version=branch_b_b1_2.0`) records all labelled decisions, not
only accepted stable decisions: action type, agreement, label entropy, margin bin, value variance,
completed determinizations, acceptable-action count, and soft-policy mass on the chosen action.

Artifacts:

- `docs/workstreams/robust_learner_v2_b1_2_train.json`
- `docs/workstreams/robust_learner_v2_b1_2_val.json`
- `docs/workstreams/robust_learner_v2_b1_2_test.json`

Summary:

| Partition | Scanned | Labelled | Stable | Unstable | Not applicable | Stable rate among labelled | Root+action expected top-1 on stable |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 227 | 226 | 64 | 162 | 1 | 0.283 | 0.984 |
| val | 253 | 253 | 64 | 189 | 0 | 0.253 | 1.000 |
| test | 171 | 170 | 64 | 106 | 1 | 0.376 | 0.969 |

Dominant action-type stability:

| Partition | SELECT_CARD `3` | PLAY `7` | ATTACH `8` |
|---|---:|---:|---:|
| train | 31/126 stable = 0.246 | 10/41 = 0.244 | 6/24 = 0.250 |
| val | 40/127 = 0.315 | 14/49 = 0.286 | 2/30 = 0.067 |
| test | 30/69 = 0.435 | 19/21 = 0.905 | 5/57 = 0.088 |

Interpretation for Branch B labels:

- Stable/high-agreement decisions are suitable for the representation-ceiling and overfit tests.
- Unstable decisions should not become hard argmax labels. Use soft policy, normalized advantage,
  acceptable-action sets, and confidence weights.
- Margin alone is not a sufficient filter in these samples; instability appears across several margin bins.
- The finding supports B1.2/B1.3 label handling. It does not ask Branch A to remake the shared snapshot/schema
  and does not prove covariate shift.

## B1.3 On-Policy Shift Tooling

Initial tool: `tools/measure_on_policy_shift.py`.

Purpose:

- run the old student (`rank` or `rankh`) in real `cabt` games;
- trace the student's actually visited single-pick decisions;
- query Teacher API V1 repeatedly on those on-policy states;
- report acceptable-set agreement, hard top-1 agreement, teacher-relative regret, before/after first
  unacceptable student action, by turn, action type, teacher margin, and rough nearest frozen-train root
  distance;
- separate stable, unstable, and not-applicable teacher calls.

Important implementation note: Kaggle accepted a plain function closure tracer but not the first class-style
callable wrapper attempt; the zero-decision smoke artifact was deleted. The current tracer was verified with
`first` vs `first`, then with `rank` vs `heuristic`.

100-game artifact:

- `docs/workstreams/robust_learner_v2_b1_3_rank_100g.json`

Command:

```text
python tools/measure_on_policy_shift.py --student rank --opponent heuristic --games 100 --n-determ 3 --time-budget 4.0 --teacher-repeats 3 --min-teacher-agreement 1.0 --min-acceptable-rate 0.5 --reference-max-files 160 --reference-max-decisions 3000 --output docs\workstreams\robust_learner_v2_b1_3_rank_100g.json
```

Summary:

```text
rank vs heuristic, 100 games: student 18-82, 0 draws/errors
student/ranker visited decisions: 2980
teacher-applicable decisions: 2939
not-applicable teacher decisions: 41
not-applicable teacher calls: 123
acceptable-set agreement: 0.688
hard top-1 agreement: 0.439
mean regret: 30694.7
p90 regret: 1110.0
high-regret decisions >=1000: 335
teacher stable/unstable among applicable: 1950 / 989
```

Pre/post first unacceptable action:

| Phase | N | Acceptable agreement | Hard top-1 agreement | Mean regret | P90 regret | High-regret |
|---|---:|---:|---:|---:|---:|---:|
| before | 633 | 0.965 | 0.626 | 1187.8 | 129.2 | 5 |
| first unacceptable | 97 | 0.103 | 0.010 | 53275.6 | 111199.0 | 21 |
| after | 2205 | 0.634 | 0.405 | 38202.4 | 1240.7 | 308 |

Teacher-stability slice:

| Teacher stability | N | Acceptable agreement | Hard top-1 agreement | Mean regret | P90 regret | High-regret |
|---|---:|---:|---:|---:|---:|---:|
| stable | 1950 | 0.690 | 0.523 | 30426.8 | 1134.4 | 239 |
| unstable | 989 | 0.683 | 0.274 | 31222.8 | 954.9 | 96 |

Dominant action-type slices:

| Student action type | N | Acceptable agreement | Hard top-1 agreement | Mean regret | High-regret |
|---|---:|---:|---:|---:|---:|
| SELECT_CARD `3` | 748 | 0.893 | 0.777 | 1315.5 | 15 |
| PLAY `7` | 565 | 0.781 | 0.431 | 16295.0 | 30 |
| EVOLVE `9` | 374 | 0.643 | 0.314 | 20371.5 | 38 |
| END `14` | 325 | 0.266 | 0.078 | 141778.9 | 130 |
| ATTACH `8` | 236 | 0.448 | 0.188 | 67519.0 | 48 |
| RETREAT `12` | 190 | 0.549 | 0.219 | 5038.0 | 36 |
| ABILITY `10` | 154 | 0.416 | 0.193 | 30442.9 | 20 |

Distance-to-reference slice:

| Nearest frozen-train L1 bin | N | Acceptable agreement | Hard top-1 agreement | Mean regret | High-regret |
|---|---:|---:|---:|---:|---:|
| <=0 | 50 | 1.000 | 1.000 | 170.0 | 2 |
| <=5 | 26 | 0.987 | 0.718 | 1.6 | 0 |
| <=20 | 493 | 0.813 | 0.481 | 10740.7 | 23 |
| <=100 | 2356 | 0.653 | 0.416 | 36037.4 | 309 |
| >100 | 14 | 0.381 | 0.310 | 263.3 | 1 |

Highest-regret examples are in the JSON artifact. The top failures are mostly after the first unacceptable
action and include stable-teacher cases with acceptable-rate 0.0, so this is not merely a teacher-noise
artifact.

Interpretation: this directly measures the old ranker on states it actually visits. The pattern supports
the Branch B diagnosis that compounding on-policy errors are a real problem: before the first unacceptable
action, acceptable agreement is high and regret is low; the first unacceptable action and subsequent states
contain most catastrophic regret. This does not authorize Student V2, DAgger, or integration yet; it closes
the B1 diagnostic evidence needed before deciding how to proceed.
