# Contextual Action Ranker V1

Branch: `exp/robust-learner-v2`

Status: implemented, trained, integrated as a search guide, and screened. No deck change, no Student V2 promotion, no main merge.

## Implementation

Files:

- `agent/contextual_ranker.py`
- `agent/contextual_ranker_v1.json`
- `tools/train_contextual_action_ranker.py`
- `agent/search.py`
- `agent/main.py`
- `agent/cabt_arena.py`

The model scores every legal sibling action from the same root decision. It consumes:

- root state features;
- option/action descriptor;
- acting-card embedding;
- decoded card effects;
- target/entity features;
- state x effect interactions;
- immediate one-step option deltas from `search.option_deltas`;
- short public history scalars.

Targets are grouped by semantic sibling-action class, not raw option index. Training uses Teacher V1 soft policy, action advantages, acceptable-action sets, teacher stability/confidence, replay winner choices as a small noisy auxiliary, and recollected Round 1/Round 2 recovery states.

Deployment is conservative: `agent_search_ctx` uses the contextual model for candidate ordering and exact-value tie breaking inside search. Search remains the final authority. Contextual scoring time is debited from the normal search budget before calling the forward-model search.

## Artifacts

- Dataset: `docs/workstreams/contextual_action_ranker_v1_dataset.json`
- Train/eval: `docs/workstreams/contextual_action_ranker_v1_train_eval.json`
- Arena screen: `docs/workstreams/contextual_action_ranker_v1_arena_search_ctx_vs_search_20g.json`
- On-policy guided search: `docs/workstreams/contextual_action_ranker_v1_on_policy_search_ctx_4g.json`
- On-policy standalone search control: `docs/workstreams/contextual_action_ranker_v1_on_policy_search_4g.json`

## Dataset

| Source | Decisions |
|---|---:|
| replay train | 48 |
| replay val | 16 |
| replay test | 16 |
| recovery Round 1 | 40 |
| recovery Round 2 | 40 |
| total | 160 |

Teacher labels: 91 stable, 69 unstable. The dataset includes 18 high-regret recollected recovery decisions, 5 players, and 2 deck hashes. Prior B1.3/Round 1/Round 2 on-policy JSONs did not retain raw observations, so recovery states were recollected from the Round 1 and Round 2 deployed ranker artifacts rather than rehydrated from those JSON reports.

## Offline Replay Test

Held-out replay test set, `n=16`:

| Model | Top-1 | Top-2 | Pairwise | MRR | NDCG | Acceptable | Mean Regret | P95 | High-Regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full contextual | 0.375 | 0.625 | 0.624 | 0.578 | 0.888 | 0.563 | 139.7 | 720.6 | 0 |
| no decoded effects | 0.250 | 0.563 | 0.636 | 0.514 | 0.926 | 0.688 | 86.6 | 285.0 | 0 |
| no card embedding | 0.188 | 0.500 | 0.535 | 0.442 | 0.864 | 0.625 | 142.5 | 720.6 | 0 |
| no option deltas | 0.375 | 0.625 | 0.550 | 0.564 | 0.882 | 0.563 | 145.3 | 720.6 | 0 |
| old ranker baseline | 0.438 | 0.688 | 0.847 | 0.638 | 0.937 | 0.875 | 86.8 | 390.6 | 0 |
| option-0 baseline | 0.375 | 0.563 | 0.491 | 0.545 | 0.902 | 0.625 | 108.7 | 358.8 | 0 |

Offline replay fidelity is mixed and does not beat the old ranker baseline on this small test slice. The no-effects ablation had lower regret than the full model, which is a calibration warning for the decoded-effect/interactions block rather than evidence to drop the intended input family.

## On-Policy Diagnostics

Small diagnostic, first 120 teacher-labelled visited decisions from 4 games vs heuristic:

| Agent | Trace | Applicable | Acceptable | Hard Top-1 | Mean Regret | P90 | P95 | High-Regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| search_ctx | 3-1 | 117 | 0.953 | 0.761 | 31.7 | 19.8 | 82.0 | 2 |
| search | 4-0 | 119 | 0.941 | 0.798 | 26.9 | 84.6 | 129.4 | 0 |

Guided search has slightly better acceptable-set agreement and lower p90/p95 regret in this tiny slice, but worse mean regret and two high-regret decisions. This does not prove an on-policy improvement over standalone search.

## Arena Screen

20-game same-deck screen, seats swapped:

| A | B | Result | Win Rate | Errors | Time |
|---|---|---:|---:|---:|---:|
| search_ctx | search | 11-9 | 0.550 | 0 | 515.2s |

This is directionally positive under the intended final-gate comparison, but it is only a cheap screen. It is not strong enough for promotion by itself.

## Decision

Contextual Action Ranker V1 is now a real integrated system, not another old-objective DAgger round. It trains on contextual sibling-action inputs, uses soft/advantage/acceptable/confidence labels, includes recovery states, runs ablations, and guides search without taking authority away from search.

Recommendation: continue, but do not promote. The next gated step should be a modest confirmation run and one calibration pass focused on why the full model underperforms the no-effects ablation and old ranker on offline replay fidelity. Promotion should require a stronger `search_ctx` vs `search` result under the same wall-clock policy plus no worsening of on-policy regret/high-regret tails.
