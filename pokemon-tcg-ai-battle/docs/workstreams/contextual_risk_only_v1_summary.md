# A-Label Risk-Only Contextual Model

Status: offline only. No live screen, no promotion, and no `agent_search` modification.

Decision: **C** - risk-only fails; request more/different labels from Model A

The A-label risk-only model did not improve the required offline safety metrics over agent_search.

## Ingest And Alignment

- Decisions loaded: 50
- Options loaded: 451
- Feature rows generated: 451
- Option alignment: 451 / 451
- Missing fields: []
- Training-ready: true

## Held-Out Test Safety

| model                          | mean regret | p95    | p99      | hi-regret | acceptable | top1  |
| ------------------------------ | ----------- | ------ | -------- | --------- | ---------- | ----- |
| agent_search                   | 1425.57     | 20.82  | 24699.32 | 1         | 1.0        | 0.409 |
| old_ranker                     | 1503.35     | 799.78 | 24869.21 | 1         | 0.727      | 0.455 |
| previous_full_teacher_v2       | 1497.68     | 799.18 | 24869.21 | 1         | 0.727      | 0.455 |
| previous_b_bootstrap_risk_only | 1425.57     | 20.82  | 24699.32 | 1         | 1.0        | 0.409 |
| new_a_label_risk_only          | 1430.51     | 104.12 | 24717.7  | 1         | 0.955      | 0.364 |
| no_effects_risk_only           | 1425.57     | 20.82  | 24699.32 | 1         | 1.0        | 0.409 |
| no_deltas_risk_only            | 1430.51     | 104.12 | 24717.7  | 1         | 0.955      | 0.364 |

## Risk Detection

- High-regret recall: 0.833
- Unacceptable-action recall: 0.692
- False-positive risk rate: 0.200

## Conservative Integration Proposal

No `agent_search_risk` integration is recommended from this run. The classifier has detection signal, but the gated intervention did not improve selected-action safety over plain `agent_search`.

## Label Request

Decision C is packaged as `data/manifests/teacher_v2_risk_label_request_for_A.json`. The request asks for targeted residual/risk enrichment around search-selected high-regret misses and safe search-choice false positives, not a generic larger batch.

