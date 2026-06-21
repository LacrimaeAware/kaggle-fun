# Residual/Risk Contextual Prototype

Status: offline only. No live screen and no `agent_search` modification.

Decision: **A** - offline improved enough to justify a small agent_search_residual screen

risk_only improved mean regret, p95/p99 regret, high-regret count, and acceptable-action recall against agent_search on the held-out bootstrap test without losing to old-ranker/option-0 safety metrics. Treat this as screen-eligible only with the caveat that these are B-bootstrap labels, not Model A's dedicated high-compute residual/risk artifact.

Important caveat: labels are a Branch B bootstrap from existing Teacher V2 artifacts plus local deployed-search estimates, not Model A's new high-compute residual/risk artifact.

## Test Metrics

| model                    | mean regret | p95    | p99    | hi-regret | acceptable | top1 |
| ------------------------ | ----------- | ------ | ------ | --------- | ---------- | ---- |
| agent_search             | 15.34       | 84.0   | 88.72  | 0         | 0.85       | 0.65 |
| old_ranker               | 70.32       | 258.49 | 701.0  | 3         | 0.8        | 0.4  |
| option0                  | 86.3        | 251.07 | 699.51 | 4         | 0.65       | 0.45 |
| previous_full_teacher_v2 | 81.05       | 294.29 | 708.16 | 4         | 0.7        | 0.3  |
| previous_no_effects      | 31.03       | 143.84 | 211.72 | 2         | 0.7        | 0.5  |
| residual_only            | 43.62       | 143.84 | 211.72 | 2         | 0.55       | 0.25 |
| risk_only                | 8.12        | 23.18  | 71.59  | 0         | 0.9        | 0.55 |
| residual_plus_risk       | 33.87       | 98.3   | 202.61 | 1         | 0.65       | 0.4  |

Risk detection: catastrophic-risk recall 0.407, false-positive rate 0.123.
