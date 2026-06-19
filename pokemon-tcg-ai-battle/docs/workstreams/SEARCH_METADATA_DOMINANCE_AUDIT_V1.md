# Search Metadata Dominance Audit V1

## Verdict

**B. LIVE N8 WARNING SIGNAL; EXTRA-SEARCH MECHANISM STILL UNPROVEN**

Strict live N=8 metadata retains strong high-regret signal and an oracle stronger-search upper bound improves, but the deployable live-next-unflagged proxy gets worse.

## Dataset

- Input: `data\manifests\continuous_terrain_v1.jsonl`
- Decisions/options/games: 702 / 5743 / 160
- Test decisions/options: 90 / 763
- Arena/live screen: not run
- `agent_search`: not modified

## High-Regret Prediction

| feature set | AP | AUROC | recall@FPR10 | positives/test rows |
|---|---:|---:|---:|---:|
| `leakage_control_stronger` | 1.000 | 1.000 | 1.000 | 33/763 |
| `live_n8_strict_postchoice` | 0.961 | 0.998 | 1.000 | 33/763 |
| `live_n8_strict_prechoice` | 0.961 | 0.998 | 1.000 | 33/763 |
| `live_n8_uncertainty_only` | 0.104 | 0.471 | 0.273 | 33/763 |
| `live_n8_values_only` | 0.964 | 0.999 | 1.000 | 33/763 |
| `previous_R1_suspect` | 0.747 | 0.990 | 1.000 | 33/763 |

## Selected-Action Trigger Proxy

Thresholds are calibrated on train selected rows for <=10% safe-action FPR, then evaluated on held-out test decisions.

| feature set | trigger rate | caught bad | missed bad | false safe triggers | before high/p95 regret | oracle extra-search high/p95 | live-next-unflagged high/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `live_n8_strict_postchoice` | 0.100 | 2 | 0 | 7 | 2/16.839 | 1/7.785 | 3/16.839 |
| `live_n8_strict_prechoice` | 0.100 | 2 | 0 | 7 | 2/16.839 | 1/7.785 | 3/16.839 |
| `previous_R1_suspect` | 0.122 | 2 | 0 | 9 | 2/16.839 | 1/7.785 | 2/16.839 |

## Integrity Findings

- Previous R1 did not use direct stronger values, but it was not strict live-only: it included dataset criticality metadata and ambiguous value-SE provenance.
- Strict live N=8 values still predict high-regret strongly, but high-regret is a residual-style label built from live-vs-stronger values, so this is target-coupled evidence.
- Strict live-only N=8 metadata must exclude stronger values, deltas, hand advantages, high-regret/unacceptable probabilities, criticality metadata, and teacher policy targets.
- The semantic/R4 path leaks `policy_prob` from the stronger soft policy through `action_scalars`, so the previous R3/R4 conclusions are diagnostic only.
- High-regret and unacceptable binary labels are identical on 5743/5743 rows.
- I did not find a score-key bug in the unacceptable table; the oddness is label identity/near-identity plus fusion leakage, not a table indexing error.

## Recommendation

Next, build a tiny offline extra-search simulator for only the triggered test states, or ask Model A for higher-k labels on the triggered false positives/false negatives. Do not promote a live risk rule from this audit alone.
