# Branch A / A3 -- search quality, selective computation, Teacher V2 labels

Branch `exp/planner-teacher-v2`. `agent_search` remains the live baseline; no merge/promotion. This is the
A3 mission per the final coordination plan: make search less noisy and produce stronger Teacher V2 labels
for Model B.

## A1 -- recorded conclusion

- `agent_search` is the best live baseline; Branch A produced no agent that beats it.
- Tactic Miner V1 (ontology/miner/`mined_tactics_v1.json`) is useful infrastructure + a labeled-tactic
  source for B; `agent_search_prior` (0.433, 13-17) is NOT a submission candidate.
- Hard tactical floors and simple mined-prior variants are stopped unless a new hypothesis is approved.
- Status: arena-evaluated; refuted/inconclusive as live wins.

## A2 -- search confounder audit (`tools/audit_search_confounders.py`, `data/manifests/search_confounder_audit.json`)

Instrumented the real determinization loop on 40 self-play (deployed, DENPA92) and 37 replay (offline-label,
deciding player's deck) decisions, at the 0.6s live budget and an 8s offline budget.

| confounder | status | risk | fix / parked | effect on Teacher V2 |
|---|---|---|---|---|
| fake Water Energy padding (id 3) | **0% of decisions** (both sets) | low | parked (doesn't trigger) | none, unless labeling cross-deck states with the our-deck prior; then use the actual player deck |
| determinizations reached @0.6s | self-play **7.8/8** (10% budget-limited); replay 8.0/8 | medium (live) | offline @8s -> **8.0/8** | label gen must use a large budget |
| per-option coverage @0.6s | self-play **0.983 / 90% full**; replay 100% | medium (live) | offline @8s -> **100% full** | offline labels fully cover every sibling |
| shared hidden worlds across siblings | **confirmed** (1 search_begin/determ, all options stepped from it) | low | already correct | paired sibling comparison is valid |
| public-zone leakage | strip removes active/bench/discard/energy/tools/pre-evo; no pool anomalies | low | already correct | hidden pool is representative |
| rollout / engine RNG variance | **~50% of decisions unstable** (A2), engine-rollout-dominated | **HIGH** | reduce via higher-N (more samples averaged) | the main lever: high-N offline labels are lower-noise |
| teacher label stability | 50% stable (A2) | high | record stability per label | B down-weights unstable labels |
| paired/seeded comparison | determinization Python-seeded; engine RNG not seedable | medium | seed determinization; treat engine RNG as MC noise | labels are MC estimates; report variance |

Net: determinization is cleaner than feared (no padding, shared worlds, full coverage at the offline
budget). The dominant confounder is engine-rollout noise, which more determinizations average down. So the
Teacher V2 lever is **selective high-N offline search on high-criticality decisions**, plus a signal
stronger than the hand evaluator alone.

## A3/A4 -- plan (building next)

1. **Criticality scorer** (cheap, from the obs): immediate KO/lethal available, prize swing, KO-back
   exposure, resource-critical, large action spread / top-two margin, low teacher stability, deep-vs-shallow
   disagreement. Spend extra compute only where criticality is high (not on every near-tie -- near-ties are
   often low-impact).
2. **Stronger evaluation on triggered decisions:** high-N (e.g. 32) determinizations at the offline budget
   (lower-noise advantages), plus a **terminal-outcome auxiliary** (longer rollouts to a decided result per
   option) so the target is more than the hand-eval leaf alone. Optional selective 2-ply on tactical
   KO/survival branches.
3. **A4 Teacher V2 label artifact** per selected decision: legal sibling list, soft policy, action
   advantage/regret, criticality score, stability/uncertainty, determinization count, action spread,
   whether deeper search changed the choice, hand-vs-outcome components, and the metadata B needs to align
   siblings + option deltas (the B5 request).
4. **A5:** only if a live selective-search variant emerges, screen it vs `agent_search` at equal wall-clock.

## A4 -- Teacher V2 implementation + label artifact (built)

`agent/teacher_api_v2.py` (criticality scorer + `query_v2`) and `tools/query_teacher_v2.py` (label gen).
`query_v2` gates on criticality, then runs high-N(32) low-noise hand advantage (Teacher V1 machinery) plus a
terminal-outcome auxiliary (K full playouts/option -> per-option win-rate). Status: implemented, validated on
a pilot; NOT a live agent (label generator only).

Pilot (`data/manifests/teacher_v2_labels_pilot.jsonl`): 8 high-criticality decisions, ~1-2 s each at k=4.

**Finding:** the terminal-outcome top action disagreed with the hand-eval argmax on **6/8** labeled
decisions -- suggestive that the outcome signal adds information beyond the hand leaf (the required
stronger-than-hand signal). **Caveat (held):** at k=4 the outcome win-rate is high-variance, so this does
not yet separate "outcome carries information" from "outcome is noisy". Confirming it needs higher-k outcome
estimates and/or a regret check; not claimed as established.

## B5 handoff -- what Model B consumes

Artifact: `data/manifests/teacher_v2_labels_pilot.jsonl`. Per decision:
- `criticality` (score + components), `soft_policy_target`, `acceptable_action_set`, `top_two_margin`,
  `action_spread`, `hand_argmax_eq_class`, `outcome_argmax_option`, `hand_outcome_agree`, `forced_action_flag`;
- `options[]` each with `index`, `semantic_action_key`, `eq_class`, `hand_mean_value`,
  `hand_value_variance`, `hand_norm_advantage`, `completed_determinizations`, `outcome_winrate`,
  `outcome_playouts`; plus `source {file, step}`.

Alignment for B: options are keyed by `index` + `semantic_action_key` (the shared `state_action_schema_v2`
key), `eq_class` groups siblings -- align your option-deltas/descriptors on these. Suggested use: primary
target = `hand_norm_advantage` (low-noise), weighted by criticality and 1/`hand_value_variance`; treat
`outcome_winrate` as an auxiliary and request higher-k before relying on it; down-weight large
`acceptable_action_set` (near-ties).

## A5 / recommendation

No live selective-search variant was built, so no A5 screen. Recommendation: **label generator only** for now
-- Teacher V2 is a labeling/analysis deliverable, not a current live win-rate lever. Next, on request: scale
the label set across more high-criticality decisions, and run a higher-k outcome study (or regret check) to
test whether the outcome divergence is information or variance. `agent_search` remains the submission baseline.
