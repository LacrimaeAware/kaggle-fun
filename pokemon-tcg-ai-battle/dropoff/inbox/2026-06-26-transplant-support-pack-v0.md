# Starmie Transplant Trace Support Pack V0 (2026-06-26)

Model B. Built the clean per-decision support data for Model A's replay-transplant value prior. **No gameplay
run; gameplay unchanged.** Verdict: **A_TRANSPLANT_SUPPORT_DATA_READY**.

## What this gives Model A
For 7182 Starmie-pilot replay decisions (3 cohorts: yushin, keidroid, old_exact), each row connects the
**runtime state** (for similarity) to **eval-only consequence targets** (for value), strictly separated:

- `runtime`: tactical_state, legal_semantic_keys, option_index_to_semantic_key, current_agent_action_key,
  search_action/candidate_order, hard_safety_flags, and per-option `action_semantics` (family / role /
  turn_ending / blocked_by_c3 / consumes_attachment|supporter / changes_active).
- `eval_only`: pilot_action_key/family, outcome_won, agreement, `same_turn` (future sequence, dev-actions-before-
  attack, pilot_attacked_later, ended-without-attacking, followed_by_* flags), `turn_end_delta`, `next_own_delta`.

This is exactly the transplant object Model A's prompt asks for: "in replay states similar along family-specific
features, what happened after this kind of action?" The same-turn + turn-end deltas are the short-horizon
consequence answers.

## Coverage + integrity (all clean)
- **100% same-turn sequence**, **100% turn-end delta resolved**; next-own-decision 6153/7182 (1029 missing = game
  ended that turn).
- **Runtime/eval separation PASS** (0 leaks; verified two ways: substring scan + agent!=pilot on 37% of rows
  proves runtime carries the agent's pick, not the pilot's).
- **0 split-leakage episodes, 0 duplicate decision IDs, 0 cross-cohort episode overlap.**
- Family support: SELECT_CARD 2614, PLAY 1949, ATTACK 978, ATTACH 796, EVOLVE 287, OTHER 239, YES_NO 224,
  **RETREAT 95 (lowest — Model A should abstain/widen neighbors for RETREAT)**.

## Value signal is real
ATTACH decisions show mean `opp_board_hp` delta **−81.4** and `my_ready_main_attackers` **+0.46** by turn-end
(attaching enables damage + readiness). The consequences are family-discriminative, not noise — the prior has
something to learn from.

## Turn-boundary method (so Model A trusts the deltas)
Turns are delimited by the replay `status` field: a contiguous ACTIVE run for a seat = one turn. turn-end state =
first non-ACTIVE step after the run (post-turn board); next-own = next ACTIVE run start. The same-turn future is
enriched from the trace's already-validated `future_same_turn_sequence`.

## Artifacts
`data/generated/starmie_transplant_support_v0/`: trace_inventory.json, same_turn_sequences.jsonl,
turn_end_deltas.jsonl, action_semantics.jsonl, model_a_transplant_join.jsonl, data_quality_report.json,
review_examples.{html,jsonl}, VERDICT.json. Tools: `tools/transplant_{inventory,sequences,join}_v0.py`.

## Note for Model A's prior design
The pack covers the replay (expert) state distribution. Recall the V2 C3 smoke lesson: the selector failure
surfaced on OUR-AGENT mirror states, which differ from expert states (distribution shift). The transplant prior
trained on these expert states should be applicability-gated (abstain on low neighbor support / OOD) so it does
not over-transfer to our-agent states.
