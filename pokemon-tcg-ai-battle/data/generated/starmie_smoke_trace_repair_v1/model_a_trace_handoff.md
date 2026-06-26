# Model A Handoff: Smoke Trace Harness Repaired for Selector V3

Model B. The live-smoke trace logging is now reliable. Verdict: **TRACE_HARNESS_READY_FOR_SELECTOR_V3_SMOKE**.

## Fields Model A can rely on
Per-game (in `*_summary.json` metrics + derivable from the trace): `game_id` (unique), `mode`, `opponent`, `seat`,
`result`, `total_selector_calls`, `total_overrides`, `total_blocked_proposals`, `total_vetoes`, first-changed id.
Per-decision (one row per override OR blocked-terminal in `*_trace.jsonl`): `game_id`, `step`, `baseline_raw`,
`selector_raw`, `baseline_family`, `selector_family`, `source` (baseline|selector|fallback|veto),
`terminal_override_blocked`, `blocked_terminal`, `blocked_override_reason`, `hard_veto`, `confidence`, `entropy`,
`top1_margin`, `tactical` (guaranteed_ko / game_winning_attack / safe_development / prize_diff), `first_changed`,
`game_result`. Full contract: `selector_v3_trace_schema.json`. The `transplant_*` fields are reserved and will be
populated when the V3 selector runtime exposes them through `SH.selector_trace`.

## What was broken before (V2 smoke) and is now fixed
1. **`game_id` was non-unique** — `{mode}:{opp}:{seat0}:{i}` collided across chunks (24 distinct ids for 360 games),
   so per-game analysis silently collapsed. **Fixed:** `{mode}:{opp}:t{TASK_INDEX}:{i}` with a globally unique task
   index. Validator: 12 distinct ids for 12 games, each mapping to one (mode, opponent, result).
2. **Blocked terminal proposals were counters only** — never logged as rows, so "did blocking a terminal cost the
   game" had no per-decision evidence. **Fixed:** blocked terminals are logged as rows (`blocked_terminal=true`,
   `blocked_override_reason`, outcome-linked). Validator: 42 blocked rows == 42 summary counter.
3. **`first_changed_outcomes` was mis-keyed** in the diagnostic (iterated mode-ids vs mode-values, reporting 0).
   **Fixed** in the diagnostic; validator confirms first_changed appears exactly once per changed game.

## Validation (micro-smoke, off + c3 x 6 opponents x 2 games, 0 errors)
`tools/validate_smoke_trace_v1.py` -> **ALL 8 checks PASS**: game_id unique/consistent; every decision has an
outcome; blocked proposals logged + counter-matched; first_changed once per changed game; mode labels normalized;
terminal flags correct (0 c3 terminal overrides); override count matches; no metadata leak.

## How to join future V3 smoke traces to transplant / proposer records
IMPORTANT: live-smoke decisions are **our-agent self-play states**, not replays — they carry **no replay
`decision_id`**. So they cannot be exact-joined to `transplant_records.jsonl` (which is keyed on replay decisions).
Join is by **action family + tactical-state similarity**, the same axes your transplant prior uses for retrieval.
The `tactical` block + `baseline_family`/`selector_family` are the join handles. The proposer is run inline by
`selector_trace`; when the V3 runtime exposes `proposer_top_k` and `transplant_*`, they are carried per row.

## Known limitations
- **Distribution shift (the load-bearing one):** the transplant prior is trained on expert replay states; the smoke
  exercises our-agent states. The trace lets you SEE V3's live behavior, but transplant scores attached at smoke
  time are out-of-distribution retrievals — gate on support/abstain, as your V3 prompt already plans.
- No exact decision_id on live rows (see above). Step is a within-game decision index, not a replay step.
- Micro-smoke is schema validation only (n=2/cell) — not win-rate evidence. The N500 / V3 smoke remains the
  performance test, judged on the deployed+mirror cells (per the V2 lesson).

## Artifacts
`data/generated/starmie_smoke_trace_repair_v1/`: harness_audit.json, micro_smoke_trace.jsonl,
micro_smoke_summary.json, trace_validation_report.json, selector_v3_trace_schema.json, model_a_trace_handoff.md.
Harness: `tools/selector_v2_smoke_v1.py` (parametrized `--modes/--out`). Validator: `tools/validate_smoke_trace_v1.py`.

VERDICT=TRACE_HARNESS_READY_FOR_SELECTOR_V3_SMOKE
