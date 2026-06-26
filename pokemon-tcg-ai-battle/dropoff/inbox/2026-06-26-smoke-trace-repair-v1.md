# Starmie Smoke Trace Repair V1 (2026-06-26)

Model B. Proved the live-smoke trace logging is reliable before any future Selector V3 / N500 smoke. **No large
smoke run (micro only, 24 games); gameplay unchanged.** Verdict: **TRACE_HARNESS_READY_FOR_SELECTOR_V3_SMOKE**.

## Why this task
The V2 smoke diagnostic (and the adversarial review) found the trace logging had 3 defects. I fixed them in the V2
round; this task PROVES the fixes with a validator + micro-smoke and formalizes the schema for V3.

## The 3 fixes, now validated
1. **game_id non-unique** -> globally-unique task index. Validator: 12 distinct ids for 12 games.
2. **blocked terminal proposals were counters only** -> now logged as outcome-linked rows. Validator: 42 rows == 42 counter.
3. **first_changed_outcomes mis-keyed** (diagnostic) -> fixed; validator: first_changed once per changed game.

## What I did
- Audited the harness (`harness_audit.json`).
- Parametrized `tools/selector_v2_smoke_v1.py` with `--modes/--out/--trace-file/--summary-file` (default full smoke
  unchanged). Caught + fixed a latent bug along the way: the summary write + final print used the hardcoded default
  dir instead of `--out`, which clobbered the committed V2 summary on the first micro run (restored from git).
- Ran the micro-smoke (off + c3 x 6 opp x 2 games, 0 errors) and `tools/validate_smoke_trace_v1.py` -> **ALL 8 checks PASS**.
- Defined `selector_v3_trace_schema.json` (per-game + per-decision, with reserved `transplant_*` fields).
- Wrote `model_a_trace_handoff.md`. Existing test suite: ALL 9 suites pass.

## Load-bearing limitation for Model A
Live-smoke decisions are **our-agent self-play states with no replay decision_id** — they can't be exact-joined to
the transplant records (keyed on replay decisions). Join is by action family + tactical-state similarity. And
transplant scores attached at smoke time are out-of-distribution retrievals (our-agent vs expert states) — gate on
support/abstain, as the V3 prompt plans. This is the same distribution-shift caveat from the V2 smoke.

## Artifacts
`data/generated/starmie_smoke_trace_repair_v1/`: harness_audit.json, micro_smoke_trace.jsonl, micro_smoke_summary.json,
trace_validation_report.json, selector_v3_trace_schema.json, model_a_trace_handoff.md, VERDICT.json.
Tools: `tools/selector_v2_smoke_v1.py` (parametrized), `tools/validate_smoke_trace_v1.py`.

Do not run a serious live smoke until Selector V3 exists; this harness is now ready to log it interpretably.
