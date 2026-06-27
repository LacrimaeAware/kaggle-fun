# Project stabilization / merge-readiness audit V0 (2026-06-27)

Model B, runtime/stabilization lane (transplant moved to a Model C theory lane; Model A back on non-transplant
feature/proposer work). **Verdict: A_STABLE_INFRA_READY_TO_MERGE_DEFAULT_OFF. Default behavior:
NOT_PIPELINE_DIRTY.** No merge, no submission, no gameplay change. 12/12 test suites pass.

## Headline: the deployed agent is doubly insulated from all the experimental work
The kaggle submission agent is `main.agent_starmie`, which calls `deck_policy_v3.best_ko_attack` +
`search_v3.best_option` -- it does **NOT** call `starmie_heuristics.choose_action`. The entire learned-selector
arc is wired only into `choose_action`, so it is **off the submission path entirely**, on top of being
default-off within `choose_action`. Verified:
- `STARMIE_SELECTOR_MODE` default `off`; `_selector_override` returns the baseline unchanged when off (off-identity).
- After a default (off) call, `_SELECTOR_RT_V3 == 'uninitialised'` -> the V3 runtime and `transplant_support_table`
  are **never loaded by default**.
- `agent/main.py` references neither `STARMIE_SELECTOR_MODE`, the selector runtime, nor `choose_action`.
- Missing selector artifacts fail closed (runtime import failure -> baseline).
- Full suite 12/12.

So nothing in the selector/transplant/V5 work can affect a submission unless someone both sets an env var AND
routes the submission through `choose_action` (neither is the case today).

## What is safe to merge default-off (read-only / dormant)
- The `choose_action` selector wiring (`_baseline_pick` / `_selector_override`, default off, fail-closed).
- Read-only / disabled feature modules: `turn_context_v0` (PREP, not wired), `learned_selector_bridge`,
  `learned_proposer_adapter` (disabled).
- `selector_trace` + repaired transplant-field logging (diagnostic only; never changes the action).
- Parity validators, the `run_all` harness (now 12 modules), and the V5 support tests.

## What stays branch-only (experimental, not promotable)
- Vendored `portable_selector_v2/v3` runtimes + `transplant_support_table.json` (inert unless mode set; V3 powered
  result is a **positive point estimate +2.6pp on deployed+mirror, p=0.263, below the 6.3pp MDE -- underpowered /
  not established, do not promote**).
- Smoke / powered-A/B / diagnostic runners + transplant tools + the V5 feasibility probe.
- Generated smoke / A/B artifacts (heavy jsonl gitignored; verdict JSONs are the durable record).

## Archive / supersede (housekeeping, non-blocking)
- `portable_selector_v1` (broad top3, regressed -35pp mirror; superseded by V2/V3).
- `starmie_selector_v3_smoke/` (the inert-D run; superseded by `_repaired`).
- Untracked `dropoff/inbox/2026-06-27-unified-project-map.md` (not created by this audit; user to track or remove).

## Lane status after this audit
- **Transplant**: parked from implementation; Model C theory lane only. No transplant table consulted by default.
- **Model A**: non-transplant feature/proposer screen (separate prompt).
- **Model B (this)**: runtime stable, branch coherent, work preserved, nothing leaks into the submission.

## Artifacts
`data/generated/project_stabilization_v0/`: current_branch_inventory.md, stable_infra_report.json,
experimental_artifact_report.json, default_behavior_report.json, merge_readiness_report.json. Generator:
`tools/project_stabilization_report_v0.py` (read-only; runs no games).
