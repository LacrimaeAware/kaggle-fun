# Starmie Selector V3 Tiny Smoke — BLOCKED on transplant table key mismatch (2026-06-27)

Model B. Imported Model A's Selector V3 export and attempted the tiny smoke. **Verdict:
D_V3_SELECTOR_UNSAFE_OR_INVALID (invalid-for-live, not unsafe). Promotion: NEEDS_SELECTOR_REPAIR.** Gameplay
unchanged, selector default off, no full smoke run.

## What passed
- **In-repo parity: PASS** (102/102 selected, terminal-block 100% [ATTACK/END/RETREAT], transplant-support 100%,
  source 100%, top-k 100%, deterministic, no obs mutation, 0 metadata failures).
- **Wiring + safety: PASS** (off-identity, fail-closed, never overrides into ATTACK/END/RETREAT). All 11 test suites pass.

## The blocker (root cause)
**V3's transplant signal is inert in live play.** Tiny confirmation (2 games x 6 opponents): S2 (c3) logged 8-24
overrides/matchup; **S3 (selector_v3_transplant) logged 0 overrides / 0 blocks across every matchup**, 0 errors.
V3 falls back to baseline on every live decision.

Root cause: the runtime resolves transplant support via
`table_key = f"{family}||{_semantic_action_key(option)}"`, and `_semantic_action_key` returns the **structured JSON**
key. But `transplant_support_table.json` is keyed with the **compact** form (`"ATTACH||ATTACH:Water:Mega"`). The
structured key never matches a compact table key -> every lookup returns missing -> fallback -> baseline.

Why parity didn't catch it: Model A's parity inputs **supply** transplant support directly
(`observation.transplant_support_by_semantic_key` + per-option `transplant` on all 21 selector-source rows), so the
runtime never exercises the table fallback. The table-key bug only surfaces on the live path, where no support map
is supplied.

Why Model B can't work around it: Model B has no transplant-scoring function (that's Model A's V0). Computing/
attaching support = the forbidden "substitute a different transplant support calculation"; silently re-keying the
export is out of scope. So the live path has no working support provider, and I correctly did NOT run a full
360-game smoke that could only reproduce baseline (S3 == off).

## Fix needed (Model A)
1. Make the runtime's table-lookup key match the table key format: either (a) derive the COMPACT semantic key for
   the table lookup, or (b) re-export `transplant_support_table.json` keyed by the STRUCTURED `_semantic_action_key`
   the runtime already computes.
2. OR make the runtime self-contained: compute transplant support from packed features without requiring
   caller-supplied support maps (so it works live without Model B attaching anything).
3. Add a parity probe that exercises the TABLE fallback path (no supplied support map) so this is caught next time.

After the fix, Model B re-runs: vendor -> parity gate -> confirm S3 produces overrides -> full tiny smoke (S0/S2/S3
x 6 opponents x 20 games) with the repaired changed-decision logging, judged on deployed+mirror combined.

## Artifacts
`data/generated/starmie_selector_v3_smoke/`: parity_report.json (PASS), baseline_manifest.json,
off_mode_reproduction.json, diagnostic_report.json, VERDICT.json, _confirm_summary.json/_confirm_trace.jsonl
(c3-vs-v3 inertness evidence). Vendored: `agent/vendor/portable_selector_v3/`. Wiring (default off, fail-closed):
`agent/starmie_heuristics.py` `selector_v3_transplant`. Tests: `tests/test_selector_v3_wiring_v1.py`.
