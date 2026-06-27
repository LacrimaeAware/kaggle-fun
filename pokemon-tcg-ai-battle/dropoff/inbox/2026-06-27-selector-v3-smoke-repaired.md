# Starmie Selector V3 Tiny Smoke — REPAIRED, non-inert, but NEUTRAL (2026-06-27)

Model B. Re-ran the V3 tiny smoke after Model A's `PORTABLE_SELECTOR_V3_TABLE_LOOKUP_REPAIRED` fix.
**Verdict: B_REPAIRED_V3_SMOKE_NEUTRAL. Promotion: DO_NOT_SUBMIT / NEEDS_N500.** Gameplay unchanged,
selector default off, 11/11 test suites pass.

## The blocker is cleared (this was the whole point of the rerun)
The prior run was `D_V3_SELECTOR_UNSAFE_OR_INVALID` because the runtime queried transplant support with a
STRUCTURED key while the exported table was keyed COMPACT -> every lookup missed -> selector inert (0 overrides,
S3 == off). **That is now fixed and PROVEN to work without any caller-supplied support:**

- **Parity gate B (table-fallback-forced):** stripped ALL caller-supplied support and forced the table path ->
  100% lookup hits. This is the gate the prior run never had.
- **Parity gate C (live-shape probe):** table non-inert, >=1 override, not baseline-identical.
- **Live smoke:** 919 table-supported overrides (source=selector) across 120 V3 games, 0 terminal overrides,
  0 errors. S3 is genuinely live now, not a baseline echo.

One Model B mechanic worth recording: the portable PACKER does not emit `compact_semantic_action_key`, so the
runtime still missed the table even after Model A's key fix. Model B bridges it in
`agent/starmie_heuristics.py::_attach_compact_keys` using `learned_proposer_adapter.option_index_to_key`
(95.5% table match). This is option IDENTITY, not a substitute support calculation -- the runtime's designed
extension point. **Ideally the portable packer should emit `compact_semantic_action_key` natively** so Model B
needs no bridge; flagging for Model A.

## Why NEUTRAL and not directional (this is the honest read, post adversarial review)
V3 is safe and now does something, but the win-rate signal does not clear the bar for "directional":

| metric | off | c3 | v3 | v3 - off |
|---|---|---|---|---|
| key (deployed+mirror, n=40) | 42.5% | 40.0% | **57.5%** | **+15pp** |
| deployed alone (n=20) | 35% | 30% | 60% | +25pp |
| true field win-rate (4 opp, real n) | 87.5% | 92.5% | 87.5% | 0pp |
| true all-opponent win-rate (n=120) | 72.5% | 75.0% | 77.5% | +5pp |

- **+15pp on key is NOT significant:** Fisher p=0.263, 95% CI ~[-7, +37]pp straddles zero. The SIGN is not
  established at n=20. Per the A/B definitions, A needs the direction established; this is a positive *point
  estimate* only -> B.
- **The +25pp deployed swing is 3-game-fragile:** 3 of v3's 12 deployed wins flipping -> +10pp; 5 -> 0pp.
- **Override -> win link is a selection artifact:** the selector touches 97.5% of games, so "win-rate among
  changed games" is meaningless. LOSS games actually carry MORE overrides (8.8) than WIN games (7.6). Override
  intensity anti-correlates with winning. The 919 overrides are churn, not a demonstrated win driver.
- **One real regression, masked by averaging:** denpa92 100% -> 85% (-15pp, p=0.23). The "field 87.5% = 87.5%"
  flatness is arithmetic cancellation (denpa92 down, first/alakazam up), not stability. Non-catastrophic
  (> -20pp), so not C, but flag it.
- **v3 vs c3 profiles can't be told apart at n=20.** c3's only individually significant cell (alakazam 75->100,
  p=0.047) fails Bonferroni. v3 has no individually significant cell.

## What this means
The plumbing is done and trustworthy. V3 is the FIRST selector variant that is simultaneously (a) non-inert
live, (b) never overrides into a terminal action, and (c) has a non-negative key-matchup point estimate. That is
real progress over V1 (crashed mirror) and V2/c3 (key-flat). But n=20 cannot establish that the +15pp is real,
and the override pattern looks like churn. **Do not promote on this.**

N500 is justified ONLY as a powered test of one pre-registered hypothesis: "V3 lifts the deployed cell", with
denpa92 watched as a regression guardrail. Not as confirmation of a result we already believe.

## For Model A
1. Have the portable packer emit `compact_semantic_action_key` (and/or `projection.raw_vector`) natively so the
   table lookup works without Model B's `_attach_compact_keys` bridge.
2. The transplant signal currently behaves like churn (overrides anti-correlate with wins). Worth checking
   whether the support table's `recommended_use`/confidence gating is selective enough -- 919 overrides over
   120 games is a lot of intervention for a flat field result.

## Known logging gap (fixed for next run, win rates unaffected)
This smoke ran before the `selector_trace` top-level transplant-field fix, so `transplant_table_hit` /
`transplant_lookup_key` / `transplant_support_source` logged null in `changed_decisions.jsonl`. `source=selector`
is the table-supported-override proxy used throughout the diagnostic. The fix
(`agent/starmie_heuristics.py::selector_trace` captures the top-level out["transplant_*"] fields;
`tools/selector_v2_smoke_v1.py::_transplant_fields` reads them) is in place for the N500 run.

## Artifacts
`data/generated/starmie_selector_v3_smoke_repaired/`: VERDICT.json, diagnostic_report.json (significance +
override-efficacy + regression flags), parity_report.json (3 gates PASS), vendor_refresh_report.json,
baseline_manifest.json, live_smoke_summary.json, changed_decisions.jsonl (2456 rows), review_examples.html.
Vendored repaired bundle: `agent/vendor/portable_selector_v3/`. Wiring (default off, fail-closed):
`agent/starmie_heuristics.py` mode `selector_v3_transplant`. Tests: `tests/test_selector_v3_wiring_v1.py`
(+ run_all 11 suites).
