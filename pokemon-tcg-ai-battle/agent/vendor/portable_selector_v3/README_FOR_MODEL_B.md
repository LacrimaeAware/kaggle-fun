# Starmie Selector V3 Portable Export

Default mode is off. Use `mode="selector_v3_transplant"` only for the requested Model B smoke.

Selected policy: `T1_C3_PLUS_TRANSPLANT_SCORE`.
Rejected policy: `T4_AXIS_MASK_SELECTOR` is not exported for smoke because it had terminal overrides.

Call:
`runtime.rank_and_select(packed, packed['packed_options'], baseline_action=packed.get('baseline_action'), search_action=packed.get('search_action'), mode='selector_v3_transplant')`

Transplant support policy:
- Prefer per-option `transplant` / `V_transplant` summaries supplied by the caller.
- Then try observation support maps or the exported semantic support table.
- If a non-baseline override lacks support, the runtime falls back/abstains loudly.
- The live runtime does not use replay ids, pilot ids, outcomes, final results, selected labels, or future sequence.

Parity passed: True
Selected-action parity: 1.0
Transplant support parity: 1.0

This bundle is disabled/offline. Do not submit, merge, tune thresholds, or enable by default from this export.
