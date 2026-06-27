# F1 energy-allocation runtime prep / attach-trace support pack (2026-06-27)

Model B prep for the C8 F1_ENERGY_ALLOCATION lab. **Verdict: A_F1_ENERGY_RUNTIME_PREP_READY. Default behavior:
NOT_PIPELINE_DIRTY.** Prep only: no attach-scoring change, no large A/B, no merge, selector + ATTACH_MEGA toggle
default off, 14/14 test suites pass.

## What runtime fields are READY (all computable live, from public obs, using the REAL repo helpers)
Per legal ATTACH option, `tools/f1_attach_context_extractor_v1.py::extract_attach_context(obs)` returns, with
**100% field population on Starmie states**:
- `target_role` via `starmie_tactical_state.semantic_role` -> **main_attacker (Mega 1031) / energy_engine
  (Cinderace 666) / setup_basic (Staryu 1030)** + generic attacker/wall/utility.
- `energy_class` via `deck_policy_v3.option_card_id` -> `card_stats` -> **basic_water / ignition / tool** (Ignition
  id 17 unique).
- `energy_units_added`: **3 for Ignition on Mega, 1 for ordinary energy, 0 for a tool** (Hero's Cape).
- `target_energy_before/after` (units), `shortfall_before/after` (vs `_VERIFIED` cheapest attack cost),
  `crosses_attack_threshold`, `already_ready`, `redundant_energy` (energy_after > max useful attack cost),
  `target_card_id/owner/zone/slot`, `semantic_key`.

Verified on 200 Starmie self-play attach options: e.g. `ATTACH:Water:Mega` at 0 energy -> crosses Jetting
threshold; `ATTACH:Ignition:Mega` at 0 -> +3 units -> crosses Nebula threshold; `ATTACH:Ignition:Mega` on a
ready Mega -> redundant (over-loads past Nebula); `ATTACH:Water:Cinderace` -> energy_engine role detected.

## What is MISSING / caveats (none blocking)
- **Golden fixtures are a different deck** (no Mega/Staryu/Cinderace ids) -> they cannot exercise the Starmie
  role/threshold logic. F1 is demonstrated on a tiny Starmie self-play capture; the repo has no committed
  Starmie fixture set. If we want a stable Starmie fixture pack, capture-and-freeze is a small follow-up.
- **card-COUNT vs energy-UNITS**: `_attach_score` scores by energy-card count; thresholds need UNITS (Ignition=3).
  The extractor standardizes on units; any candidate MUST too.
- `energy_engine` (Cinderace) attaches are RARE in self-play (the agent already mostly attaches to the Mega line)
  -- consistent with the prior ATTACH_MEGA probe being win-rate neutral. F1 must be contextual, not "always Mega".

## Are Model A's candidate specs implementable? YES, for the public-feature mechanisms
Mechanisms expressible from the ready fields: M1 main-attacker threshold (shortfall/crosses), M2 backup
continuity (role + ready flags across bench -- needs a per-bench scan, available), M3 engine-starvation
exception (energy_engine role + whether any Mega/Staryu attach materially reduces shortfall), M4 redundant
over-attach penalty (`redundant_energy`), M5 target-role policy (`target_role`), M6 energy-type policy
(`energy_class`/units), M7 turn-context policy (turn_context_v0 fields). M8 matchup/threat needs the opponent-
threat composite (available via tactical_state but coarser). The deployed path uses search, so the toggle must
be wired where the candidate actually decides the pick (heuristic/baseline path), then measured.

## Fixed-state tests Model B should REQUIRE before any F1 toggle
1. off-identity (toggle off == current pick) and fail-closed.
2. never starves a legitimate Cinderace turn (engine_starvation_events ~ 0 on fixed states where Cinderace is
   the only enabling attach).
3. abstains on non-Starmie / unknown target roles (shortfall None) and on already-ready+no-higher-attack targets.
4. Ignition handled as 3 units on Mega; tool (Cape) is not an energy decision.
5. redundant over-attach avoided on a ready Mega already at >=3 units.
6. no obs mutation, no result/outcome/pilot leakage (covered by test_f1_attach_context_v1).

## local_meta_v1 command template for the probe (n100 -> n500)
Wire the candidate behind a default-off env toggle, then:
```
# Stage A (n~60/cell primary + sentinels)
python tools/local_meta_harness_v1.py --modes off,<f1_toggle> \
  --opponents deployed:60,mirror:60,denpa92:40,lucario:40 --stage A --out data/generated/f1_probe
# analyze (primary deployed+mirror, Holm-corrected sentinels, early-stopping guard)
python tools/local_meta_analyze_v1.py --dir data/generated/f1_probe \
  --baseline off --treatment <f1_toggle> --primary deployed,mirror \
  --sentinels denpa92,alakazam,lucario,koraidon,abomasnow --neg first,random
# Stage B (->200) then Stage C (->500) ONLY if non-regressive; promote only on significant PRIMARY above MDE.
```

## Artifacts
`data/generated/f1_energy_runtime_prep_v0/`: current_attach_logic_audit.json, runtime_feature_feasibility.json,
attach_context_examples.jsonl (200), logging_support_report.json, default_behavior_report.json, closeout.json.
Tool: `tools/f1_attach_context_extractor_v1.py`. Test: `tests/test_f1_attach_context_v1.py` (in run_all).

If Model A returns F1_ENERGY_ALLOCATION_READY_FOR_RUNTIME_PROBE, Model B can wire the exact candidate as a
default-off toggle with these fixed-state tests, then run the local_meta_v1 staged probe. If Model A returns
B/C/D, do not implement F1 -- move to F7_TUTOR_TARGETS or F3_DRAW_DECKOUT_SEQUENCING.
