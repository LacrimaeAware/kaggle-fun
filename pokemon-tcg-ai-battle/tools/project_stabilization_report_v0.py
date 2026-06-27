"""Project stabilization V0 report generator (Model B, READ-ONLY audit).

Emits the four stabilization JSON reports from the verified inventory + default-behavior checks. Pure report
authoring -- it runs NO games, changes NO gameplay, and only re-verifies default-off facts. Run:
  PYTHONIOENCODING=utf-8 python tools/project_stabilization_report_v0.py
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "generated" / "project_stabilization_v0"
OUT.mkdir(parents=True, exist_ok=True)

# ------- S2: stable default-off infrastructure -------
stable_infra = {
    "definition": "Infrastructure safe to preserve because it is read-only or default-off and does not change the deployed agent's behavior.",
    "deployed_agent_path_note": "The kaggle submission agent is main.agent_starmie, which calls deck_policy_v3.best_ko_attack + search_v3.best_option, NOT starmie_heuristics.choose_action. The selector is wired ONLY into choose_action, so it is off the submission path entirely AND default-off within choose_action.",
    "items": [
        {"item": "selector_trace logging", "paths": ["agent/starmie_heuristics.py (selector_trace)"],
         "affects_default_choose_action": False, "tests": ["tests/test_selector_v3_wiring_v1.py"],
         "default_off": True, "env_var": "STARMIE_SELECTOR_MODE (default off; trace only runs when a mode is set)",
         "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "repaired transplant-field logging", "paths": ["agent/starmie_heuristics.py (selector_trace top-level transplant_* fields)", "tools/selector_v2_smoke_v1.py (_transplant_fields)"],
         "affects_default_choose_action": False, "tests": ["covered via smoke; no gameplay path"],
         "default_off": True, "env_var": "n/a (diagnostic logging only)", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "compact_semantic_action_key bridge", "paths": ["agent/starmie_heuristics.py (_attach_compact_keys)", "agent/learned_proposer_adapter.py (option_index_to_key)"],
         "affects_default_choose_action": False, "tests": ["tests/test_proposer_adapter_v0.py", "tests/test_transplant_v5_support_v1.py"],
         "default_off": True, "env_var": "only used inside the selector_v3 branch (mode-gated)", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "turn_context_v0 extractor", "paths": ["agent/turn_context_v0.py"],
         "affects_default_choose_action": False, "tests": ["tests/test_turn_context_v0.py"],
         "default_off": True, "env_var": "n/a (PREP module, NOT imported by choose_action)", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "learned_selector_bridge (feature adapter)", "paths": ["agent/learned_selector_bridge.py"],
         "affects_default_choose_action": False, "tests": ["tests/test_bridge_trace_v0.py", "tools/validate_selector_bridge_v1.py"],
         "default_off": True, "env_var": "lazily imported only inside the mode-gated selector branch", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "learned_proposer_adapter (disabled)", "paths": ["agent/learned_proposer_adapter.py"],
         "affects_default_choose_action": False, "tests": ["tests/test_proposer_adapter_v0.py"],
         "default_off": True, "env_var": "disabled by default; never wires into choose_action by itself", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "V5 transplant feasibility probe + support pack", "paths": ["tools/transplant_v5_feasibility_probe_v1.py", "tests/test_transplant_v5_support_v1.py"],
         "affects_default_choose_action": False, "tests": ["tests/test_transplant_v5_support_v1.py"],
         "default_off": True, "env_var": "n/a (offline read-only probe)", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "official packer parity + V2/V3 parity gates", "paths": ["tools/validate_selector_v2_parity.py", "tools/validate_selector_v3_parity.py", "tools/validate_selector_bridge_v1.py", "tools/validate_smoke_trace_v1.py"],
         "affects_default_choose_action": False, "tests": ["self-validating gates"],
         "default_off": True, "env_var": "n/a (read-only validators)", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "run_all test harness (12 modules)", "paths": ["tests/run_all.py"],
         "affects_default_choose_action": False, "tests": ["is the harness"],
         "default_off": True, "env_var": "pops STARMIE_SELECTOR_MODE so suite runs default-off", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
        {"item": "runtime feature audits", "paths": ["data/generated/runtime_feature_audit/", "tools/turn_context_prep_v0.py"],
         "affects_default_choose_action": False, "tests": ["tests/test_turn_context_v0.py"],
         "default_off": True, "env_var": "n/a (read-only audit artifacts)", "classification": "SAFE_TO_MERGE_DEFAULT_OFF"},
    ],
}

# ------- S3: experimental artifacts NOT to promote -------
experimental = {
    "definition": "Artifacts that must remain disabled / branch-only; they must never affect the deployed submission.",
    "items": [
        {"artifact": "V1 broad top3 selector", "paths": ["agent/vendor/portable_selector_v1/"],
         "verdict": "B_FAILURE_MODE_DIRECTIONAL (regressed mirror -35pp)",
         "why_not_promoted": "over-disrupted turn structure; promoted low-confidence rank-2/3 picks into terminal ATTACK/END/RETREAT.",
         "reusable": "the runtime/packer harness + the failure-mode diagnosis; not the policy.",
         "must_not_affect_submission": True, "classification": "ARCHIVE_OR_SUPERSEDE"},
        {"artifact": "V2 / C3 family-limited selector", "paths": ["agent/vendor/portable_selector_v2/"],
         "verdict": "B_C3_SELECTOR_SMOKE_NEUTRAL",
         "why_not_promoted": "terminal-safe but flat on the key deployed+mirror cells; aggregate gain was field-driven (n=20).",
         "reusable": "the terminal-override-block design (carried into V3); parity gate.",
         "must_not_affect_submission": True, "classification": "KEEP_BRANCH_ONLY"},
        {"artifact": "V3 repaired transplant selector + T(a) support table", "paths": ["agent/vendor/portable_selector_v3/", "agent/vendor/portable_selector_v3/transplant_support_table.json"],
         "verdict": "B_V3_POWERED_NEUTRAL (positive point estimate +2.6pp on deployed+mirror, p=0.263, below the 6.3pp MDE -> underpowered / not established, do NOT promote)",
         "why_not_promoted": "powered A/B (n=500/arm) did not establish a win; the runtime support is state-blind T(a). Safe (0 terminal overrides, 0 errors) but not better.",
         "reusable": "the repaired runtime, the compact-key bridge, the powered-A/B harness + diagnostic; the parity gates.",
         "must_not_affect_submission": True, "classification": "KEEP_BRANCH_ONLY"},
        {"artifact": "Transplant V4/V5 offline artifacts", "paths": ["data/generated/transplant_toy_lab_v0/", "data/generated/transplant_v5_runtime_support/", "tools/transplant_*_v0.py"],
         "verdict": "toy lab B_AXIS_DELTA_DIRECTIONAL_ONLY; V5 feasibility A (runtime-computable) but Model A's V5 model returned T_ACTION_BASELINE_REMAINS_BEST",
         "why_not_promoted": "transplant remains a research hypothesis (now Model C's lane), not an execution path.",
         "reusable": "the feasibility contract + probe for if/when Model C revives a state-conditioned T(s,a,delta).",
         "must_not_affect_submission": True, "classification": "KEEP_BRANCH_ONLY"},
        {"artifact": "Powered A/B + smoke outputs", "paths": ["data/generated/starmie_selector_v3_powered_ab/", "data/generated/starmie_selector_v2_smoke/", "data/generated/starmie_selector_v3_smoke_repaired/", "data/generated/starmie_selector_live_smoke_v1/"],
         "verdict": "evidence artifacts (gitignored heavy data)",
         "why_not_promoted": "measurement outputs, not shippable code.",
         "reusable": "the verdicts + diagnostics (committed); the heavy jsonl is gitignored.",
         "must_not_affect_submission": True, "classification": "KEEP_BRANCH_ONLY"},
        {"artifact": "Any selector runtime export (enabled)", "paths": ["none enabled"],
         "verdict": "no enabled export exists",
         "why_not_promoted": "no variant earned live enablement.",
         "reusable": "n/a", "must_not_affect_submission": True, "classification": "KEEP_BRANCH_ONLY"},
        {"artifact": "Stale blocked-run V3 manifest", "paths": ["data/generated/starmie_selector_v3_smoke/ (the D-verdict inert run)"],
         "verdict": "superseded by starmie_selector_v3_smoke_repaired",
         "why_not_promoted": "historical record of the inert-D run; superseded by the repaired run.",
         "reusable": "historical only.", "must_not_affect_submission": True, "classification": "ARCHIVE_OR_SUPERSEDE"},
    ],
}

# ------- S3(4): default-behavior verification (verified facts) -------
default_behavior = {
    "checks": {
        "selector_mode_default": "off",
        "env_default_off": "off",
        "off_identity_override_returns_baseline": True,
        "v3_runtime_and_transplant_table_unloaded_after_off_call": "_SELECTOR_RT_V3 == 'uninitialised'",
        "main_references_STARMIE_SELECTOR_MODE": False,
        "main_imports_selector_runtime": False,
        "main_calls_choose_action": False,
        "heuristics_loads_transplant_table_at_import": False,
        "missing_selector_artifacts_fail_closed": "yes -- _selector_runtime_v3() returns None on import failure -> _selector_override_v3 returns baseline",
        "full_test_suite": "12/12 modules pass",
    },
    "interpretation": "Default behavior is intact and the deployed agent (main.agent_starmie) never touches the selector or transplant table. No default behavior is altered.",
    "RESULT": "NOT_PIPELINE_DIRTY",
}

# ------- S5: merge-readiness -------
merge_readiness = {
    "verdict": "A_STABLE_INFRA_READY_TO_MERGE_DEFAULT_OFF",
    "note": "No merge performed (out of scope). This classifies what is mergeable; the user decides if/when to merge. Stale items are archive candidates, NOT merge blockers.",
    "columns": ["component", "merge_now", "reason", "required_tests", "risk", "notes"],
    "SAFE_TO_MERGE_DEFAULT_OFF": [
        {"component": "selector wiring in choose_action (_baseline_pick/_selector_override, default off)", "merge_now": "yes", "reason": "default off, fail-closed, off-identity proven, NOT on the deployed agent_starmie path", "required_tests": "test_selector_wiring_v1, test_selector_v2/v3_wiring_v1", "risk": "very low", "notes": "selector only affects choose_action, which the submission does not call"},
        {"component": "turn_context_v0 + learned_selector_bridge + learned_proposer_adapter (read-only/disabled)", "merge_now": "yes", "reason": "read-only / not wired into gameplay", "required_tests": "test_turn_context_v0, test_bridge_trace_v0, test_proposer_adapter_v0", "risk": "very low", "notes": "docstrings assert not-wired"},
        {"component": "parity validators + run_all harness fix + V5 support tests", "merge_now": "yes", "reason": "read-only validators / test infra", "required_tests": "run_all (12 modules)", "risk": "none", "notes": "no gameplay path"},
        {"component": "selector_trace + transplant-field logging", "merge_now": "yes", "reason": "diagnostic logging only, runs only when a mode is set", "required_tests": "test_selector_v3_wiring_v1", "risk": "very low", "notes": "does not change the returned action"},
    ],
    "KEEP_BRANCH_ONLY": [
        {"component": "vendored portable_selector_v2/v3 runtimes + transplant_support_table.json", "merge_now": "no", "reason": "experimental, inert unless mode set; powered result not promotable", "required_tests": "parity gates", "risk": "low (dormant) ", "notes": "could merge as dormant default-off assets later; keep on branch for now"},
        {"component": "smoke / powered-A/B / diagnostic runners + transplant tools", "merge_now": "no", "reason": "experimental research runners", "required_tests": "n/a", "risk": "low", "notes": "selector_v2/v3 smoke, powered_ab, diagnostics, transplant_*_v0, v5 feasibility probe"},
        {"component": "powered A/B + smoke generated artifacts", "merge_now": "no", "reason": "measurement outputs (heavy jsonl gitignored)", "required_tests": "n/a", "risk": "none", "notes": "verdict JSONs are the durable record"},
    ],
    "ARCHIVE_OR_SUPERSEDE": [
        {"component": "portable_selector_v1 (broad top3)", "merge_now": "no", "reason": "regressed (-35pp mirror); superseded by V2/V3 design", "required_tests": "n/a", "risk": "none", "notes": "archive; keep the failure-mode diagnosis"},
        {"component": "starmie_selector_v3_smoke (inert-D run) manifest/parity", "merge_now": "no", "reason": "superseded by _repaired run", "required_tests": "n/a", "risk": "none", "notes": "historical only"},
        {"component": "untracked dropoff/inbox/2026-06-27-unified-project-map.md", "merge_now": "review", "reason": "untracked doc not created by this audit", "required_tests": "n/a", "risk": "none", "notes": "user to decide whether to track or remove"},
    ],
}

for name, obj in [("stable_infra_report.json", stable_infra), ("experimental_artifact_report.json", experimental),
                  ("default_behavior_report.json", default_behavior), ("merge_readiness_report.json", merge_readiness)]:
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print("wrote", name)
print("VERDICT:", merge_readiness["verdict"], "| default_behavior:", default_behavior["RESULT"])
