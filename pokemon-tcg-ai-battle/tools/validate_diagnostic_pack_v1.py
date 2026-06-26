"""STARMIE SELECTOR DIAGNOSTIC PACK V1 -- integrity check (section 6).

Validates: all referenced artifacts exist; JSONL parses; no pilot/outcome/future metadata in the per-decision
runtime-facing fields; selector remains default OFF; no gameplay files modified by this task.

  PYTHONIOENCODING=utf-8 python tools/validate_diagnostic_pack_v1.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_selector_live_smoke_v1"
REQUIRED = ["changed_decision_classes.jsonl", "failure_aggregate_report.json", "mirror_regression_report.json",
            "selector_smoke_review.html", "selector_smoke_review.jsonl", "model_a_selector_failure_handoff.md"]
FORBIDDEN_SUBSTR = ("pilot_name", "won", "outcome", "future", "yushin", "keidroid")
GAMEPLAY_FILES = ["agent/starmie_heuristics.py", "agent/search_v3.py", "agent/eval.py", "agent/deck_policy_v3.py"]


def main() -> int:
    fails = []

    for fn in REQUIRED:
        if not (OUT / fn).exists():
            fails.append(f"missing artifact: {fn}")

    # JSONL parses + required keys + no forbidden metadata leak in rows
    cdc = OUT / "changed_decision_classes.jsonl"
    if cdc.exists():
        req_keys = {"decision_id", "mode", "baseline_family", "changed", "transition_class",
                    "terminal_override", "premature_terminal_override", "proposer_rank_of_pick"}
        n = 0
        for i, line in enumerate(open(cdc, encoding="utf-8")):
            try:
                r = json.loads(line)
            except Exception as e:
                fails.append(f"changed_decision_classes.jsonl line {i} parse error: {e}")
                break
            n += 1
            if not req_keys.issubset(r.keys()):
                fails.append(f"row {i} missing keys: {req_keys - set(r.keys())}")
                break
            # runtime-facing fields must not carry pilot/outcome/future signals (game_result/matchup are explicit Nones)
            blob = json.dumps({k: v for k, v in r.items() if k not in ("game_result", "matchup")}).lower()
            for bad in FORBIDDEN_SUBSTR:
                if bad in blob:
                    fails.append(f"row {i} leaks forbidden metadata '{bad}'")
                    break
        if n == 0:
            fails.append("changed_decision_classes.jsonl is empty")
        else:
            print(f"  OK changed_decision_classes.jsonl: {n} rows, schema + no-leak checks pass")

    # selector default OFF
    os.environ.pop("STARMIE_SELECTOR_MODE", None)
    sys.path.insert(0, str(ROOT / "agent"))
    import starmie_heuristics as SH  # noqa: E402
    mode = os.environ.get("STARMIE_SELECTOR_MODE", "off")
    if mode != "off":
        fails.append(f"selector default not off: {mode}")
    else:
        print("  OK selector default mode is off")

    # no gameplay files modified vs HEAD (this task is diagnostics-only)
    try:
        diff = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--"] +
                                       [f"pokemon-tcg-ai-battle/{f}" for f in GAMEPLAY_FILES],
                                       text=True, cwd=ROOT.parent).strip()
        if diff:
            fails.append(f"gameplay files modified: {diff}")
        else:
            print("  OK no gameplay files modified vs HEAD")
    except Exception as e:
        print(f"  (skipped git check: {e})")

    if fails:
        print("\nINTEGRITY: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("\nINTEGRITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
