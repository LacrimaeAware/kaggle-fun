"""Run every test module in tests/ with one command (stdlib, no pytest required).

    PYTHONIOENCODING=utf-8 python tests/run_all.py

Each module is run as its own subprocess (its __main__ block) and judged by exit code, so modules may use
main()/sys.exit/module-level execution interchangeably and one module cannot abort the rest. Add modules to MODULES.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES = [
    "test_split_base_v2",          # schema / encoding / teacher parity
    "test_heuristics_fixed_state",  # heuristics + leaf eval + agent legality on frozen states
    "test_starmie_audit_fixes_v1",  # Model B audit-fix correctness on frozen states
    "test_attacker_continuity_v1",  # ATTACKER_CONTINUITY_V1 leaf term (tactical-leaf task)
    "test_attach_mega_not_engine_v1",  # ATTACH_MEGA_NOT_ENGINE_V1 attach-targeting probe
    "test_bridge_trace_v0",         # proposer-bridge trace logger (runtime/eval separation, no behaviour change)
    "test_proposer_adapter_v0",     # learned-proposer adapter + safety spec (disabled, cannot change action)
    "test_selector_wiring_v1",      # STARMIE_SELECTOR_MODE wiring (off = identity, modes legal, fail-closed)
]


def main() -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("STARMIE_SELECTOR_MODE", None)  # suite runs against the default (off) agent
    failed = []
    for name in MODULES:
        print(f"\n===== {name} =====")
        proc = subprocess.run([sys.executable, str(HERE / f"{name}.py")], env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        sys.stdout.write(proc.stdout)
        if proc.stderr.strip():
            sys.stdout.write(proc.stderr)
        if proc.returncode != 0:
            failed.append(name)
    print(f"\n{'ALL SUITES PASS' if not failed else 'SOME SUITES FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
