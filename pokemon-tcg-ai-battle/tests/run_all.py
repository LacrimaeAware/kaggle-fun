"""Run every test module in tests/ with one command (stdlib, no pytest required).

    PYTHONIOENCODING=utf-8 python tests/run_all.py

Each module exposes a main() -> int (0 = pass). Add new test modules to MODULES.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = [
    "test_split_base_v2",          # schema / encoding / teacher parity
    "test_heuristics_fixed_state",  # heuristics + leaf eval + agent legality on frozen states
    "test_starmie_audit_fixes_v1",  # Model B audit-fix correctness on frozen states
    "test_attacker_continuity_v1",  # ATTACKER_CONTINUITY_V1 leaf term (tactical-leaf task)
    "test_attach_mega_not_engine_v1",  # ATTACH_MEGA_NOT_ENGINE_V1 attach-targeting probe
    "test_bridge_trace_v0",         # proposer-bridge trace logger (runtime/eval separation, no behaviour change)
    "test_proposer_adapter_v0",     # learned-proposer adapter + safety spec (disabled, cannot change action)
]


def main() -> int:
    rc = 0
    for name in MODULES:
        print(f"\n===== {name} =====")
        mod = importlib.import_module(name)
        rc |= mod.main()
    print(f"\n{'ALL SUITES PASS' if rc == 0 else 'SOME SUITES FAILED'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
