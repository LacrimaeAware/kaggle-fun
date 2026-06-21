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
