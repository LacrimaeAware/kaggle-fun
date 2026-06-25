"""Frozen-state correctness tests: the agent must never remove its last Pokemon from play.

These run in milliseconds (no games) and encode the instant-loss blunder observed in a real match:
with Dudunsparce as the only Pokemon, the agent picked Run Away Draw (shuffle self into deck) -> 0
Pokemon -> loss, instead of ending the turn. This is how to validate heuristic correctness fast,
instead of slow + mirror-blind win-rate games.

    python tests/test_no_suicide_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import main as M  # noqa: E402

# Run Away Draw ability (type 10) vs End Turn (type 14). Dudunsparce(66) is the only Pokemon.
SUICIDE_OBS = {
    "select": {"maxCount": 1, "minCount": 1, "option": [{"type": 10, "abilityId": 1}, {"type": 14}]},
    "current": {"yourIndex": 0, "players": [
        {"active": [{"id": 66, "hp": 140, "energies": [5]}], "bench": [], "prize": [1, 2, 3],
         "deckCount": 12, "handCount": 4},
        {"active": [{"id": 999, "hp": 100}], "bench": []},
    ]},
}

# Same prompt but with a benched Abra: shuffling the active just promotes the bench, so the rule must
# NOT fire here (it should let normal play proceed).
HAS_BENCH_OBS = {
    "select": {"maxCount": 1, "minCount": 1, "option": [{"type": 10, "abilityId": 1}, {"type": 14}]},
    "current": {"yourIndex": 0, "players": [
        {"active": [{"id": 66, "hp": 140, "energies": [5]}], "bench": [{"id": 741, "hp": 50}],
         "prize": [1, 2, 3], "deckCount": 12, "handCount": 4},
        {"active": [{"id": 999, "hp": 100}], "bench": []},
    ]},
}


def main() -> int:
    fails = []

    end_idx = 1  # the End Turn option above
    for name, fn in (("agent_phaware", M.agent_phaware), ("agent_planner", M.agent_planner)):
        got = fn(SUICIDE_OBS)
        if got != [end_idx]:
            fails.append(f"{name} suicided: returned {got}, want [{end_idx}] (End, not Run Away Draw)")
        else:
            print(f"  [PASS] {name} ends the turn instead of shuffling away its last Pokemon")

    if M._no_suicide(HAS_BENCH_OBS) is not None:
        fails.append("_no_suicide fired with a bench present (should stay inactive there)")
    else:
        print("  [PASS] _no_suicide stays inactive when a bench Pokemon exists")

    if len(M.DECK) != 60 or 305 not in M.DECK or 65 in M.DECK:
        fails.append(f"deck wrong: size={len(M.DECK)}, has305={305 in M.DECK}, has65={65 in M.DECK}")
    else:
        print("  [PASS] deck is the hiroingk list (Dunsparce 305, 60 cards)")

    print("\n" + ("ALL PASS" if not fails else "FAILURES:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
