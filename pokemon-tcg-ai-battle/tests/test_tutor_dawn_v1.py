"""Frozen-state tests for the heuristic-first submission (submissions/sub_heuristic): the Dawn/tutor fix
and the heuristic-first entry. Millisecond assertions, no games. Tests the VENDORED (patched) package in
the submission, not the read-only source.

    python tests/test_tutor_dawn_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submissions" / "sub_heuristic"
sys.path.insert(0, str(SUB))

from pokemon_ai_agent.policy.heuristics.tutor_targets import choose_tutor_targets  # noqa: E402


def _dawn_obs(deck_ids, hand_ids):
    return {
        "select": {
            "effect": {"id": 1231}, "maxCount": 1, "minCount": 0,
            "option": [{"type": 3, "area": 1, "index": i} for i in range(len(deck_ids))],
            "deck": [{"id": c} for c in deck_ids],
        },
        "current": {"yourIndex": 0, "players": [
            {"active": [], "bench": [], "hand": [{"id": c} for c in hand_ids], "discard": [], "benchMax": 5},
            {"active": [], "bench": []},
        ]},
    }


def main() -> int:
    fails = []

    # Hold Kadabra+Alakazam+2 Dunsparce, no Abra. Deck offers [Dunsparce, Abra, BattleCage] -> fetch Abra.
    got = choose_tutor_targets(_dawn_obs([305, 741, 1264], [742, 743, 305, 305]))
    if got != [1]:
        fails.append(f"Dawn should fetch Abra (the missing prerequisite), got {got} (0=Dunsparce,1=Abra,2=Cage)")
    else:
        print("  [PASS] Dawn fetches Abra to complete a held Kadabra+Alakazam line, not a redundant Dunsparce")

    # Only [Dunsparce, BattleCage] offered, already hold 2 Dunsparce -> must NOT grab the redundant Dunsparce.
    got2 = choose_tutor_targets(_dawn_obs([305, 1264], [742, 743, 305, 305]))
    if got2 == [0]:
        fails.append(f"Dawn grabbed a redundant 3rd Dunsparce, got {got2}")
    else:
        print(f"  [PASS] Dawn refuses the redundant Dunsparce (picked {got2}, Dunsparce capped to 0)")

    # Regression: Abra in play, no Kadabra, Kadabra offered -> still fetch Kadabra.
    obs3 = _dawn_obs([742, 1264], [])
    obs3["current"]["players"][0]["active"] = [{"id": 741}]   # Abra active
    got3 = choose_tutor_targets(obs3)
    if got3 != [0]:
        fails.append(f"Dawn with Abra in play should fetch Kadabra, got {got3}")
    else:
        print("  [PASS] Dawn still fetches Kadabra for an in-play Abra (no regression)")

    print("\n" + ("ALL PASS" if not fails else "FAILURES:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
