"""Read-only PUBLIC turn-context extractor for Model A's future Feature V3 packer.

This module is PREP only. It is intentionally NOT imported by choose_action / the live selector; it does not change
gameplay. It surfaces the public temporal fields that the runtime audit found present in every observation but
unused by the adapter/model. It never reads `current.result` (game outcome) or any future/pilot/outcome data, and
it never mutates the observation. Missing fields are returned as explicit null with a `field_status` map.

Verified facts (runtime_feature_audit_v0): the raw `current.*` fields below are present at 100% of mid-game obs.
Turn-phase nuance (empirically observed): global turn 0 is a SHARED setup phase (both seats act, `firstPlayer` is
-1 until resolved); after setup the first player acts on ODD global turns. Player-turn-index derivations are
therefore marked best_effort and left for Model A to finalize with full engine knowledge.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import deck_policy_v3 as DP  # noqa: E402  read-only obs helpers

ATTACK, END, CARD, PLAY, ATTACH, EVOLVE, ABILITY = 13, 14, 3, 7, 8, 9, 10
STATUS_CONDS = ("asleep", "paralyzed", "confused", "burned", "poisoned")
# the explicit field set this extractor guarantees (stable schema)
FIELDS = (
    "global_turn_number", "first_player", "first_player_resolved", "am_i_first_player", "is_setup_phase",
    "turn_action_count", "decision_index_in_turn", "supporter_used_this_turn", "energy_attached_this_turn",
    "retreated_this_turn", "stadium_played_this_turn", "stadium_in_play", "active_appeared_this_turn",
    "bench_appeared_this_turn_count", "status_conditions", "attack_available", "end_available",
    "terminal_legal_option_count", "nonterminal_legal_option_count", "information_revealing_legal_count",
    "safe_development_legal_count", "is_our_first_turn_best_effort", "is_first_player_first_turn_best_effort",
)
# fields that must never be surfaced as features
FORBIDDEN = ("result", "future", "pilot", "outcome", "won", "replay")


def _int(v, d=None):
    try:
        return int(v)
    except Exception:
        return d


def extract_turn_context(obs: Any) -> dict:
    """Return a flat, stable dict of public turn-context features + a `field_status` map. Read-only."""
    cur = DP._get(obs, "current", {}) or {}
    sel = DP._get(obs, "select", {}) or {}
    opts = DP._items(DP._get(sel, "option", []))
    me = _int(DP._get(cur, "yourIndex", 0), 0)
    first = DP._get(cur, "firstPlayer", None)
    turn = DP._get(cur, "turn", None)
    first_resolved = isinstance(first, int) and first >= 0
    is_setup = (turn == 0) or (not first_resolved)

    types = [DP._get(o, "type", None) for o in opts]
    terminal = sum(1 for t in types if t in (ATTACK, END))
    info_reveal = sum(1 for t in types if t == CARD)
    safe_dev = sum(1 for t in types if t in (PLAY, ATTACH, EVOLVE))

    players = DP._players(cur)
    meP = players[me] if isinstance(players, list) and me < len(players) else {}
    act = DP._active(meP)
    bench = DP._bench(meP)
    am_first = (me == first) if first_resolved else None

    # best-effort player-turn-index (post-setup the first player acts on odd global turns)
    if is_setup or turn is None:
        is_our_first, is_fp_first = None, None
    else:
        is_fp_first = (turn == 1)
        is_our_first = (turn == 1) if am_first else ((turn == 2) if am_first is not None else None)

    out = {
        "global_turn_number": turn,
        "first_player": first,
        "first_player_resolved": first_resolved,
        "am_i_first_player": am_first,
        "is_setup_phase": is_setup,
        "turn_action_count": DP._get(cur, "turnActionCount", None),
        "decision_index_in_turn": DP._get(cur, "turnActionCount", None),
        "supporter_used_this_turn": DP._get(cur, "supporterPlayed", None),
        "energy_attached_this_turn": DP._get(cur, "energyAttached", None),
        "retreated_this_turn": DP._get(cur, "retreated", None),
        "stadium_played_this_turn": DP._get(cur, "stadiumPlayed", None),
        "stadium_in_play": bool(DP._get(cur, "stadium", None)),
        "active_appeared_this_turn": bool(DP._get(act, "appearThisTurn", False)) if act else None,
        "bench_appeared_this_turn_count": sum(1 for b in bench if DP._get(b, "appearThisTurn", False)),
        "status_conditions": {c: bool(DP._get(meP, c, False)) for c in STATUS_CONDS},
        "attack_available": ATTACK in types,
        "end_available": END in types,
        "terminal_legal_option_count": terminal,
        "nonterminal_legal_option_count": len(types) - terminal,
        "information_revealing_legal_count": info_reveal,
        "safe_development_legal_count": safe_dev,
        "is_our_first_turn_best_effort": is_our_first,
        "is_first_player_first_turn_best_effort": is_fp_first,
    }
    # explicit status per field: present (from obs) | derived | best_effort | unsupported
    status = {}
    raw = {"global_turn_number": "turn", "first_player": "firstPlayer", "turn_action_count": "turnActionCount",
            "decision_index_in_turn": "turnActionCount", "supporter_used_this_turn": "supporterPlayed",
            "energy_attached_this_turn": "energyAttached", "retreated_this_turn": "retreated",
            "stadium_played_this_turn": "stadiumPlayed"}
    for f in FIELDS:
        if f in raw:
            status[f] = "present" if DP._get(cur, raw[f], None) is not None else "null_missing"
        elif f.endswith("_best_effort"):
            status[f] = "best_effort_setup_caveat"
        else:
            status[f] = "derived"
    out["field_status"] = status
    return out


if __name__ == "__main__":  # pragma: no cover
    print("turn_context_v0: read-only extractor; not wired into gameplay. Fields:", len(FIELDS))
