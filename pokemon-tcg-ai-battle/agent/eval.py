"""L2 v0 value: a transparent linear score of a state from a fixed player's perspective.

Used at the leaves of the forward-model search (agent/search.py). Higher = better for `me`.
Works on the plain-dict form of a State (the live obs["current"], or dataclasses.asdict of a
simulated State), so the same function scores both real and simulated states.

Design (deliberately simple and interpretable for v0):
  - PRIZES dominate: you win by taking all your prizes, so prize differential is weighted far
    above everything. A line that takes a knockout (a prize) should always outrank one that
    does not, regardless of board fluff.
  - DAMAGE race and BOARD presence are tie-breakers that give search a gradient between
    equal-prize lines (chip damage, keeping bodies, not over-extending).
  - ENERGY on the active rewards developing a real attacker.
The weights are named constants so Gate A can tune them without touching the structure.
"""
from __future__ import annotations

WIN, LOSS = 1_000_000.0, -1_000_000.0
W_PRIZE = 1000.0      # prizes win the game -> dominates
W_HP = 1.0            # damage race (hp is 0..340-ish)
W_BODY = 30.0         # board presence; a KO costs the opponent a body
W_ENERGY = 8.0        # energy developed on my active attacker


def _board_hp(p: dict) -> int:
    tot = 0
    for a in (p.get("active") or []):
        if a:
            tot += a.get("hp", 0) or 0
    for b in (p.get("bench") or []):
        if b:
            tot += b.get("hp", 0) or 0
    return tot


def _n_pokemon(p: dict) -> int:
    n = sum(1 for a in (p.get("active") or []) if a)
    return n + len(p.get("bench") or [])


def _active_energy(p: dict) -> int:
    a = p.get("active") or []
    if a and a[0]:
        return len(a[0].get("energies") or [])
    return 0


def evaluate(cur: dict, me: int) -> float:
    """Score State-dict `cur` from player `me`'s point of view. Terminal states return +/-WIN."""
    res = cur.get("result", -1)
    if res == me:
        return WIN
    if res == (1 - me):
        return LOSS
    if res == 2:
        return 0.0
    players = cur.get("players") or []
    if len(players) < 2:
        return 0.0
    P, O = players[me], players[1 - me]
    my_left = len(P.get("prize") or [])
    op_left = len(O.get("prize") or [])
    score = W_PRIZE * (op_left - my_left)
    score += W_HP * (_board_hp(P) - _board_hp(O))
    score += W_BODY * (_n_pokemon(P) - _n_pokemon(O))
    score += W_ENERGY * _active_energy(P)
    return score


def _terminal(cur: dict, me: int):
    res = cur.get("result", -1)
    if res == me:
        return WIN
    if res == (1 - me):
        return LOSS
    if res == 2:
        return 0.0
    return None


def evaluate_obs(obs: dict, me: int) -> float:
    """Hand eval over the State inside an observation dict."""
    return evaluate(obs.get("current") or obs, me)


def evaluate_learned(obs: dict, me: int) -> float:
    """Learned-value leaf eval: terminal states dominate (+/-WIN); otherwise the value model's
    P(win) from `me`'s perspective (so search compares leaves by predicted win probability).
    Falls back to the hand eval if no trained weights are loaded."""
    import value_model as VM
    cur = obs.get("current") or {}
    t = _terminal(cur, me)
    if t is not None:
        return t
    p = VM.score_obs(obs)
    if p is None:
        return 0.5                            # neutral: never mix thousands-scale hand eval into a P(win) argmax
    if cur.get("yourIndex", me) != me:        # value is from the to-move side; flip if not me
        p = 1.0 - p
    return p
