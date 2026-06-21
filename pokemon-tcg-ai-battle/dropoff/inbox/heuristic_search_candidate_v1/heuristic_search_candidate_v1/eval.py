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

import math

import features as FT

WIN, LOSS = 1_000_000.0, -1_000_000.0
BLEND_LAMBDA = 0.4    # weight of the learned value in the blended leaf eval (hand gets 1-lambda)
BLEND_SCALE = 2000.0  # squash scale: maps the hand score (prize-dominated) into (0,1)
W_PRIZE = 1000.0      # prizes win the game -> dominates
W_HP = 1.0            # damage race (hp is 0..340-ish)
W_BODY = 30.0         # board presence; a KO costs the opponent a body
W_ENERGY = 8.0        # energy developed on my active attacker

# Candidate deck-policy terms. Keep OFF for the first structural A/B so continuation/target
# improvements are not confounded with new leaf weights. Flip this one flag for the later eval-term arm.
ENABLE_EXPERIMENTAL_DECK_EVAL = False
W_HAND_BASE = 1.5 if ENABLE_EXPERIMENTAL_DECK_EVAL else 0.0
W_POWERFUL_HAND = 8.0 if ENABLE_EXPERIMENTAL_DECK_EVAL else 0.0
HAND_CAP = 12
W_OPP_THREAT_DAMAGE = 0.25 if ENABLE_EXPERIMENTAL_DECK_EVAL else 0.0
W_OPP_KO_THREAT = 160.0 if ENABLE_EXPERIMENTAL_DECK_EVAL else 0.0
_POWERFUL_HAND_IDS = (743,)   # Alakazam


def _powerful_hand_online(p: dict) -> bool:
    for slot in (p.get("active") or []) + (p.get("bench") or []):
        if isinstance(slot, dict) and slot.get("id") in _POWERFUL_HAND_IDS:
            return True
    return False


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

    hand = P.get("handCount")
    if hand is None:
        hand = len(P.get("hand") or [])
    hand = max(0, min(int(hand or 0), HAND_CAP))
    score += W_HAND_BASE * hand
    if W_POWERFUL_HAND and _powerful_hand_online(P):
        score += W_POWERFUL_HAND * hand

    # The search already simulates one opponent reply. This modest term values whether the resulting
    # active is still exposed on the following turn; it uses public board/energy only.
    try:
        f = FT.encode_state({"current": cur}, perspective=me)
        threat = float(f.get("opp_active_affordable_dmg", 0.0) or 0.0)
        score -= W_OPP_THREAT_DAMAGE * min(threat, float(f.get("my_active_hp", 0.0) or 0.0))
        if f.get("opp_can_ko_my_active_now"):
            score -= W_OPP_KO_THREAT * max(1.0, float(f.get("my_active_prize", 1.0) or 1.0))
    except Exception:
        pass
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


def _sigmoid(x: float) -> float:
    if x <= -60:
        return 0.0
    if x >= 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def evaluate_blend(obs: dict, me: int, lam: float = BLEND_LAMBDA) -> float:
    """Combine the hand eval (sharp LOCAL ranking of nearby leaves) with the learned value
    (global positional judgment) on ONE [0,1] scale: (1-lam)*sigmoid(hand) + lam*P(win).
    Terminal states map to 1/0/0.5 so they stay comparable. Falls back to the squashed hand
    eval alone if no learned weights are present."""
    import value_model as VM  # was missing -> NameError made blend silently fall back to hand-only
    cur = obs.get("current") or {}
    t = _terminal(cur, me)
    if t is not None:
        return 1.0 if t == WIN else (0.0 if t == LOSS else 0.5)
    hand01 = _sigmoid(evaluate(cur, me) / BLEND_SCALE)
    p = VM.score_obs(obs)
    if p is None:
        return hand01
    if cur.get("yourIndex", me) != me:
        p = 1.0 - p
    return (1.0 - lam) * hand01 + lam * p
