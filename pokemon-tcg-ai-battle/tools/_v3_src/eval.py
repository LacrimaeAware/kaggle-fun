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

WIN, LOSS = 1_000_000.0, -1_000_000.0
BLEND_LAMBDA = 0.4    # weight of the learned value in the blended leaf eval (hand gets 1-lambda)
BLEND_SCALE = 2000.0  # squash scale: maps the hand score (prize-dominated) into (0,1)
W_PRIZE = 1000.0      # prizes win the game -> dominates
W_HP = 1.0            # damage race (hp is 0..340-ish)
W_BODY = 30.0         # board presence; a KO costs the opponent a body
W_ENERGY = 8.0        # energy developed on my active attacker
# Gated "hoard for Powerful Hand" term. OFF (0.0) by default -> agent_search is UNCHANGED. When >0 AND a
# Powerful Hand attacker (Alakazam) is in play for `me`, value each card held in hand. Rationale: the 4 terms
# above are hand-blind, and W_BODY actively rewards emptying the hand into bodies; but Alakazam's Powerful Hand
# deals 2 damage counters (20 HP) per card in hand, so this deck wants to HOARD. Gated on Alakazam-in-play so
# it cannot distort normal decks. ~15/card ~= 0.75 of a future damage-counter pair, discounted, << W_PRIZE.
W_POWERFUL_HAND = 0.0
# Candidate-only deck terms. They are intentionally OFF in evaluate(); use
# evaluate_deck_v3 explicitly so resolver/search tests do not silently change the leaf.
W_V3_PH_POTENTIAL = 0.0
W_V3_BACKUP_ATTACKER = 0.0
W_V3_DECKOUT = 0.0
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


def _active(p: dict):
    active = p.get("active") or []
    return active[0] if active and active[0] else None


def _attached_count(entity: dict | None) -> int:
    if not entity:
        return 0
    cards = entity.get("energyCards")
    return len(cards) if cards is not None else len(entity.get("energies") or [])


def _hand_count(p: dict) -> int:
    value = p.get("handCount")
    return int(value if value is not None else len(p.get("hand") or []))


def evaluate_components(cur: dict, me: int) -> dict[str, float]:
    """Expose the exact baseline terms for traces and audits."""
    players = cur.get("players") or []
    if len(players) < 2:
        return {"prize": 0.0, "hp": 0.0, "body": 0.0, "energy": 0.0, "hoard": 0.0}
    player, opponent = players[me], players[1 - me]
    hand = _hand_count(player)
    return {
        "prize": W_PRIZE * (len(opponent.get("prize") or []) - len(player.get("prize") or [])),
        "hp": W_HP * (_board_hp(player) - _board_hp(opponent)),
        "body": W_BODY * (_n_pokemon(player) - _n_pokemon(opponent)),
        "energy": W_ENERGY * _active_energy(player),
        "hoard": (W_POWERFUL_HAND * hand if W_POWERFUL_HAND and _powerful_hand_online(player) else 0.0),
    }


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
    return sum(evaluate_components(cur, me).values())


def evaluate_deck_v3(cur: dict, me: int, *, ph_weight: float = W_V3_PH_POTENTIAL,
                     backup_weight: float = W_V3_BACKUP_ATTACKER,
                     deckout_weight: float = W_V3_DECKOUT) -> float:
    """Optional, separately gated deck-aware leaf.

    This function is not used by the default candidate.  The terms are tied to
    concrete mechanics rather than generic card value, but still require their
    own A/B tests before activation.
    """
    terminal = cur.get("result", -1)
    if terminal == me:
        return WIN
    if terminal == (1 - me):
        return LOSS
    players = cur.get("players") or []
    if len(players) < 2:
        return 0.0
    player, opponent = players[me], players[1 - me]
    score = evaluate(cur, me)
    active = _active(player)
    opp_active = _active(opponent)
    if ph_weight and active and active.get("id") == 743 and _attached_count(active) >= 1:
        potential = 20.0 * _hand_count(player)
        if opp_active:
            potential = min(potential, float(opp_active.get("hp", 0) or 0))
        score += ph_weight * potential
    if backup_weight:
        ready = 0
        for entity in player.get("bench") or []:
            if entity and entity.get("id") == 743 and _attached_count(entity) >= 1:
                ready += 1
        score += backup_weight * ready
    if deckout_weight:
        deck_left = float(player.get("deckCount", 0) or 0)
        if deck_left <= 5:
            score -= deckout_weight * (6.0 - deck_left)
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
