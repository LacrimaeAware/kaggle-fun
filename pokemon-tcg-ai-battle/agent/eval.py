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
_POWERFUL_HAND_IDS = (743,)   # Alakazam

# V3 deck-aware leaf terms (separately gated; all OFF by default so the deployed leaf_mode="hand" is unchanged).
W_V3_PH_POTENTIAL = 0.0       # realized Powerful Hand damage (Alakazam active + energy), 20/card, capped at opp HP
W_V3_BACKUP_ATTACKER = 0.0    # a benched, energized Alakazam ready to promote
W_V3_DECKOUT = 0.0            # penalize MY deck approaching empty (the self-deck-out loss the board eval can't see)
# Magnitudes used by the dedicated A/B leaf_modes ("deckout"/"ph"/"deck"); kept separate from the off-by-default
# module weights above so turning a test mode on never touches the deployed path.
DECKOUT_TEST = 100.0
PH_TEST = 1.0


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
    a = p.get("active") or []
    return a[0] if a and a[0] else None


def _attached_count(slot: dict) -> int:
    return len((slot or {}).get("energies") or [])


def _hand_count(p: dict) -> int:
    h = p.get("handCount")
    return h if h is not None else len(p.get("hand") or [])


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
    if W_POWERFUL_HAND and _powerful_hand_online(P):
        hand = P.get("handCount")
        if hand is None:
            hand = len(P.get("hand") or [])
        score += W_POWERFUL_HAND * hand
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


W_CA_HAND = 25.0   # card-advantage term: value each card in MY hand. A line that draws/tutors more ends
                   # with a bigger hand -> higher score, so search prefers the draw/tutor cascade.


def evaluate_ca(cur: dict, me: int) -> float:
    """Board eval plus card advantage: each card in my hand is worth W_CA_HAND. Terminals still dominate
    (evaluate() returns +/-WIN), so a KO/prize line always outranks a hand-hoarding one."""
    base = evaluate(cur, me)
    if _terminal(cur, me) is not None:
        return base
    players = cur.get("players") or []
    if len(players) < 2:
        return base
    P = players[me]
    hand = P.get("handCount")
    if hand is None:
        hand = len(P.get("hand") or [])
    return base + W_CA_HAND * (hand or 0)


def evaluate_ca_obs(obs: dict, me: int) -> float:
    return evaluate_ca(obs.get("current") or obs, me)


def evaluate_deck_v3(cur: dict, me: int, *, ph_weight: float = W_V3_PH_POTENTIAL,
                     backup_weight: float = W_V3_BACKUP_ATTACKER,
                     deckout_weight: float = W_V3_DECKOUT) -> float:
    """Board eval plus separately-gated deck-aware terms (all weights default OFF):
      - ph_weight: realized Powerful Hand damage when Alakazam(743) is active and energized, 20/card,
        capped at the opponent active's current HP (a hoarded hand beyond a KO is not over-rewarded).
      - backup_weight: a benched, energized Alakazam ready to promote.
      - deckout_weight: hinge penalty as MY deck approaches empty (deck <= 5), the self-deck-out loss
        that prizes/HP/bodies/energy are all blind to.
    Terminal states still dominate via evaluate() (+/-WIN)."""
    base = evaluate(cur, me)
    if _terminal(cur, me) is not None:
        return base
    players = cur.get("players") or []
    if len(players) < 2:
        return base
    P, O = players[me], players[1 - me]
    score = base
    a = _active(P)
    if ph_weight and a and a.get("id") == 743 and _attached_count(a) >= 1:
        potential = 20.0 * _hand_count(P)
        oa = _active(O)
        if oa:
            potential = min(potential, float(oa.get("hp", 0) or 0))
        score += ph_weight * potential
    if backup_weight:
        ready = sum(1 for b in (P.get("bench") or [])
                    if b and b.get("id") == 743 and _attached_count(b) >= 1)
        score += backup_weight * ready
    if deckout_weight:
        deck_left = float(P.get("deckCount", 0) or 0)
        if deck_left <= 5:
            score -= deckout_weight * (6.0 - deck_left)
    return score


def evaluate_deck_v3_obs(obs: dict, me: int, **kw) -> float:
    return evaluate_deck_v3(obs.get("current") or obs, me, **kw)


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
