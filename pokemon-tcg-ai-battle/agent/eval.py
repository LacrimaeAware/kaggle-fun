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

# ---- ATTACKER_CONTINUITY_V1 (tactical-leaf task, Section 4) -- DISABLED BY DEFAULT ---------------------------
# The current deckout leaf is DECK-BLIND about the Starmie attacker: W_ENERGY * active-energy-CARDS rewards energy
# on the Cinderace ENGINE exactly as much as on the Mega Starmie ATTACKER, and counts an Ignition (3 functional
# units) as one card, so search cannot prefer a ready/continuing Mega attacker over Cinderace development
# (structurally proven + 2339/22083 Cinderace-active-while-Mega-ready decisions in the corpus audit). This single
# term gives the leaf a MAIN-ATTACKER-CONTINUITY signal: reward a ready Mega attacker (active or bench) and main
# continuity; PENALIZE energy wasted on the engine, redundant energy that crosses no readiness threshold, and
# energy piled on an exposed 3-prize Mega. It is Ignition-UNIT-aware and BOUNDED well under W_PRIZE/KO terms (it
# never flips a prize/KO decision). Enable with env STARMIE_LEAF_ATTACKER_CONTINUITY=1.
import os as _os
ATTACKER_CONTINUITY_ON = _os.environ.get("STARMIE_LEAF_ATTACKER_CONTINUITY", "") == "1"
_MEGA_STARMIE, _CINDERACE, _IGNITION = 1031, 666, 17
# Frozen weights (hand-set + bounded; validated by the offline audit + A/B, NOT learned). w1..w4 reward; w5..w8 penalize.
ACW = {"ready_main_active": 20.0, "ready_main_bench": 15.0, "one_short_main": 8.0, "viable_backups": 6.0,
       "no_main_online": 25.0, "engine_overinvest": 6.0, "redundant_energy": 4.0, "exposed_concentration": 5.0}


def _cont_units(slot: dict) -> int:
    """Energy UNITS on a slot, Ignition-aware (3 on Mega Starmie, 1 otherwise) -- matches the live agent."""
    if not slot:
        return 0
    cards = slot.get("energies") or slot.get("energyCards") or []
    is_mega = slot.get("id") == _MEGA_STARMIE
    u = 0
    for c in cards:
        cid = c.get("id") if isinstance(c, dict) else c
        u += 3 if (cid == _IGNITION and is_mega) else 1
    return u


def _cont_slots(p: dict) -> list:
    return [a for a in (p.get("active") or []) if a] + [b for b in (p.get("bench") or []) if b]


def attacker_continuity_vector(P: dict, O: dict) -> dict:
    """Public diagnostic components (Section 4). Mega Starmie (1031)=main attacker; Cinderace (666)=energy engine.
    Ready main = a Mega with >=1 energy unit (Jetting-ready). One-short = a Mega in play with 0 units (one
    ordinary attachment makes it ready)."""
    active = (P.get("active") or [None])[0] if P.get("active") else None
    bench = [b for b in (P.get("bench") or []) if b]

    def is_mega(s):
        return bool(s) and s.get("id") == _MEGA_STARMIE

    ready_main_active = 1 if (is_mega(active) and _cont_units(active) >= 1) else 0
    ready_main_bench = sum(1 for b in bench if is_mega(b) and _cont_units(b) >= 1)
    one_short_main = sum(1 for s in _cont_slots(P) if is_mega(s) and _cont_units(s) == 0)
    total_ready_main = ready_main_active + ready_main_bench
    viable_backups = max(0, total_ready_main - 1)            # ready Megas beyond the one you'd attack with
    no_main_online = 1 if total_ready_main == 0 else 0
    engine_overinvest = sum(max(0, _cont_units(s) - 1) for s in _cont_slots(P) if s.get("id") == _CINDERACE)
    redundant_energy = 0
    for s in _cont_slots(P):
        if is_mega(s):
            u = _cont_units(s)
            if u == 2:           # past Jetting(1), one short of Nebula(3): crosses no new threshold
                redundant_energy += 1
            elif u > 3:          # beyond Nebula(3): overkill this turn
                redundant_energy += (u - 3)
    exposed_concentration = 0
    for s in _cont_slots(P):
        if is_mega(s) and _cont_units(s) >= 3:
            hp = float(s.get("hp", 0) or 0)
            if 0 < hp <= 0.5 * 330.0:    # a damaged Mega holding 3 prizes + invested energy -> exposed
                exposed_concentration += 1
    return {"ready_main_active": ready_main_active, "ready_main_bench": ready_main_bench,
            "one_short_main": one_short_main, "viable_backups": viable_backups, "no_main_online": no_main_online,
            "engine_overinvest": engine_overinvest, "redundant_energy": redundant_energy,
            "exposed_concentration": exposed_concentration}


def attacker_continuity_score(P: dict, O: dict) -> float:
    v = attacker_continuity_vector(P, O)
    return (ACW["ready_main_active"] * v["ready_main_active"]
            + ACW["ready_main_bench"] * v["ready_main_bench"]
            + ACW["one_short_main"] * v["one_short_main"]
            + ACW["viable_backups"] * v["viable_backups"]
            - ACW["no_main_online"] * v["no_main_online"]
            - ACW["engine_overinvest"] * v["engine_overinvest"]
            - ACW["redundant_energy"] * v["redundant_energy"]
            - ACW["exposed_concentration"] * v["exposed_concentration"])


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
                     deckout_weight: float = W_V3_DECKOUT, continuity: bool | None = None) -> float:
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
    # ATTACKER_CONTINUITY_V1 (disabled by default): a bounded main-attacker-continuity nudge between non-terminal
    # states. Terminal/prize/KO terms already returned/dominate, so this never flips a win/KO/deck-out decision.
    if (ATTACKER_CONTINUITY_ON if continuity is None else continuity):
        score += attacker_continuity_score(P, O)
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
