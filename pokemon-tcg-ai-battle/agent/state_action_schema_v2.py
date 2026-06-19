"""SPLIT_BASE_V2 / P2 -- the shared semantic state+action schema.

This is the ONE module both research branches import for state/action identity:
  * Branch A (Planner/Teacher V2)  -- agent/teacher_api_v1.py builds on it,
  * Branch B (Robust Learner V2)   -- the entity/action encoder builds on it.

It is FROZEN after the split: changes require auditor approval and a cherry-pick into both
branches (see docs/workstreams/SPLIT_BASE_V2.md). Edit the JSONL/code canon, never a generated copy.

WHY THIS EXISTS (the bug class it kills):
  * PLAY options (type 7) carry NO `area` field; their `index` is a hand index directly. A
    card-identity join keyed on area==HAND therefore went dead for EVERY play, leaving card
    stats / effects / embedding zeroed and collapsing all PLAY options into one equivalence class.
  * The canonical key must include the acting-card identity so two PLAY options for DIFFERENT
    hand cards stay distinct, while two genuinely-identical moves collapse.
  * Orderless zones (hand / discard / bench) must be multisets, not arbitrary positional vectors,
    or option-order permutations silently change the representation.

This module is pure representation. It does NOT choose moves, run search, or train. Forced-move
classification and any teacher value live in agent/teacher_api_v1.py.

Self-check: `python agent/state_action_schema_v2.py` prints the schema version + constants.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

import features as FT  # the canonical L1 encoder (shared 47-dim vector); import-cheap, no engine

SCHEMA_VERSION = "split_base_v2.0"


# --- option / area type codes (decoded from the cabt option schema; cross-checked vs main.py,
#     features.py, tools/build_action_dataset.py) -------------------------------------------------
class OptType:
    NUMBER = 0
    YES = 1            # menu "yes" (also used for go-first)
    MENU2 = 2
    SELECT_CARD = 3
    PLAY = 7           # play a hand card (NO area field; index is a hand index)
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    RETREAT = 12
    ATTACK = 13
    END = 14


class AreaType:
    HAND = 2
    ACTIVE = 4
    BENCH = 5
    DISCARD = 6


# the major action types the golden fixtures must cover (P3 acceptance)
MAJOR_ACTION_TYPES = (
    OptType.PLAY, OptType.ATTACH, OptType.EVOLVE, OptType.ABILITY,
    OptType.RETREAT, OptType.ATTACK, OptType.END, OptType.SELECT_CARD,
)

# energy int -> cost letter (mirrors features.EN; Rainbow(10)/Team Rocket(11) = wildcard '*')
ENERGY_LETTER = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M", 9: "N",
                 10: "*", 11: "*"}

# decoded card-effect keys carried on the action descriptor (mirror tools/build_action_dataset)
EFFECT_KEYS = ["draw", "search", "search_to_bench", "energy_accel", "heal", "switch_gust",
               "recover_discard", "disrupt", "discard_cost", "status", "has_ability"]


def _load(fn: str) -> dict:
    """Tolerant multi-path load (same contract as main/features): a missing file degrades to {}."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (fn, os.path.join(here, fn), os.path.join("/kaggle_simulations/agent", fn)):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


CF = _load("card_features.json")   # id(str) -> functional features (ct, stage, ex, hp, atks, tags, ty, prize)
CE = _load("card_effects.json")    # id(str) -> decoded effects (draw, search, energy_accel, ...)
CS = _load("card_stats.json")      # id(str) -> raw stats (n, hp, ex, mega, wk, rs, ty, atk)


def _slot_id(c):
    """A board/hand slot may be a card dict or a bare int id."""
    return c.get("id") if isinstance(c, dict) else c


# === card / entity identity ======================================================================

def card_identity(opt: dict, me_player: dict):
    """Canonical acting-card id for one legal option -- the verified join.

    PLAY (type 7) has no `area`; its `index` is a hand index (the landmine the old area==HAND-only
    check missed). attach(8)/evolve(9)/select-from-hand(3) carry area==HAND. ability/select on a
    board pokemon carry area ACTIVE/BENCH. attack(13)/retreat(12) act with the active pokemon.

    Returns an int card id or None. Mirrors tools/build_action_dataset.opt_card_id (golden-tested).
    """
    if not isinstance(opt, dict) or not me_player:
        return None
    t, idx, area = opt.get("type"), opt.get("index"), opt.get("area")
    hand = me_player.get("hand") or []
    active = me_player.get("active") or []
    bench = me_player.get("bench") or []
    discard = me_player.get("discard") or []

    def at(seq, i):
        return _slot_id(seq[i]) if isinstance(i, int) and 0 <= i < len(seq) else None

    if t == OptType.PLAY:          # implicit hand play; index is a hand index
        return at(hand, idx)
    if area == AreaType.HAND:      # attach / evolve / select-from-hand
        return at(hand, idx)
    if area == AreaType.ACTIVE:    # ability / select on the active
        return at(active, 0)
    if area == AreaType.BENCH:     # ability / select on a bench pokemon
        return at(bench, idx)
    if area == AreaType.DISCARD:   # select from discard
        return at(discard, idx)
    if t in (OptType.ATTACK, OptType.RETREAT):
        return at(active, 0)
    return None


# === semantic equivalence ========================================================================

def semantic_action_key(opt: dict, current: dict, me: int):
    """Canonical key: options sharing a key are strategically the SAME move. Includes the acting-card
    identity so two PLAY options for different hand cards stay distinct (the old type-only key
    collapsed them). Stable under option-order permutation. Mirrors build_action_dataset.opt_key.
    """
    players = current.get("players") or []
    me_player = players[me] if me < len(players) else {}
    cid = card_identity(opt, me_player)
    return (opt.get("type"), cid, opt.get("attackId"), opt.get("inPlayArea"), opt.get("inPlayIndex"))


def equivalence_classes(opts: list, current: dict, me: int) -> list:
    """Per-option equivalence-class index (first-seen order). opts that are non-dicts get class -1.
    Two options collapse iff their semantic_action_key matches."""
    seen: dict = {}
    out = []
    for o in opts:
        if not isinstance(o, dict):
            out.append(-1)
            continue
        k = semantic_action_key(o, current, me)
        if k not in seen:
            seen[k] = len(seen)
        out.append(seen[k])
    return out


# === action descriptor ===========================================================================

def target_entity(opt: dict) -> dict:
    """The board entity an option targets (for abilities/attacks/attaches that reference a slot)."""
    return {
        "in_play_area": opt.get("inPlayArea"),
        "in_play_index": opt.get("inPlayIndex"),
        "player_index": opt.get("playerIndex"),
    }


def action_descriptor(opt: dict, current: dict, me: int) -> dict:
    """Full semantic descriptor for one legal option: type, acting card + its stats/effects,
    attack params, target, resource consumed, and the equivalence key. Pure, JSON-serializable.
    """
    players = current.get("players") or []
    me_player = players[me] if me < len(players) else {}
    t = opt.get("type")
    cid = card_identity(opt, me_player)
    cf = CF.get(str(cid), {}) if cid is not None else {}
    ce = CE.get(str(cid), {}) if cid is not None else {}
    d = {
        "type": t if isinstance(t, int) else -1,
        "card_id": int(cid) if cid is not None else -1,
        "is_pokemon": 1 if cf.get("ct") == 0 else 0,
        "is_trainer": 1 if cf.get("ct") in (1, 2, 3, 4) else 0,
        "is_energy": 1 if cf.get("ct") in (5, 6) else 0,
        "is_basic": 1 if cf.get("stage") == "basic" else 0,
        "is_evolution": 1 if cf.get("stage") in ("stage1", "stage2") else 0,
        "is_ex_or_mega": 1 if (cf.get("ex") or cf.get("mega")) else 0,
        "attack_id": opt.get("attackId"),
        "target": target_entity(opt),
        "resource_consumed": _resource_consumed(opt, t, cf),
        "semantic_action_key": semantic_action_key(opt, current, me),
    }
    for k in EFFECT_KEYS:
        d["effect_" + k] = float(ce.get(k, 0) or 0)
    return d


def _resource_consumed(opt: dict, t, cf: dict) -> dict:
    """Best-effort description of what a move spends (the substrate B can learn irreversibility on)."""
    return {
        "spends_hand_card": 1 if t in (OptType.PLAY, OptType.ATTACH, OptType.EVOLVE) else 0,
        "attaches_energy": 1 if t == OptType.ATTACH else 0,
        "is_supporter": 1 if cf.get("ct") == 2 else 0,   # ct: 2 = supporter (per card_features)
        "is_stadium": 1 if cf.get("ct") == 4 else 0,
        "uses_attack": 1 if t == OptType.ATTACK else 0,
        "retreats": 1 if t == OptType.RETREAT else 0,
    }


# === structural state (orderless zones as multisets) =============================================

def _energy_multiset(pkmn: dict) -> dict:
    """Attached energy as a {letter: count} multiset (type-aware; orderless)."""
    letters = [ENERGY_LETTER.get(e, "C") for e in (pkmn.get("energies") or [])]
    return dict(Counter(letters))


def _entity(pkmn) -> dict | None:
    """Structured descriptor for one in-play pokemon (identity, hp, attached energy, evo stack)."""
    if not isinstance(pkmn, dict):
        return None
    return {
        "card_id": _slot_id(pkmn),
        "hp": pkmn.get("hp", 0),
        "energy": _energy_multiset(pkmn),
        "n_energy": len(pkmn.get("energies") or []),
        "tools": [_slot_id(t) for t in (pkmn.get("tools") or [])],
        "evolution_stack": [_slot_id(e) for e in (pkmn.get("preEvolution") or [])],
    }


def _multiset_ids(cards) -> dict:
    """{card_id: count} for an orderless zone (hand/discard/bench-as-ids)."""
    return dict(Counter(_slot_id(c) for c in (cards or []) if _slot_id(c) is not None))


def _player_state(p: dict) -> dict:
    active = (p.get("active") or [None])
    return {
        "active": _entity(active[0]) if active and active[0] else None,
        "bench": [_entity(b) for b in (p.get("bench") or []) if isinstance(b, dict)],
        "hand_multiset": _multiset_ids(p.get("hand")),
        "hand_count": p.get("handCount", len(p.get("hand") or [])),
        "discard_multiset": _multiset_ids(p.get("discard")),
        "prize_count": len(p.get("prize") or []),
        "deck_count": p.get("deckCount", 0),
        "status": {c: 1 for c in ("poisoned", "burned", "asleep", "paralyzed", "confused")
                   if p.get(c)},
    }


def semantic_state(obs: dict, history: list | None = None) -> dict:
    """Structural, perspective-aware state: both players' zones as multisets/entities, public
    per-turn resource flags, and an optional recent public action/event history (populated by the
    dataset builder from the replay step sequence; see public_history). Orderless zones are
    multisets, never positional vectors."""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    me = cur.get("yourIndex", 0)
    opp = 1 - me
    return {
        "schema_version": SCHEMA_VERSION,
        "your_index": me,
        "turn": cur.get("turn"),
        "result": cur.get("result", -1),
        "me": _player_state(players[me]) if me < len(players) else {},
        "opp": _player_state(players[opp]) if opp < len(players) else {},
        "public": {
            "stadium": cur.get("stadium"),
            "supporter_played": 1 if cur.get("supporterPlayed") else 0,
            "stadium_played": 1 if cur.get("stadiumPlayed") else 0,
            "energy_attached": 1 if cur.get("energyAttached") else 0,
            "retreated": 1 if cur.get("retreated") else 0,
        },
        "history": list(history or []),
    }


def public_history(steps: list, upto_step: int, me: int, max_events: int = 12) -> list:
    """Recent public action/event history from a replay step sequence, up to (not including)
    upto_step, as (player, option_type, card_id) triples. Public info only. The fixture builder
    passes this into semantic_state so the representation has a defined history slot."""
    out = []
    for s in steps[:upto_step]:
        for ai, agent in enumerate(s):
            if not isinstance(agent, dict):
                continue
            act = agent.get("action")
            if not isinstance(act, list) or len(act) != 1:
                continue
            obs = agent.get("observation") or {}
            opts = (obs.get("select") or {}).get("option") or []
            i = act[0]
            if not (isinstance(i, int) and 0 <= i < len(opts)) or not isinstance(opts[i], dict):
                continue
            cur = obs.get("current") or {}
            pl = (cur.get("players") or [{}])
            mp = pl[ai] if ai < len(pl) else {}
            out.append((ai, opts[i].get("type"), card_identity(opts[i], mp)))
    return out[-max_events:]


# === canonical encoding + deck identity ==========================================================

# the shared L1 feature vector both trainer and live MUST use (proves train/serve parity, P3)
FEATURE_KEYS = FT.FEATURE_KEYS


def encode_vector(obs: dict) -> list:
    """THE canonical numeric encoding shared by trainer and live inference: the L1 47-dim vector.
    Any encoding parity test compares this against itself across the trainer and serve paths."""
    return FT.vectorize(FT.encode_state(obs))


def deck_signature(deck: list) -> dict:
    """Order-independent identity of a 60-card deck (archetype id where allowed)."""
    sig = tuple(sorted(int(c) for c in deck))
    h = hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:16]
    return {"hash": h, "n_cards": len(sig), "n_distinct": len(set(sig))}


def is_single_pick_decision(obs: dict) -> bool:
    """A genuine single-pick decision with a real choice (maxCount==1 and >=2 legal options)."""
    sel = obs.get("select") or {}
    return (sel.get("maxCount") or 0) == 1 and len(sel.get("option") or []) >= 2


if __name__ == "__main__":
    print(f"schema {SCHEMA_VERSION}")
    print(f"  major action types: {MAJOR_ACTION_TYPES}")
    print(f"  card DBs loaded: features={len(CF)} effects={len(CE)} stats={len(CS)}")
    print(f"  L1 feature dim: {len(FEATURE_KEYS)}")
