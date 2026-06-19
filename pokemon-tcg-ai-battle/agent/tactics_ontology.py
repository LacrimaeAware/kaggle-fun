"""Branch A / Tactic Miner V1 -- tactical ontology + per-option classifier + entity properties.

Maps each legal option to a tactic type and exposes target/entity properties, so the miner can compare a
strong player's CHOSEN action against its legal siblings IN CONTEXT (not as an unconditional rule). Pure,
no engine. Consumed by tools/tactic_miner.py and the state-conditioned search priors.
"""
from __future__ import annotations

import json
import os

import state_action_schema_v2 as SCH
import features as FT

TACTICS = [
    "attack", "ko", "gust", "switch", "draw", "tutor", "attach", "accelerate", "evolve",
    "ability_unlock", "heal", "disruption", "retreat", "develop_board", "preserve_attacker",
    "prepare_replacement", "select", "play", "end", "other",
]


def _load(fn: str) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (fn, os.path.join(here, fn), os.path.join("/kaggle_simulations/agent", fn)):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


CE = _load("card_effects.json")    # decoded effects per card id
CF = _load("card_features.json")   # functional features (stage, tags, prize, hp, atks, ct)
ATK = _load("attack_stats.json")   # attackId -> {d(dmg), c(cost), n}


def _cf(cid):
    return CF.get(str(cid), {}) if cid is not None else {}


def _ce(cid):
    return CE.get(str(cid), {}) if cid is not None else {}


def _active(p):
    a = (p or {}).get("active") or []
    return a[0] if a and a[0] else None


def _attack_kos(o: dict, me: dict, opp: dict) -> bool:
    """Does this ATTACK option knock out the opponent's active right now? (type-aware weakness/resist.)"""
    dmg = ATK.get(str(o.get("attackId")), {}).get("d", 0) or 0
    opA = _active(opp)
    if not opA:
        return dmg > 0
    myA = _active(me)
    oc = _cf(opA.get("id"))
    myty = _cf((myA or {}).get("id")).get("ty", "")
    if oc.get("wk") and myty and oc.get("wk") == myty:
        dmg *= 2
    if oc.get("rs") and myty and oc.get("rs") == myty:
        dmg = max(0, dmg - 30)
    ohp = opA.get("hp", 0) or 0
    return dmg >= ohp and ohp > 0


def classify_tactic(o: dict, obs: dict, me_player: dict) -> str:
    """Primary tactic label for one legal option. Best-effort; uses option type + decoded card effects."""
    if not isinstance(o, dict):
        return "other"
    t = o.get("type")
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    me = cur.get("yourIndex", 0)
    opp = players[1 - me] if len(players) > 1 else {}
    cid = SCH.card_identity(o, me_player)
    ce, cf = _ce(cid), _cf(cid)

    if t == SCH.OptType.ATTACK:
        return "ko" if _attack_kos(o, me_player, opp) else "attack"
    if t == SCH.OptType.RETREAT:
        return "retreat"
    if t == SCH.OptType.EVOLVE:
        return "evolve"
    if t == SCH.OptType.END:
        return "end"
    if t == SCH.OptType.ABILITY:
        if ce.get("switch_gust"):
            return "gust"
        if ce.get("draw"):
            return "draw"
        if ce.get("energy_accel"):
            return "accelerate"
        if ce.get("heal"):
            return "heal"
        if ce.get("disrupt"):
            return "disruption"
        return "ability_unlock"
    if t == SCH.OptType.ATTACH:
        return "accelerate" if ce.get("energy_accel") else "attach"
    if t == SCH.OptType.SELECT_CARD:
        if o.get("area") == SCH.AreaType.BENCH and o.get("playerIndex") not in (None, me):
            return "gust"          # picking an opponent benched mon to drag active
        return "select"
    if t == SCH.OptType.PLAY:
        if ce.get("switch_gust"):
            return "gust"
        if ce.get("search_to_bench"):
            return "develop_board"
        if ce.get("draw"):
            return "draw"
        if ce.get("search"):
            return "tutor"
        if ce.get("energy_accel"):
            return "accelerate"
        if ce.get("heal"):
            return "heal"
        if ce.get("disrupt"):
            return "disruption"
        if ce.get("recover_discard"):
            return "tutor"
        if cf.get("ct") == 0:      # a Pokemon -> board development
            return "develop_board"
        return "play"
    return "other"


def entity_properties(o: dict, obs: dict, me_player: dict) -> dict:
    """Target/entity properties for one option: the acting card and (for gust/attacks) the target."""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    me = cur.get("yourIndex", 0)
    opp = players[1 - me] if len(players) > 1 else {}
    cid = SCH.card_identity(o, me_player)
    cf = _cf(cid)
    props = {
        "acting_card_id": cid if cid is not None else -1,
        "acting_stage": cf.get("stage"),
        "acting_prize_value": cf.get("prize", 1),
        "acting_engine_role": next((r for r in ("draw", "tutor", "gust", "energy_accel", "ability_engine")
                                    if r in (cf.get("tags") or [])), None),
        "target_hp": None, "target_ko_able": None, "target_prize_value": None,
        "target_energy": None, "target_stage": None,
    }
    # gust target = an opponent benched mon at o.index
    if o.get("type") == SCH.OptType.SELECT_CARD and o.get("area") == SCH.AreaType.BENCH \
            and o.get("playerIndex") not in (None, me):
        bench = opp.get("bench") or []
        idx = o.get("index")
        tgt = bench[idx] if isinstance(idx, int) and 0 <= idx < len(bench) else None
        if tgt:
            myA = _active(me_player)
            my_dmg = FT._best_affordable(myA)[0] if myA else 0
            hp = tgt.get("hp", 0) or 0
            tcf = _cf(tgt.get("id"))
            props.update(target_hp=hp, target_ko_able=int(my_dmg > 0 and my_dmg >= hp),
                         target_prize_value=tcf.get("prize", 1),
                         target_energy=len(tgt.get("energies") or []), target_stage=tcf.get("stage"))
    return props


def context_features(obs: dict) -> dict:
    """Compact, discrete root-state context the miner buckets patterns by (from the L1 encoder)."""
    f = FT.encode_state(obs)
    cur = obs.get("current") or {}
    return {
        "can_ko_now": int(f.get("can_ko_opp_now", 0) > 0),
        "active_can_attack": int(f.get("active_can_attack_now", 0) > 0),
        "active_energy_short": int(f.get("active_energy_short", 0) > 0),
        "prize_lead_bucket": ("ahead" if f.get("prize_lead", 0) > 0 else "behind" if f.get("prize_lead", 0) < 0 else "even"),
        "my_prizes_left": int(f.get("my_prizes_left", 6)),
        "hand_low": int(f.get("hand_size", 7) <= 3),
        "deckout_risk": int(f.get("deckout_risk", 0) > 0),
        "early": int((cur.get("turn", 0) or 0) <= 6),
        "supporter_available": int(f.get("supporter_available", 0) > 0),
        "fixing_available": int(f.get("fixing_available", 0) > 0),
    }


if __name__ == "__main__":
    print(f"tactic ontology: {len(TACTICS)} tactics; effects={len(CE)} cards, features={len(CF)} cards")
