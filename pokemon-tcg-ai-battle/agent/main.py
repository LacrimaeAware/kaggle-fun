"""cabt agents: `agent` (heuristic), `agent_search` (forward-model search, hand eval),
`agent_search_v` (search with the learned tree value). STRONGEST = agent_search; ship that.

Measured findings (cabt_arena.py, same deck both sides, real engine; 95% Wilson CIs):
  - Hand-scoring EVERY option by board value LOST to first_agent (0.44/200): net harmful.
  - The feature heuristic (keep default order; override only for a lethal/KO, go first, attach
    energy to an energy-short active) TIES first_agent (0.513/300). Board-aware hand rules do
    not beat the baseline.
  - Forward model CONFIRMED local (cg.dll search_begin/search_step; H001). `agent_search` does
    1-ply lookahead with determinization-averaged rollouts + aggressive opponent reply; the hand
    leaf eval. It BEATS first_agent (0.585/800, CI [0.551,0.619]) and edges the heuristic
    (0.543/300, CI [0.487,0.599]). This is our best agent.
  - `agent_search_v` swaps in a learned gradient-boosted value (AUC 0.735) at the leaves. It
    LOSES: 0.427/400 vs the heuristic, CI [0.380,0.476] (entirely below 0.5); 0.467 vs hand
    search. A tree MC value with high global AUC is a worse 1-ply leaf eval than the hand eval,
    because global classification != local ranking of nearby candidate leaves (the tree is
    piecewise-constant; the hand eval's continuous prize/HP gradient ranks siblings finely).
    Kept for the record / future deeper-search work, not shipped. See registry H023.

All agents are always legal and never raise (mirror validation forfeits on any exception).
Stats/features bundled (card_stats.json, attack_stats.json, features.py, card_features.json).
"""
from __future__ import annotations

import json
import os

import features as FT        # L1 state encoder (card_features.json bundled alongside)

ATTACK, YES, ATTACH = 13, 1, 8   # OptionType
A_ACTIVE = 4                 # AreaType.ACTIVE (attach target)
IS_FIRST_CTX = 41            # SelectContext.IS_FIRST


def _load(fn: str) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (fn, os.path.join(here, fn), os.path.join("/kaggle_simulations/agent", fn)):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


CDB = _load("card_stats.json")     # id(str) -> {n,hp,ex,mega,wk,rs,ty,atk:[...]}
ATK = _load("attack_stats.json")   # attackId(str) -> {d(dmg),c(cost),n(name)}

DECK = (
    [721] * 2 + [722] * 4 + [723] * 4 + [1092] + [1121] * 2 + [1145] * 2
    + [1163] * 2 + [1219] * 4 + [1227] * 4 + [1262] * 2 + [3] * 33
)


def _cs(cid) -> dict:
    return CDB.get(str(cid), {})


def _prize_value(cid) -> int:
    c = _cs(cid)
    return 3 if c.get("mega") else 2 if c.get("ex") else 1


def _active(p):
    a = p.get("active") or []
    return a[0] if a and a[0] else None


def _attack_value(o, me, opp) -> float:
    """Value of an ATTACK option. >= 8000 means it knocks out the opponent's active (or wins)."""
    dmg = ATK.get(str(o.get("attackId")), {}).get("d", 0)
    opA = _active(opp)
    if not opA:
        return 1500 + dmg
    myA = _active(me)
    oc = _cs(opA.get("id"))
    myty = _cs(myA.get("id")).get("ty", "") if myA else ""
    if oc.get("wk") and myty and oc["wk"] == myty:
        dmg *= 2
    if oc.get("rs") and myty and oc["rs"] == myty:
        dmg = max(0, dmg - 30)
    ohp = opA.get("hp", 0)
    if dmg >= ohp and ohp > 0:                       # knockout
        gained = _prize_value(opA.get("id"))
        if gained >= len(me.get("prize") or []):     # this KO wins the game
            return 90000.0
        return 8000.0 + gained * 1000 + dmg
    return 2000.0 + dmg


def _choose(obs: dict) -> list[int]:
    sel = obs.get("select")
    if sel is None:
        return list(DECK)
    opts = sel.get("option") or []
    k = sel.get("maxCount") or 0
    mn = sel.get("minCount") or 0
    n = len(opts)
    if n == 0 or k <= 0:
        return []
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    yi = cur.get("yourIndex", 0)
    me = players[yi] if yi < len(players) else {}
    opp = players[1 - yi] if len(players) > 1 else {}

    if k == 1:
        attacks = [(i, o) for i, o in enumerate(opts) if o.get("type") == ATTACK]
        if attacks:
            i, o = max(attacks, key=lambda io: _attack_value(io[1], me, opp))
            if _attack_value(o, me, opp) >= 8000:        # take a knockout / game-winning attack
                return [i]
        # feature-informed: if my active still needs energy for its attack and the manual
        # attachment is unused, attach to the ACTIVE attacker rather than a default target.
        f = FT.encode_state(obs)
        if f.get("active_energy_short", 0) > 0 and not f.get("energy_attach_done"):
            for i, o in enumerate(opts):
                if o.get("type") == ATTACH and o.get("inPlayArea") == A_ACTIVE:
                    return [i]
        if sel.get("context") == IS_FIRST_CTX:           # go first (small real edge)
            for i, o in enumerate(opts):
                if o.get("type") == YES:
                    return [i]

    return list(range(max(min(k, n), min(mn, n))))       # else: the safe default order


def agent(obs: dict) -> list[int]:
    """Always returns a legal selection; never raises."""
    try:
        return _choose(obs)
    except Exception:
        sel = obs.get("select")
        if sel is None:
            return list(DECK)
        n = len((sel or {}).get("option") or [])
        k = (sel or {}).get("minCount") or 1
        return list(range(min(max(k, 1), n))) if n else []


def agent_search(obs: dict) -> list[int]:
    """Forward-model search with the HAND leaf eval; heuristic elsewhere. Never raises."""
    try:
        if obs.get("select") is None:
            return list(DECK)
        import search
        mv = search.best_option(obs, DECK, use_learned=False)
        if mv is not None:
            return mv
    except Exception:
        pass
    return agent(obs)


def agent_search_v(obs: dict) -> list[int]:
    """Forward-model search with the LEARNED value at the leaves (L2); heuristic elsewhere."""
    try:
        if obs.get("select") is None:
            return list(DECK)
        import search
        mv = search.best_option(obs, DECK, use_learned=True)
        if mv is not None:
            return mv
    except Exception:
        pass
    return agent(obs)
