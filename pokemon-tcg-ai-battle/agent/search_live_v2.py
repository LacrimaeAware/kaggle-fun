"""Branch A live-search candidate v2 -- agent_search + high-confidence tactical FLOORS.

Floors apply BEFORE search (like main._forced_move); search stays the authority on open decisions. They
patch places where the 1-ply hand-eval leaf is provably blind, not general tuning.

Floor 1 -- DRAW ENGINE. The leaf eval (eval.evaluate) scores prizes + board HP + active energy, with NO
card-advantage term, so drawing cards changes the leaf by ~0 and the search is indifferent to its own draw
abilities. Measured: of 174 non-forced decisions where a draw-engine ability was available, production
agent_search used it only 27 times (15%). This deck IS a draw engine (Dudunsparce/Kadabra/Alakazam), so we
fire available draw abilities (deckout-guarded) instead of leaving them unused.

This is a screening candidate on Branch A; it does not modify the production agent (main.py) and is not a
submission until a cheap A/B clears it.
"""
from __future__ import annotations

import json
import os

import main as M
import search as S
import state_action_schema_v2 as SCH
import features as FT
import tactics_ontology as TO

# draw-engine abilities in the DENPA92 deck (card ids; all carry a draw effect)
DRAW_ABILITY_IDS = {66, 742, 743}   # Dudunsparce, Kadabra, Alakazam
MIN_DECK_FOR_DRAW = 6               # deckout guard: don't dig when the deck is getting thin


def _draw_floor(obs: dict):
    """If a draw-engine ability is legal and we are not deckout-risky, fire it. [i] or None. Pure."""
    sel = obs.get("select") or {}
    if (sel.get("maxCount") or 0) != 1:
        return None
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if not players:
        return None
    me = cur.get("yourIndex", 0)
    mp = players[me] if me < len(players) else {}
    if (mp.get("deckCount", 0) or 0) < MIN_DECK_FOR_DRAW:
        return None
    for i, o in enumerate(sel.get("option") or []):
        if isinstance(o, dict) and o.get("type") == SCH.OptType.ABILITY and SCH.card_identity(o, mp) in DRAW_ABILITY_IDS:
            return [i]
    return None


def agent_search_draw(obs: dict):
    """agent_search + draw-engine floor. Order: lethal/go-first floor (main) -> draw floor -> search ->
    heuristic fallback. Never raises."""
    try:
        if obs.get("select") is None:
            return list(M.DECK)
        mv = M._forced_move(obs)          # lethal KO / go-first stays top priority
        if mv is not None:
            return mv
        mv = _draw_floor(obs)             # fire the under-used draw engine
        if mv is not None:
            return mv
        mv = S.best_option(obs, M.DECK, leaf_mode="hand")
        if mv is not None:
            return mv
    except Exception:
        pass
    return M.agent(obs)


# --- Floor 2: GUST TARGET (Boss's Orders). No self-cost, unlike the draw abilities. -----------------
def _gust_floor(obs: dict):
    """When choosing which opponent BENCHED Pokemon to drag active, pick the best target: one we can KO
    this turn (prefer highest prize), else the highest prize-value threat. [i] or None. Pure."""
    sel = obs.get("select") or {}
    if (sel.get("maxCount") or 0) != 1:
        return None
    opts = sel.get("option") or []
    dict_opts = [o for o in opts if isinstance(o, dict)]
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return None
    me = cur.get("yourIndex", 0)
    mine, opp = players[me], players[1 - me]
    gust = [(i, o) for i, o in enumerate(opts)
            if isinstance(o, dict) and o.get("type") == SCH.OptType.SELECT_CARD
            and o.get("area") == SCH.AreaType.BENCH and o.get("playerIndex") not in (None, me)]
    if len(gust) < 2 or len(gust) != len(dict_opts):   # only a clean "pick an opponent benched mon" select
        return None
    bench = opp.get("bench") or []
    myA = (mine.get("active") or [None])
    myA = myA[0] if myA else None
    my_dmg = FT._best_affordable(myA)[0] if myA else 0
    best = None
    for i, o in gust:
        idx = o.get("index")
        tgt = bench[idx] if isinstance(idx, int) and 0 <= idx < len(bench) else None
        if not tgt:
            continue
        hp = tgt.get("hp", 0) or 0
        pv = FT.cf(tgt.get("id")).get("prize", 1) or 1
        koable = 1 if (my_dmg > 0 and my_dmg >= hp) else 0
        score = (koable, pv, -hp)                       # KO-able first, then prize, then easiest KO
        if best is None or score > best[0]:
            best = (score, i)
    return [best[1]] if best else None


# --- Floor 3: EVOLVE our line. Kadabra/Alakazam draw on evolve (Psychic Draw); no self-sacrifice. ---
EVOLVE_TARGETS = {66, 742, 743}   # Dudunsparce, Kadabra, Alakazam


def _evolve_floor(obs: dict):
    """If we can evolve into one of our key evolutions (develops the attacker; Kadabra/Alakazam also draw
    via Psychic Draw), do it. [i] or None. Pure."""
    sel = obs.get("select") or {}
    if (sel.get("maxCount") or 0) != 1:
        return None
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if not players:
        return None
    me = cur.get("yourIndex", 0)
    mp = players[me] if me < len(players) else {}
    for i, o in enumerate(sel.get("option") or []):
        if isinstance(o, dict) and o.get("type") == SCH.OptType.EVOLVE and SCH.card_identity(o, mp) in EVOLVE_TARGETS:
            return [i]
    return None


def _floored(obs: dict, floors):
    """Shared wrapper: lethal/go-first floor (main) -> the given floors in order -> search -> heuristic."""
    try:
        if obs.get("select") is None:
            return list(M.DECK)
        mv = M._forced_move(obs)
        if mv is not None:
            return mv
        for f in floors:
            mv = f(obs)
            if mv is not None:
                return mv
        mv = S.best_option(obs, M.DECK, leaf_mode="hand")
        if mv is not None:
            return mv
    except Exception:
        pass
    return M.agent(obs)


def agent_search_gust(obs: dict):
    return _floored(obs, [_gust_floor])


def agent_search_evolve(obs: dict):
    return _floored(obs, [_evolve_floor])


def agent_search_tactical(obs: dict):
    """agent_search + gust-target + evolve-line floors (no draw floor; that one sacrificed board)."""
    return _floored(obs, [_gust_floor, _evolve_floor])


# --- Tactic Miner V1: SOFT state-conditioned prior. Acts ONLY on the search's near-ties (where A2 showed
#     the search is noisy/indifferent), breaking them toward the mined high-confidence pattern. Not a floor;
#     search decides whenever it has a clear value preference. -----------------------------------------
_MINED = None
ACCEPT_FRAC = 0.15          # an option is a "near-tie" with the best if within this frac of the value spread
MIN_MECH_CONF = 0.85        # only use clean-mechanics patterns (attack/ko/gust), not draw etc.


def _mined_lifts():
    global _MINED
    if _MINED is not None:
        return _MINED
    _MINED = {}
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "manifests",
                            "mined_tactics_v1.json")
        art = json.load(open(path, encoding="utf-8"))
        for p in art.get("context_patterns_ranked", []):
            if p.get("mechanics_confidence", 0) >= MIN_MECH_CONF and "=" in p["context"]:
                ck, cv = p["context"].split("=", 1)
                _MINED[(ck, cv, p["tactic"])] = p["lift"]
    except Exception:
        pass
    return _MINED


def _prior_score(o: dict, obs: dict, mp: dict) -> float:
    """How much strong players favor this option's tactic in the current context (sum of mined lift-1)."""
    mined = _mined_lifts()
    if not mined:
        return 0.0
    tactic = TO.classify_tactic(o, obs, mp)
    s = 0.0
    for ck, cv in TO.context_features(obs).items():
        lift = mined.get((ck, str(cv), tactic))
        if lift is not None:
            s += (lift - 1.0)
    return s


def agent_search_prior(obs: dict):
    """Search evaluates; the mined prior only breaks near-ties toward the strong-player tactic."""
    try:
        if obs.get("select") is None:
            return list(M.DECK)
        mv = M._forced_move(obs)
        if mv is not None:
            return mv
        ev = S.option_evals(obs, M.DECK, leaf_mode="hand")
        if ev:
            vals = [(e[0] if e else None) for e in ev]
            present = [v for v in vals if v is not None]
            if present:
                best, worst = max(present), min(present)
                margin = ACCEPT_FRAC * (best - worst) if best > worst else 0.0
                acceptable = [i for i, v in enumerate(vals) if v is not None and v >= best - margin]
                if len(acceptable) == 1:
                    return [acceptable[0]]
                cur = obs.get("current") or {}
                players = cur.get("players") or []
                me = cur.get("yourIndex", 0)
                mp = players[me] if me < len(players) else {}
                opts = obs["select"]["option"]
                best_i = max(acceptable, key=lambda i: (_prior_score(opts[i], obs, mp), vals[i]))
                return [best_i]
        mv = S.best_option(obs, M.DECK, leaf_mode="hand")
        if mv is not None:
            return mv
    except Exception:
        pass
    return M.agent(obs)
