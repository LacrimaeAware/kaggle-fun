"""Heuristic Search V2 -- FAIL-CLOSED proposals.

Contract (do not violate):
  * heuristics only PROPOSE alternative candidate selections; they never pick the final move;
  * the baseline/default selection is ALWAYS a candidate and is chosen unless a proposal is
    clearly better, or is statistically indistinguishable and used only as a tie-break;
  * unknown contexts return [] (no change);
  * any failure (incomplete coverage, exception, identity unresolved, low time) -> default.

This module is pure: it loads card data and operates on the obs dict. The search VALIDATOR
(`compare_selections`) is given the isolated search/eval modules so it never hard-couples to a
specific copy, and so it can never bypass search -- it IS search, restricted to the candidates.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

# ---- OptionType / AreaType / SelectContext (from cg/api.py) ----
NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY = 0, 1, 2, 3, 4, 5, 6
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END = 7, 8, 9, 10, 11, 12, 13, 14
A_DECK, A_HAND, A_DISCARD, A_ACTIVE, A_BENCH, A_PRIZE = 1, 2, 3, 4, 5, 6
CTX_TO_BENCH = 5
CTX_TO_HAND = 7
CTX_IS_FIRST = 41

# ---- deck card ids ----
BASIC_PSYCHIC, ENRICHING = 5, 13
TELEPATH = 19
DUNSPARCE_IDS = {65, 305}
DUDUNSPARCE = 66
ABRA, KADABRA, ALAKAZAM = 741, 742, 743
RARE_CANDY, ENH_HAMMER, POFFIN = 1079, 1081, 1086
NIGHT_STRETCHER, SACRED_ASH, POKE_PAD = 1097, 1129, 1152
BOSS, LANA_AID, HILDA, DAWN, BATTLE_CAGE = 1182, 1184, 1225, 1231, 1264


def _load(fn: str) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (fn, os.path.join(here, fn), os.path.join("/kaggle_simulations/agent", fn)):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


CDB = _load("card_stats.json")
CF = _load("card_features.json")
ATK = _load("attack_stats.json")


def _meta(cid) -> dict:
    return CDB.get(str(cid), {}) or CF.get(str(cid), {}) or {}


def _is_pokemon(cid) -> bool:
    return bool(_meta(cid).get("hp"))


# ---------------- obs helpers (dict form, live path) ----------------
def _cur(obs):
    return obs.get("current") or {}


def _sel(obs):
    return obs.get("select")


def _players(cur):
    return cur.get("players") or []


def _me(cur):
    return cur.get("yourIndex", 0) or 0


def _player(cur, i):
    ps = _players(cur)
    return ps[i] if 0 <= i < len(ps) else {}


def _active(p):
    a = p.get("active") or []
    return a[0] if a and a[0] else None


def _bench(p):
    return [x for x in (p.get("bench") or []) if x]


def _hand(p):
    return p.get("hand") or []


def _cid(card):
    if isinstance(card, (int, float)):
        return int(card)
    if isinstance(card, dict):
        v = card.get("id")
        return int(v) if v is not None else None
    return None


def _inplay_ids(p):
    out = []
    a = _active(p)
    if a:
        out.append(_cid(a))
    out += [_cid(x) for x in _bench(p)]
    return [x for x in out if x is not None]


def _hand_ids(p):
    return [x for x in (_cid(c) for c in _hand(p)) if x is not None]


def _bench_space(p):
    return max(0, int(p.get("benchMax", 5) or 5) - len(_bench(p)))


def line_state(p) -> dict:
    board = Counter(_inplay_ids(p))
    hand = Counter(_hand_ids(p))
    duns = sum(board[x] for x in DUNSPARCE_IDS)
    return {
        "abra_board": board[ABRA], "kadabra_board": board[KADABRA], "alakazam_board": board[ALAKAZAM],
        "dunsparce_board": duns, "dudunsparce_board": board[DUDUNSPARCE],
        "alakazam_hand": hand[ALAKAZAM], "rare_candy_hand": hand[RARE_CANDY],
    }


# ---------------- M0: mechanical fixes (no proposal needed) ----------------
def powerful_hand_damage(obs, me: int) -> float:
    """Alakazam Powerful Hand = 20 * current hand size, when my active is Alakazam."""
    cur = _cur(obs)
    P = _player(cur, me)
    if _cid(_active(P)) != ALAKAZAM:
        return 0.0
    hand = P.get("handCount")
    if hand is None:
        hand = len(_hand(P))
    return 20.0 * float(hand or 0)


def attack_value_m0(option, obs, me: int) -> float:
    """Baseline _attack_value semantics plus dynamic Powerful Hand. >=90000 => confirmed game win."""
    cur = _cur(obs)
    P, O = _player(cur, me), _player(cur, 1 - me)
    opA = _active(O)
    static = float((ATK.get(str(option.get("attackId")), {}) or {}).get("d", 0) or 0)
    ph = powerful_hand_damage(obs, me)
    dmg = ph if (ph > 0 and static <= 0) else static
    if not opA:
        return 1500.0 + dmg
    oc = _meta(_cid(opA))
    myty = _meta(_cid(_active(P))).get("ty", "") if _active(P) else ""
    if oc.get("wk") and myty and oc["wk"] == myty:
        dmg *= 2.0
    if oc.get("rs") and myty and oc["rs"] == myty:
        dmg = max(0.0, dmg - 30.0)
    ohp = float(opA.get("hp", 0) or 0)
    if ohp > 0 and dmg >= ohp:
        gained = 3 if oc.get("mega") else 2 if oc.get("ex") else 1
        if gained >= len(P.get("prize") or []):
            return 90000.0 + gained * 1000 + dmg
        return 8000.0 + gained * 1000 + dmg
    return 2000.0 + dmg


def forced_move_m0(obs):
    """M0 forced rule: force ONLY a confirmed game-ending prize attack; go first. Ordinary KOs
    fall through to search."""
    sel = _sel(obs)
    if not sel or (sel.get("maxCount") or 0) != 1:
        return None
    opts = sel.get("option") or []
    if not opts:
        return None
    cur = _cur(obs)
    me = _me(cur)
    attacks = [(i, o) for i, o in enumerate(opts) if o.get("type") == ATTACK]
    if attacks:
        i, o = max(attacks, key=lambda io: attack_value_m0(io[1], obs, me))
        if attack_value_m0(o, obs, me) >= 90000.0:
            return [i]
    if sel.get("context") == CTX_IS_FIRST:
        for i, o in enumerate(opts):
            if o.get("type") == YES:
                return [i]
    return None


def forced_move_ko_phaware(obs):
    """Baseline forced rule (force ANY KO, go first) but with PH-AWARE attack value, so a lethal
    Powerful Hand is recognized and taken. Keeps auto-KO -- does NOT defer ordinary KOs to search."""
    sel = _sel(obs)
    if not sel or (sel.get("maxCount") or 0) != 1:
        return None
    opts = sel.get("option") or []
    if not opts:
        return None
    me = _me(_cur(obs))
    attacks = [(i, o) for i, o in enumerate(opts) if o.get("type") == ATTACK]
    if attacks:
        i, o = max(attacks, key=lambda io: attack_value_m0(io[1], obs, me))
        if attack_value_m0(o, obs, me) >= 8000.0:
            return [i]
    if sel.get("context") == CTX_IS_FIRST:
        for i, o in enumerate(opts):
            if o.get("type") == YES:
                return [i]
    return None


# ---------------- resolvers: PROPOSE only ----------------
def _deck_card_of_option(sel, option):
    """For a deck-search CARD option, the fetched card id = sel.deck[option.index]."""
    if option.get("type") != CARD or option.get("area") not in (A_DECK, None):
        return None
    deck = sel.get("deck") or []
    idx = option.get("index")
    if isinstance(idx, int) and 0 <= idx < len(deck):
        return _cid(deck[idx])
    return None


def _poffin_resolver(obs):
    """R1: a 'put Basic Pokemon onto Bench' prompt (context TO_BENCH, CARD options from deck).
    Propose a BALANCED setup: one Abra + one Dunsparce when both are legal and not yet on board.
    Returns candidate selections (lists of option indices) or []. Fail-closed: [] if unsure."""
    sel = _sel(obs)
    if not sel or sel.get("context") != CTX_TO_BENCH:
        return []
    opts = sel.get("option") or []
    k = int(sel.get("maxCount") or 0)
    if k < 2 or len(opts) < 2:
        return []
    # map each option to the basic it fetches
    by_card = {}
    for i, o in enumerate(opts):
        cid = _deck_card_of_option(sel, o)
        if cid is not None and _is_pokemon(cid):
            by_card.setdefault(cid, []).append(i)
    abra_i = (by_card.get(ABRA) or [None])[0]
    duns_i = next((by_card[c][0] for c in DUNSPARCE_IDS if c in by_card), None)
    if abra_i is None or duns_i is None or abra_i == duns_i:
        return []
    ls = line_state(_player(_cur(obs), _me(_cur(obs))))
    # only propose the balanced fetch when it actually develops missing lines
    if ls["abra_board"] and ls["dunsparce_board"]:
        return []
    return [sorted([abra_i, duns_i])]


_RESOLVERS = {"poffin": _poffin_resolver}


def propose(obs, enabled=None) -> list:
    """Return 0+ legal heuristic candidate selections. Never the final move. Never deletes the
    default. [] for unknown contexts. `enabled` = iterable of resolver names (default: all)."""
    out = []
    names = _RESOLVERS.keys() if enabled is None else [n for n in enabled if n in _RESOLVERS]
    for n in names:
        try:
            for cand in (_RESOLVERS[n](obs) or []):
                if cand and cand not in out:
                    out.append(list(cand))
        except Exception:
            continue
    return out


def default_selection(obs) -> list:
    """What the baseline default order picks for this prompt (mirrors main._choose's final line)."""
    sel = _sel(obs)
    if sel is None:
        return []
    opts = sel.get("option") or []
    k = int(sel.get("maxCount") or 0)
    mn = int(sel.get("minCount") or 0)
    n = len(opts)
    if n == 0 or k <= 0:
        return []
    return list(range(max(min(k, n), min(mn, n))))


# ---------------- the VALIDATOR: paired-world search over candidates ----------------
def _simulate_selection(A, root_id, selection, me, S):
    """Apply `selection` (a list of option indices) at the root, then roll out my turn + one
    opponent reply with the BASELINE rollout, and hand-eval at the start of my next turn."""
    st = A.search_step(root_id, list(selection))
    saw_opp = False
    for _ in range(S.DEPTH_CAP):
        ob = st.observation
        cur = ob.current
        if cur is not None and cur.result != -1:
            break
        sub = ob.select
        if sub is None:
            break
        my_move = cur is not None and cur.yourIndex == me
        if saw_opp and my_move:
            break
        if not my_move:
            saw_opp = True
        st = A.search_step(st.searchId, S._rollout_pick(sub, is_me=my_move))
    obs = S._obs_dict(st.observation)
    return S.EV.evaluate_obs(obs, me)


def compare_selections(obs, deck, candidates, default, S, time_budget=0.6, n_determ=8):
    """Choose among {default} + candidates by paired-world forward-model value. FAIL-CLOSED.

    Returns (chosen_selection, stats). chosen is always `default` unless a candidate's paired
    mean value is clearly higher (by > tie_eps) or indistinguishable-and-used-as-tiebreak.
    On ANY problem (no engine, exception, incomplete coverage, low budget) returns default.
    """
    stats = {"n_determ_done": 0, "candidates": len(candidates), "chosen": "default",
             "accepted": False, "tiebreak": False, "incomplete": False, "error": False}
    A = S._api()
    if A is None:
        stats["error"] = True
        return default, stats
    sel, cur = _sel(obs), _cur(obs)
    players = cur.get("players") or []
    if not sel or len(players) < 2:
        return default, stats
    me = _me(cur)
    P, O = players[me], players[1 - me]
    oa = O.get("active") or []
    if oa and oa[0] is None:        # face-down opp active -> don't search; fail closed
        return default, stats
    # all selections to score: default first
    sels = [list(default)] + [list(c) for c in candidates if list(c) != list(default)]
    if len(sels) < 2:
        return default, stats
    n_my_deck = P.get("deckCount", 0) or 0
    n_op_deck = O.get("deckCount", 0) or 0
    n_my_prize = len(P.get("prize") or [])
    n_op_prize = len(O.get("prize") or [])
    n_op_hand = O.get("handCount", 0) or 0
    obsd = A.to_observation_class(obs)
    nC = len(sels)
    sums = [0.0] * nC
    counts = [0] * nC
    t0 = time.time()
    for _ in range(n_determ):
        if time.time() - t0 > time_budget:
            break
        mp = S._hidden_pool(deck, P, exclude_hand=False)
        mp += [3] * max(0, (n_my_deck + n_my_prize) - len(mp))
        op = S._hidden_pool(deck, O, exclude_hand=True)
        op += [3] * max(0, (n_op_deck + n_op_prize + n_op_hand) - len(op))
        try:
            root = A.search_begin(
                obsd, your_deck=mp[:n_my_deck], your_prize=mp[n_my_deck:n_my_deck + n_my_prize],
                opponent_deck=op[n_op_hand + n_op_prize:n_op_hand + n_op_prize + n_op_deck],
                opponent_prize=op[n_op_hand:n_op_hand + n_op_prize], opponent_hand=op[:n_op_hand],
                opponent_active=[])
        except Exception:
            continue
        ok_world = True
        world_vals = [None] * nC
        for ci, s in enumerate(sels):
            try:
                world_vals[ci] = _simulate_selection(A, root.searchId, s, me, S)
            except Exception:
                ok_world = False
                break
        try:
            A.search_end()
        except Exception:
            pass
        if ok_world and all(v is not None for v in world_vals):   # paired: count world only if ALL scored
            for ci in range(nC):
                sums[ci] += world_vals[ci]
                counts[ci] += 1
        stats["n_determ_done"] += 1

    if any(c == 0 for c in counts):
        stats["incomplete"] = True
        return default, stats
    means = [sums[i] / counts[i] for i in range(nC)]
    base_mean = means[0]
    best_i = max(range(nC), key=lambda i: means[i])
    if best_i == 0:
        return default, stats
    tie_eps = 1.0    # hand-eval units; prize term is 1000, HP is 1.0 -> ~tie within a point
    if means[best_i] > base_mean + tie_eps:
        stats["chosen"] = "heuristic"
        stats["accepted"] = True
        return sels[best_i], stats
    # indistinguishable -> heuristic only as a tie-break
    if means[best_i] >= base_mean - tie_eps:
        stats["chosen"] = "heuristic"
        stats["tiebreak"] = True
        return sels[best_i], stats
    return default, stats
