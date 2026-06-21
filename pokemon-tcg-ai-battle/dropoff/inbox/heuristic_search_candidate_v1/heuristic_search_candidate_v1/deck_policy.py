"""Deck-specific policy support for Dudunsparce / Alakazam search.

This module is deliberately conservative:

* root decisions remain search-authoritative;
* deck heuristics only order candidates and break exact search ties;
* selection prompts created by tutors, gust, recovery, evolution and multi-select
  are handled here because the baseline otherwise chooses option zero;
* rollout choices use the same policy so search evaluates coherent continuations.

Every public function is crash-safe and returns ``None`` when it cannot make a
confident, legal choice.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from collections import Counter
from typing import Any, Iterable

import features as FT

# OptionType
NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY = range(7)
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END = 7, 8, 9, 10, 11, 12, 13, 14

# Common AreaType values observed in the replay corpus.
A_DECK, A_HAND, A_PRIZE, A_ACTIVE, A_BENCH, A_DISCARD = 1, 2, 3, 4, 5, 6

# Deck card ids. Support both Dunsparce printings seen in the project.
ABRA = 741
KADABRA = 742
ALAKAZAM = 743
DUNSPARCE_IDS = {65, 305}
DUDUNSPARCE = 66
RARE_CANDY = 1079
ENHANCED_HAMMER = 1081
POFFIN = 1086
NIGHT_STRETCHER = 1097
SACRED_ASH = 1129
POKE_PAD = 1152
BOSS = 1182
LANA_AID = 1184
HILDA = 1225
DAWN = 1231
BATTLE_CAGE = 1264
BASIC_PSYCHIC = 5
TELEPATH_PSYCHIC = 19
ENRICHING_ENERGY = 13

PSYCHIC_LINE = {ABRA, KADABRA, ALAKAZAM}
DUNSPARCE_LINE = DUNSPARCE_IDS | {DUDUNSPARCE}
ENERGY_IDS = {BASIC_PSYCHIC, TELEPATH_PSYCHIC, ENRICHING_ENERGY}


def _load(fn: str) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (fn, os.path.join(here, fn), os.path.join('/kaggle_simulations/agent', fn)):
        try:
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            continue
    return {}


CDB = _load('card_stats.json')
CF = _load('card_features.json')
ATK = _load('attack_stats.json')
CEFF = _load('card_effects.json')


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        return dataclasses.asdict(obj)
    except Exception:
        out = {}
        for key in ('current', 'select'):
            val = getattr(obj, key, None)
            if val is not None:
                out[key] = _as_dict(val)
        return out


def _items(value: Any) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return []


def _current(obs: Any) -> Any:
    return _get(obs, 'current', {}) or {}


def _selection(obs: Any) -> Any:
    return _get(obs, 'select', None)


def _players(cur: Any) -> list:
    return _items(_get(cur, 'players', []))


def _perspective(cur: Any) -> int:
    try:
        return int(_get(cur, 'yourIndex', 0) or 0)
    except Exception:
        return 0


def _player(cur: Any, idx: int) -> Any:
    ps = _players(cur)
    return ps[idx] if 0 <= idx < len(ps) else {}


def _active(player: Any) -> Any:
    arr = _items(_get(player, 'active', []))
    return arr[0] if arr and arr[0] else None


def _bench(player: Any) -> list:
    return [x for x in _items(_get(player, 'bench', [])) if x]


def _hand(player: Any) -> list:
    return _items(_get(player, 'hand', []))


def _card_id(card: Any):
    if card is None:
        return None
    if isinstance(card, (int, float)):
        return int(card)
    value = _get(card, 'id', None)
    try:
        return int(value) if value is not None else None
    except Exception:
        return value


def _card_name(cid) -> str:
    if cid is None:
        return ''
    row = CDB.get(str(cid), {}) or CF.get(str(cid), {}) or {}
    return str(row.get('n') or row.get('name') or '')


def _normalise_name(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', (text or '').lower()).strip()


def _card_meta(cid) -> dict:
    if cid is None:
        return {}
    out = {}
    out.update(CF.get(str(cid), {}) or {})
    out.update(CDB.get(str(cid), {}) or {})
    return out


def _prize_value(cid) -> int:
    c = _card_meta(cid)
    if c.get('mega'):
        return 3
    if c.get('ex'):
        return 2
    return int(c.get('prize', 1) or 1)


def _attached_count(entity: Any) -> int:
    raw = _get(entity, 'energies', None)
    if raw is not None:
        return len(_items(raw))
    return len(_items(_get(entity, 'energyCards', [])))


def _entity_hp(entity: Any) -> float:
    return float(_get(entity, 'hp', 0) or 0)


def _entity_id(entity: Any):
    return _card_id(entity)


def _inplay_ids(player: Any) -> list:
    out = []
    a = _active(player)
    if a:
        out.append(_entity_id(a))
    out.extend(_entity_id(x) for x in _bench(player))
    return [x for x in out if x is not None]


def _hand_ids(player: Any) -> list:
    return [x for x in (_card_id(c) for c in _hand(player)) if x is not None]


def _discard_ids(player: Any) -> list:
    return [x for x in (_card_id(c) for c in _items(_get(player, 'discard', []))) if x is not None]


def _zone(player: Any, area: Any) -> list:
    try:
        area = int(area)
    except Exception:
        return []
    if area == A_DECK:
        return _items(_get(player, 'deck', []))
    if area == A_HAND:
        return _hand(player)
    if area == A_PRIZE:
        return _items(_get(player, 'prize', []))
    if area == A_ACTIVE:
        return [x for x in _items(_get(player, 'active', [])) if x]
    if area == A_BENCH:
        return _bench(player)
    if area == A_DISCARD:
        return _items(_get(player, 'discard', []))
    # Some search observations expose other named zones. Try them conservatively.
    for key in ('lostZone', 'stadium'):
        vals = _items(_get(player, key, []))
        if vals:
            return vals
    return []


def option_card_id(option: Any, observation: Any, perspective: int | None = None):
    """Best-effort acting/selected card identity for an option."""
    for key in ('cardId', 'card_id', 'pokemonId', 'energyId', 'id'):
        cid = _get(option, key, None)
        if cid is not None:
            try:
                return int(cid)
            except Exception:
                return cid

    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    pidx = _get(option, 'playerIndex', me)
    try:
        pidx = int(pidx)
    except Exception:
        pidx = me
    player = _player(cur, pidx)
    idx = _get(option, 'index', None)
    try:
        idx = int(idx)
    except Exception:
        return None

    typ = _get(option, 'type', None)
    # PLAY has no area in replay observations: index is directly into hand.
    if typ == PLAY and _get(option, 'area', None) is None:
        cards = _hand(player)
    else:
        cards = _zone(player, _get(option, 'area', None))
        if not cards and typ in (PLAY, ATTACH, EVOLVE):
            cards = _hand(player)
    if 0 <= idx < len(cards):
        return _card_id(cards[idx])
    return None


def option_target_entity(option: Any, observation: Any, perspective: int | None = None):
    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    pidx = _get(option, 'playerIndex', me)
    try:
        pidx = int(pidx)
    except Exception:
        pidx = me
    player = _player(cur, pidx)
    area = _get(option, 'inPlayArea', None)
    idx = _get(option, 'inPlayIndex', _get(option, 'targetIndex', _get(option, 'index', None)))
    try:
        idx = int(idx)
    except Exception:
        return None
    cards = _zone(player, area)
    return cards[idx] if 0 <= idx < len(cards) else None


def _attack_static(option: Any) -> tuple[float, str]:
    aid = _get(option, 'attackId', None)
    row = ATK.get(str(aid), {}) or {}
    return float(row.get('d', 0) or 0), str(row.get('n') or row.get('name') or '')


def attack_damage(option: Any, observation: Any, perspective: int | None = None,
                  apply_type: bool = True) -> float:
    """Estimated attack damage, including Alakazam's dynamic Powerful Hand."""
    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    P, O = _player(cur, me), _player(cur, 1 - me)
    attacker, defender = _active(P), _active(O)
    dmg, name = _attack_static(option)
    aid = _entity_id(attacker)
    norm = _normalise_name(name)
    is_powerful = aid == ALAKAZAM and ('powerful hand' in norm or dmg <= 0)
    if is_powerful:
        hand_count = _get(P, 'handCount', None)
        if hand_count is None:
            hand_count = len(_hand(P))
        return max(0.0, 20.0 * float(hand_count or 0))

    if not apply_type or not attacker or not defender:
        return dmg
    ac, dc = _card_meta(_entity_id(attacker)), _card_meta(_entity_id(defender))
    aty = str(ac.get('ty', '') or '')
    if dc.get('wk') and aty and dc.get('wk') == aty:
        dmg *= 2.0
    if dc.get('rs') and aty and dc.get('rs') == aty:
        dmg = max(0.0, dmg - 30.0)
    return dmg


def attack_value(option: Any, observation: Any, perspective: int | None = None) -> float:
    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    P, O = _player(cur, me), _player(cur, 1 - me)
    defender = _active(O)
    dmg = attack_damage(option, observation, me)
    if not defender:
        return 1500.0 + dmg
    hp = _entity_hp(defender)
    if hp > 0 and dmg >= hp:
        gained = _prize_value(_entity_id(defender))
        prizes_left = len(_items(_get(P, 'prize', [])))
        if gained >= prizes_left:
            return 90000.0 + gained * 1000.0 + dmg
        return 8000.0 + gained * 1000.0 + dmg
    return 2000.0 + dmg


def final_prize_attack(observation: Any) -> list[int] | None:
    sel = _selection(observation)
    if sel is None or int(_get(sel, 'maxCount', 0) or 0) != 1:
        return None
    opts = _items(_get(sel, 'option', []))
    attacks = [(i, o, attack_value(o, observation)) for i, o in enumerate(opts)
               if _get(o, 'type', None) == ATTACK]
    if not attacks:
        return None
    i, _o, value = max(attacks, key=lambda row: row[2])
    return [i] if value >= 90000.0 else None


def _line_state(player: Any) -> dict[str, int]:
    board = Counter(_inplay_ids(player))
    hand = Counter(_hand_ids(player))
    disc = Counter(_discard_ids(player))
    duns = sum(board[x] for x in DUNSPARCE_IDS)
    return {
        'abra_board': board[ABRA],
        'kadabra_board': board[KADABRA],
        'alakazam_board': board[ALAKAZAM],
        'dunsparce_board': duns,
        'dudunsparce_board': board[DUDUNSPARCE],
        'abra_hand': hand[ABRA],
        'kadabra_hand': hand[KADABRA],
        'alakazam_hand': hand[ALAKAZAM],
        'dudunsparce_hand': hand[DUDUNSPARCE],
        'rare_candy_hand': hand[RARE_CANDY],
        'psychic_energy_hand': hand[BASIC_PSYCHIC] + hand[TELEPATH_PSYCHIC],
        'pokemon_discard': sum(1 for cid in disc.elements() if _card_meta(cid).get('hp')),
    }


def _bench_space(player: Any) -> int:
    max_bench = int(_get(player, 'benchMax', 5) or 5)
    return max(0, max_bench - len(_bench(player)))


def _powerful_hand_context(observation: Any, perspective: int) -> tuple[bool, int, int]:
    cur = _current(observation)
    P, O = _player(cur, perspective), _player(cur, 1 - perspective)
    active = _active(P)
    opp = _active(O)
    online = _entity_id(active) == ALAKAZAM
    hand = int(_get(P, 'handCount', len(_hand(P))) or 0)
    needed = math.ceil(_entity_hp(opp) / 20.0) if online and opp else 0
    return online, hand, needed


def _estimated_hand_delta(option: Any, observation: Any, perspective: int) -> float:
    typ = _get(option, 'type', None)
    cid = option_card_id(option, observation, perspective)
    if typ == ATTACK or typ == END:
        return 0.0
    if typ == ATTACH:
        return 3.0 if cid == ENRICHING_ENERGY else -1.0
    if typ == EVOLVE:
        if cid == KADABRA:
            return 1.0  # spend one, draw two
        if cid == ALAKAZAM:
            return 2.0  # spend one, draw three
        return -1.0
    if typ == ABILITY:
        source = option_target_entity(option, observation, perspective)
        if _entity_id(source) == DUDUNSPARCE:
            return 3.0
        return 0.0
    if typ == PLAY:
        effects = CEFF.get(str(cid), {}) or {}
        draw = float(effects.get('draw', 0) or 0)
        search = float(effects.get('search', 0) or 0)
        recover = float(effects.get('recover_discard', 0) or 0)
        # Tutor/recovery cards generally replace themselves with one or more cards.
        return -1.0 + max(draw, search, recover)
    return 0.0


def _opponent_special_energy_count(player: Any) -> int:
    n = 0
    for entity in ([_active(player)] if _active(player) else []) + _bench(player):
        for ec in _items(_get(entity, 'energyCards', [])):
            cid = _card_id(ec)
            meta = _card_meta(cid)
            tags = set(meta.get('tags', []) or [])
            if cid not in (None, BASIC_PSYCHIC) and ('special_energy' in tags or cid != BASIC_PSYCHIC):
                n += 1
    return n


def _boss_target_value(observation: Any, perspective: int) -> float:
    cur = _current(observation)
    P, O = _player(cur, perspective), _player(cur, 1 - perspective)
    if not _bench(O):
        return -20.0
    hand = int(_get(P, 'handCount', len(_hand(P))) or 0)
    active = _active(P)
    best = -5.0
    for target in _bench(O):
        hp = _entity_hp(target)
        prize = _prize_value(_entity_id(target))
        # After playing Boss, Powerful Hand loses one hand card.
        ph_dmg = 20.0 * max(0, hand - 1) if _entity_id(active) == ALAKAZAM else 0.0
        static_best = 0.0
        if active:
            for atk in _card_meta(_entity_id(active)).get('atk', []) or []:
                static_best = max(static_best, float(atk.get('d', atk.get('dmg', 0)) or 0))
        can_ko = max(ph_dmg, static_best) >= hp > 0
        attached = _attached_count(target)
        value = (40.0 if can_ko else 0.0) + 12.0 * prize + 2.0 * attached - 0.03 * hp
        best = max(best, value)
    return best


def _run_away_score(observation: Any, perspective: int, source: Any | None) -> float:
    cur = _current(observation)
    P = _player(cur, perspective)
    deck_left = int(_get(P, 'deckCount', 0) or 0)
    bodies = len(_inplay_ids(P))
    source = source or next((x for x in ([_active(P)] if _active(P) else []) + _bench(P)
                             if _entity_id(x) == DUDUNSPARCE), None)
    if source is None:
        return 0.0
    score = 10.0
    hp = _entity_hp(source)
    meta_hp = float(_card_meta(DUDUNSPARCE).get('hp', 140) or 140)
    if hp < meta_hp * 0.55:
        score += 10.0  # recycle a damaged body / deny a prize
    if _attached_count(source):
        score += 3.0  # resources are recycled, not discarded
    if bodies <= 1:
        score -= 30.0
    if deck_left <= 6:
        score -= 18.0
    if _bench_space(P) <= 0:
        score += 4.0
    online, hand, needed = _powerful_hand_context(observation, perspective)
    if online and needed and hand < needed <= hand + 3:
        score += 18.0
    return score


def root_option_priors(observation: Any, perspective: int | None = None) -> list[float] | None:
    """Conservative priors for root option ordering and exact-value tie-breaking."""
    sel = _selection(observation)
    if sel is None:
        return None
    opts = _items(_get(sel, 'option', []))
    if not opts:
        return None
    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    P, O = _player(cur, me), _player(cur, 1 - me)
    lines = _line_state(P)
    f = FT.encode_state(_as_dict(observation), perspective=me)
    online, hand, needed = _powerful_hand_context(observation, me)
    values = []
    for option in opts:
        typ = _get(option, 'type', None)
        cid = option_card_id(option, observation, me)
        score = 0.0
        if typ == ATTACK:
            score = attack_value(option, observation, me) / 100.0
        elif typ == END:
            score = -12.0 if (f.get('tutor_playable_now') or f.get('draw_playable_now') or f.get('can_evolve')) else 0.0
        elif typ == EVOLVE:
            if cid == ALAKAZAM:
                score += 22.0
                if lines['kadabra_board'] or (lines['abra_board'] and lines['rare_candy_hand']):
                    score += 8.0
            elif cid == KADABRA:
                score += 14.0
            elif cid == DUDUNSPARCE:
                score += 10.0
        elif typ == ABILITY:
            source = option_target_entity(option, observation, me)
            if _entity_id(source) == DUDUNSPARCE:
                score += _run_away_score(observation, me, source)
            else:
                score += 3.0
        elif typ == ATTACH:
            target = option_target_entity(option, observation, me)
            tid = _entity_id(target)
            if tid == ALAKAZAM:
                score += 14.0
            elif tid in (ABRA, KADABRA):
                score += 7.0
            elif tid == DUDUNSPARCE:
                score += 1.0
            if cid == ENRICHING_ENERGY and tid == DUDUNSPARCE and lines['alakazam_board']:
                score += 16.0
        elif typ == PLAY:
            if cid == POFFIN and _bench_space(P) > 0:
                score += 12.0 + (5.0 if not lines['abra_board'] else 0.0) + (4.0 if not lines['dunsparce_board'] else 0.0)
            elif cid == RARE_CANDY:
                score += 18.0 if lines['abra_board'] and lines['alakazam_hand'] else -3.0
            elif cid == HILDA:
                score += 12.0 if (lines['abra_board'] or lines['kadabra_board']) and not lines['alakazam_hand'] else 5.0
            elif cid == DAWN:
                missing = int(not lines['abra_board']) + int(not lines['kadabra_board']) + int(not lines['alakazam_hand'])
                score += 6.0 + 3.0 * missing
            elif cid == POKE_PAD:
                score += 7.0
            elif cid == NIGHT_STRETCHER:
                score += 9.0 if any(x in _discard_ids(P) for x in PSYCHIC_LINE | DUNSPARCE_LINE | ENERGY_IDS) else -2.0
            elif cid == SACRED_ASH:
                score += 8.0 if (int(_get(P, 'deckCount', 0) or 0) <= 10 or lines['pokemon_discard'] >= 3) else -4.0
            elif cid == BOSS:
                score += _boss_target_value(observation, me)
            elif cid == ENHANCED_HAMMER:
                score += 10.0 if _opponent_special_energy_count(O) else -6.0
            elif cid == BATTLE_CAGE:
                score -= 3.0  # hold unless a future explicit spread-threat feature says otherwise
            else:
                e = CEFF.get(str(cid), {}) or {}
                score += 2.0 * float(e.get('draw', 0) or 0)
                score += 1.5 * float(e.get('search', 0) or 0)
                score += 4.0 * float(bool(e.get('search_to_bench')))

        # Powerful Hand hand-budget guard. Priors only; search remains authoritative.
        if online and needed > 0 and hand >= needed:
            projected = hand + _estimated_hand_delta(option, observation, me)
            if projected < needed and typ not in (ATTACK,):
                score -= 24.0
        values.append(float(score))
    return values


def _target_score(option: Any, observation: Any, perspective: int) -> float:
    cur = _current(observation)
    me = perspective
    pidx = _get(option, 'playerIndex', me)
    try:
        pidx = int(pidx)
    except Exception:
        pidx = me
    entity = option_target_entity(option, observation, me)
    cid = option_card_id(option, observation, me)
    if entity is not None:
        cid = _entity_id(entity)

    # Opponent target: prefer prize-rich, low-HP, highly powered, evolved threats.
    if pidx != me or int(_get(option, 'area', -1) or -1) not in (A_DECK, A_HAND, A_DISCARD, A_PRIZE):
        if entity is not None and pidx != me:
            hp = _entity_hp(entity)
            return 35.0 * _prize_value(cid) + 4.0 * _attached_count(entity) - 0.08 * hp

    P = _player(cur, me)
    lines = _line_state(P)
    score = card_need_score(cid, observation, me)
    # Own in-play target selection: prefer the live attacker/evolution line.
    if entity is not None:
        if cid == ALAKAZAM:
            score += 25.0
        elif cid in (ABRA, KADABRA):
            score += 14.0
        elif cid == DUDUNSPARCE:
            score += 6.0
        if entity is _active(P):
            score += 5.0
    return score


def card_need_score(cid, observation: Any, perspective: int) -> float:
    if cid is None:
        return 0.0
    cur = _current(observation)
    P = _player(cur, perspective)
    lines = _line_state(P)
    deck_left = int(_get(P, 'deckCount', 0) or 0)
    score = 0.0
    if cid == ALAKAZAM:
        score = 34.0 if (lines['kadabra_board'] or (lines['abra_board'] and lines['rare_candy_hand'])) else 16.0
    elif cid == KADABRA:
        score = 28.0 if lines['abra_board'] and not lines['kadabra_board'] else 10.0
    elif cid == ABRA:
        score = 25.0 if not lines['abra_board'] and _bench_space(P) > 0 else 6.0
    elif cid == DUDUNSPARCE:
        score = 22.0 if lines['dunsparce_board'] and not lines['dudunsparce_board'] else 8.0
    elif cid in DUNSPARCE_IDS:
        score = 18.0 if not lines['dunsparce_board'] and _bench_space(P) > 0 else 4.0
    elif cid == RARE_CANDY:
        score = 25.0 if lines['abra_board'] and lines['alakazam_hand'] else 8.0
    elif cid in (BASIC_PSYCHIC, TELEPATH_PSYCHIC):
        f = FT.encode_state(_as_dict(observation), perspective=perspective)
        score = 22.0 if f.get('active_energy_short', 0) > 0 else 6.0
    elif cid == ENRICHING_ENERGY:
        score = 14.0 if lines['alakazam_board'] else 6.0
    elif cid in PSYCHIC_LINE | DUNSPARCE_LINE:
        score = 10.0
    elif cid in ENERGY_IDS:
        score = 8.0
    else:
        e = CEFF.get(str(cid), {}) or {}
        score += 4.0 * float(e.get('draw', 0) or 0)
        score += 2.0 * float(e.get('search', 0) or 0)
        score += 5.0 * float(bool(e.get('recover_discard')))
    if deck_left <= 5 and cid not in (SACRED_ASH,):
        score -= 3.0
    return score


def _selection_only(sel: Any) -> bool:
    opts = _items(_get(sel, 'option', []))
    if not opts:
        return False
    strategic = {PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END}
    return all(_get(o, 'type', None) not in strategic for o in opts)


def choose_subdecision(observation: Any, perspective: int | None = None) -> list[int] | None:
    """Choose tutor/target/multi-select prompts; leave strategic root decisions to search."""
    sel = _selection(observation)
    if sel is None:
        return None
    opts = _items(_get(sel, 'option', []))
    if not opts:
        return []
    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    k = int(_get(sel, 'maxCount', 0) or 0)
    mn = int(_get(sel, 'minCount', 0) or 0)
    if k <= 0:
        return []
    # Do not override normal root strategic choices.
    if k == 1 and not _selection_only(sel):
        return None

    scores = []
    for i, option in enumerate(opts):
        typ = _get(option, 'type', None)
        cid = option_card_id(option, observation, me)
        score = _target_score(option, observation, me)
        if typ in (YES, NO):
            score = 0.0
        elif typ == NUMBER:
            raw = _get(option, 'number', _get(option, 'index', 0))
            try:
                score = float(raw)
            except Exception:
                score = 0.0
        scores.append((float(score), i))
    scores.sort(key=lambda row: (-row[0], row[1]))

    if k == 1:
        if mn == 0 and scores[0][0] <= 0:
            return []
        return [scores[0][1]]
    positive = [i for s, i in scores if s > 0]
    take = max(mn, min(k, len(positive)))
    if take == 0 and mn == 0:
        return []
    if take < mn:
        take = min(k, len(scores))
    chosen = [i for _s, i in scores[:take]]
    return sorted(chosen)


def rollout_choice(observation: Any, is_me: bool = False) -> list[int] | None:
    """Policy for simulated continuation decisions."""
    sel = _selection(observation)
    if sel is None:
        return None
    sub = choose_subdecision(observation)
    if sub is not None:
        return sub
    opts = _items(_get(sel, 'option', []))
    if not opts:
        return []
    cur = _current(observation)
    me = _perspective(cur)
    k = int(_get(sel, 'maxCount', 0) or 0)
    mn = int(_get(sel, 'minCount', 0) or 0)
    if k <= 0:
        return []
    if k == 1:
        # Terminal prize wins always dominate.
        win = final_prize_attack(observation)
        if win is not None:
            return win
        if is_me:
            priors = root_option_priors(observation, me)
            if priors:
                return [max(range(len(priors)), key=lambda i: (priors[i], -i))]
        attacks = [(attack_value(o, observation, me), i) for i, o in enumerate(opts)
                   if _get(o, 'type', None) == ATTACK]
        if attacks:
            return [max(attacks)[1]]
    return list(range(max(min(k, len(opts)), min(mn, len(opts)))))
