"""Conservative Dudunsparce/Alakazam support for ``agent_search``.

The important architectural rule is that this module never becomes a policy
controller.  It does three narrow jobs:

1. estimate attacks that have dynamic text (especially Alakazam's Powerful
   Hand, which places damage counters and therefore ignores weakness and
   resistance);
2. identify a *small* set of mechanically safe pre-attack development moves so
   search can compare "attack now" with "develop once, then attack" instead of
   blindly ending the turn at the first available knockout;
3. propose card-specific target selections for subdecisions.  The caller must
   compare every proposal with the baseline/default selection on paired hidden
   worlds and fail closed to the default.

Unknown schemas and uncertain situations return no proposal.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import statistics
import time
from collections import Counter
from typing import Any, Iterable

# OptionType / common AreaType values.
NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY = range(7)
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END = 7, 8, 9, 10, 11, 12, 13, 14
A_DECK, A_HAND, A_ACTIVE, A_BENCH = 1, 2, 4, 5
CTX_TO_BENCH = 5
CTX_IS_FIRST = 41

# Deck card ids.
BASIC_PSYCHIC, ENRICHING, TELEPATH = 5, 13, 19
DUNSPARCE_IDS = {65, 305}
DUDUNSPARCE = 66
ABRA, KADABRA, ALAKAZAM = 741, 742, 743
RARE_CANDY, ENH_HAMMER, POFFIN = 1079, 1081, 1086
NIGHT_STRETCHER, SACRED_ASH, POKE_PAD = 1097, 1129, 1152
BOSS, LANA_AID, HILDA, DAWN, BATTLE_CAGE = 1182, 1184, 1225, 1231, 1264


def _load(fn: str) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (fn, os.path.join(here, fn), os.path.join('/kaggle_simulations/agent', fn)):
        try:
            with open(path, encoding='utf-8') as handle:
                return json.load(handle)
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


def _items(value: Any) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return []


def _as_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    try:
        return dataclasses.asdict(obj)
    except Exception:
        return {}


def _current(obs: Any):
    return _get(obs, 'current', {}) or {}


def _selection(obs: Any):
    return _get(obs, 'select', None)


def _players(cur: Any) -> list:
    return _items(_get(cur, 'players', []))


def _perspective(cur: Any) -> int:
    try:
        return int(_get(cur, 'yourIndex', 0) or 0)
    except Exception:
        return 0


def _player(cur: Any, index: int):
    players = _players(cur)
    return players[index] if 0 <= index < len(players) else {}


def _active(player: Any):
    active = _items(_get(player, 'active', []))
    return active[0] if active and active[0] else None


def _bench(player: Any) -> list:
    return [entity for entity in _items(_get(player, 'bench', [])) if entity]


def _hand(player: Any) -> list:
    return _items(_get(player, 'hand', []))


def _cid(card: Any):
    if card is None:
        return None
    if isinstance(card, (int, float)):
        return int(card)
    value = _get(card, 'id', None)
    try:
        return int(value) if value is not None else None
    except Exception:
        return value


def _meta(cid) -> dict:
    if cid is None:
        return {}
    out = {}
    out.update(CF.get(str(cid), {}) or {})
    out.update(CDB.get(str(cid), {}) or {})
    return out


def _prize_value(cid) -> int:
    row = _meta(cid)
    if row.get('mega'):
        return 3
    if row.get('ex'):
        return 2
    return int(row.get('prize', 1) or 1)


def _attached_count(entity: Any) -> int:
    energies = _get(entity, 'energyCards', None)
    if energies is not None:
        return len(_items(energies))
    return len(_items(_get(entity, 'energies', [])))


def _hand_count(player: Any) -> int:
    value = _get(player, 'handCount', None)
    return int(value if value is not None else len(_hand(player)))


def _entity_at(player: Any, area: Any, index: Any):
    try:
        area = int(area)
        index = int(index)
    except Exception:
        return None
    if area == A_ACTIVE:
        zone = [entity for entity in _items(_get(player, 'active', [])) if entity]
    elif area == A_BENCH:
        zone = _bench(player)
    elif area == A_HAND:
        zone = _hand(player)
    else:
        return None
    return zone[index] if 0 <= index < len(zone) else None


def option_card_id(option: Any, observation: Any, perspective: int | None = None):
    """Resolve the acting/selected card without guessing when the schema is unclear."""
    for key in ('cardId', 'card_id', 'pokemonId', 'energyId'):
        value = _get(option, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return value

    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    pidx = _get(option, 'playerIndex', me)
    try:
        pidx = int(pidx)
    except Exception:
        pidx = me
    player = _player(cur, pidx)
    typ = _get(option, 'type', None)
    idx = _get(option, 'index', None)
    try:
        idx = int(idx)
    except Exception:
        return None

    # PLAY / ATTACH / EVOLVE root options normally index the acting player's hand.
    if typ in (PLAY, ATTACH, EVOLVE):
        hand = _hand(player)
        if 0 <= idx < len(hand):
            return _cid(hand[idx])

    # CARD target prompts usually expose a selection-local deck array.
    sel = _selection(observation)
    if typ == CARD and sel is not None:
        for key in ('deck', 'discard', 'prize', 'hand'):
            zone = _items(_get(sel, key, []))
            if 0 <= idx < len(zone):
                return _cid(zone[idx])
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
    area = _get(option, 'inPlayArea', _get(option, 'area', None))
    index = _get(option, 'inPlayIndex', _get(option, 'targetIndex', None))
    if index is None and area in (A_ACTIVE, A_BENCH):
        index = _get(option, 'index', None)
    return _entity_at(player, area, index)


def _attack_static(option: Any) -> tuple[float, str]:
    row = ATK.get(str(_get(option, 'attackId', None)), {}) or {}
    return float(row.get('d', row.get('dmg', 0)) or 0), str(row.get('n', row.get('name', '')) or '')


def attack_profile(option: Any, observation: Any, perspective: int | None = None) -> dict:
    """Return attack amount and KO value.

    Powerful Hand places damage counters.  It therefore ignores weakness and
    resistance; treating it as ordinary damage was a subtle bug in prior
    candidates.
    """
    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    player, opponent = _player(cur, me), _player(cur, 1 - me)
    attacker, defender = _active(player), _active(opponent)
    static_damage, attack_name = _attack_static(option)
    attacker_id = _cid(attacker)
    normalized = attack_name.lower().replace("'", '')
    is_powerful_hand = attacker_id == ALAKAZAM and ('powerful hand' in normalized or static_damage <= 0)

    if is_powerful_hand:
        amount = 20.0 * _hand_count(player)
        is_counters = True
    else:
        amount = static_damage
        is_counters = False
        if attacker and defender:
            attacker_type = str(_meta(attacker_id).get('ty', '') or '')
            defender_meta = _meta(_cid(defender))
            if defender_meta.get('wk') and attacker_type and defender_meta.get('wk') == attacker_type:
                amount *= 2.0
            if defender_meta.get('rs') and attacker_type and defender_meta.get('rs') == attacker_type:
                amount = max(0.0, amount - 30.0)

    hp = float(_get(defender, 'hp', 0) or 0) if defender else 0.0
    ko = bool(defender and hp > 0 and amount >= hp)
    prizes = _prize_value(_cid(defender)) if ko else 0
    prizes_left = len(_items(_get(player, 'prize', [])))
    game_win = bool(ko and prizes >= prizes_left)
    value = (90000.0 + prizes * 1000.0 + amount if game_win else
             8000.0 + prizes * 1000.0 + amount if ko else
             2000.0 + amount)
    return {
        'amount': amount,
        'damage_counters': is_counters,
        'ko': ko,
        'prizes': prizes,
        'game_win': game_win,
        'value': value,
    }


def best_ko_attack(observation: Any, perspective: int | None = None):
    sel = _selection(observation)
    if sel is None or int(_get(sel, 'maxCount', 0) or 0) != 1:
        return None
    options = _items(_get(sel, 'option', []))
    candidates = []
    for index, option in enumerate(options):
        if _get(option, 'type', None) == ATTACK:
            profile = attack_profile(option, observation, perspective)
            if profile['ko']:
                candidates.append((profile['value'], index, profile))
    if not candidates:
        return None
    _value, index, profile = max(candidates)
    return index, profile


def _projected_hand_after(option: Any, observation: Any, perspective: int) -> int:
    player = _player(_current(observation), perspective)
    hand = _hand_count(player)
    typ = _get(option, 'type', None)
    cid = option_card_id(option, observation, perspective)
    if typ == EVOLVE:
        # Evolve-from-hand consumes one card, then Psychic Draw draws 2/3.
        if cid == KADABRA:
            return hand + 1
        if cid == ALAKAZAM:
            return hand + 2
        return hand - 1
    if typ == ATTACH:
        return hand + 3 if cid == ENRICHING else hand - 1
    return hand


def safe_pre_attack_indices(observation: Any, attack_index: int, perspective: int | None = None) -> list[int]:
    """Return a deliberately tiny set of setup actions worth comparing with attack-now.

    Attacking ends the turn.  The old forced-KO rule therefore skipped all
    development whenever a KO was already visible.  We do *not* broadly disable
    the KO floor.  Instead, we allow search to compare attack-now with only
    mechanically constrained, one-step actions that do not touch the current
    attacker and do not make a Powerful Hand KO mathematically impossible.
    """
    sel = _selection(observation)
    if sel is None:
        return []
    options = _items(_get(sel, 'option', []))
    if not (0 <= attack_index < len(options)):
        return []
    cur = _current(observation)
    me = _perspective(cur) if perspective is None else perspective
    player, opponent = _player(cur, me), _player(cur, 1 - me)
    active = _active(player)
    defender = _active(opponent)
    attack = attack_profile(options[attack_index], observation, me)
    if not attack['ko'] or attack['game_win']:
        return []

    needed_hand = math.ceil(float(_get(defender, 'hp', 0) or 0) / 20.0) if attack['damage_counters'] else 0
    safe: list[int] = []
    for index, option in enumerate(options):
        if index == attack_index:
            continue
        typ = _get(option, 'type', None)
        target = option_target_entity(option, observation, me)
        target_is_active = target is not None and target is active
        cid = option_card_id(option, observation, me)

        if typ == EVOLVE:
            # Never evolve the current attacker while protecting a known KO.
            if target is None or target_is_active:
                continue
            if cid not in (KADABRA, ALAKAZAM, DUDUNSPARCE):
                continue
            if needed_hand and _projected_hand_after(option, observation, me) < needed_hand:
                continue
            safe.append(index)

        elif typ == ATTACH:
            # Only prepare a bench evolution-line attacker.  Telepath Energy can
            # open an additional target prompt, so it is intentionally excluded
            # from this first fail-closed implementation.
            if target is None or target_is_active or _cid(target) not in (ABRA, KADABRA, ALAKAZAM):
                continue
            if cid not in (BASIC_PSYCHIC, ENRICHING):
                continue
            if needed_hand and _projected_hand_after(option, observation, me) < needed_hand:
                continue
            safe.append(index)

    return safe[:3]


def _inplay_ids(player: Any) -> list:
    out = []
    if _active(player):
        out.append(_cid(_active(player)))
    out.extend(_cid(entity) for entity in _bench(player))
    return [cid for cid in out if cid is not None]


def _selection_card_id(sel: Any, option: Any):
    idx = _get(option, 'index', None)
    try:
        idx = int(idx)
    except Exception:
        return None
    for key in ('deck', 'discard', 'prize', 'hand'):
        zone = _items(_get(sel, key, []))
        if 0 <= idx < len(zone):
            return _cid(zone[idx])
    return None


def _poffin_proposals(observation: Any) -> list[list[int]]:
    sel = _selection(observation)
    if sel is None or int(_get(sel, 'context', -1) or -1) != CTX_TO_BENCH:
        return []
    options = _items(_get(sel, 'option', []))
    maximum = int(_get(sel, 'maxCount', 0) or 0)
    if maximum < 2 or len(options) < 2:
        return []
    by_card: dict[int, list[int]] = {}
    for index, option in enumerate(options):
        if _get(option, 'type', None) != CARD:
            continue
        cid = _selection_card_id(sel, option)
        if cid is not None:
            by_card.setdefault(cid, []).append(index)
    abra = (by_card.get(ABRA) or [None])[0]
    duns = next((by_card[c][0] for c in DUNSPARCE_IDS if by_card.get(c)), None)
    if abra is None or duns is None or abra == duns:
        return []

    me = _player(_current(observation), _perspective(_current(observation)))
    board = Counter(_inplay_ids(me))
    have_abra = board[ABRA] > 0
    have_duns = any(board[c] > 0 for c in DUNSPARCE_IDS)
    if have_abra and have_duns:
        return []
    return [sorted([abra, duns])]


def _retreat_cost(entity: Any) -> int:
    row = _meta(_cid(entity))
    for key in ('retreat', 'retreatCost', 'rc'):
        if row.get(key) is not None:
            try:
                return int(row[key])
            except Exception:
                pass
    return 0


def _best_affordable_attack_amount(player: Any, target: Any) -> float:
    active = _active(player)
    if not active or not target:
        return 0.0
    if _cid(active) == ALAKAZAM and _attached_count(active) >= 1:
        return 20.0 * _hand_count(player)
    best = 0.0
    for attack in (_meta(_cid(active)).get('atks', []) or _meta(_cid(active)).get('atk', []) or []):
        cost = attack.get('cE', attack.get('cost', [])) or []
        if _attached_count(active) < len(cost):
            continue
        damage = float(attack.get('dmg', attack.get('d', 0)) or 0)
        attacker_type = str(_meta(_cid(active)).get('ty', '') or '')
        target_meta = _meta(_cid(target))
        if target_meta.get('wk') and attacker_type and target_meta.get('wk') == attacker_type:
            damage *= 2.0
        if target_meta.get('rs') and attacker_type and target_meta.get('rs') == attacker_type:
            damage = max(0.0, damage - 30.0)
        best = max(best, damage)
    return best


def _boss_proposals(observation: Any) -> list[list[int]]:
    """Propose only a concrete KO or a clearly stranded opponent bench target."""
    sel = _selection(observation)
    if sel is None or int(_get(sel, 'maxCount', 0) or 0) != 1:
        return []
    options = _items(_get(sel, 'option', []))
    if len(options) < 2:
        return []
    cur = _current(observation)
    me = _perspective(cur)
    player, opponent = _player(cur, me), _player(cur, 1 - me)
    bench = _bench(opponent)
    if not bench:
        return []

    rows = []
    for index, option in enumerate(options):
        pidx = _get(option, 'playerIndex', None)
        try:
            pidx = int(pidx)
        except Exception:
            pidx = None
        target = option_target_entity(option, observation, me)
        if target is None or (pidx is not None and pidx == me):
            continue
        hp = float(_get(target, 'hp', 0) or 0)
        damage = _best_affordable_attack_amount(player, target)
        can_ko = hp > 0 and damage >= hp
        prize = _prize_value(_cid(target))
        retreat_gap = max(0, _retreat_cost(target) - _attached_count(target))
        game_win = can_ko and prize >= len(_items(_get(player, 'prize', [])))
        score = (10000.0 if game_win else 1000.0 if can_ko else 0.0)
        score += 80.0 * prize + 25.0 * retreat_gap + 4.0 * _attached_count(target) - 0.03 * hp
        rows.append((score, index, can_ko, retreat_gap))
    if not rows:
        return []
    rows.sort(reverse=True)
    score, index, can_ko, retreat_gap = rows[0]
    default = default_selection(observation)
    if default and default[0] == index:
        return []
    # A low-HP target alone is not enough.  Require a KO or a meaningful strand.
    if not can_ko and retreat_gap < 2:
        return []
    return [[index]]


_RESOLVERS = {
    'poffin': _poffin_proposals,
    'boss': _boss_proposals,
}


def propose(observation: Any, enabled: Iterable[str] | None = None) -> list[list[int]]:
    names = list(_RESOLVERS) if enabled is None else [name for name in enabled if name in _RESOLVERS]
    out: list[list[int]] = []
    for name in names:
        try:
            for candidate in _RESOLVERS[name](observation) or []:
                candidate = list(candidate)
                if candidate and candidate not in out and valid_selection(observation, candidate):
                    out.append(candidate)
        except Exception:
            continue
    return out


def default_selection(observation: Any) -> list[int]:
    sel = _selection(observation)
    if sel is None:
        return []
    options = _items(_get(sel, 'option', []))
    maximum = int(_get(sel, 'maxCount', 0) or 0)
    minimum = int(_get(sel, 'minCount', 0) or 0)
    if not options or maximum <= 0:
        return []
    count = max(min(maximum, len(options)), min(minimum, len(options)))
    return list(range(count))


def valid_selection(observation: Any, selection: list[int]) -> bool:
    sel = _selection(observation)
    if sel is None:
        return False
    n = len(_items(_get(sel, 'option', [])))
    minimum = int(_get(sel, 'minCount', 0) or 0)
    maximum = int(_get(sel, 'maxCount', 0) or 0)
    unique = list(dict.fromkeys(selection))
    return unique == selection and minimum <= len(selection) <= maximum and all(0 <= index < n for index in selection)


def _simulate_selection(api, root_id, selection, me, search_module):
    state = api.search_step(root_id, list(selection))
    saw_opponent = False
    for _ in range(search_module.DEPTH_CAP):
        observation = state.observation
        cur = observation.current
        if cur is not None and cur.result != -1:
            break
        sel = observation.select
        if sel is None:
            break
        my_move = cur is not None and cur.yourIndex == me
        if saw_opponent and my_move:
            break
        if not my_move:
            saw_opponent = True
        state = api.search_step(state.searchId, search_module._rollout_pick(observation, is_me=my_move))
    return search_module.EV.evaluate_obs(search_module._obs_dict(state.observation), me)


def compare_selections(observation: dict, deck: list, candidates: list[list[int]], default: list[int],
                       search_module, *, time_budget: float = 1.0, n_determ: int = 8,
                       minimum_worlds: int = 4, z: float = 1.64,
                       minimum_margin: float = 3.0, allow_tie: bool = False):
    """Paired-world validation.  Any uncertainty returns the baseline default.

    ``z=1.64`` is a one-sided ~95% threshold.  This is intentionally stricter
    than the previous one-point tie rule, which accepted proposals without
    measuring paired uncertainty.
    """
    stats = {
        'worlds': 0, 'accepted': False, 'tiebreak': False, 'incomplete': False,
        'error': False, 'mean_diff': 0.0, 'se_diff': None, 'chosen': 'default',
    }
    if not valid_selection(observation, default):
        stats['error'] = True
        return default, stats
    selections = [list(default)]
    for candidate in candidates:
        candidate = list(candidate)
        if candidate != default and valid_selection(observation, candidate) and candidate not in selections:
            selections.append(candidate)
    if len(selections) < 2:
        return default, stats

    api = search_module._api()
    sel, cur = observation.get('select'), observation.get('current')
    players = (cur or {}).get('players') or []
    if api is None or not sel or len(players) < 2:
        stats['error'] = True
        return default, stats
    me = int((cur or {}).get('yourIndex', 0) or 0)
    player, opponent = players[me], players[1 - me]
    opp_active = opponent.get('active') or []
    if opp_active and opp_active[0] is None:
        return default, stats

    n_my_deck = int(player.get('deckCount', 0) or 0)
    n_opp_deck = int(opponent.get('deckCount', 0) or 0)
    n_my_prize = len(player.get('prize') or [])
    n_opp_prize = len(opponent.get('prize') or [])
    n_opp_hand = int(opponent.get('handCount', 0) or 0)
    observation_class = api.to_observation_class(observation)
    paired_values: list[list[float]] = []
    started = time.time()

    for _ in range(n_determ):
        if time.time() - started >= time_budget:
            stats['incomplete'] = True
            break
        my_pool = search_module._hidden_pool(deck, player, exclude_hand=False)
        my_pool += [3] * max(0, n_my_deck + n_my_prize - len(my_pool))
        opp_pool = search_module._hidden_pool(deck, opponent, exclude_hand=True)
        opp_pool += [3] * max(0, n_opp_deck + n_opp_prize + n_opp_hand - len(opp_pool))
        try:
            root = api.search_begin(
                observation_class,
                your_deck=my_pool[:n_my_deck],
                your_prize=my_pool[n_my_deck:n_my_deck + n_my_prize],
                opponent_hand=opp_pool[:n_opp_hand],
                opponent_prize=opp_pool[n_opp_hand:n_opp_hand + n_opp_prize],
                opponent_deck=opp_pool[n_opp_hand + n_opp_prize:n_opp_hand + n_opp_prize + n_opp_deck],
                opponent_active=[],
            )
        except Exception:
            continue
        values = []
        complete = True
        try:
            for selection in selections:
                values.append(_simulate_selection(api, root.searchId, selection, me, search_module))
        except Exception:
            complete = False
        finally:
            try:
                api.search_end()
            except Exception:
                pass
        if complete and len(values) == len(selections):
            paired_values.append(values)

    stats['worlds'] = len(paired_values)
    if len(paired_values) < minimum_worlds or stats['incomplete']:
        stats['incomplete'] = True
        return default, stats

    best_index = 0
    best_mean = 0.0
    best_se = None
    for index in range(1, len(selections)):
        diffs = [row[index] - row[0] for row in paired_values]
        mean = statistics.fmean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float('inf')
        if mean > best_mean:
            best_index, best_mean, best_se = index, mean, se
    stats['mean_diff'] = best_mean
    stats['se_diff'] = best_se
    if best_index == 0:
        return default, stats

    threshold = max(minimum_margin, z * (best_se if best_se is not None else float('inf')))
    if best_mean > threshold:
        stats['accepted'] = True
        stats['chosen'] = 'heuristic'
        return selections[best_index], stats
    if allow_tie and best_mean >= -minimum_margin and best_se is not None and best_se <= minimum_margin:
        stats['tiebreak'] = True
        stats['chosen'] = 'heuristic_tiebreak'
        return selections[best_index], stats
    return default, stats
