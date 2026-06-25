"""Starmie/Cinderace heuristic layer (verified card mechanics + imitation-gap findings, 2026-06-25).

Composition: this layer decides the moves it has high confidence on; search_v3 fills the rest; legal-first
is the last resort. The CENTRAL rule, from the imitation-gap analysis (we attacked when top pilots developed
339 vs 68), is DEVELOP-BEFORE-ATTACK: attacking ends the turn, so do every useful development first and
attack LAST -- unless the attack wins the game now.

Card mechanics (from the deck's card images; see pokemon-ai-agent/docs/starmie_deck_and_heuristics.md):
- Mega Starmie ex (1031): Stage1 from Staryu, 330HP, KO = 3 prizes. Jetting Blow [W] 120 + 50 bench snipe;
  Nebula Beam [CCC] 210 flat (ignores weakness/resistance).
- Cinderace (666): Turbo Flare [C] 50 + accelerate 3 Basic energy to the bench (the energy engine).
- Ignition Energy (17): [CCC] on an Evolution; discarded end of turn -> only attach to fund a Nebula KO now.
- Hero's Cape (1159): +100 HP tool. Wally's Compassion (1229): full-heal a Mega ex + energy to hand.

attack_stats.json is a flat number -> advisory only. We rely on engine affordability (an attack is offered
only if payable) and special-case Nebula Beam's "ignores weakness/resistance".
"""
from __future__ import annotations

import json
import os

import deck_policy_v3 as DP   # option/board helpers; loads card stats; no cg import

# OptionType (mirror deck_policy_v3)
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END = 7, 8, 9, 10, 11, 12, 13, 14
CARD, YES, NO = 3, 1, 2
CTX_IS_FIRST = 41

# Deck card ids
BASIC_WATER, IGNITION, CINDERACE, STARYU, MEGA_STARMIE = 3, 17, 666, 1030, 1031
POFFIN, NIGHT_STRETCHER, CRUSHING_HAMMER, ULTRA_BALL, POKEGEAR = 1086, 1097, 1120, 1121, 1122
MEGA_SIGNAL, HEROS_CAPE, BOSS, SALVATORE, HARLEQUIN, HILDA, LILLIE, WALLYS = 1145, 1159, 1182, 1189, 1223, 1225, 1227, 1229

# Attack ids (attack_stats.json)
JETTING_BLOW, NEBULA_BEAM, TURBO_FLARE = 1487, 1488, 965

STARMIE_DECK = (
    [3] * 9 + [17] * 4 + [666] * 4 + [1030] * 3 + [1031] * 3 + [1086] * 4 + [1097] * 2 + [1120] * 4
    + [1121] * 1 + [1122] * 4 + [1145] * 4 + [1159] * 1 + [1182] * 1 + [1189] * 4 + [1223] * 2
    + [1225] * 2 + [1227] * 4 + [1229] * 4
)
assert len(STARMIE_DECK) == 60


def _load(fn):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, fn), fn, os.path.join("/kaggle_simulations/agent", fn)):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


CDB = _load("card_stats.json")
CEFF = _load("card_effects.json")
SEARCH_TRAINERS = {MEGA_SIGNAL, POFFIN, ULTRA_BALL, POKEGEAR, SALVATORE, HILDA}
DRAW_SUPPORTERS = {LILLIE, HARLEQUIN}


# ---------- board helpers ----------
def _me_opp(obs):
    cur = DP._current(obs)
    me = DP._perspective(cur)
    return DP._player(cur, me), DP._player(cur, 1 - me), me


def _ids_in_play(player):
    out = []
    a = DP._active(player)
    if a:
        out.append(DP._cid(a))
    out += [DP._cid(b) for b in DP._bench(player)]
    return [c for c in out if c is not None]


def _ids_hand(player):
    return [DP._cid(c) for c in DP._hand(player)]


def _maxhp(cid):
    try:
        return float((CDB.get(str(cid), {}) or {}).get("hp", 0) or 0)
    except Exception:
        return 0.0


def _opt_type(opts, i):
    return opts[i].get("type") if 0 <= i < len(opts) else None


# ---------- attack reasoning ----------
def _attack_kos_active(obs, opt, opp):
    """Does this attack option KO the opponent's active? Handles Nebula's ignore-weakness/resistance."""
    aid = opt.get("attackId")
    defender = DP._active(opp)
    if not defender:
        return False, False
    hp = float(DP._get(defender, "hp", 0) or 0)
    if hp <= 0:
        return False, False
    if aid == NEBULA_BEAM:
        dmg = 210.0                          # flat; ignores weakness/resistance/effects
    else:
        prof = DP.attack_profile(opt, obs)   # applies weakness/resistance for normal attacks
        dmg = prof.get("amount", 0.0)
    ko = dmg >= hp
    prizes_left = len(DP._items(DP._get(DP._player(DP._current(obs), DP._perspective(DP._current(obs))), "prize", [])))
    game_win = ko and DP._prize_value(DP._cid(defender)) >= prizes_left
    return ko, game_win


def _attack_options(opts):
    return [(i, o) for i, o in enumerate(opts) if o.get("type") == ATTACK]


def _best_attack_index(obs, opts, opp):
    """User rule: prefer the Jetting Blow KO; use Nebula only when Jetting can't KO but Nebula can.
    Otherwise chip with Jetting (cheap); Turbo Flare if Cinderace is the only attacker."""
    atks = _attack_options(opts)
    if not atks:
        return None
    by_id = {}
    for i, o in atks:
        by_id.setdefault(o.get("attackId"), (i, o))
    jet = by_id.get(JETTING_BLOW)
    neb = by_id.get(NEBULA_BEAM)
    turbo = by_id.get(TURBO_FLARE)
    jet_ko = neb_ko = False
    if jet:
        jet_ko, jw = _attack_kos_active(obs, jet[1], opp)
        if jw:
            return jet[0]
    if neb:
        neb_ko, nw = _attack_kos_active(obs, neb[1], opp)
        if nw:
            return neb[0]
    if jet and jet_ko:
        return jet[0]
    if neb and neb_ko:
        return neb[0]
    # No KO -> chip. Prefer Nebula's 210 (focus a big threat) when it is affordable WITHOUT spending an
    # Ignition (active already has >=3 real energy); otherwise Jetting's 120 + 50 bench snipe (cheap, conserves
    # the Ignition). This matches pilots racing big threats while honoring the user's "don't waste Ignition" rule.
    player = _me_opp(obs)[0]
    active = DP._active(player)
    active_energy = DP._attached_count(active) if active else 0
    if neb and jet:
        return neb[0] if active_energy >= 3 else jet[0]
    if jet:
        return jet[0]
    if turbo:
        return turbo[0]          # Cinderace energy engine (+50)
    if neb:
        return neb[0]
    return atks[0][0]


def _game_winning_attack(obs, opts, opp):
    for i, o in _attack_options(opts):
        _ko, gw = _attack_kos_active(obs, o, opp)
        if gw:
            return i
    return None


# ---------- development usefulness ----------
def _needs(player):
    play = _ids_in_play(player)
    hand = _ids_hand(player)
    n_basics_play = sum(1 for c in play if (CDB.get(str(c), {}) or {}).get("hp") and c in (STARYU, CINDERACE))
    return {
        "have_mega_play": MEGA_STARMIE in play,
        "have_mega_hand": MEGA_STARMIE in hand,
        "have_staryu": STARYU in (play + hand),
        "n_basics_play": n_basics_play,
        "bench_count": len(DP._bench(player)),
    }


def _attach_score(obs, opt, player):
    """Score an ATTACH option onto our Starmie line. 0 = not useful."""
    try:
        tgt = DP.option_target_entity(opt, obs)
    except Exception:
        tgt = None
    if tgt is None:
        return 0.0
    tcid = DP._cid(tgt)
    ecid = DP.option_card_id(opt, obs)
    e_on_tgt = DP._attached_count(tgt)
    active = DP._active(player)
    tgt_is_active = active is not None and tgt is active
    if ecid == HEROS_CAPE:
        return 80.0 if tcid == MEGA_STARMIE else 30.0     # +100 HP on the win con
    if ecid == IGNITION:
        # Only to fund a Nebula KO this turn: active Mega Starmie, not already at 3 energy.
        if tgt_is_active and tcid == MEGA_STARMIE and e_on_tgt < 3:
            opp = _me_opp(obs)[1]
            d = DP._active(opp)
            if d and 0 < float(DP._get(d, "hp", 0) or 0) <= 210:
                return 88.0
        return 0.0                                        # never waste Ignition otherwise
    if ecid == BASIC_WATER:
        if tcid == MEGA_STARMIE:
            if tgt_is_active and e_on_tgt < 1:
                return 86.0                               # enable Jetting Blow now
            if e_on_tgt < 3:
                return 62.0 if tgt_is_active else 56.0    # build toward Nebula / develop the line
            return 0.0
        if tcid == CINDERACE and e_on_tgt < 1:
            return 70.0                                   # enable Turbo Flare (the engine)
        if tcid == STARYU:
            return 54.0
    return 0.0


def _best_ko_index(obs, opts, opp):
    """Index of a KOing attack, Jetting-Blow-preferred (conserve Ignition/Nebula for when Jetting can't KO)."""
    atks = _attack_options(opts)
    if not atks:
        return None
    by_id = {}
    for i, o in atks:
        by_id.setdefault(o.get("attackId"), (i, o))
    jet, neb, turbo = by_id.get(JETTING_BLOW), by_id.get(NEBULA_BEAM), by_id.get(TURBO_FLARE)
    if jet and _attack_kos_active(obs, jet[1], opp)[0]:
        return jet[0]
    if neb and _attack_kos_active(obs, neb[1], opp)[0]:
        return neb[0]
    if turbo and _attack_kos_active(obs, turbo[1], opp)[0]:
        return turbo[0]
    for i, o in atks:
        if _attack_kos_active(obs, o, opp)[0]:
            return i
    return None


def _high_value_play(obs, opts, player, opp):
    """Free, high-confidence plays worth doing before the turn-ending attack: gust a 2+ prize KO, heal a
    damaged Mega Starmie, tool it up. Returns an option index or None (everything else defers to search)."""
    idx = {}
    for i, o in enumerate(opts):
        if o.get("type") != PLAY:
            continue
        cid = DP.option_card_id(o, obs)
        idx.setdefault(cid, i)
    if BOSS in idx and _boss_enables_ko(obs, player, opp):
        return idx[BOSS]
    if WALLYS in idx and _wally_useful(player):
        return idx[WALLYS]
    a = DP._active(player)
    if HEROS_CAPE in idx and a and DP._cid(a) == MEGA_STARMIE:
        return idx[HEROS_CAPE]
    return None


def _best_attach_index(obs, opts, player):
    best_i, best_s = None, 0.0
    for i, o in enumerate(opts):
        if o.get("type") != ATTACH:
            continue
        s = _attach_score(obs, o, player)
        if s > best_s:
            best_s, best_i = s, i
    return best_i if best_s > 0.0 else None


def _retreat_pivot(obs, opts, player):
    """Active Cinderace (the energy engine, not an attacker) with a ready benched Mega Starmie ex -> retreat to
    promote the attacker. Pilots do this consistently; our agent never retreated (0% agreement)."""
    a = DP._active(player)
    if not a or DP._cid(a) != CINDERACE:
        return None
    ridx = next((i for i, o in enumerate(opts) if o.get("type") == RETREAT), None)
    if ridx is None:
        return None
    for b in DP._bench(player):
        if DP._cid(b) == MEGA_STARMIE and DP._attached_count(b) >= 1:
            return ridx
    return None


def _our_max_hit(player):
    """Our active's best single-hit this turn, by what it can actually pay for (engine offers payable attacks
    only; we approximate from attached energy). Mega Starmie: Nebula 210 at >=3 energy, Jetting 120 at >=1.
    Cinderace: Turbo Flare 50. Used to judge whether a gusted bench target is KO-able."""
    a = DP._active(player)
    if not a:
        return 0.0
    e = DP._attached_count(a)
    cid = DP._cid(a)
    if cid == MEGA_STARMIE:
        return 210.0 if e >= 3 else (120.0 if e >= 1 else 0.0)
    if cid == CINDERACE:
        return 50.0 if e >= 1 else 0.0
    return 0.0


def _boss_enables_ko(obs, player, opp):
    """Gust worth the card when we CANNOT KO the opponent's active (ensured by call order, after the KO floor)
    but CAN KO a benched target this turn -- a gust KO of ANY prize value is usually worth it (user's rule;
    the old 2+-prize requirement was far too strict -- most decks have no ex/mega to gust)."""
    bench = DP._bench(opp)
    if not bench:
        return False
    cap = _our_max_hit(player)
    if cap <= 0:
        return False
    return any(0 < float(DP._get(e, "hp", 0) or 0) <= cap for e in bench)


def _wally_useful(player):
    """Wally's heals a Mega ex to full + returns its energy to hand. Only worth a Supporter when the active
    Mega Starmie is genuinely in KO range next turn -- proxied as remaining HP <= half (<=165 of 330) so a
    typical big hit would otherwise KO it -- and it actually has energy to recover. NEVER on a near-full Mega
    Starmie (the deployed agent's pointless-Wally's bug). (Proxy for "opponent can KO me next turn"; a precise
    incoming-damage check is a refinement -- attack_stats is unreliable for conditional damage.)"""
    a = DP._active(player)
    if not a or DP._cid(a) != MEGA_STARMIE:
        return False
    hp = float(DP._get(a, "hp", 0) or 0)
    mx = _maxhp(MEGA_STARMIE)
    return mx > 0 and 0 < hp <= mx * 0.5 and DP._attached_count(a) >= 1


# ---------- rules ----------
def _go_first(obs):
    sel = DP._selection(obs)
    if not sel or int(DP._get(sel, "context", -1) or -1) != CTX_IS_FIRST:
        return None
    opts = DP._items(DP._get(sel, "option", []))
    for i, o in enumerate(opts):
        if o.get("type") == YES:
            return [i]
    return None


def _no_suicide(obs):
    sel = DP._selection(obs)
    if not sel or int(DP._get(sel, "maxCount", 0) or 0) != 1:
        return None
    opts = DP._items(DP._get(sel, "option", []))
    me = _me_opp(obs)[0]
    in_play = (1 if DP._active(me) else 0) + len(DP._bench(me))
    if in_play > 1:
        return None
    end_idx = next((i for i, o in enumerate(opts) if o.get("type") == END), None)
    if end_idx is not None and any(o.get("type") == ABILITY for o in opts):
        return [end_idx]
    return None


def _main_action(obs):
    """Force only the high-confidence mechanical moves; defer the ambiguous "which setup card / chip vs
    one-more-play" judgment to search (returning None). High-confidence (in order): win now; take a KO
    (Jetting-preferred); evolve Staryu->Mega Starmie; gust/heal/tool (free high-value plays); route energy onto
    the line; pivot Cinderace -> Mega Starmie. These are the imitation-gap wins that search gets wrong."""
    sel = DP._selection(obs)
    if not sel or int(DP._get(sel, "maxCount", 0) or 0) != 1:
        return None
    opts = DP._items(DP._get(sel, "option", []))
    if len(opts) < 2:
        return None
    types = {o.get("type") for o in opts}
    if not (types & {ATTACK, ATTACH, EVOLVE, ABILITY, PLAY, RETREAT, END}):
        return None
    player, opp, _ = _me_opp(obs)

    gw = _game_winning_attack(obs, opts, opp)
    if gw is not None:
        return [gw]
    ko = _best_ko_index(obs, opts, opp)
    if ko is not None:
        return [ko]
    for i, o in enumerate(opts):
        if o.get("type") == EVOLVE and DP.option_card_id(o, obs) == MEGA_STARMIE:
            return [i]
    hv = _high_value_play(obs, opts, player, opp)
    if hv is not None:
        return [hv]
    ea = _best_attach_index(obs, opts, player)
    if ea is not None:
        return [ea]
    rp = _retreat_pivot(obs, opts, player)
    if rp is not None:
        return [rp]
    return None   # defer the rest (which trainer / chip vs setup / end) to search


def _tutor_target(obs):
    """CARD-selection prompt (search/fetch): pick the card that fills our biggest need."""
    sel = DP._selection(obs)
    if not sel or int(DP._get(sel, "maxCount", 0) or 0) != 1:
        return None
    opts = DP._items(DP._get(sel, "option", []))
    if len(opts) < 2 or not all(o.get("type") == CARD for o in opts):
        return None
    player = _me_opp(obs)[0]
    need = _needs(player)
    ranked = []
    for i, o in enumerate(opts):
        try:
            cid = DP.option_card_id(o, obs)
        except Exception:
            cid = None
        if cid is None:
            cid = DP._selection_card_id(sel, o)
        ranked.append((_need_value(cid, need), i))
    ranked.sort(reverse=True)
    if ranked and ranked[0][0] > 0:
        return [ranked[0][1]]
    return None


def _need_value(cid, need):
    if cid is None:
        return 0.0
    if cid == MEGA_STARMIE:
        return 100.0 if not need["have_mega_play"] else 30.0     # the win condition
    if cid == STARYU:
        return 80.0 if not need["have_staryu"] else 25.0
    if cid == CINDERACE:
        return 70.0 if need["n_basics_play"] < 1 else 20.0       # the energy engine
    if cid == IGNITION:
        return 40.0
    if cid == BASIC_WATER:
        return 35.0
    if cid == HEROS_CAPE:
        return 45.0
    if cid == BOSS:
        return 30.0
    eff = CEFF.get(str(cid), {}) or {}
    if eff.get("search") or eff.get("search_to_bench"):
        return 28.0
    if eff.get("draw"):
        return 22.0
    return 10.0


# _tutor_target is defined but left OUT of RULES for now: search picks search-targets at ~63% agreement and a
# naive need-rank regressed it (52.6%). Re-enable once card resolution on deck-search prompts is solid.
RULES = (_go_first, _no_suicide, _main_action)


def _veto_search_pick(obs, mv):
    """Veto a KNOWN-bad search pick and replace it (don't just decline to force the good move -- prevent the
    bad one). Currently: search must not play Wally's Compassion on a Mega Starmie that isn't in KO range
    (the pointless full-HP Wally's); replace with the chip attack if one is available."""
    sel = DP._selection(obs)
    if not sel or int(DP._get(sel, "maxCount", 0) or 0) != 1 or len(mv) != 1:
        return None
    opts = DP._items(DP._get(sel, "option", []))
    i = mv[0]
    if not (0 <= i < len(opts)):
        return None
    o = opts[i]
    if o.get("type") == PLAY and DP.option_card_id(o, obs) == WALLYS:
        player, opp, _ = _me_opp(obs)
        if not _wally_useful(player):
            atk = _best_attack_index(obs, opts, opp)
            if atk is not None:
                return [atk]
    return None


def choose(obs):
    for rule in RULES:
        try:
            r = rule(obs)
        except Exception:
            r = None
        if r is not None and DP.valid_selection(obs, r):
            return r
    return None


def choose_action(obs, deck=STARMIE_DECK):
    """Heuristic-first, then forward-model search (with a veto on known-bad picks), then legal default."""
    if obs.get("select") is None:
        return list(deck)
    h = choose(obs)
    if h is not None:
        return list(h)
    try:
        import search_v3 as S
        S.USE_DYNAMIC_ATTACKS = True
        mv = S.best_option(obs, list(deck), leaf_mode="deckout", rollout_mode="develop")
        if mv:
            veto = _veto_search_pick(obs, list(mv))
            return list(veto) if (veto and DP.valid_selection(obs, veto)) else list(mv)
    except Exception:
        pass
    return DP.default_selection(obs)


def agent(obs):
    try:
        return choose_action(obs, STARMIE_DECK)
    except Exception:
        sel = obs.get("select")
        if sel is None:
            return list(STARMIE_DECK)
        n = len((sel or {}).get("option") or [])
        k = (sel or {}).get("minCount") or 1
        return list(range(min(max(k, 1), n))) if n else []
