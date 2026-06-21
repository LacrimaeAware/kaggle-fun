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
import time
from collections import Counter

import features as FT        # L1 state encoder (card_features.json bundled alongside)

ATTACK, YES, ATTACH = 13, 1, 8   # OptionType
PLAY, EVOLVE, ABILITY, END = 7, 9, 10, 14   # OptionType (play hand card / evolve / ability / end turn)
A_ACTIVE = 4                 # AreaType.ACTIVE (attach target)
A_HAND = 2                   # AreaType.HAND (an option that plays from hand -> index into hand)
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
CEFF = _load("card_effects.json")  # id(str) -> {draw,search,energy_accel,heal,...} (decoded effects)

# Deck = DENPA92's Dudunsparce/Alakazam list, lifted from the replay DB (tools/build_replay_db.py).
# Chosen by measurement, not by ladder rank: deck strength is policy-coupled. The #1 player's
# Iono-Bellibolt evolution deck scores 0.83 on the ladder but only ~0.21 under OUR heuristic (our
# agent can't pilot an evolution engine). DENPA92's deck suits our policy: 0.738 head-to-head vs the
# old deck under our heuristic (Wilson95 [0.63,0.82], n=80), ~0.55 under search. It also has 8 basic
# Pokemon vs the old 6, fixing a mulligan outlier (our old deck shuffled up to 6x at setup; field
# mean 0.26). Old deck (Kyogre/Snover/Mega-Abomasnow, 6 basics, 0.44 ladder wr) kept for reference:
#   [721]*2 + [722]*4 + [723]*4 + [1092] + [1121]*2 + [1145]*2 + [1163]*2 + [1219]*4 + [1227]*4 + [1262]*2 + [3]*33
DECK = (
    [5] * 3 + [19] * 4 + [65] * 4 + [66] * 4 + [741] * 4 + [742] * 4 + [743] * 3
    + [1079] * 3 + [1081] * 3 + [1086] * 4 + [1097] + [1129] + [1146] + [1152] * 4
    + [1159] + [1182] * 3 + [1184] + [1225] * 4 + [1231] * 4 + [1264] * 4
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


def _forced_move(obs: dict):
    """Clean, always-correct overrides to bake into search (no lookahead needed): take a listed
    lethal/KO attack, or go first when offered. Returns [i] or None. (Energy-to-active is left to
    search/heuristic since it is borderline and worth learning.)"""
    sel = obs.get("select")
    if not sel or (sel.get("maxCount") or 0) != 1:
        return None
    opts = sel.get("option") or []
    if not opts:
        return None
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    yi = cur.get("yourIndex", 0)
    me = players[yi] if yi < len(players) else {}
    opp = players[1 - yi] if len(players) > 1 else {}
    attacks = [(i, o) for i, o in enumerate(opts) if o.get("type") == ATTACK]
    if attacks:
        i, o = max(attacks, key=lambda io: _attack_value(io[1], me, opp))
        if _attack_value(o, me, opp) >= 8000:            # knockout / game-winning attack
            return [i]
    if sel.get("context") == IS_FIRST_CTX:               # go first
        for i, o in enumerate(opts):
            if o.get("type") == YES:
                return [i]
    return None


def _agent_search(obs: dict, leaf_mode: str, opp_k: int = 0) -> list[int]:
    try:
        if obs.get("select") is None:
            return list(DECK)
        mv = _forced_move(obs)                            # bake in the clean heuristics first
        if mv is not None:
            return mv
        import search
        mv = search.best_option(obs, DECK, leaf_mode=leaf_mode, opp_k=opp_k)
        if mv is not None:
            return mv
    except Exception:
        pass
    return agent(obs)


def agent_search_ctx(obs: dict) -> list[int]:
    """Contextual-ranker-guided hand-search.

    The learned model only orders/tie-breaks legal siblings. Search still
    evaluates candidates with the forward model and remains the final chooser.
    Contextual scoring time is debited from the normal per-decision budget so
    screens compare against standalone search under the same wall-clock cap.
    """
    try:
        if obs.get("select") is None:
            return list(DECK)
        mv = _forced_move(obs)
        if mv is not None:
            return mv
        import contextual_ranker
        import search
        t0 = time.time()
        scores = contextual_ranker.score_options(obs, DECK)
        elapsed = time.time() - t0
        if scores:
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            budget = max(0.03, search.DEFAULT_BUDGET - elapsed)
            mv = search.best_option(
                obs,
                DECK,
                time_budget=budget,
                leaf_mode="hand",
                option_order=order,
                option_prior=scores,
            )
            if mv is not None:
                return mv
    except Exception:
        pass
    return agent_search(obs)


def agent_search(obs: dict) -> list[int]:
    """Forward-model search with the HAND leaf eval; heuristic floor + fallback. Never raises."""
    return _agent_search(obs, "hand")


def agent_search2(obs: dict) -> list[int]:
    """2-ply: hand-eval search that branches on the opponent's best reply (min over top-k punishes)."""
    return _agent_search(obs, "hand", opp_k=2)


def agent_search_v(obs: dict) -> list[int]:
    """Forward-model search with the LEARNED value at the leaves (L2); heuristic floor + fallback."""
    return _agent_search(obs, "learned")


def agent_combine(obs: dict) -> list[int]:
    """Combine v1: clean heuristic floor + forward-model search with the BLENDED leaf eval
    (hand eval for local ranking + learned value for global judgment). Never raises."""
    return _agent_search(obs, "blend")


def agent_rank(obs: dict) -> list[int]:
    """The DISTILLED learned policy: pick the option the trained action-ranker scores highest (card
    embedding + effects + action descriptor + root + forward-model deltas), instant, no search.
    Clean forced-move floor first (lethal/go-first), then the net, then the heuristic fallback."""
    try:
        if obs.get("select") is None:
            return list(DECK)
        mv = _forced_move(obs)
        if mv is not None:
            return mv
        import ranker
        mv = ranker.predict(obs, DECK)
        if mv is not None:
            return mv
    except Exception:
        pass
    return agent(obs)


def agent_rank_hybrid(obs: dict) -> list[int]:
    """Conservative integration: the learned net decides ONLY on STRATEGIC decisions (its training
    distribution); the strong heuristic pilots everything else. Standalone agent_rank loses because it
    is off-distribution on the non-strategic majority of a game; this isolates the net's strategic picks."""
    try:
        if obs.get("select") is None:
            return list(DECK)
        mv = _forced_move(obs)
        if mv is not None:
            return mv
        import ranker
        if ranker.is_strategic(obs):
            mv = ranker.predict(obs, DECK)
            if mv is not None:
                return mv
    except Exception:
        pass
    return agent(obs)


def _opt_card_id(o: dict, me: dict):
    """The card a hand-option plays: area==HAND -> me.hand[index].id (the reliable join)."""
    if o.get("area") == A_HAND:
        idx = o.get("index")
        hand = me.get("hand") or []
        if isinstance(idx, int) and 0 <= idx < len(hand):
            c = hand[idx]
            return (c.get("id") if isinstance(c, dict) else c)
    return None


def _eff_score(o: dict, me: dict, opp: dict, obs: dict) -> float:
    """Effect-aware score for one option, using decoded card effects (CEFF) + floor rules. Scale is
    shared with _attack_value so setup plays can outrank a weak attack but never a KO (>=8000)."""
    t = o.get("type")
    if t == ATTACK:
        return _attack_value(o, me, opp)
    if t == END:
        return -1000.0
    if t == ABILITY:
        cid = _opt_card_id(o, me)
        return 3000.0 if CEFF.get(str(cid), {}).get("energy_accel") else 1500.0   # use engine abilities
    if t == ATTACH and o.get("inPlayArea") == A_ACTIVE:
        f = FT.encode_state(obs)
        return 3000.0 if (f.get("active_energy_short", 0) > 0 and not f.get("energy_attach_done")) else 600.0
    if t == EVOLVE:
        return 3500.0                                       # evolve toward the attacker
    cid = _opt_card_id(o, me)
    if cid is None:
        return 0.0
    e = CEFF.get(str(cid), {})
    cs = _cs(cid)
    s = 0.0
    if e.get("search_to_bench"):
        s += 4000.0                                         # Poffin-class: fetch basics, develop
    elif e.get("search"):
        s += 3000.0 + 80.0 * e.get("search", 0)
    if e.get("draw"):
        s += 2200.0 + 150.0 * e.get("draw", 0)
    if e.get("energy_accel"):
        s += 2600.0
    if e.get("recover_discard"):
        s += 1500.0
    if e.get("switch_gust"):
        s += 1200.0
    if s == 0.0 and cs.get("hp"):                           # a plain Pokemon: play it to develop board
        s = 2500.0 if not _active(me) else 1800.0
    return s


def _choose_eff(obs: dict) -> list[int]:
    """Effect-aware heuristic: KO/lethal floor, then pick the option with the best effect-aware score
    (setup plays valued via CEFF), else the safe default order. Tests whether decoded card effects let
    a no-search policy pilot engine decks better."""
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
            if _attack_value(o, me, opp) >= 8000:           # take a KO / game-winning attack
                return [i]
        if sel.get("context") == IS_FIRST_CTX:
            for i, o in enumerate(opts):
                if o.get("type") == YES:
                    return [i]
        best_i, best_s = None, 0.0
        for i, o in enumerate(opts):
            s = _eff_score(o, me, opp, obs)
            if best_i is None or s > best_s:
                best_s, best_i = s, i
        if best_i is not None and best_s > 0.0:
            return [best_i]
    return list(range(max(min(k, n), min(mn, n))))


def agent_eff(obs: dict) -> list[int]:
    """Effect-aware heuristic (no search): uses decoded card effects (card_effects.json) to value
    setup plays. Never raises."""
    try:
        return _choose_eff(obs)
    except Exception:
        return agent(obs)

# ============================================================================
# Candidate V3: dynamic attacks + turn-aware KO sequencing + fail-closed targets
# ============================================================================

_V3_RESOLVERS: tuple[str, ...] = ()
V3_STATS = Counter()


def reset_v3_stats() -> None:
    V3_STATS.clear()


def get_v3_stats() -> dict:
    return dict(V3_STATS)


def _v3_is_subdecision(obs: dict) -> bool:
    """Target/multi-select prompt rather than a normal strategic root choice."""
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    if not options:
        return False
    if (sel.get("maxCount") or 0) != 1:
        return True
    strategic = {PLAY, ATTACH, EVOLVE, ABILITY, 12, ATTACK, END}
    return all(option.get("type") not in strategic for option in options)


def _v3_budget(obs: dict, *, n_determ: int) -> float:
    """Conservative use of the cumulative time bank when it is exposed.

    This is intentionally separate from the strategy logic.  It reserves at
    least 120 seconds and caps any one decision.  Missing/invalid timing fields
    fall back to fixed safe values.
    """
    fallback = 4.0 if n_determ >= 32 else 1.0
    raw = obs.get("remainingOverageTime")
    try:
        remaining = float(raw)
    except Exception:
        return fallback
    if remaining <= 0:
        return 0.05
    spendable = max(0.0, remaining - 120.0)
    # Budget as though roughly 80 strategic decisions remain; cap hard.
    return max(0.15, min(6.0 if n_determ >= 32 else 1.5, spendable / 80.0))


def _v3_search_call(obs: dict, *, n_determ: int, allowed_indices=None):
    import search_v3 as S
    previous = getattr(S, "USE_DYNAMIC_ATTACKS", False)
    S.USE_DYNAMIC_ATTACKS = True
    try:
        return S.best_option(
            obs,
            DECK,
            leaf_mode="hand",
            time_budget=_v3_budget(obs, n_determ=n_determ),
            allowed_indices=allowed_indices,
            n_determ=n_determ,
            require_complete_world=True,
        )
    finally:
        S.USE_DYNAMIC_ATTACKS = previous


def _agent_search_v3(obs: dict, *, n_determ: int = 8, resolvers=_V3_RESOLVERS) -> list[int]:
    """Search-authoritative candidate.

    Important sequencing rule: an attack ends the turn.  If an ordinary KO is
    already available, V3 does not blindly skip all setup and does not broadly
    disable the KO floor.  It lets search compare ATTACK NOW against a tiny set
    of mechanically constrained, bench-only setup actions, while excluding END
    and unrelated cards.  A failed comparison falls back to taking the KO.
    """
    try:
        V3_STATS["decisions"] += 1
        if obs.get("select") is None:
            V3_STATS["deck_requests"] += 1
            return list(DECK)
        sel = obs.get("select") or {}
        options = sel.get("option") or []
        if not options:
            return []

        import deck_policy_v3 as DP3
        import search_v3 as S

        # Go first remains a deterministic setup choice.
        if sel.get("context") == IS_FIRST_CTX:
            for index, option in enumerate(options):
                if option.get("type") == YES:
                    V3_STATS["go_first"] += 1
                    return [index]

        # Target and multi-select prompts: heuristic may propose, paired search
        # validates, and any uncertainty returns the original default.
        if _v3_is_subdecision(obs):
            V3_STATS["subdecisions"] += 1
            default = DP3.default_selection(obs)
            candidates = DP3.propose(obs, enabled=resolvers)
            V3_STATS["proposal_count"] += len(candidates)
            if candidates:
                previous = getattr(S, "USE_DYNAMIC_ATTACKS", False)
                S.USE_DYNAMIC_ATTACKS = True
                try:
                    chosen, stats = DP3.compare_selections(
                        obs,
                        DECK,
                        candidates,
                        default,
                        S,
                        time_budget=min(1.5, _v3_budget(obs, n_determ=n_determ)),
                        n_determ=min(n_determ, 12),
                        minimum_worlds=4,
                        allow_tie=False,
                    )
                finally:
                    S.USE_DYNAMIC_ATTACKS = previous
                V3_STATS["proposal_worlds"] += int(stats.get("worlds", 0) or 0)
                V3_STATS["proposal_incomplete"] += int(bool(stats.get("incomplete")))
                if stats.get("accepted") and chosen is not None:
                    V3_STATS["proposal_accepted"] += 1
                    return list(chosen)
                V3_STATS["proposal_rejected"] += 1
            V3_STATS["subdecision_default"] += 1
            return default

        # Dynamic KO detection, including Powerful Hand damage counters.
        ko = DP3.best_ko_attack(obs)
        if ko is not None:
            V3_STATS["ko_available"] += 1
            attack_index, profile = ko
            if profile.get("game_win"):
                V3_STATS["game_win_forced"] += 1
                return [attack_index]

            # Preserve the proven ordinary-KO floor, but permit one safe setup
            # action before the attack when search validates it.
            safe_setup = DP3.safe_pre_attack_indices(obs, attack_index)
            V3_STATS["safe_pre_attack_options"] += len(safe_setup)
            if safe_setup:
                V3_STATS["ko_window_search"] += 1
                move = _v3_search_call(
                    obs,
                    n_determ=n_determ,
                    allowed_indices=[attack_index] + safe_setup,
                )
                if move is not None:
                    V3_STATS["ko_window_search_used"] += 1
                    V3_STATS["ko_window_deferred"] += int(move[0] != attack_index)
                    return move
            V3_STATS["ordinary_ko_forced"] += 1
            return [attack_index]

        V3_STATS["root_search_calls"] += 1
        move = _v3_search_call(obs, n_determ=n_determ)
        if move is not None:
            V3_STATS["root_search_used"] += 1
            return move
    except Exception:
        V3_STATS["exceptions"] += 1

    # Fail closed.  Before the old fallback, preserve a dynamic KO that the
    # printed-zero Powerful Hand bug would otherwise miss.
    try:
        import deck_policy_v3 as DP3
        ko = DP3.best_ko_attack(obs)
        if ko is not None:
            V3_STATS["fallback_dynamic_ko"] += 1
            return [ko[0]]
    except Exception:
        V3_STATS["fallback_exceptions"] += 1
    V3_STATS["fallback_agent"] += 1
    return agent(obs)


def agent_search_v3(obs: dict) -> list[int]:
    """Dynamic PH + turn-aware KO window; no target resolver enabled."""
    return _agent_search_v3(obs, n_determ=8, resolvers=())


def agent_search_v3_poffin(obs: dict) -> list[int]:
    """V3 plus only the Poffin target proposer."""
    return _agent_search_v3(obs, n_determ=8, resolvers=("poffin",))


def agent_search_v3_boss(obs: dict) -> list[int]:
    """V3 plus only the Boss target proposer; trace schema before promotion."""
    return _agent_search_v3(obs, n_determ=8, resolvers=("boss",))


def agent_search_v3_targets(obs: dict) -> list[int]:
    """V3 plus both implemented target proposers; combine only after ablations."""
    return _agent_search_v3(obs, n_determ=8, resolvers=("poffin", "boss"))


def agent_search_v3_n32(obs: dict) -> list[int]:
    """V3 plus the separately promising N=32 sampling axis, no resolvers."""
    return _agent_search_v3(obs, n_determ=32, resolvers=())


def agent_search_v3_n32_poffin(obs: dict) -> list[int]:
    """N32 + Poffin only; keep separate until both axes survive alone."""
    return _agent_search_v3(obs, n_determ=32, resolvers=("poffin",))


def agent_search_phfix(obs: dict) -> list[int]:
    """Minimal ablation: legacy search plus PH-aware automatic KO detection."""
    try:
        if obs.get("select") is None:
            return list(DECK)
        import deck_policy_v3 as DP3
        ko = DP3.best_ko_attack(obs)
        if ko is not None:
            return [ko[0]]
        if (obs.get("select") or {}).get("context") == IS_FIRST_CTX:
            for index, option in enumerate((obs.get("select") or {}).get("option") or []):
                if option.get("type") == YES:
                    return [index]
        move = _v3_search_call(obs, n_determ=8)
        if move is not None:
            return move
    except Exception:
        pass
    return agent(obs)


# --- PH as VISIBILITY, not a forced command (the pro's decomposition) ------------------
def _agent_search_phvis(obs: dict, *, n_determ: int = 8, dynamic: bool = True,
                        force_final: bool = False) -> list[int]:
    """Production agent_search flow, but the forward search/rollout SEE Powerful Hand = 20 x hand
    (dynamic=True). The production forced rule (_forced_move: static KO + go-first) is unchanged, so
    PH is never force-attacked; search decides it. force_final additionally forces PH ONLY when it
    immediately wins the game. This isolates PH visibility (PH_DAMAGE_FIX) from the 'force every KO'
    strategic policy, and the game-win-only force (PH_FORCE_FINAL)."""
    try:
        if obs.get("select") is None:
            return list(DECK)
        if force_final:
            import deck_policy_v3 as DP3
            ko = DP3.best_ko_attack(obs)
            if ko is not None and ko[1].get("game_win"):
                return [ko[0]]
        mv = _forced_move(obs)                       # production: static KO + go-first (misses PH by design)
        if mv is not None:
            return mv
        import search_v3 as S
        previous = getattr(S, "USE_DYNAMIC_ATTACKS", False)
        S.USE_DYNAMIC_ATTACKS = bool(dynamic)
        try:
            mv = S.best_option(obs, DECK, leaf_mode="hand", n_determ=n_determ,
                               time_budget=_v3_budget(obs, n_determ=n_determ),
                               require_complete_world=True)
        finally:
            S.USE_DYNAMIC_ATTACKS = previous
        if mv is not None:
            return mv
    except Exception:
        pass
    return agent(obs)


def agent_search_ph_vis(obs: dict) -> list[int]:
    """PH_DAMAGE_FIX: search sees Powerful Hand; nothing PH-forced (static KO + go-first only)."""
    return _agent_search_phvis(obs, n_determ=8, dynamic=True, force_final=False)


def agent_search_ph_vis_final(obs: dict) -> list[int]:
    """PH_DAMAGE_FIX + PH_FORCE_FINAL: + force Powerful Hand only when it wins the game."""
    return _agent_search_phvis(obs, n_determ=8, dynamic=True, force_final=True)


def agent_search_s32_plain(obs: dict) -> list[int]:
    """S32 sampling axis: production forced rule + N=32 STATIC search (no PH visibility)."""
    return _agent_search_phvis(obs, n_determ=32, dynamic=False, force_final=False)


def agent_search_s32_ph_vis(obs: dict) -> list[int]:
    """S32 + PH_DAMAGE_FIX."""
    return _agent_search_phvis(obs, n_determ=32, dynamic=True, force_final=False)


def agent_search_s32_ph_vis_final(obs: dict) -> list[int]:
    """S32 + PH_DAMAGE_FIX + PH_FORCE_FINAL."""
    return _agent_search_phvis(obs, n_determ=32, dynamic=True, force_final=True)
