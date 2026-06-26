"""Learned-proposer ADAPTER + offline SAFETY-FILTER spec (bridge adapter v0). DISABLED BY DEFAULT.

Prepares the Starmie agent to LATER consume a Model A runtime proposer artifact, WITHOUT enabling it and WITHOUT
changing gameplay. This module is standalone: it is NOT imported by starmie_heuristics.choose_action, so it
cannot affect the live agent. It provides:
  - a canonical semantic-action-key mapper (must match the Model B trace format + Model A proposer keys);
  - a disabled adapter interface (load_proposer / rank_actions) that returns DISABLED until a real artifact is
    wired and explicitly enabled (env STARMIE_PROPOSER_ENABLED=1, default off);
  - offline hard-veto SAFETY filters (V0..V8) -- callable, never active in live selection.

Nothing here selects a live action. The proposer remains a candidate generator behind safety gates until an
explicit, separately-tested integration.
"""
from __future__ import annotations

import os

import deck_policy_v3 as DP
import starmie_heuristics as SH

# Disabled by default. Even when "enabled", this module never wires into choose_action by itself.
PROPOSER_ENABLED = os.environ.get("STARMIE_PROPOSER_ENABLED", "") == "1"

PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END, CARD, YES, NO = 7, 8, 9, 10, 11, 12, 13, 14, 3, 1, 2
_ROLE = {SH.MEGA_STARMIE: "Mega", SH.CINDERACE: "Cinderace", SH.STARYU: "Staryu",
         SH.IGNITION: "Ignition", SH.BASIC_WATER: "Water"}
_ATTACK_NAME = {SH.JETTING_BLOW: "Jetting", SH.NEBULA_BEAM: "Nebula", SH.TURBO_FLARE: "Turbo Flare"}


# ---------------------------------------------------------------------------- semantic action keys
def _name(cid):
    return (SH.CDB.get(str(cid), {}) or {}).get("n", f"#{cid}") if cid is not None else None


def semantic_key(option, obs):
    """Canonical semantic action key for a RAW obs option. MUST stay consistent with build_bridge_trace_v0._semkey
    so Model A proposer outputs (semantic keys) join to the live legal options."""
    t = option.get("type")
    if t == ATTACK:
        aid = option.get("attackId")
        return f"ATTACK:{_ATTACK_NAME.get(aid, aid)}"
    if t in (PLAY, ATTACH, EVOLVE, ABILITY, CARD):
        try:
            cid = DP.option_card_id(option, obs)
        except Exception:
            cid = None
        if t == ATTACH:
            try:
                tgt = DP.option_target_entity(option, obs)
            except Exception:
                tgt = None
            trole = _ROLE.get(DP._cid(tgt)) if tgt else "?"
            return f"ATTACH:{_ROLE.get(cid, _name(cid))}:{trole}"
        fam = {PLAY: "PLAY", EVOLVE: "EVOLVE", ABILITY: "ABILITY", CARD: "SELECT_CARD"}[t]
        return f"{fam}:{_ROLE.get(cid) or _name(cid)}"
    if t == RETREAT:
        # role of the active being retreated (target-role for symmetry with the spec)
        a = None
        try:
            a = DP._active((obs.get("current") or {}).get("players", [{}])[(obs.get("current") or {}).get("yourIndex", 0)])
        except Exception:
            a = None
        return f"RETREAT:{_ROLE.get(DP._cid(a)) if a else '?'}"
    return {END: "END", YES: "YES", NO: "NO", DISCARD: "DISCARD"}.get(t, str(t))


def option_index_to_key(obs):
    sel = DP._selection(obs) or {}
    opts = DP._items(DP._get(sel, "option", []))
    return {i: semantic_key(o, obs) for i, o in enumerate(opts)}


def key_to_indices(obs):
    """Reverse map: semantic key -> [raw option indices] (a key can cover several equivalent indices)."""
    out = {}
    for i, k in option_index_to_key(obs).items():
        out.setdefault(k, []).append(i)
    return out


# ---------------------------------------------------------------------------- disabled adapter interface
class ProposerHandle:
    def __init__(self, path=None, status="DISABLED", model_hash=None, feature_hash=None):
        self.path, self.status = path, status
        self.model_hash, self.feature_hash = model_hash, feature_hash


def load_proposer(path=None):
    """Load a runtime proposer artifact. Returns a handle whose status is MISSING (no file / not provided) or
    DISABLED (file present but proposer not enabled). A real runtime artifact + inference is Model A's to provide;
    until then this never produces rankings."""
    if not path or not os.path.exists(path):
        return ProposerHandle(path=path, status="MISSING")
    return ProposerHandle(path=path, status="DISABLED")   # present but intentionally not run here


def rank_actions(handle, obs, legal_options=None, top_k=5):
    """Disabled adapter: returns a well-formed ProposerResult with status DISABLED/MISSING and NO ranked actions,
    so it can never change the agent's action. When a real runtime proposer is wired (Model A), this is where its
    inference output would be normalized into this schema."""
    status = "DISABLED" if (handle and handle.status == "DISABLED" and PROPOSER_ENABLED) else \
             (handle.status if handle else "MISSING")
    return {
        "status": status if status in ("MISSING", "DISABLED", "UNSUPPORTED", "ERROR", "READY") else "DISABLED",
        "model_hash": getattr(handle, "model_hash", None), "feature_hash": getattr(handle, "feature_hash", None),
        "ranked_actions": [],          # always empty here -> cannot influence selection
        "entropy": None, "top1_margin": None,
        "diagnostics": {"note": "adapter is offline/disabled; no live integration", "index_to_key": option_index_to_key(obs)},
    }


# ---------------------------------------------------------------------------- offline hard-veto safety filters
def _ctx(obs):
    sel = DP._selection(obs) or {}
    opts = DP._items(DP._get(sel, "option", []))
    me, opp, _ = SH._me_opp(obs)
    return sel, opts, me, opp


def _opt(opts, i):
    return opts[i] if (isinstance(i, int) and 0 <= i < len(opts)) else None


def v0_illegal_or_stale(obs, i):
    sel, opts, _, _ = _ctx(obs)
    ok = isinstance(i, int) and 0 <= i < len(opts)
    return {"veto": not ok, "reason": "proposed option index is illegal/stale" if not ok else "", "confidence": "hard", "evidence": {"n_legal": len(opts), "index": i}}


def v1_misses_game_winning_attack(obs, i):
    sel, opts, me, opp = _ctx(obs)
    gw = SH._game_winning_attack(obs, opts, opp)
    veto = gw is not None and i != gw
    return {"veto": veto, "reason": "a game-winning attack is available but not proposed" if veto else "", "confidence": "hard", "evidence": {"game_winning_index": gw, "proposed": i}}


def v2_loses_guaranteed_ko(obs, i):
    sel, opts, me, opp = _ctx(obs)
    ko = SH._best_ko_index(obs, opts, opp)
    o = _opt(opts, i)
    # a guaranteed KO exists, the proposed action is not an attack, and it is not a higher-value Boss/gust KO setup
    proposed_is_attack = bool(o and o.get("type") == ATTACK)
    veto = ko is not None and not proposed_is_attack
    return {"veto": veto, "reason": "a guaranteed KO is available but the proposed action is not an attack" if veto else "", "confidence": "soft", "evidence": {"ko_index": ko, "proposed_is_attack": proposed_is_attack}}


def v3_causes_deckout(obs, i):
    sel, opts, me, opp = _ctx(obs)
    o = _opt(opts, i)
    deck = int(DP._get(me, "deckCount", 99) or 99)
    cid = DP.option_card_id(o, obs) if o else None
    eff = (SH.CEFF.get(str(cid), {}) or {}) if cid else {}
    draws = int(eff.get("draw", 0) or 0)
    veto = bool(o and o.get("type") == PLAY and deck <= 4 and draws >= deck)
    return {"veto": veto, "reason": "play would draw past an near-empty deck (deck-out)" if veto else "", "confidence": "hard", "evidence": {"deck": deck, "draws": draws}}


def v4_wally_strips_ko_energy(obs, i):
    sel, opts, me, opp = _ctx(obs)
    o = _opt(opts, i)
    ko = SH._best_ko_index(obs, opts, opp)
    is_wally = bool(o and o.get("type") == PLAY and DP.option_card_id(o, obs) == SH.WALLYS)
    veto = is_wally and ko is not None   # playing Wally returns the attacker's energy -> would lose the available KO
    return {"veto": veto, "reason": "Wally's Compassion would return energy needed for an available KO" if veto else "", "confidence": "soft", "evidence": {"ko_index": ko, "is_wally": is_wally}}


def v5_boss_retargets_active_ko(obs, i):
    sel, opts, me, opp = _ctx(obs)
    o = _opt(opts, i)
    ko = SH._best_ko_index(obs, opts, opp)
    is_boss = bool(o and o.get("type") == PLAY and DP.option_card_id(o, obs) == SH.BOSS)
    veto = is_boss and ko is not None   # gusting when the active is already KO-able re-targets the KO worse
    return {"veto": veto, "reason": "Boss gust when the active is already KO-able re-targets the KO" if veto else "", "confidence": "soft", "evidence": {"ko_index": ko, "is_boss": is_boss}}


def v6_ignition_misuse(obs, i):
    sel, opts, me, opp = _ctx(obs)
    o = _opt(opts, i)
    if not (o and o.get("type") == ATTACH and DP.option_card_id(o, obs) == SH.IGNITION):
        return {"veto": False, "reason": "", "confidence": "soft", "evidence": {}}
    score = SH._attach_score(obs, o, me)   # >0 only when Ignition funds a Nebula KO this turn
    veto = score <= 0
    return {"veto": veto, "reason": "Ignition attached without same-turn Nebula value" if veto else "", "confidence": "soft", "evidence": {"attach_score": score}}


def v7_bad_retreat(obs, i):
    sel, opts, me, opp = _ctx(obs)
    o = _opt(opts, i)
    if not (o and o.get("type") == RETREAT):
        return {"veto": False, "reason": "", "confidence": "soft", "evidence": {}}
    # retreat is only clearly safe when a ready Mega attacker is benched to promote
    has_ready_backup = any(DP._cid(b) == SH.MEGA_STARMIE and SH._energy_units(b) >= 1 for b in DP._bench(me))
    veto = not has_ready_backup
    return {"veto": veto, "reason": "retreat with no ready benched Mega attacker to promote" if veto else "", "confidence": "soft", "evidence": {"has_ready_backup": has_ready_backup}}


def v8_unsupported_action(obs, i):
    sel, opts, _, _ = _ctx(obs)
    o = _opt(opts, i)
    deck = list(SH.STARMIE_DECK)
    # OOD = the deck in play is not our exact Starmie deck (support cannot be assumed) OR mapping unknown
    key = semantic_key(o, obs) if o else None
    veto = key is None
    return {"veto": veto, "reason": "proposed action has no semantic-key mapping" if veto else "", "confidence": "hard", "evidence": {"semantic_key": key}}


SAFETY_FILTERS = [v0_illegal_or_stale, v1_misses_game_winning_attack, v2_loses_guaranteed_ko, v3_causes_deckout,
                  v4_wally_strips_ko_energy, v5_boss_retargets_active_ko, v6_ignition_misuse, v7_bad_retreat,
                  v8_unsupported_action]


def safety_check(obs, proposed_index):
    """Run all offline hard-veto filters on a hypothetical proposer override. Returns the per-filter verdicts +
    an aggregate veto. NOT active in live selection -- a first gate spec only (the list is not claimed complete)."""
    results = {}
    veto = False
    for f in SAFETY_FILTERS:
        try:
            r = f(obs, proposed_index)
        except Exception as e:
            r = {"veto": False, "reason": f"error: {e!r}", "confidence": "error", "evidence": {}}
        results[f.__name__] = r
        veto = veto or (r["veto"] and r["confidence"] == "hard")
    soft = [k for k, r in results.items() if r["veto"] and r["confidence"] != "hard"]
    return {"hard_veto": veto, "soft_flags": soft, "filters": results}
