"""L1: encode a live cabt observation into a feature vector a model/heuristic can read.

The point (per the plan): turn the raw board into the things a human reads instantly, using
the L0 card features (card_features.json). Two sources, deliberately:
  1. obs["current"] (the board): HP, prizes, attached energy -> energy AFFORDANCE.
  2. obs["select"]["option"] (the LEGAL plays): what you can ACTUALLY do right now, tagged by
     each card's functional role -> playability. A draw card in hand that is not playable this
     turn does not count; the engine's legal-option list is the truth.

Energy affordance is TYPE-AWARE: holding 4 Water when your attacker needs 4 Lightning is zero
valid energy even though you hold 4 cards (colorless cost = any; Rainbow/Team-Rocket energy =
wildcard). That mismatch is the feature, complemented by whether fixing (draw/tutor/accel) is
available to dig out of it.

Run `python agent/features.py` to print the encoded features for a real mid-game state.
"""
from __future__ import annotations

import json
import os
from collections import Counter

def _load_cf() -> dict:
    """Tolerant multi-path load (like main/_load, value_model, search): a missing file degrades to
    an empty dict -> zero-valued features, never an import-time crash that forfeits every game."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("card_features.json", os.path.join(here, "card_features.json"),
              os.path.join("/kaggle_simulations/agent", "card_features.json")):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


CF = _load_cf()

# EnergyType int (obs) -> cost letter (card data). Rainbow(10)/Team Rocket(11) = wildcard.
EN = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M", 9: "N", 10: "*", 11: "*"}
# OptionType
PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END = 7, 8, 9, 10, 12, 13, 14


def cf(cid) -> dict:
    return CF.get(str(cid), {})


def energy_shortfall(cost: list[str], attached: list[str]) -> int:
    """How many energy pips of `cost` are unmet by `attached`, type-aware. 0 = affordable."""
    avail = Counter(attached)
    wild = avail.pop("*", 0)
    short = 0
    colorless = 0
    for c in cost:
        if c == "C":
            colorless += 1
        elif avail.get(c, 0) > 0:
            avail[c] -= 1
        elif wild > 0:
            wild -= 1
        else:
            short += 1
    pool = sum(v for v in avail.values() if v > 0) + wild
    for _ in range(colorless):
        if pool > 0:
            pool -= 1
        else:
            short += 1
    return short


def _attached_letters(pkmn: dict) -> list[str]:
    return [EN.get(e, "C") for e in (pkmn.get("energies") or [])]


def _best_affordable(pkmn: dict):
    """(best affordable damage, min shortfall to any attack) for an in-play Pokemon."""
    c = cf(pkmn.get("id"))
    attached = _attached_letters(pkmn)
    best_dmg, min_short = 0, 99
    for a in c.get("atks", []):
        sh = energy_shortfall(a.get("cE", []), attached)
        min_short = min(min_short, sh)
        if sh <= 0:
            best_dmg = max(best_dmg, a.get("dmg", 0))
    return best_dmg, (min_short if min_short != 99 else 0)


def _active(p):
    a = p.get("active") or []
    return a[0] if a and a[0] else None


def _has_tag(cards, tag) -> int:
    return sum(1 for c in cards if tag in cf(c.get("id")).get("tags", []))


def encode_state(obs: dict) -> dict:
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    yi = cur.get("yourIndex", 0)
    me = players[yi] if yi < len(players) else {}
    opp = players[1 - yi] if len(players) > 1 else {}
    myA, opA = _active(me), _active(opp)
    hand = me.get("hand") or []

    f: dict[str, float] = {}
    # --- prize race & board ---
    f["my_prizes_left"] = len(me.get("prize") or [])
    f["opp_prizes_left"] = len(opp.get("prize") or [])
    f["prize_lead"] = f["opp_prizes_left"] - f["my_prizes_left"]
    f["my_bench"] = len(me.get("bench") or [])
    f["opp_bench"] = len(opp.get("bench") or [])
    f["hand_size"] = me.get("handCount", len(hand))
    f["deck_left"] = me.get("deckCount", 0)
    f["deckout_risk"] = 1.0 if f["deck_left"] <= 5 else 0.0
    f["my_active_hp"] = myA.get("hp", 0) if myA else 0
    f["opp_active_hp"] = opA.get("hp", 0) if opA else 0
    for cond in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        f["my_" + cond] = 1.0 if me.get(cond) else 0.0

    # --- rules: prize values & per-turn play limits (deterministic from card attrs + obs) ---
    f["opp_active_prize"] = cf(opA.get("id")).get("prize", 0) if opA else 0   # prizes I gain by KOing it
    f["my_active_prize"] = cf(myA.get("id")).get("prize", 0) if myA else 0    # prizes I give up if KO'd
    f["my_active_is_ex"] = 1.0 if (myA and (cf(myA.get("id")).get("ex") or cf(myA.get("id")).get("mega"))) else 0.0
    f["opp_active_is_ex"] = 1.0 if (opA and (cf(opA.get("id")).get("ex") or cf(opA.get("id")).get("mega"))) else 0.0
    f["supporter_available"] = 0.0 if cur.get("supporterPlayed") else 1.0
    f["stadium_available"] = 0.0 if cur.get("stadiumPlayed") else 1.0
    f["energy_attach_done"] = 1.0 if cur.get("energyAttached") else 0.0
    f["retreated_this_turn"] = 1.0 if cur.get("retreated") else 0.0

    # --- energy affordance on my active attacker (type-aware) ---
    if myA:
        dmg, short = _best_affordable(myA)
        f["active_energy_short"] = short            # pips still needed for its cheapest attack
        f["active_can_attack_now"] = 1.0 if short <= 0 else 0.0
        f["active_affordable_dmg"] = dmg
        # can that attack KO the opponent's active this turn?
        f["can_ko_opp_now"] = 1.0 if (opA and dmg >= opA.get("hp", 0) and dmg > 0) else 0.0
        f["ko_prize_value"] = f["opp_active_prize"] if f["can_ko_opp_now"] else 0.0
        f["active_n_energy"] = len(_attached_letters(myA))
    else:
        f["active_energy_short"] = 99
        f["active_can_attack_now"] = f["active_affordable_dmg"] = f["can_ko_opp_now"] = f["ko_prize_value"] = f["active_n_energy"] = 0.0

    # --- hand resources (what you hold, by role) ---
    f["hand_energy"] = _has_tag(hand, "basic_energy") + _has_tag(hand, "special_energy")
    f["hand_energy_accel"] = _has_tag(hand, "energy_accel")
    f["hand_draw"] = _has_tag(hand, "draw") + _has_tag(hand, "cycle")
    f["hand_tutor"] = _has_tag(hand, "tutor")
    f["hand_gust"] = _has_tag(hand, "gust")
    f["hand_heal"] = _has_tag(hand, "heal")
    f["fixing_available"] = 1.0 if (f["hand_draw"] or f["hand_tutor"] or f["hand_energy_accel"]) else 0.0

    # --- color match: distinct energy my attackers NEED vs what I can produce ---
    need = set()
    for p in ([myA] if myA else []) + (me.get("bench") or []):
        for a in cf((p or {}).get("id")).get("atks", []):
            need |= {c for c in a.get("cE", []) if c not in ("C", "*")}
    have = set(_attached_letters(myA) if myA else [])
    for c in hand:
        have |= {l for l in cf(c.get("id")).get("ty", "") if l}
    f["needed_colors"] = len(need)
    f["color_mismatch"] = len(need - have)          # colors I need but cannot currently produce

    # --- playability: what is ACTUALLY legal right now (engine truth), tagged by role ---
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    f["n_legal"] = len(opts)
    types = Counter(o.get("type") for o in opts)
    f["can_attack_legal"] = 1.0 if types.get(ATTACK) else 0.0
    f["can_attach_energy"] = 1.0 if types.get(ATTACH) else 0.0
    f["can_evolve"] = 1.0 if types.get(EVOLVE) else 0.0
    f["can_use_ability"] = 1.0 if types.get(ABILITY) else 0.0
    f["can_retreat"] = 1.0 if types.get(RETREAT) else 0.0
    # role of cards that are legally PLAYABLE from hand this decision
    play_roles = Counter()
    for o in opts:
        if o.get("type") == PLAY and o.get("index") is not None and o["index"] < len(hand):
            for t in cf(hand[o["index"]].get("id")).get("tags", []):
                play_roles[t] += 1
    f["draw_playable_now"] = 1.0 if play_roles.get("draw") or play_roles.get("cycle") else 0.0
    f["tutor_playable_now"] = 1.0 if play_roles.get("tutor") else 0.0
    f["gust_playable_now"] = 1.0 if play_roles.get("gust") else 0.0
    return f


FEATURE_KEYS = sorted(encode_state({"current": {"players": [{}, {}], "yourIndex": 0}, "select": {}}).keys())


def vectorize(f: dict) -> list[float]:
    return [float(f.get(k, 0.0)) for k in FEATURE_KEYS]


if __name__ == "__main__":
    import contextlib, io, logging
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import kaggle_environments.envs.cabt.cabt as cabt
        from kaggle_environments import make
    grabbed = {}

    def probe(o):
        if o.get("select") is not None and o.get("current") and 12 <= o["current"].get("turn", 0) <= 18 and "f" not in grabbed:
            grabbed["f"] = encode_state(o)
            grabbed["turn"] = o["current"]["turn"]
        sel = o.get("select")
        return cabt.deck if sel is None else list(range(sel.get("maxCount", 1) or 1))

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        make("cabt").run([probe, cabt.first_agent])
    if "f" in grabbed:
        print(f"encoded state at turn {grabbed['turn']} ({len(grabbed['f'])} features):")
        for k in FEATURE_KEYS:
            print(f"  {k:22s} {grabbed['f'][k]}")
    else:
        print("no mid-game decision captured; run again")
