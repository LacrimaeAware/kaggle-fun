"""Contextual sibling-action ranker V1.

Scores every legal sibling action from the same root state. This is not a global
state-value model and it is not a hard floor: search may use the scores as an
ordering/prior while keeping simulated values as final authority.
"""
from __future__ import annotations

import json
import os

import numpy as np

import features as FT
import search as S
import state_action_schema_v2 as SCH

OTYPES = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14]
EFFECT_KEYS = ["draw", "search", "search_to_bench", "energy_accel", "heal", "switch_gust",
               "recover_discard", "disrupt", "discard_cost", "status", "has_ability"]
DELTA_KEYS = ["prizes_taken", "opp_prizes_taken", "opp_ko", "dmg_dealt", "cards_drawn",
              "energy_attached", "board_dev", "deck_used", "discard_gain", "ends_turn",
              "wins_now", "loses_now"]
CTX_KEYS = ["active_energy_short", "hand_size", "my_bench", "opp_bench", "opp_active_hp",
            "can_ko_opp_now", "can_attach_energy", "supporter_available", "energy_attach_done",
            "tutor_playable_now", "gust_playable_now", "draw_playable_now"]
PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END = 7, 8, 9, 10, 12, 13, 14
A_ACTIVE, A_BENCH = 4, 5

_MODEL = None
_MODEL_NAME = None


def _load_json(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, name), name, os.path.join("/kaggle_simulations/agent", name)):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None


CF = _load_json("card_features.json") or {}
CE = _load_json("card_effects.json") or {}
ATK = _load_json("attack_stats.json") or {}


def _slot_id(c):
    return c.get("id") if isinstance(c, dict) else c


def _model():
    global _MODEL, _MODEL_NAME
    model_name = os.environ.get("CABT_CONTEXTUAL_MODEL", "contextual_ranker_v1.json")
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL or None
    blob = _load_json(model_name)
    if not blob:
        _MODEL = False
        _MODEL_NAME = model_name
        return None
    sd = blob["state_dict"]
    m = {
        "emb": np.array(sd["emb.weight"], dtype=np.float32),
        "W0": np.array(sd["net.0.weight"], dtype=np.float32),
        "b0": np.array(sd["net.0.bias"], dtype=np.float32),
        "W2": np.array(sd["net.2.weight"], dtype=np.float32),
        "b2": np.array(sd["net.2.bias"], dtype=np.float32),
        "W4": np.array(sd["net.4.weight"], dtype=np.float32),
        "b4": np.array(sd["net.4.bias"], dtype=np.float32),
        "mean": np.array(blob["mean"], dtype=np.float32),
        "std": np.array(blob["std"], dtype=np.float32),
        "id2ix": {int(c): i for i, c in enumerate(blob["card_ids"])},
        "emb_dim": int(blob.get("emb", 24)),
        "use_emb": bool(blob.get("use_emb", True)),
        "ablate": blob.get("ablate") or {},
    }
    _MODEL = m
    _MODEL_NAME = model_name
    return m


def _player(cur, idx):
    players = cur.get("players") or []
    return players[idx] if isinstance(idx, int) and 0 <= idx < len(players) else {}


def _zone_entity(cur, me, area, index, player_index=None):
    pi = me if player_index is None else player_index
    p = _player(cur, pi)
    if area == A_ACTIVE:
        active = p.get("active") or []
        return active[0] if active and isinstance(active[0], dict) else None, pi, "active"
    if area == A_BENCH:
        bench = p.get("bench") or []
        return (bench[index], pi, "bench") if isinstance(index, int) and 0 <= index < len(bench) else (None, pi, "bench")
    return None, pi, "none"


def _active_entity(cur, idx):
    p = _player(cur, idx)
    active = p.get("active") or []
    return active[0] if active and isinstance(active[0], dict) else None


def _entity_card_id(ent):
    return _slot_id(ent) if isinstance(ent, dict) else None


def _energy_count(ent) -> int:
    if not isinstance(ent, dict):
        return 0
    return len(ent.get("energyCards") or ent.get("energies") or [])


def _hp_left(ent) -> float:
    if not isinstance(ent, dict):
        return 0.0
    return float(ent.get("hp", ent.get("hp_left", 0)) or 0.0)


def _prize_value(cid) -> float:
    cf = CF.get(str(cid), {}) if cid is not None else {}
    if cf.get("mega"):
        return 3.0
    if cf.get("ex"):
        return 2.0
    return 1.0 if cid is not None and cf.get("ct") == 0 else 0.0


def _best_damage(cid) -> float:
    cf = CF.get(str(cid), {}) if cid is not None else {}
    return float(cf.get("best_dmg", 0.0) or 0.0)


def _engine_role(cid) -> float:
    ce = CE.get(str(cid), {}) if cid is not None else {}
    return float(
        1.5 * (ce.get("search", 0) or ce.get("search_to_bench", 0) or 0)
        + 1.2 * (ce.get("draw", 0) or 0)
        + 1.4 * (ce.get("energy_accel", 0) or 0)
        + 1.0 * (ce.get("switch_gust", 0) or 0)
        + 0.8 * (ce.get("has_ability", 0) or 0)
    )


def _attack_damage(opt) -> float:
    if opt.get("type") != ATTACK:
        return 0.0
    a = ATK.get(str(opt.get("attackId")), {})
    return float(a.get("d", 0.0) or 0.0)


def _target_features(opt: dict, cur: dict, me: int) -> list[float]:
    t = opt.get("type")
    if t == ATTACK:
        ent, pi, zone = _active_entity(cur, 1 - me), 1 - me, "active"
    else:
        ent, pi, zone = _zone_entity(cur, me, opt.get("inPlayArea"), opt.get("inPlayIndex"), opt.get("playerIndex"))
    cid = _entity_card_id(ent)
    cf = CF.get(str(cid), {}) if cid is not None else {}
    hp = _hp_left(ent)
    dmg = _attack_damage(opt)
    is_opp = 1.0 if pi == 1 - me else 0.0
    is_active = 1.0 if zone == "active" else 0.0
    is_bench = 1.0 if zone == "bench" else 0.0
    ko_now = 1.0 if hp > 0 and dmg >= hp and t == ATTACK else 0.0
    strand = 0.0
    if is_opp and t in (PLAY, ABILITY, ATTACK):
        strand = min(1.0, float(cf.get("retreat", 0) or 0) / 4.0)
    return [
        hp / 300.0,
        ko_now,
        _prize_value(cid) / 3.0,
        min(1.0, _energy_count(ent) / 6.0),
        _best_damage(cid) / 300.0,
        _engine_role(cid) / 8.0,
        strand,
        1.0 if cf.get("stage") == "basic" else 0.0,
        1.0 if cf.get("stage") in ("stage1", "stage2") else 0.0,
        1.0 if (cf.get("ex") or cf.get("mega")) else 0.0,
        is_opp,
        is_active,
        is_bench,
        1.0 if t == RETREAT else 0.0,
    ]


def _action_dense(opt: dict, cur: dict, me: int) -> tuple[int, list[float], list[float]]:
    cid = SCH.card_identity(opt, _player(cur, me))
    cf = CF.get(str(cid), {}) if cid is not None else {}
    ce = CE.get(str(cid), {}) if cid is not None else {}
    attack = ATK.get(str(opt.get("attackId")), {}) if opt.get("type") == ATTACK else {}
    cost = attack.get("c", 0)
    a_cost = float(len(cost) if isinstance(cost, (list, str)) else (cost or 0))
    effects = [float(ce.get(k, 0) or 0) for k in EFFECT_KEYS]
    dense = [1.0 if opt.get("type") == t else 0.0 for t in OTYPES]
    dense += [
        1.0 if cf.get("ct") == 0 else 0.0,
        1.0 if cf.get("ct") in (1, 2, 3, 4) else 0.0,
        1.0 if cf.get("ct") in (5, 6) else 0.0,
        1.0 if cf.get("stage") == "basic" else 0.0,
        1.0 if cf.get("stage") in ("stage1", "stage2") else 0.0,
        1.0 if (cf.get("ex") or cf.get("mega")) else 0.0,
        float(cf.get("hp", 0) or 0) / 300.0,
        float(cf.get("best_dmg", 0) or 0) / 300.0,
        float(attack.get("d", 0) or 0) / 300.0,
        a_cost / 4.0,
        1.0 if opt.get("inPlayIndex") is not None else 0.0,
        1.0 if (opt.get("playerIndex") not in (None, me)) else 0.0,
    ]
    return int(cid) if cid is not None else -1, dense, effects


def _scaled_delta(dd: dict | None) -> list[float]:
    raw = [float((dd or {}).get(k, 0.0) or 0.0) for k in DELTA_KEYS]
    return [raw[0] / 3, raw[1] / 3, raw[2], raw[3] / 100, raw[4] / 5, raw[5] / 3,
            raw[6] / 3, raw[7] / 5, raw[8] / 5, raw[9], raw[10], raw[11]]


def _history_features(cur: dict) -> list[float]:
    return [
        float(cur.get("turn", 0) or 0) / 30.0,
        float(cur.get("turnActionCount", 0) or 0) / 20.0,
        1.0 if cur.get("supporterPlayed") else 0.0,
        1.0 if cur.get("stadiumPlayed") else 0.0,
        1.0 if cur.get("energyAttached") else 0.0,
        1.0 if cur.get("retreated") else 0.0,
    ]


def dense_for_option(opt: dict, obs: dict, root: list[float], root_map: dict, delta: dict | None) -> tuple[int, list[float]]:
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    cid, action_base, effects = _action_dense(opt, cur, me)
    target = _target_features(opt, cur, me)
    delta_vec = _scaled_delta(delta)
    ctx = [float(root_map.get(k, 0.0) or 0.0) for k in CTX_KEYS]
    interactions = [e * c for e in effects for c in ctx]
    dense = []
    dense += action_base
    dense += effects
    dense += target
    dense += delta_vec
    dense += list(root)
    dense += interactions
    dense += _history_features(cur)
    return cid, dense


def section_slices() -> dict[str, tuple[int, int]]:
    pos = 0
    out = {}
    out["action_base"] = (pos, pos + len(OTYPES) + 12); pos = out["action_base"][1]
    out["effects"] = (pos, pos + len(EFFECT_KEYS)); pos = out["effects"][1]
    out["target"] = (pos, pos + 14); pos = out["target"][1]
    out["deltas"] = (pos, pos + len(DELTA_KEYS)); pos = out["deltas"][1]
    out["root"] = (pos, pos + len(FT.FEATURE_KEYS)); pos = out["root"][1]
    out["interactions"] = (pos, pos + len(EFFECT_KEYS) * len(CTX_KEYS)); pos = out["interactions"][1]
    out["history"] = (pos, pos + 6); pos = out["history"][1]
    out["dense_dim"] = (0, pos)
    return out


SLICES = section_slices()


def apply_ablation(dense: np.ndarray, ablate: dict | None) -> np.ndarray:
    if not ablate:
        return dense
    out = np.array(dense, dtype=np.float32, copy=True)
    if ablate.get("effects"):
        a, b = SLICES["effects"]; out[..., a:b] = 0.0
        a, b = SLICES["interactions"]; out[..., a:b] = 0.0
    if ablate.get("deltas"):
        a, b = SLICES["deltas"]; out[..., a:b] = 0.0
    if ablate.get("target"):
        a, b = SLICES["target"]; out[..., a:b] = 0.0
    if ablate.get("history"):
        a, b = SLICES["history"]; out[..., a:b] = 0.0
    return out


def decision_features(obs: dict, deck: list) -> dict | None:
    sel, cur = obs.get("select"), obs.get("current")
    if not sel or not cur or (sel.get("maxCount") or 0) != 1:
        return None
    opts = sel.get("option") or []
    if len(opts) < 2:
        return None
    me = cur.get("yourIndex", 0)
    try:
        root_map = FT.encode_state(obs)
        root = FT.vectorize(root_map)
    except Exception:
        return None
    try:
        deltas = S.option_deltas(obs, deck)
    except Exception:
        deltas = None
    cids, dense, eqs, keys = [], [], [], []
    seen = {}
    for j, opt in enumerate(opts):
        if not isinstance(opt, dict):
            continue
        key = tuple(SCH.semantic_action_key(opt, cur, me))
        if key not in seen:
            seen[key] = len(seen)
        dd = deltas[j] if (deltas and j < len(deltas)) else None
        cid, row = dense_for_option(opt, obs, root, root_map, dd)
        cids.append(cid)
        dense.append(row)
        eqs.append(seen[key])
        keys.append(key)
    if len(dense) < 2 or len(set(eqs)) < 2:
        return None
    return {
        "cids": cids,
        "dense": dense,
        "eq": eqs,
        "keys": [list(k) for k in keys],
        "key_to_eq": {k: v for k, v in seen.items()},
    }


def _forward(m, cid, dense):
    dn = (dense - m["mean"]) / m["std"]
    dn = apply_ablation(dn, m.get("ablate"))
    if m["use_emb"]:
        ix = m["id2ix"].get(int(cid))
        e = m["emb"][ix] if ix is not None else np.zeros(m["emb_dim"], dtype=np.float32)
        x = np.concatenate([e, dn])
    else:
        x = dn
    h = np.maximum(0.0, m["W0"] @ x + m["b0"])
    h = np.maximum(0.0, m["W2"] @ h + m["b2"])
    return float((m["W4"] @ h + m["b4"])[0])


def score_options(obs: dict, deck: list):
    try:
        m = _model()
        if m is None:
            return None
        feat = decision_features(obs, deck)
        if feat is None:
            return None
        dense = np.array(feat["dense"], dtype=np.float32)
        scores = [_forward(m, cid, row) for cid, row in zip(feat["cids"], dense)]
        return scores
    except Exception:
        return None


def order(obs: dict, deck: list):
    scores = score_options(obs, deck)
    if not scores:
        return None
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def predict(obs: dict, deck: list):
    ords = order(obs, deck)
    return [ords[0]] if ords else None
