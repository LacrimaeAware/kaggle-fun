"""Match-time inference for the distilled action ranker (the learned policy).

Loads the model saved by tools/train_action_ranker.py --save (a JSON blob of weights + the exact
feature spec) and, at a single-pick decision, scores each legal option with the SAME features the
trainer used -- card-id EMBEDDING + decoded effects + action descriptor + root-state features +
forward-model one-step DELTAS -- then returns the argmax option. A plain numpy forward (embedding
lookup + 2 hidden ReLU layers), so NO torch dependency at match time and it is ~0 cost.

This is the SPEED payoff of distillation: it reproduces the 0.6s/decision search's move choice with an
instant net (offline distill top-1 ~0.62 overall, ~0.51 on the slice where option-0 is not the
teacher's pick). Crash-safe: predict() returns None on ANY problem so the caller falls back.

Feature construction MUST stay byte-identical to tools/build_action_dataset.py (option_features /
opt_card_id) and tools/train_action_ranker.py (dense_vec); this module is the agent/-only copy (tools/
is not bundled in the submission). The saved blob carries OTYPES/EFFECT_KEYS/DELTA_KEYS so the layout
cannot silently drift.
"""
from __future__ import annotations

import json
import os

import numpy as np

import features as FT
import search as S

PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK = 7, 8, 9, 10, 12, 13
A_HAND, A_ACTIVE, A_BENCH, A_DISCARD = 2, 4, 5, 6

_MODEL = None


def _load_json(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, name), name, os.path.join("/kaggle_simulations/agent", name)):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None


def _model():
    """Load + cache the deploy blob once. Returns a dict of numpy arrays + spec, or None (no model)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL or None
    blob = _load_json("ranker_model.json")
    if not blob:
        _MODEL = False
        return None
    sd = blob["state_dict"]
    m = {
        "emb": np.array(sd["emb.weight"], dtype=np.float32),
        "W0": np.array(sd["net.0.weight"], dtype=np.float32), "b0": np.array(sd["net.0.bias"], dtype=np.float32),
        "W2": np.array(sd["net.2.weight"], dtype=np.float32), "b2": np.array(sd["net.2.bias"], dtype=np.float32),
        "W4": np.array(sd["net.4.weight"], dtype=np.float32), "b4": np.array(sd["net.4.bias"], dtype=np.float32),
        "mean": np.array(blob["mean"], dtype=np.float32), "std": np.array(blob["std"], dtype=np.float32),
        "id2ix": {int(c): i for i, c in enumerate(blob["card_ids"])},
        "emb_dim": int(blob["emb"]), "use_emb": bool(blob.get("use_emb", True)),
        "otypes": blob["otypes"], "effect_keys": blob["effect_keys"], "delta_keys": blob["delta_keys"],
        "cf": _load_json("card_features.json") or {}, "ce": _load_json("card_effects.json") or {},
        "atk": _load_json("attack_stats.json") or {},
    }
    _MODEL = m
    return m


def _slot_id(c):
    return (c.get("id") if isinstance(c, dict) else c)


def _card_id(o, me_player):
    """The acting-card identity per option (mirrors build_action_dataset.opt_card_id)."""
    if not isinstance(o, dict) or not me_player:
        return None
    t, idx, area = o.get("type"), o.get("index"), o.get("area")
    hand = me_player.get("hand") or []
    active = me_player.get("active") or []
    bench = me_player.get("bench") or []
    discard = me_player.get("discard") or []

    def at(seq, i):
        return _slot_id(seq[i]) if isinstance(i, int) and 0 <= i < len(seq) else None

    if t == PLAY:
        return at(hand, idx)
    if area == A_HAND:
        return at(hand, idx)
    if area == A_ACTIVE:
        return at(active, 0)
    if area == A_BENCH:
        return at(bench, idx)
    if area == A_DISCARD:
        return at(discard, idx)
    if t in (ATTACK, RETREAT):
        return at(active, 0)
    return None


def _option_dense(m, o, cur, me, dd, root):
    """Per-option dense feature vector (mirrors option_features + dense_vec, in order)."""
    cid = _card_id(o, (cur.get("players") or [{}])[me])
    cf = m["cf"].get(str(cid), {}) if cid else {}
    ce = m["ce"].get(str(cid), {}) if cid else {}
    t = o.get("type")
    a = m["atk"].get(str(o.get("attackId")), {}) if t == ATTACK else {}
    cost = a.get("c", 0)
    a_cost = float(len(cost) if isinstance(cost, (list, str)) else (cost or 0))
    d = [1.0 if t == ot else 0.0 for ot in m["otypes"]]
    d += [
        1.0 if cf.get("ct") == 0 else 0.0,
        1.0 if cf.get("ct") in (1, 2, 3, 4) else 0.0,
        1.0 if cf.get("ct") in (5, 6) else 0.0,
        1.0 if cf.get("stage") == "basic" else 0.0,
        1.0 if cf.get("stage") in ("stage1", "stage2") else 0.0,
        1.0 if (cf.get("ex") or cf.get("mega")) else 0.0,
        float(cf.get("hp", 0) or 0) / 300.0,
        float(cf.get("best_dmg", 0) or 0) / 300.0,
        float(a.get("d", 0) or 0) / 300.0,
        a_cost / 4.0,
        1.0 if o.get("inPlayIndex") is not None else 0.0,
        1.0 if (o.get("playerIndex") not in (None, me)) else 0.0,
    ]
    d += [float(ce.get(k, 0) or 0) for k in m["effect_keys"]]
    raw = [float(dd.get(k, 0.0)) for k in m["delta_keys"]]
    d += [raw[0] / 3, raw[1] / 3, raw[2], raw[3] / 100, raw[4] / 5, raw[5] / 3,
          raw[6] / 3, raw[7] / 5, raw[8] / 5, raw[9], raw[10], raw[11]]
    d += list(root)
    return cid, np.array(d, dtype=np.float32)


def _forward(m, cid, dense):
    dn = (dense - m["mean"]) / m["std"]
    if m["use_emb"]:
        ix = m["id2ix"].get(int(cid)) if cid is not None else None
        e = m["emb"][ix] if ix is not None else np.zeros(m["emb_dim"], dtype=np.float32)
        x = np.concatenate([e, dn])
    else:
        x = dn
    h = np.maximum(0.0, m["W0"] @ x + m["b0"])
    h = np.maximum(0.0, m["W2"] @ h + m["b2"])
    return float((m["W4"] @ h + m["b4"])[0])


def is_strategic(obs: dict) -> bool:
    """Mirror the training filter (build_action_dataset --strategic-only): a single-pick decision with
    >1 option TYPE and >1 distinct move (by the canonical key). The net was trained ONLY on these, so
    it is off-distribution elsewhere; the hybrid agent restricts the net to this in-distribution slice."""
    sel, cur = obs.get("select"), obs.get("current")
    if not sel or not cur or (sel.get("maxCount") or 0) != 1:
        return False
    opts = sel.get("option") or []
    if len(opts) < 2 or not cur.get("players"):
        return False
    me = cur.get("yourIndex", 0)
    me_player = (cur.get("players") or [{}])[me]
    types, keys = set(), set()
    for o in opts:
        if isinstance(o, dict):
            types.add(o.get("type"))
            keys.add((o.get("type"), _card_id(o, me_player), o.get("attackId"),
                      o.get("inPlayArea"), o.get("inPlayIndex")))
    return len(types) > 1 and len(keys) > 1


def predict(obs: dict, deck: list):
    """Return [best_option_index] by the learned ranker, or None if it does not apply (caller falls
    back to the heuristic). Never raises."""
    try:
        m = _model()
        if m is None:
            return None
        sel, cur = obs.get("select"), obs.get("current")
        if not sel or not cur or (sel.get("maxCount") or 0) != 1:
            return None
        opts = sel.get("option") or []
        if len(opts) < 2:
            return None
        players = cur.get("players") or []
        if len(players) < 2:
            return None
        me = cur.get("yourIndex", 0)
        try:
            root = FT.vectorize(FT.encode_state(obs))
        except Exception:
            return None
        try:
            deltas = S.option_deltas(obs, deck)
        except Exception:
            deltas = None
        best_i, best_v = None, None
        for j, o in enumerate(opts):
            if not isinstance(o, dict):
                continue
            dd = deltas[j] if (deltas and j < len(deltas) and deltas[j]) else {}
            cid, dense = _option_dense(m, o, cur, me, dd, root)
            v = _forward(m, cid, dense)
            if best_v is None or v > best_v:
                best_v, best_i = v, j
        return [best_i] if best_i is not None else None
    except Exception:
        return None
