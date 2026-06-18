"""L2 inference: the learned value model (gradient-boosted tree), from value_weights.json.

Pure numpy/json (no sklearn at inference), so it bundles with the agent. The model is a set of
small decision trees: decision_function(x) = init + lr * sum_t leaf_value_t(x), and the value is
P(win) = sigmoid(decision_function). score(feat_dict) returns P(win) in (0,1), or None if no
trained weights are present so callers fall back to the hand eval.

This is the value search calls at its leaves. Trained on self-play outcomes over the L1
representation, it expresses thresholds/interactions (e.g. the prize/KO line) a linear eval
cannot, which is why the model is a tree, not a dot product.
"""
from __future__ import annotations

import json
import math
import os

import features as FT

_W = None
_LOADED = False


def _load():
    global _W, _LOADED
    if _LOADED:
        return _W
    _LOADED = True
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("value_weights.json",
              os.path.join(here, "value_weights.json"),
              os.path.join("/kaggle_simulations/agent", "value_weights.json")):
        try:
            with open(p, encoding="utf-8") as f:
                w = json.load(f)
            if w.get("kind") != "gbm" or "trees" not in w or "keys" not in w:
                continue
            # integrity: every stored key still exists in the live feature set (a stale
            # value_weights.json vs a changed encode_state would silently score garbage)
            if any(k not in FT.FEATURE_KEYS for k in w["keys"]):
                continue
            _W = w
            return _W
        except Exception:
            continue
    _W = None
    return None


def available() -> bool:
    return _load() is not None


def _traverse(tree: dict, x: list) -> float:
    feat, thr, left, right, val = tree["feature"], tree["threshold"], tree["left"], tree["right"], tree["value"]
    node = 0
    while feat[node] != -2:                       # -2 == leaf (sklearn TREE_UNDEFINED)
        node = left[node] if x[feat[node]] <= thr[node] else right[node]
    return val[node]


def proba_from_export(model: dict, feat: dict) -> float:
    """P(win) for `feat` under an in-memory export dict (used by the trainer's verification too)."""
    keys = model["keys"]
    x = [float(feat.get(k, 0.0)) for k in keys]
    z = model["init"] + model["lr"] * sum(_traverse(t, x) for t in model["trees"])
    if z <= -60:
        return 0.0
    if z >= 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def score(feat: dict) -> float | None:
    """P(win) for a feature dict, or None if no weights are loaded."""
    w = _load()
    if w is None:
        return None
    return proba_from_export(w, feat)


def score_obs(obs: dict) -> float | None:
    """P(win) for an observation dict (uses features.encode_state from yourIndex's perspective)."""
    try:
        return score(FT.encode_state(obs))
    except Exception:
        return None
