"""Learned-adviser interface (Model B, section 6) -- DISABLED BY DEFAULT.

A narrow, fail-closed seam for the C8 learned model (Model A's lane) to advise the Starmie heuristic agent on
SAME-FAMILY alternatives only (initially ATTACH; SELECT_CARD/TARGET only if Model A reports support). It does
NOT decide moves on its own and is OFF until Model A supplies a Starmie coverage artifact with status SUPPORTED
or DIRECTIONAL_ONLY. No weights are fabricated here; this is the contract + a safe no-op until a real model and
its coverage artifact exist.

Contract (the agent may call this only after its hard rules):
    rank_same_family(observation, legal_options, action_family) ->
        {ranking, confidence, support_status, model_id}
Guarantees:
- pre-action observable features only;
- ranks ONLY same-family alternatives that are already legal;
- never overrides a game-winning attack or a hard mechanical safety rule (the CALLER enforces this by only
  consulting the adviser on non-safety, same-family ties);
- returns support_status OUT_OF_SUPPORT / MODEL_ARTIFACT_NOT_AVAILABLE and ranking=None when it cannot help;
- abstains (ranking=None) below a frozen confidence threshold;
- exposes the exact model/artifact hash via model_id.
"""
from __future__ import annotations

import hashlib
import json
import os

# Hard master switch. Must remain False until Model A's coverage artifact is supplied AND offline-validated.
ENABLED = False

# Where Model A's Starmie coverage artifact would land (adapted to kaggle-fun layout). When present and status
# is SUPPORTED/DIRECTIONAL_ONLY, an actual model could be loaded here. Absent -> MODEL_ARTIFACT_NOT_AVAILABLE.
COVERAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "starmie_audit",
                             "c8a5_starmie_adviser_coverage_v1.json")
CONFIDENCE_FLOOR = 0.0   # frozen from validation before any enable; 0 here = abstain on everything.

SUPPORTED_FAMILIES = {"ATTACH"}   # SELECT_CARD/TARGET added only if Model A reports support.


def _coverage():
    try:
        with open(COVERAGE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def model_id():
    cov = _coverage()
    if not cov:
        return None
    blob = json.dumps(cov, sort_keys=True).encode()
    return "cov:" + hashlib.sha1(blob).hexdigest()[:12]


def rank_same_family(observation, legal_options, action_family):
    """Return a ranking over same-family legal options, or abstain. Safe no-op until a real, supported,
    validated model exists. The caller MUST only consult this for non-safety, same-family decisions and must
    ignore ranking=None."""
    out = {"ranking": None, "confidence": 0.0, "support_status": "MODEL_ARTIFACT_NOT_AVAILABLE", "model_id": None}
    if not ENABLED:
        out["support_status"] = "DISABLED"
        return out
    if action_family not in SUPPORTED_FAMILIES:
        out["support_status"] = "FAMILY_NOT_SUPPORTED"
        return out
    cov = _coverage()
    if not cov:
        return out                      # MODEL_ARTIFACT_NOT_AVAILABLE
    status = str(cov.get("status", "OUT_OF_SUPPORT"))
    out["model_id"] = model_id()
    if status not in ("SUPPORTED", "DIRECTIONAL_ONLY"):
        out["support_status"] = "OUT_OF_SUPPORT"
        return out
    # A real model + the support gate would go here. Until then we abstain (never fabricate a ranking).
    out["support_status"] = status
    out["confidence"] = 0.0
    out["ranking"] = None               # abstain: confidence below CONFIDENCE_FLOOR
    return out
