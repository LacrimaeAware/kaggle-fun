"""Augment the atlas tables with RESOLVED option targets. The corpus legal_actions were stripped of
inPlayArea/inPlayIndex, so option_target_entity couldn't resolve attach/play targets. This re-resolves the
target role from the RAW observation: pilot_action comes from the V2 corpus (joined by decision_id), agent_action
is already in the table. No agent run. Rewrites the atlas tables in place, then you re-run the analysis.

  python tools/augment_atlas_targets.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import deck_policy_v3 as DP            # noqa: E402
import starmie_tactical_state as TS    # noqa: E402

ATLAS = ROOT / "data" / "starmie_audit" / "v2_behavior_atlas"
V2 = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v2.jsonl"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
ROLE = {1031: "Mega", 666: "Cinderace", 1030: "Staryu", 17: "Ignition", 3: "Water"}
_EPC = {}


def _obs(e, step, seat):
    if e not in _EPC:
        if len(_EPC) > 24:
            _EPC.clear()
        try:
            _EPC[e] = json.load(open(REPLAYS / f"{e}.json", encoding="utf-8"))
        except Exception:
            _EPC[e] = None
    try:
        return _EPC[e]["steps"][step][seat].get("observation")
    except Exception:
        return None


def _resolve(obs, action):
    """(target_role, target_units, card_role) for the raw chosen option."""
    if not obs or not action:
        return None, None, None
    opts = (obs.get("select") or {}).get("option") or []
    i = action[0]
    if not (0 <= i < len(opts)):
        return None, None, None
    o = opts[i]
    try:
        cr = ROLE.get(DP.option_card_id(o, obs))
    except Exception:
        cr = None
    try:
        ent = DP.option_target_entity(o, obs)
    except Exception:
        ent = None
    if ent:
        return ROLE.get(DP._cid(ent), f"#{DP._cid(ent)}"), TS.energy_units(ent), cr
    return None, None, cr


def _pilot_action_map():
    """decision_id -> pilot_action indices, for both pilots, from V2."""
    m = {}
    for line in open(V2, encoding="utf-8"):
        r = json.loads(line)
        m[r["id"]] = r["pilot_action"]
    return m


def augment(path, pa_map, has_agent):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    nfix = 0
    for r in rows:
        obs = _obs(r["episode"], r["step"], r["seat"])
        pa = pa_map.get(r["decision_id"])
        tr, tu, cr = _resolve(obs, pa)
        r["pilot_target_role"], r["pilot_target_units"], r["pilot_card_role"] = tr, tu, cr
        if tr is not None or cr is not None:
            nfix += 1
        if has_agent and r.get("agent_action"):
            atr, atu, acr = _resolve(obs, r["agent_action"])
            r["agent_target_role"], r["agent_card_role"] = atr, acr
    with open(path, "w", encoding="utf-8") as o:
        for r in rows:
            o.write(json.dumps(r, default=str) + "\n")
    return len(rows), nfix


def main():
    pa = _pilot_action_map()
    for name, has_agent in (("newtop1_decisions.jsonl", True), ("keidroid_decisions.jsonl", False)):
        p = ATLAS / name
        if not p.exists():
            continue
        n, fix = augment(p, pa, has_agent)
        print(f"{name}: {n} rows, {fix} targets/card-roles resolved")


if __name__ == "__main__":
    main()
