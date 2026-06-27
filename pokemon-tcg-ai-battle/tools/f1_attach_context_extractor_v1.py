"""F1 energy-allocation ATTACH-context extractor (Model B runtime prep, READ-ONLY).

extract_attach_context(obs, legal_actions) -> per-ATTACH payload with target role, energy type/units, and
energy-shortfall / attack-threshold context, computed from PUBLIC obs using the REAL repo helpers (no
reinvented thresholds): roles from starmie_tactical_state.semantic_role, attack costs from TS._VERIFIED,
energy id->type from card_stats, units via the Ignition=3-on-Mega rule. Does not mutate obs, never reads
result/outcome/pilot/future, uses explicit nulls (not silent zero), changes NO gameplay.

Run (emits 200 examples + feasibility report):
  PYTHONIOENCODING=utf-8 python tools/f1_attach_context_extractor_v1.py
"""
from __future__ import annotations
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
OUT = ROOT / "data" / "generated" / "f1_energy_runtime_prep_v0"
FIXTURES = ROOT / "tests" / "golden_state_action_fixtures" / "fixtures.json"
ATTACH_TYPE = 8
IGNITION, BASIC_WATER, MEGA_STARMIE, CINDERACE, STARYU = 17, 3, 1031, 666, 1030
_FORBIDDEN = ("result", "outcome", "won", "pilot", "replay", "future", "reward")

with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    import deck_policy_v3 as DP
    import starmie_tactical_state as TS
    import starmie_heuristics as SH
    import learned_selector_bridge as BR
    import learned_proposer_adapter as AD


def _energy_info(ecid):
    """(name, class) from card_stats via DP._meta. class in water/lightning/ignition/tool/other."""
    if ecid is None:
        return None, None
    try:
        m = DP._meta(ecid) or {}
    except Exception:
        return None, None
    name = m.get("n")
    if ecid == IGNITION or (name and "Ignition" in name):
        cls = "ignition"
    elif name and "{W}" in name:
        cls = "basic_water"
    elif name and "{L}" in name:
        cls = "basic_lightning"
    elif name and ("Cape" in name or m.get("type") == "tool"):
        cls = "tool"
    elif m.get("type") == "basic_energy":
        cls = "basic_energy_other"
    else:
        cls = m.get("type") or "other"
    return name, cls


def _units_on(entity):
    if not isinstance(entity, dict):
        return None
    try:
        return TS.energy_units(entity)
    except Exception:
        try:
            return DP._attached_count(entity)   # fallback: card count
        except Exception:
            return None


def _costs(cid):
    try:
        cs = [a.get("units") for a in TS._VERIFIED.get(cid, []) if a.get("units") is not None]
        return (min(cs), max(cs)) if cs else (None, None)
    except Exception:
        return (None, None)


def extract_attach_context(obs, legal_actions=None):
    """Return a list of per-ATTACH-option context payloads (read-only). legal_actions defaults to obs.select.option."""
    sel = obs.get("select") or {}
    options = legal_actions if legal_actions is not None else (sel.get("option") or [])
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            keys = AD.option_index_to_key(obs)
    except Exception:
        keys = {}
    out = []
    for i, opt in enumerate(options):
        if not isinstance(opt, dict):
            continue
        fam_type = opt.get("type")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                of = BR.option_features(opt, obs)
        except Exception:
            of = {}
        if fam_type != ATTACH_TYPE and of.get("action_family") != "ATTACH":
            continue
        try:
            ecid = DP.option_card_id(opt, obs)
        except Exception:
            ecid = None
        try:
            tgt = DP.option_target_entity(opt, obs)
        except Exception:
            tgt = None
        try:
            tcid = DP._cid(tgt) if tgt is not None else of.get("target_card_id")
        except Exception:
            tcid = of.get("target_card_id")
        ename, etype = _energy_info(ecid)
        is_ign = (ecid == IGNITION)
        if etype == "tool":
            units_added = 0                                   # a tool (Hero's Cape) adds HP, not energy units
        elif is_ign and tcid == MEGA_STARMIE:
            units_added = 3                                   # Ignition on Mega = 3 units
        elif ecid is not None:
            units_added = 1
        else:
            units_added = None
        before = _units_on(tgt)
        after = (before + units_added) if (before is not None and units_added is not None) else None
        cheapest, max_useful = _costs(tcid)
        role = None
        with contextlib.suppress(Exception):
            role = TS.semantic_role(tcid, True) if tcid is not None else None

        def _short(u):
            return max(0, cheapest - u) if (cheapest is not None and u is not None) else None
        sb, sa = _short(before), _short(after)
        already_ready = (before >= cheapest) if (cheapest is not None and before is not None) else None
        crosses = (sb is not None and sa is not None and sb > 0 and sa == 0)
        if units_added == 0:                                  # a tool attach is never "redundant energy"
            redundant = False
        else:
            redundant = (after > max_useful) if (max_useful is not None and after is not None) else None
        payload = {
            "raw_option_index": i,
            "semantic_key": keys.get(i),
            "energy_card_id": ecid,
            "energy_card_name": ename,
            "energy_class": etype,
            "energy_is_ignition": is_ign,
            "energy_units_added": units_added,
            "target_role": role,
            "target_card_id": tcid,
            "target_owner": of.get("target_owner"),
            "target_zone": of.get("target_zone"),
            "target_slot": of.get("target_slot"),
            "target_energy_before": before,
            "target_energy_after": after,
            "attack_cheapest_units": cheapest,
            "attack_max_useful_units": max_useful,
            "shortfall_before": sb,
            "shortfall_after": sa,
            "crosses_attack_threshold": crosses if cheapest is not None else None,
            "already_ready": already_ready,
            "redundant_energy": redundant,
        }
        payload["runtime_safe"] = True
        payload["missing_fields"] = [k for k, v in payload.items() if v is None]
        out.append(payload)
    return out


def capture_starmie_attach_obs(n_games=2, budget=0.15, cap=400):
    """Tiny Starmie self-play (<=2 games, no policy change) to capture obs at ATTACH-legal single-pick decisions.
    The golden fixtures are a DIFFERENT deck (no Mega/Staryu/Cinderace ids), so we need real Starmie states to
    exercise the role/threshold logic. Records deep-copied obs only; returns the list."""
    captured = []
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make
        import main as M
        import search_v3 as S
        with contextlib.suppress(Exception):
            S.DEFAULT_BUDGET = budget

        def rec(obs):
            sel = obs.get("select")
            if (sel and (sel.get("maxCount") or 0) == 1
                    and any(isinstance(o, dict) and o.get("type") == ATTACH_TYPE for o in (sel.get("option") or []))
                    and len(captured) < cap):
                with contextlib.suppress(Exception):
                    captured.append(json.loads(json.dumps(obs)))   # deep copy, no mutation of live obs
            return M.agent_starmie(obs)
        for _ in range(n_games):
            if len(captured) >= cap:
                break
            with contextlib.suppress(Exception):
                env = make("cabt")
                env.run([rec, M.agent_starmie])
    return captured


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    obs_list = capture_starmie_attach_obs(n_games=10, cap=240)   # data capture for the example pack (not an A/B)
    payloads, role_dist, type_dist = [], {}, {}
    n_with_attach = 0
    field_pop = {}
    for di, obs in enumerate(obs_list):
        rows = extract_attach_context(obs)
        if rows:
            n_with_attach += 1
        for r in rows:
            role_dist[r["target_role"]] = role_dist.get(r["target_role"], 0) + 1
            type_dist[r["energy_class"]] = type_dist.get(r["energy_class"], 0) + 1
            for k, v in r.items():
                if k in ("missing_fields", "runtime_safe"):
                    continue
                field_pop[k] = field_pop.get(k, 0) + (1 if v is not None else 0)
            if len(payloads) < 200:
                payloads.append({"decision_id": f"starmie_selfplay:{di}", **r})
    n = max(1, sum(role_dist.values()))
    feasibility = {
        "source": "Starmie agent_starmie self-play capture (2 games; golden fixtures are a different deck)",
        "decisions_with_attach": n_with_attach, "attach_options_seen": sum(role_dist.values()),
        "target_role_distribution": role_dist, "energy_class_distribution": type_dist,
        "field_population_pct": {k: round(100 * c / n, 1) for k, c in sorted(field_pop.items())},
        "mechanisms": {
            "target_role": "starmie_tactical_state.semantic_role (1031 main_attacker / 666 energy_engine / 1030 setup_basic)",
            "energy_type": "deck_policy_v3.option_card_id -> card_stats ty (W/L/CCC); Ignition=17 unique",
            "energy_units_added": "Ignition on Mega = 3 units, else 1 (matches _energy_units)",
            "shortfall/threshold": "TS._VERIFIED cheapest/max attack-unit costs; shortfall = cheapest - units; crosses = shortfall_before>0 & shortfall_after==0",
            "redundant_energy": "energy_after > max_useful attack-unit cost (no further attack enabled)",
        },
        "runtime_safe": "all fields from public obs + static cost table; no result/outcome/hidden/future; explicit nulls.",
        "note": "shortfall uses energy UNITS (Ignition=3); _attach_score scores by card COUNT -- a candidate must standardize on units.",
    }
    (OUT / "runtime_feature_feasibility.json").write_text(json.dumps(feasibility, indent=2, default=str), encoding="utf-8")
    with open(OUT / "attach_context_examples.jsonl", "w", encoding="utf-8") as fh:
        for p in payloads:
            fh.write(json.dumps(p, default=str) + "\n")
    print(json.dumps({"decisions_with_attach": n_with_attach, "attach_options": sum(role_dist.values()),
                      "payloads": len(payloads), "role_dist": role_dist, "energy_class_dist": type_dist,
                      "field_population_pct": feasibility["field_population_pct"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
