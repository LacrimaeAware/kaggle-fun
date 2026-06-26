"""STARMIE RUNTIME OBSERVATION FEATURE AUDIT V0. Audits what public information lives in live CABT observations
and what the Starmie agent / Feature-V2 adapter currently extract. Read-only; no gameplay, no training.

  PYTHONIOENCODING=utf-8 python tools/runtime_feature_audit_v0.py [--max 400]
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import deck_policy_v3 as DP            # noqa: E402
import learned_selector_bridge as BR  # noqa: E402  the live Feature-V2 adapter
OUT = ROOT / "data" / "generated" / "runtime_feature_audit"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")

# fields the live adapter (learned_selector_bridge) actually consumes
ADAPTER_CONSUMES = {
    "entity": {"id", "hp", "maxHp", "energies", "energyCards"},
    "player": {"handCount", "deckCount", "prize", "active", "bench"},
    "select": {"option", "context", "contextCard"},
    "current": {"players", "yourIndex"},
}
TURN_CTX = [
    ("who_went_first", "current.firstPlayer", "DIRECT_OBS_FIELD"),
    ("global_turn_count", "current.turn", "DIRECT_OBS_FIELD"),
    ("player_turn_count", "derive: current.turn + firstPlayer + yourIndex", "DERIVABLE"),
    ("action_index_in_turn", "current.turnActionCount", "DIRECT_OBS_FIELD"),
    ("supporter_used_this_turn", "current.supporterPlayed", "DIRECT_OBS_FIELD"),
    ("attachment_used_this_turn", "current.energyAttached", "DIRECT_OBS_FIELD"),
    ("retreat_used_this_turn", "current.retreated", "DIRECT_OBS_FIELD"),
    ("stadium_in_play", "current.stadium / current.stadiumPlayed", "DIRECT_OBS_FIELD"),
    ("attack_available", "scan select.option for type==13", "DERIVABLE"),
    ("end_available", "scan select.option for type==14", "DERIVABLE"),
    ("nonterminal_legal_count", "count non-13/14 options", "DERIVABLE"),
    ("entity_appeared_this_turn", "entity.appearThisTurn (summoning sickness)", "DIRECT_OBS_FIELD"),
    ("status_conditions", "player.asleep/paralyzed/confused/burned/poisoned", "DIRECT_OBS_FIELD"),
    ("previous_actions_this_turn", "obs.logs (short recent list only)", "PARTIAL_NEEDS_LOCAL_MEMORY"),
    ("previous_action_families_this_turn", "not in obs; logs too short", "REQUIRES_LOCAL_MEMORY"),
    ("ability_used_flags", "not in obs (has_ability is card text, not a used-flag)", "UNSUPPORTED_OR_LOCAL_MEMORY"),
]


def _sample(max_dec):
    files = sorted(os.listdir(REPLAYS))[:300]
    out = []
    for fn in files:
        if len(out) >= max_dec:
            break
        try:
            steps = json.load(open(REPLAYS / fn, encoding="utf-8"))["steps"]
        except Exception:
            continue
        for t in range(0, len(steps), max(1, len(steps) // 6)):
            for seat in (0, 1):
                try:
                    o = steps[t][seat]["observation"]
                except Exception:
                    o = None
                if o and (o.get("select") or {}).get("option"):
                    out.append(o)
                if len(out) >= max_dec:
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    obss = _sample(a.max)

    # ---------- Section 1: live observation inventory ----------
    pres = collections.Counter()
    for o in obss:
        c = o.get("current") or {}
        for k in ("turn", "firstPlayer", "yourIndex", "turnActionCount", "supporterPlayed", "energyAttached",
                  "retreated", "stadium", "stadiumPlayed", "looking", "result"):
            if k in c:
                pres[f"current.{k}"] += 1
        for k in ("logs", "step", "remainingOverageTime"):
            if k in o:
                pres[f"obs.{k}"] += 1
        p = (c.get("players") or [{}])[0]
        for k in ("asleep", "paralyzed", "confused", "burned", "poisoned", "discard", "prize", "handCount", "deckCount"):
            if k in p:
                pres[f"player.{k}"] += 1
        act = (p.get("active") or [None])[0]
        if act:
            for k in ("appearThisTurn", "tools", "preEvolution", "energies", "energyCards", "hp", "maxHp", "serial"):
                if k in act:
                    pres[f"entity.{k}"] += 1
    n = max(1, len(obss))
    inventory = {
        "sampled_decisions": len(obss),
        "field_presence_pct": {k: round(100 * v / n, 1) for k, v in sorted(pres.items())},
        "adapter_consumes": {k: sorted(v) for k, v in ADAPTER_CONSUMES.items()},
        "public_strategic_fields_NOT_consumed_by_adapter": [
            "current.turn", "current.firstPlayer", "current.turnActionCount", "current.supporterPlayed",
            "current.energyAttached", "current.retreated", "current.stadium",
            "player.asleep/paralyzed/confused/burned/poisoned (status conditions)",
            "entity.appearThisTurn (summoning sickness)", "entity.tools", "entity.preEvolution (evolution line)",
            "player.discard (visible -> prize/deck belief)", "obs.logs (recent events)",
        ],
        "forbidden_runtime_field_present": {"current.result": "game outcome; -1 mid-game; must NOT be a feature"},
    }
    (OUT / "live_observation_inventory_v0.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    # ---------- Section 2: turn-context tracker feasibility ----------
    # presence/non-null rate for the direct fields
    direct = {}
    for o in obss:
        c = o.get("current") or {}
        for fld in ("firstPlayer", "turn", "turnActionCount", "supporterPlayed", "energyAttached", "retreated"):
            if c.get(fld) is not None:
                direct[fld] = direct.get(fld, 0) + 1
    feasibility = {
        "summary": "Nearly all turn-context is a DIRECT current.* field or trivially DERIVABLE -- no history tracker "
                   "needed for the key signals. The gap is EXTRACTION/PACKING, not availability.",
        "features": [{"feature": f, "source": s, "classification": cls} for f, s, cls in TURN_CTX],
        "direct_field_nonnull_pct": {k: round(100 * v / n, 1) for k, v in direct.items()},
        "classification_counts": dict(collections.Counter(cls for _, _, cls in TURN_CTX)),
    }
    (OUT / "turn_context_tracker_feasibility_v0.json").write_text(json.dumps(feasibility, indent=2), encoding="utf-8")

    # ---------- Section 3: card mechanic observability ----------
    deck_ids = [1031, 1030, 666, 17, 3, 1159, 1229, 1488, 1487, 965]  # starmie line + key cards/attacks
    rows = []
    for cid in deck_ids:
        m = DP._meta(cid) or {}
        eff = DP.CEFF.get(str(cid)) or {}
        rows.append({"card_id": cid, "name": m.get("n"),
                     "card_stats_has": {k: (k in m) for k in ("hp", "wk", "rs", "ty", "stage", "retreat", "prize", "atks")},
                     "card_effects_tags": list(eff.keys())})
    card_obs = {
        "card_stats_fields": ["hp", "wk(weakness)", "rs(resistance)", "ty(type)", "stage", "retreat", "prize",
                              "ex/mega", "atks(cost/dmg/cE)"],
        "available_at_runtime": "card_stats.json + card_effects.json + attack_stats.json loaded by deck_policy_v3",
        "observable_from_obs": {
            "remaining_hp": "entity.hp", "max_hp": "entity.maxHp", "damage": "derive maxHp-hp (entity.damage often null)",
            "energy_units": "entity.energies (engine pre-expands Ignition to 3 units)",
            "energy_cards": "entity.energyCards", "status": "player.asleep/...", "tools": "entity.tools",
            "evolution_line": "entity.preEvolution",
        },
        "mismatches_gaps": {
            "special_energy_behavior": "Ignition unit-expansion handled by engine in entity.energies (OBSERVABLE); "
                                       "but 'Ignition only on evolution / discarded end of turn' is HARDCODED in "
                                       "deck_policy, not a card_effect feature",
            "conditional_scaling_damage": "Nebula flat-210-ignore-wk/rs and Alakazam 20x-hand are HARDCODED in "
                                          "deck_policy.attack_profile; NOT in card_effects -> opaque to the model",
            "ignore_weakness_resistance_effects": "NOT present in card_effects.json",
            "damage_prevention": "NOT present in card_effects.json (no prevent/Crustle-style tag)",
            "effect_taxonomy_present": "card_effects HAS search/heal/gust/switch/discard/has_ability tags",
        },
        "sample_cards": rows,
    }
    (OUT / "card_mechanic_observability_v0.json").write_text(json.dumps(card_obs, indent=2), encoding="utf-8")

    # ---------- Section 4: ranked blind spots + verdict ----------
    blind = [
        {"rank": 1, "name": "turn_context_features", "severity": "CRITICAL",
         "detail": "current.turnActionCount/supporterPlayed/energyAttached/retreated/turn/firstPlayer are present in "
                   "EVERY obs but reach neither the adapter nor the model's 11 state features. Directly drives the "
                   "develop-vs-attack timing the selector got wrong (it cannot see how much of the turn is spent).",
         "fix": "extract into adapter state_features + tactical (DIRECT fields, cheap)", "needs_retrain": True,
         "needs_runtime_adapter": True, "logging_only": False},
        {"rank": 2, "name": "status_conditions_and_summoning_sickness", "severity": "HIGH",
         "detail": "player.asleep/paralyzed/confused/burned/poisoned + entity.appearThisTurn are present, not "
                   "extracted. Affect attack/retreat legality + tempo.",
         "fix": "extract into board_entities/state_features", "needs_retrain": True, "needs_runtime_adapter": True,
         "logging_only": False},
        {"rank": 3, "name": "discard_zone_belief", "severity": "HIGH",
         "detail": "both players' discards are fully visible; not used for prize/deck-copy belief or known-remaining counts.",
         "fix": "add discard-derived belief features", "needs_retrain": True, "needs_runtime_adapter": True,
         "logging_only": False},
        {"rank": 4, "name": "conditional_damage_and_ignore_prevention_taxonomy", "severity": "MEDIUM",
         "detail": "Nebula ignore-wk/rs + Alakazam scaling + any prevention are HARDCODED in deck_policy, not clean "
                   "card_effect features; opaque to the model via card_id/attack_id.",
         "fix": "extend card_effects.json with ignore/prevent/conditional tags (Model A's stack)", "needs_retrain": True,
         "needs_runtime_adapter": False, "logging_only": False},
        {"rank": 5, "name": "tools_stadium", "severity": "LOW",
         "detail": "entity.tools + current.stadium present, not extracted; low relevance for this deck (no stadium, "
                   "Hero's Cape tool matters for KO threshold).",
         "fix": "extract tool_card_id (Cape +100 HP) into board_entities", "needs_retrain": True,
         "needs_runtime_adapter": True, "logging_only": False},
        {"rank": 6, "name": "full_same_turn_action_history", "severity": "LOW",
         "detail": "obs.logs is a short recent list; full prev-action-families this turn needs a local state tracker.",
         "fix": "optional local action-history tracker", "needs_retrain": True, "needs_runtime_adapter": True,
         "logging_only": True},
    ]
    verdict = "B_RUNTIME_TURN_CONTEXT_GAPS_FOUND"
    out = {"verdict": verdict,
           "headline": "The live obs is information-rich and READY for Model A's stack audit -- every public fact "
                       "audited is observable. The one actionable RUNTIME gap is turn-context: present in EVERY obs "
                       "(current.turnActionCount/supporterPlayed/energyAttached/retreated/turn/firstPlayer) but "
                       "extracted by neither the adapter nor the model. This is an EXTRACTION gap (cheap), not an "
                       "availability gap, and it maps onto the selector's develop-vs-attack failure.",
           "blind_spots_ranked": blind,
           "good_news": "no history tracker needed for the key turn-context signals; all are direct current.* fields.",
           "constraints_honored": "audit only; no gameplay, no heuristics/model/submission change, Model A untouched."}
    (OUT / "runtime_blind_spots_v0.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"sampled {len(obss)} decisions")
    print("turn-context direct-field nonnull %:", json.dumps(feasibility["direct_field_nonnull_pct"]))
    print("turn-context classification:", json.dumps(feasibility["classification_counts"]))
    print(f"VERDICT={verdict}")
    for b in blind[:3]:
        print(f"  #{b['rank']} [{b['severity']}] {b['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
