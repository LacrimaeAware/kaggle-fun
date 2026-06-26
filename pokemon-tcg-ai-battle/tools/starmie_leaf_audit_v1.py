"""STARMIE TACTICAL-LEAF V1 -- Section 3: AUDIT THE CURRENT SEARCH LEAF (does NOT edit eval.py).

Two parts:
  A. STRUCTURAL proof of deck-blindness: build fixed states and show the current leaf scores a Cinderace engine
     and a Mega Starmie attacker IDENTICALLY (the W_ENERGY term counts energy CARDS on the active regardless of
     which Pokemon it is; there is no main-attacker-continuity term). Logs every leaf contribution.
  B. CORPUS PREVALENCE: scan the exported tactical-state dataset for the deck-blind decision patterns the leaf
     cannot see (Cinderace-over-Mega, energy-on-engine overinvestment, no main-attacker continuity, redundant
     concentration that crosses no readiness threshold), with pilot-disagreement overlap.

Output: data/generated/starmie_tactical_leaf_v1/current_leaf_failure_audit.json

  python tools/starmie_leaf_audit_v1.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent"
EXPORT = ROOT / "data" / "generated" / "starmie_tactical_leaf_v1" / "starmie_tactical_state_v1.jsonl"
OUT = ROOT / "data" / "generated" / "starmie_tactical_leaf_v1" / "current_leaf_failure_audit.json"

MEGA_STARMIE, CINDERACE, IGNITION, BASIC_WATER = 1031, 666, 17, 3


def _state(active_id, active_energy_ids, bench):
    """A minimal State-dict the eval functions accept: players[me]=this side, players[1]=a fixed opponent."""
    me = {"active": [{"id": active_id, "hp": 330 if active_id == MEGA_STARMIE else 160,
                      "energies": [{"id": e} for e in active_energy_ids],
                      "energyCards": [{"id": e} for e in active_energy_ids]}],
          "bench": [{"id": b["id"], "hp": b.get("hp", 160),
                     "energies": [{"id": e} for e in b.get("energy", [])],
                     "energyCards": [{"id": e} for e in b.get("energy", [])]} for b in bench],
          "prize": [1] * 4, "deckCount": 30, "handCount": 5}
    opp = {"active": [{"id": 743, "hp": 120, "energies": []}], "bench": [],
           "prize": [1] * 4, "deckCount": 30, "handCount": 5}
    return {"players": [me, opp], "result": -1}


def structural_proof(EV):
    """Show identical leaf score for energy-on-Cinderace vs energy-on-Mega (deck-blind), and Ignition undercount."""
    def contrib(cur):
        P, O = cur["players"][0], cur["players"][1]
        return {
            "W_PRIZE_term": EV.W_PRIZE * (len(O["prize"]) - len(P["prize"])),
            "W_HP_term": EV.W_HP * (EV._board_hp(P) - EV._board_hp(O)),
            "W_BODY_term": EV.W_BODY * (EV._n_pokemon(P) - EV._n_pokemon(O)),
            "W_ENERGY_term": EV.W_ENERGY * EV._active_energy(P),
            "active_energy_cards_counted": EV._active_energy(P),
            "total_evaluate": EV.evaluate(cur, 0),
            "total_deckout_leaf": EV.evaluate_deck_v3(cur, 0, ph_weight=0.0, deckout_weight=EV.DECKOUT_TEST),
        }

    # Case 1: 3 basic Water on the Cinderace ENGINE (Mega benched, bare) vs 3 basic Water on the Mega ATTACKER.
    cind_active = _state(CINDERACE, [BASIC_WATER]*3, [{"id": MEGA_STARMIE, "hp": 330, "energy": []}])
    mega_active = _state(MEGA_STARMIE, [BASIC_WATER]*3, [{"id": CINDERACE, "hp": 160, "energy": []}])
    c_cind, c_mega = contrib(cind_active), contrib(mega_active)

    # Case 2: Ignition (3 UNITS on a Mega) counted as ONE card by the leaf's _active_energy.
    mega_ign = _state(MEGA_STARMIE, [IGNITION], [])
    mega_3w = _state(MEGA_STARMIE, [BASIC_WATER]*3, [])
    c_ign, c_3w = contrib(mega_ign), contrib(mega_3w)

    return {
        "weights": {k: getattr(EV, k) for k in ("W_PRIZE", "W_HP", "W_BODY", "W_ENERGY") if hasattr(EV, k)},
        "case1_cinderace_vs_mega_active_same_3_water": {
            "cinderace_active": c_cind, "mega_active": c_mega,
            "W_ENERGY_identical": c_cind["W_ENERGY_term"] == c_mega["W_ENERGY_term"],
            "leaf_prefers_mega_attacker": c_mega["total_deckout_leaf"] > c_cind["total_deckout_leaf"],
            "finding": ("DECK-BLIND on the attacker: W_ENERGY rewards energy on the active engine (Cinderace) "
                        "exactly as much as on the Mega Starmie attacker; the only difference is raw board HP "
                        "(same bodies either way), so the leaf has NO signal to prefer promoting/keeping the "
                        "Mega attacker over the Cinderace engine."),
        },
        "case2_ignition_undercounted": {
            "mega_one_ignition_3_units": c_ign, "mega_three_basic_water": c_3w,
            "W_ENERGY_ignition": c_ign["W_ENERGY_term"], "W_ENERGY_3water": c_3w["W_ENERGY_term"],
            "finding": ("The leaf's _active_energy counts energy CARDS, so one Ignition (3 functional units, "
                        "Nebula-ready) scores LESS than three Basic Water though both enable the same attacks; "
                        "the leaf undervalues an Ignition-funded Nebula."),
        },
    }


def corpus_prevalence():
    """Scan the export for deck-blind patterns the current leaf cannot see."""
    if not EXPORT.exists():
        return {"error": f"export not found: {EXPORT}; run starmie_tactical_export_v1.py first"}
    rows = [json.loads(l) for l in open(EXPORT, encoding="utf-8")]
    pat = {"cinderace_active_while_mega_ready_or_short": 0, "energy_on_engine_overinvestment": 0,
           "no_main_attacker_continuity": 0, "redundant_concentration_no_threshold": 0,
           "mega_one_attachment_short": 0}
    disagree = {k: 0 for k in pat}
    n = 0
    examples = {k: [] for k in pat}
    for r in rows:
        rt = r["runtime"]; bf = rt["board_features"]; ents = rt["entity_features"]
        em = r["eval_meta"]
        n += 1
        my_active = next((e for e in ents if e["owner"] == "me" and e["slot"] == "active"), None)
        mega_on_bench_ready = any(e for e in ents if e["owner"] == "me" and e["slot"] == "bench"
                                  and e["is_main_attacker"] and (e["attack_ready"] or e["one_attachment_from_ready"]))
        dis = em.get("in_disagreement_class")

        def hit(key):
            pat[key] += 1
            if dis:
                disagree[key] += 1
            if len(examples[key]) < 6:
                examples[key].append({"decision_id": r["decision_id"], "family": rt["action_family"],
                                      "baseline_action": rt["baseline_action"], "pilot_action": em["pilot_action"],
                                      "prize_diff": bf["prize_diff"], "my_ready_main": bf["my_ready_main_attackers"]})

        if my_active and my_active["is_energy_engine"] and mega_on_bench_ready:
            hit("cinderace_active_while_mega_ready_or_short")
        if bf["engine_overinvestment_units"] > 0:
            hit("energy_on_engine_overinvestment")
        if bf["my_ready_main_attackers"] == 0 and bf["my_main_one_short"] >= 1:
            hit("no_main_attacker_continuity")
        if bf["max_energy_concentration"] >= 0.6 and bf["my_ready_main_attackers"] == 0 and bf["my_main_one_short"] == 0:
            hit("redundant_concentration_no_threshold")
        if bf["my_main_one_short"] >= 1:
            hit("mega_one_attachment_short")
    return {"rows_scanned": n, "pattern_counts": pat, "pattern_counts_in_disagreement": disagree,
            "examples": examples,
            "interpretation": ("These are decisions whose tactical distinction (engine vs main attacker, ready vs "
                               "one-short, redundant energy) is INVISIBLE to the current leaf -- the exact cases an "
                               "ATTACKER_CONTINUITY leaf term targets. Disagreement overlap = where our baseline "
                               "already diverged from the pilot in such a state.")}


def main():
    sys.path.insert(0, str(AGENT))
    import eval as EV
    audit = {
        "section": "S3_current_leaf_failure_audit",
        "leaf": "evaluate_deck_v3(leaf_mode='deckout') = W_PRIZE*prizes + W_HP*boardHP + W_BODY*bodies + W_ENERGY*activeEnergyCards - deckout",
        "structural_proof": structural_proof(EV),
        "corpus_prevalence": corpus_prevalence(),
    }
    OUT.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    sp = audit["structural_proof"]["case1_cinderace_vs_mega_active_same_3_water"]
    print("STRUCTURAL: W_ENERGY identical for Cinderace-active vs Mega-active 3-Water:", sp["W_ENERGY_identical"])
    print("           leaf prefers the Mega attacker?", sp["leaf_prefers_mega_attacker"])
    cp = audit["corpus_prevalence"]
    if "pattern_counts" in cp:
        print(f"PREVALENCE over {cp['rows_scanned']} decisions:")
        for k, v in cp["pattern_counts"].items():
            print(f"  {k}: {v}  (in-disagreement {cp['pattern_counts_in_disagreement'][k]})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
