"""STARMIE TACTICAL-LEAF V1 -- Section 8: MODEL A HANDOFF artifacts.

Writes:
  tactical_feature_schema.json     -- the feature taxonomy + which features are deck-independent / Starmie
                                      semantic-role dependent / exact-card-id dependent / public-at-runtime /
                                      evaluation-only.
  tactical_coordinate_summary.json -- aggregate distributions of the tactical coordinates over the export, so
                                      Model A can see the situation-representation signal without re-running.

  python tools/starmie_handoff_v1.py
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "generated" / "starmie_tactical_leaf_v1"
EXPORT = GEN / "starmie_tactical_state_v1.jsonl"

SCHEMA = {
    "schema_version": "starmie_tactical_state_v1",
    "row_layout": {
        "decision_id": "stable id episode_step_seat",
        "runtime": "PUBLIC feature payload consumable at runtime (no hidden info, no outcome, no pilot/replay id)",
        "eval_meta": "EVALUATION-ONLY metadata; MUST NOT be a runtime model input",
    },
    "public_at_runtime": "EVERY field under runtime.* is computed only from the seat's public observation.",
    "evaluation_only": ["eval_meta.pilot_action", "eval_meta.pilot", "eval_meta.won",
                        "eval_meta.same_turn_sequence", "eval_meta.replay_id", "eval_meta.split",
                        "eval_meta.in_disagreement_class", "runtime.baseline_action (advisory)"],
    "feature_classes": {
        "deck_independent": [
            "board_features.prize_diff", "board_features.my_prizes_left", "board_features.opp_prizes_left",
            "board_features.my_pokemon", "board_features.opp_pokemon", "board_features.my_board_hp",
            "board_features.opp_board_hp", "board_features.my_units", "board_features.opp_units",
            "board_features.my_hand_count", "board_features.opp_hand_count", "board_features.my_deck_count",
            "board_features.opp_deck_count", "board_features.my_deckout_risk", "board_features.my_bench_capacity",
            "board_features.max_energy_concentration", "entity.hp_remaining", "entity.hp_max", "entity.damage",
            "entity.prize_liability", "entity.attached_cards", "entity.retreat_cost", "entity.retreat_affordable",
            "coordinates.VALUE_STATE.*", "coordinates.RACE_STATE.prize_diff",
        ],
        "starmie_semantic_role_dependent": [
            "entity.role", "entity.is_main_attacker", "entity.is_energy_engine", "board_features.my_ready_main_attackers",
            "board_features.my_backup_ready", "board_features.my_main_one_short", "board_features.my_engine_count",
            "board_features.engine_overinvestment_units", "board_features.energy_on_main_attackers",
            "coordinates.SWEEP_PRESSURE.*", "coordinates.RACE_STATE.both_mains_online",
        ],
        "exact_card_id_dependent": [
            "role classification: Mega Starmie ex=1031 (main_attacker), Cinderace=666 (energy_engine), Staryu=1030 (setup_basic)",
            "Ignition Energy=17 -> 3 energy UNITS on a Mega (unit-aware affordability)",
            "verified attack ids: Jetting Blow 1487 (>=1 unit, 120 +50 snipe), Nebula Beam 1488 (>=3 units, 210 flat), Turbo Flare 965 (>=1 unit, 50)",
            "entity.attached_units", "entity.affordable_attacks", "entity.max_affordable_damage",
            "entity.can_ko_opposing_active", "entity.attack_ready", "entity.one_attachment_from_ready",
        ],
        "uncertain_for_opponents": [
            "entity.damage_uncertainty=True for non-Starmie cards (damage estimated from card_stats, cost field "
            "unreliable for conditional/special-energy attacks); opponent attack readiness is approximate.",
        ],
    },
    "coordinates": {
        "RACE_STATE": "immediate-KO capability both sides, both mains online, prize + backup-continuity diff",
        "SWEEP_PRESSURE": "ready main + backup vs opponent response; expected consecutive KOs",
        "WALL_PRESSURE": "opponent active HP vs my max damage; bench-behind; (gust/retreat are agent-side, not encoded here)",
        "VALUE_STATE": "prize / ready-attacker / backup / energy / hand / board-HP diffs; deckout pressure",
        "COMMITMENT_STATE": "which action families are available now (game-win / KO / nonterminal attack / develop / attach / supporter / info-reveal / retreat / end)",
    },
    "consumption_note": ("Model A may consume runtime.* as a situation representation for specialist/value "
                         "experiments. Do NOT feed any eval_meta field as a model input. The deck-independent "
                         "block transfers to other decks; the role-dependent and card-id blocks are Starmie-specific."),
}


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def main():
    (GEN / "tactical_feature_schema.json").write_text(json.dumps(SCHEMA, indent=2), encoding="utf-8")

    # coordinate summary over the export
    rows = 0
    cols = {}
    fam = {}
    bool_counts = {}
    for line in open(EXPORT, encoding="utf-8"):
        r = json.loads(line)
        rows += 1
        fam[r["runtime"]["action_family"]] = fam.get(r["runtime"]["action_family"], 0) + 1
        bf = r["runtime"]["board_features"]
        for k in ("prize_diff", "my_ready_main_attackers", "my_backup_ready", "my_main_one_short",
                  "my_units", "opp_units", "max_energy_concentration", "engine_overinvestment_units",
                  "my_immediate_ko", "opp_immediate_ko", "my_deck_count"):
            if _num(bf.get(k)):
                cols.setdefault(k, []).append(bf[k])
        cs = r["runtime"]["tactical_coordinates"]["COMMITMENT_STATE"]
        for k, v in cs.items():
            if isinstance(v, bool):
                bool_counts.setdefault(k, [0, 0])
                bool_counts[k][0 if v else 1] += 1

    def desc(v):
        v = sorted(v)
        return {"n": len(v), "mean": round(sum(v) / len(v), 3), "median": v[len(v) // 2],
                "p10": v[int(0.1 * len(v))], "p90": v[int(0.9 * len(v))], "min": v[0], "max": v[-1]}

    summary = {"rows": rows, "by_family": fam,
               "board_feature_distributions": {k: desc(v) for k, v in cols.items() if v},
               "commitment_state_true_rate": {k: round(t / (t + f), 3) for k, (t, f) in bool_counts.items()}}
    (GEN / "tactical_coordinate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote tactical_feature_schema.json + tactical_coordinate_summary.json")
    print(f"rows summarized: {rows}")
    print("commitment_state true-rate:", json.dumps(summary["commitment_state_true_rate"]))


if __name__ == "__main__":
    main()
