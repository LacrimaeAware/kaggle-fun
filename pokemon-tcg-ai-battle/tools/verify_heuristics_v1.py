"""Per-heuristic verification: for each heuristic, BUILD the exact situation where it applies and
check what the bot actually does (baseline vs the proposed fix). Pure function calls on synthetic
observations -- no games, fast, deterministic. Reports FACTS per heuristic.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import main as M               # production baseline
import deck_policy_v2 as DP2   # proposed fixes
import features as FT

ALAKAZAM, KADABRA = 743, 742
PSYCHIC = 5
PASS, FAIL = "PASS", "FAIL"


def obs_of(me, opp, options, yi=0):
    return {"current": {"yourIndex": yi, "players": [me, opp]},
            "select": {"maxCount": 1, "minCount": 1, "option": options}}


def hr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------
hr("H1. Powerful Hand KO recognition  (Alakazam active, big hand vs low-HP target)")
# Situation: my Alakazam, 7 cards in hand -> Powerful Hand = 20*7 = 140 damage.
# Opponent active has 130 HP. So Powerful Hand is LETHAL.
me = {"active": [{"id": ALAKAZAM, "hp": 140, "energies": [PSYCHIC]}], "bench": [],
      "handCount": 7, "prize": [1, 2, 3]}
opp = {"active": [{"id": 99999, "hp": 130}], "bench": []}
ph_attack = {"type": 13, "attackId": 88888}        # Alakazam's only attack option = Powerful Hand (0 static dmg)
obs = obs_of(me, opp, [ph_attack])
base = M._attack_value(ph_attack, me, opp)
fix = DP2.attack_value_m0(ph_attack, obs, 0)
print(f"  hand=7 -> Powerful Hand should do 20*7=140 vs 130 HP = LETHAL")
print(f"  BASELINE _attack_value : {base:>9.0f}   sees KO? {base >= 8000}")
print(f"  FIXED  attack_value_m0 : {fix:>9.0f}   sees KO? {fix >= 8000}")
v1 = PASS if (base < 8000 and fix >= 8000) else FAIL
print(f"  => [{v1}] baseline MISSES the lethal Powerful Hand; fix CATCHES it.")

# ---------------------------------------------------------------
hr("H1b. Powerful Hand NOT lethal  (small hand) -- fix must not hallucinate a KO")
me2 = dict(me); me2["handCount"] = 3                # PH = 60 < 130
obs2 = obs_of(me2, opp, [ph_attack])
fix2 = DP2.attack_value_m0(ph_attack, obs2, 0)
print(f"  hand=3 -> Powerful Hand = 60 vs 130 HP = NOT lethal")
print(f"  FIXED attack_value_m0  : {fix2:>9.0f}   sees KO? {fix2 >= 8000}")
v1b = PASS if fix2 < 8000 else FAIL
print(f"  => [{v1b}] fix correctly does NOT call this a KO.")

# ---------------------------------------------------------------
hr("H1c. forced_move only forces a GAME-WINNING Powerful Hand, not an ordinary one")
# ordinary lethal PH but opponent still has 2 prizes -> NOT game-ending -> must NOT be auto-forced (falls to search)
me_g = {"active": [{"id": ALAKAZAM, "hp": 140, "energies": [PSYCHIC]}], "bench": [], "handCount": 7, "prize": [1, 2]}
fm = DP2.forced_move_m0(obs_of(me_g, opp, [ph_attack]))
# game-ending: I have 1 prize left, KO takes it -> should force
me_w = dict(me_g); me_w["prize"] = [1]
fmw = DP2.forced_move_m0(obs_of(me_w, opp, [ph_attack]))
print(f"  2 prizes left, lethal PH -> forced_move_m0 = {fm}  (should be None: let search decide)")
print(f"  1 prize  left, lethal PH -> forced_move_m0 = {fmw}  (should be [0]: confirmed game win)")
v1c = PASS if (fm is None and fmw == [0]) else FAIL
print(f"  => [{v1c}]")

# ---------------------------------------------------------------
hr("H2. Energy-on-active rule  (active can't attack yet, psychic energy attachable)")
# Kadabra active with 0 energy needs 1 Psychic for Super Psy Bolt. Options: END, or ATTACH psychic to active.
me_e = {"active": [{"id": KADABRA, "hp": 80, "energies": []}], "bench": [],
        "hand": [{"id": PSYCHIC}], "handCount": 1, "prize": [1]}
opp_e = {"active": [{"id": 99999, "hp": 100}], "bench": []}
A_ACTIVE = 4
opts_e = [{"type": 14}, {"type": 8, "inPlayArea": A_ACTIVE, "area": 2, "index": 0}]  # END, ATTACH->active
obs_e = obs_of(me_e, opp_e, opts_e)
f = FT.encode_state(obs_e)
choice = M._choose(obs_e)
chosen_type = opts_e[choice[0]]["type"] if choice and 0 <= choice[0] < len(opts_e) else None
print(f"  feature active_energy_short = {f.get('active_energy_short')}  (should be >0: Kadabra needs 1 P)")
print(f"  baseline _choose picked option {choice} (type {chosen_type})  -- 8=ATTACH, 14=END")
v2 = PASS if chosen_type == 8 else FAIL
print(f"  => [{v2}] baseline {'attaches energy to the active' if v2==PASS else 'does NOT attach to active'}.")

print("\n" + "#" * 70)
print(f"SUMMARY: H1 {v1} | H1b {v1b} | H1c {v1c} | H2 {v2}")
print("#" * 70)
