"""Deterministic post-deploy check: prove the SHIPPED production agent has the PH-fix + N=32,
without playing any games (the engine is shared; only agent code differs, so a unit check on the
exact behavioral change is sufficient and noise-free).

Run BEFORE deploy -> expected FAIL (proves the test targets the real change).
Run AFTER  deploy -> expected PASS.

    python tools/verify_ship_v1.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import main as M
import search as S

ALAKAZAM, PSYCHIC = 743, 5
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")


# --- 1. lethal Powerful Hand is recognized as a KO by PRODUCTION _attack_value ---
# Alakazam active, 7-card hand -> PH = 20*7 = 140 vs a 130-HP target = lethal.
me = {"active": [{"id": ALAKAZAM, "hp": 140, "energies": [PSYCHIC]}], "bench": [],
      "handCount": 7, "prize": [1, 2, 3]}
opp = {"active": [{"id": 99999, "hp": 130}], "bench": []}
ph_attack = {"type": 13, "attackId": 88888}        # Powerful Hand option (0 static dmg in ATK)
v_lethal = M._attack_value(ph_attack, me, opp)
check("lethal PH recognized", v_lethal >= 8000,
      f"_attack_value={v_lethal:.0f} (need >=8000; pre-fix baseline returns 2000)")

# --- 2. NON-lethal PH must NOT hallucinate a KO ---
me3 = dict(me); me3["handCount"] = 3                # PH = 60 < 130
v_small = M._attack_value(ph_attack, me3, opp)
check("non-lethal PH not a KO", v_small < 8000,
      f"_attack_value={v_small:.0f} (need <8000; hand=3 -> PH=60 < 130 HP)")

# --- 3. _forced_move takes the lethal PH (so the bot stops walking past its own KO) ---
obs = {"current": {"yourIndex": 0, "players": [me, opp]},
       "select": {"maxCount": 1, "minCount": 1, "option": [{"type": 14}, ph_attack]}}  # END, PH
fm = M._forced_move(obs)
check("forced_move takes lethal PH", fm == [1],
      f"forced_move={fm} (need [1] = the Powerful Hand option, not END)")

# --- 4. a non-Alakazam active with a real 0-damage situation is untouched (port is gated) ---
me_k = {"active": [{"id": 742, "hp": 80, "energies": []}], "bench": [], "handCount": 7, "prize": [1]}
v_kad = M._attack_value(ph_attack, me_k, opp)       # active is Kadabra, not Alakazam
check("port gated on Alakazam", v_kad < 8000,
      f"_attack_value={v_kad:.0f} (Kadabra active -> no PH substitution, must stay <8000)")

# --- 5. N=32 sampling deployed ---
check("N_DETERM == 32", S.N_DETERM == 32, f"search.N_DETERM={S.N_DETERM}")

# --- 6. budget lets all 32 worlds run (need ~0.56s; must be clearly > that, < forfeit risk) ---
check("budget raised for N=32", S.DEFAULT_BUDGET >= 1.0,
      f"search.DEFAULT_BUDGET={S.DEFAULT_BUDGET} (need >=1.0 so 32 worlds complete)")

print("\n" + "#" * 60)
ok = all(results)
print(f"SHIP VERIFY: {'ALL PASS' if ok else 'FAILURES PRESENT'}  ({sum(results)}/{len(results)})")
print("#" * 60)
sys.exit(0 if ok else 1)
