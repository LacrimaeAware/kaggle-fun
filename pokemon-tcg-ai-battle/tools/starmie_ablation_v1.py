"""Stage-A rule ablation: one-rule-off imitation screen for the Starmie heavy agent.

For BASE (all rules on) and each rule Rk disabled (via STARMIE_DISABLE env), run the imitation-gap harness on
N top-pilot games and record overall + per-category agreement. A rule HELPS if disabling it DROPS agreement;
it HURTS if disabling RAISES agreement. This isolates which forced rules carry their weight before any local
A/B. (Imitation is a proxy, not the objective -- escalate suspicious rules to a field A/B separately.)

  python tools/starmie_ablation_v1.py --games 24 --budget 0.3
Output: data/starmie_audit/starmie_rule_ablation_v2.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAVY_JSON = ROOT / "data" / "imitation_gap_heavy.json"
OUT = ROOT / "data" / "starmie_audit" / "starmie_rule_ablation_v2.json"

RULES = {
    "R0": "go_first", "R1": "no_suicide", "R2": "bench_dev", "R3": "evolve_mega", "R4": "boss_gust",
    "R5": "wally", "R6": "heros_cape", "R7": "energy_attach", "R8": "crushing_hammer", "R9": "ko_floor",
    "R10": "retreat_pivot", "R11": "tutor_target", "R12": "bench_target", "R13": "wally_veto",
    "R14": "no_third_mega_guard",
}
CATS = ["play", "attack", "select_card", "energy_attach", "evolve", "retreat", "end_turn"]


def _run(disable, games, budget):
    env = dict(os.environ)
    env["STARMIE_DISABLE"] = disable
    cmd = [sys.executable, "tools/imitation_gap_v1.py", "--agent", "heavy",
           "--max-games", str(games), "--budget", str(budget), "--workers", "6"]
    subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    d = json.loads(HEAVY_JSON.read_text(encoding="utf-8"))
    ca = d.get("category_agreement") or {}
    return {"overall": d.get("agreement_rate"), "decisions": d.get("total_decisions"),
            "cats": {c: (ca.get(c) or {}).get("rate") for c in CATS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--budget", type=float, default=0.3)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    order = ["BASE"] + list(RULES)
    rows = {}
    print(f"ablation: {len(order)} configs x {a.games} games (imitation screen)", flush=True)
    for k in order:
        rows[k] = _run("" if k == "BASE" else k, a.games, a.budget)
        o = rows[k]["overall"]
        print(f"  {k:5} ({RULES.get(k,'all on'):20}) overall {o*100:.1f}%" if o is not None else f"  {k}: ERROR", flush=True)

    base = rows["BASE"]["overall"] or 0
    table = []
    for k in order:
        r = rows[k]
        d = (r["overall"] or 0) - base
        table.append({"rule": k, "name": RULES.get(k, "all_on"), "overall": r["overall"],
                      "delta_vs_base": round(d, 4), "decisions": r["decisions"], "cats": r["cats"]})
    OUT.write_text(json.dumps({"games": a.games, "base_overall": base, "rows": table}, indent=2), encoding="utf-8")

    print("\n=== one-rule-OFF deltas vs BASE (negative = rule HELPS, positive = rule HURTS) ===", flush=True)
    print(f"  {'rule':5} {'name':20} {'overall':>8} {'delta':>7}", flush=True)
    for t in sorted(table, key=lambda x: (x["delta_vs_base"])):
        if t["rule"] == "BASE":
            continue
        mark = "  <- HELPS" if t["delta_vs_base"] <= -0.01 else ("  <- hurts?" if t["delta_vs_base"] >= 0.01 else "")
        print(f"  {t['rule']:5} {t['name']:20} {(t['overall'] or 0)*100:7.1f}% {t['delta_vs_base']*100:+6.1f}{mark}", flush=True)
    print(f"\nBASE overall {base*100:.1f}% | wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
