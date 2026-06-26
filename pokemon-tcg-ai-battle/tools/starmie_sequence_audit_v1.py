"""Turn-sequence validation of the develop-then-attack hypothesis (Model B sequence-gap task).

The imitation-gap finding ("pilot developed when our agent attacked") does NOT prove "the pilot attacked
later in the same turn." This checks it directly on the replays: for every pilot decision where an ATTACK was
available but the pilot chose a NON-attack (developed), we walk the rest of that same turn and classify what
the pilot did. Agent-independent (no search). Runs on ALL games of top Starmie pilots (wins AND losses) to
remove the winning-game survivorship bias, with a wins-only subset for sensitivity.

Classes (per root develop-when-could-attack case):
  S1 develops then makes the SAME attack later this turn
  S2 develops then a DIFFERENT attack later
  S3 develops then ENDS without attacking
  S4 changes active/target (retreat) then attacks
  S5 ambiguous/incomplete
KO axis (when a KO was available at the root): S6 lost the KO through development; S7 preserved/landed a KO.

  python tools/starmie_sequence_audit_v1.py --max-games 250
Output: data/starmie_audit/starmie_sequence_audit_v1.json
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
OUT = ROOT / "data" / "starmie_audit" / "starmie_sequence_audit_v1.json"
ATTACK, RETREAT, END = 13, 12, 14
MEGA = 1031

import deck_policy_v3 as DP  # noqa: E402

ATK = json.load(open(ROOT / "agent" / "attack_stats.json", encoding="utf-8"))


def _winner(d):
    r = d.get("rewards") or []
    if len(r) < 2 or r[0] is None or r[1] is None or r[0] == r[1]:
        return None
    return 0 if r[0] > r[1] else 1


def _deck_of(d, seat):
    for st in (d.get("steps") or [])[:8]:
        if isinstance(st, list) and len(st) > seat and isinstance(st[seat], dict):
            a = st[seat].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def _chosen(opts, action):
    """(set of chosen option types, set of chosen attackIds) for an action over a menu."""
    types, aids = set(), set()
    if isinstance(action, list):
        for i in action:
            if isinstance(i, int) and 0 <= i < len(opts):
                o = opts[i]
                types.add(o.get("type"))
                if o.get("type") == ATTACK:
                    aids.add(o.get("attackId"))
    return types, aids


def _classify_turn(steps, seat, t):
    """At root step t the pilot had an attack option but developed. Walk the rest of this turn; return a class
    plus whether a root KO was available and whether a KO eventually landed."""
    obs = steps[t][seat].get("observation") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    root_attack_aids = {o.get("attackId") for o in opts if o.get("type") == ATTACK}
    try:
        root_ko = DP.best_ko_attack(obs) is not None
    except Exception:
        root_ko = False

    later_attack_aids, retreated, ended, ndev, attacked = set(), False, False, 0, False
    t2 = t
    while t2 < len(steps) - 1 and steps[t2][seat].get("status") == "ACTIVE":
        o2 = (steps[t2][seat].get("observation") or {}).get("select") or {}
        opts2 = o2.get("option") or []
        types, aids = _chosen(opts2, steps[t2 + 1][seat].get("action"))
        if t2 > t:                         # actions AFTER the root develop
            if ATTACK in types:
                attacked = True
                later_attack_aids |= aids
                break
            if RETREAT in types:
                retreated = True
            if END in types:
                ended = True
                break
            ndev += 1
        t2 += 1

    if attacked:
        if retreated:
            cls = "S4_switch_then_attack"
        elif later_attack_aids & root_attack_aids:
            cls = "S1_same_attack_later"
        else:
            cls = "S2_diff_attack_later"
    elif ended:
        cls = "S3_end_no_attack"
    else:
        cls = "S5_ambiguous"
    ko_axis = None
    if root_ko:
        ko_axis = "S7_ko_preserved" if attacked else "S6_ko_lost"
    return cls, ko_axis, ndev, bool(root_attack_aids)


def audit_game(d, seat):
    steps = d.get("steps") or []
    rows = []
    rev_attack = 0   # reverse: pilot attacked at root
    for t in range(len(steps) - 1):
        e = steps[t][seat]
        if e.get("status") != "ACTIVE":
            continue
        sel = (e.get("observation") or {}).get("select") or {}
        opts = sel.get("option") or []
        if len(opts) < 2:
            continue
        types, _ = _chosen(opts, steps[t + 1][seat].get("action"))
        has_attack = any(o.get("type") == ATTACK for o in opts)
        if not has_attack:
            continue
        if ATTACK in types:
            rev_attack += 1
            continue
        # pilot could attack but developed
        cls, ko_axis, ndev, _ = _classify_turn(steps, seat, t)
        rows.append({"step": t, "cls": cls, "ko_axis": ko_axis, "ndev": ndev})
    return rows, rev_attack


def run_subset(games, label):
    cls_c, ko_c = Counter(), Counter()
    ndevs, n_cases, n_rev = [], 0, 0
    for fn, seat in games:
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        rows, rev = audit_game(d, seat)
        n_rev += rev
        for r in rows:
            cls_c[r["cls"]] += 1
            if r["ko_axis"]:
                ko_c[r["ko_axis"]] += 1
            ndevs.append(r["ndev"])
            n_cases += 1
    tot = max(1, n_cases)
    later = cls_c["S1_same_attack_later"] + cls_c["S2_diff_attack_later"] + cls_c["S4_switch_then_attack"]
    return {
        "label": label, "n_games": len(games), "n_develop_when_could_attack": n_cases,
        "n_reverse_pilot_attacked_at_root": n_rev,
        "classes": {k: cls_c[k] for k in sorted(cls_c)},
        "class_pct": {k: round(cls_c[k] / tot * 100, 1) for k in sorted(cls_c)},
        "attacked_later_same_turn_pct": round(later / tot * 100, 1),
        "same_attack_pct": round(cls_c["S1_same_attack_later"] / tot * 100, 1),
        "ended_no_attack_pct": round(cls_c["S3_end_no_attack"] / tot * 100, 1),
        "ko_axis": dict(ko_c),
        "ko_preserved_pct": (round(ko_c["S7_ko_preserved"] / max(1, ko_c["S7_ko_preserved"] + ko_c["S6_ko_lost"]) * 100, 1)),
        "avg_dev_actions_before_attack": round(sum(ndevs) / max(1, len(ndevs)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=250)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pilots = json.loads((ROOT / "data" / "starmie_top_pilots.json").read_text(encoding="utf-8"))
    names = {p["name"] for p in pilots["pilots"]}
    print(f"scanning corpus for ALL Starmie games of {len(names)} top pilots (wins+losses)...", flush=True)

    all_games, win_games = [], []
    files = sorted(glob.glob(REPLAYS + "/*.json"))
    for i, fn in enumerate(files):
        if len(all_games) >= a.max_games:
            break
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        w = _winner(d)
        if w is None:
            continue
        tn = (d.get("info") or {}).get("TeamNames") or []
        for seat in (0, 1):
            nm = str(tn[seat]) if seat < len(tn) and tn[seat] else ""
            if nm not in names:
                continue
            dk = _deck_of(d, seat)
            if not dk or MEGA not in dk:
                continue
            all_games.append((fn, seat))
            if seat == w:
                win_games.append((fn, seat))
        if (i + 1) % 1500 == 0:
            print(f"  {i+1}/{len(files)} scanned, {len(all_games)} games", flush=True)

    print(f"collected {len(all_games)} all-games ({len(win_games)} wins). auditing...", flush=True)
    res_all = run_subset(all_games, "ALL_GAMES")
    res_win = run_subset(win_games, "WINS_ONLY")

    # FINDING GATE
    g = res_all
    gate = (g["attacked_later_same_turn_pct"] >= 60 and g["same_attack_pct"] >= 40
            and (g["ko_preserved_pct"] if g["ko_axis"] else 100) >= 80)
    if gate:
        verdict = "A. DEVELOP_THEN_ATTACK_CONFIRMED"
    elif g["attacked_later_same_turn_pct"] >= 45:
        verdict = "B. DIRECTIONAL_SEQUENCE_SIGNAL"
    elif res_win["attacked_later_same_turn_pct"] - res_all["attacked_later_same_turn_pct"] >= 12:
        verdict = "C. WINNING_GAME_SELECTION_ARTIFACT"
    else:
        verdict = "D. NO_GENERAL_SEQUENCE_RULE"

    payload = {"all_games": res_all, "wins_only": res_win, "finding_gate_passed": gate, "SEQUENCE_VERDICT": verdict,
               "note": "Agent-independent: classifies what PILOTS do after developing when an attack was available. The 339-asymmetry was generated by the DEPLOYED agent (no develop-before-attack), which attacks at most of these roots."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for r in (res_all, res_win):
        print(f"\n=== {r['label']} ({r['n_games']} games, {r['n_develop_when_could_attack']} develop-when-could-attack cases) ===", flush=True)
        for k in sorted(r["classes"]):
            print(f"  {k:24} {r['classes'][k]:5d}  {r['class_pct'][k]:5.1f}%", flush=True)
        print(f"  -> attacked later same turn: {r['attacked_later_same_turn_pct']}% (same attack {r['same_attack_pct']}%) | "
              f"ended no-attack {r['ended_no_attack_pct']}% | KO preserved {r['ko_preserved_pct']}% | avg dev before attack {r['avg_dev_actions_before_attack']}", flush=True)
    print(f"\nFINDING GATE passed: {gate}\nSEQUENCE_VERDICT: {verdict}\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
