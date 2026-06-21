"""Diagnose the piloting gap: on the expert pilot's OWN replay decisions, what would OUR agent_search pick
(plain, and with the hoard term), and how often does it agree with what the expert actually did? Broken down
by decision type, with concrete divergence examples. Agreement is a diagnostic, not a target (the engine
option-0 ordering inflates it), but the BREAKDOWN and examples show WHERE our piloting diverges from his.

    python tools/compare_choices_v1.py --sample 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import main as M               # noqa: E402
import eval as EV               # noqa: E402
import search as S              # noqa: E402
import search_sprint as SS      # noqa: E402
import state_action_schema_v2 as SCH   # noqa: E402

CARD = json.load(open(ROOT / "agent" / "card_features.json", encoding="utf-8"))
name = lambda c: (CARD.get(str(c), {}) or {}).get("n", "?")
OPT = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL", 5: "ENERGY_CARD", 6: "ENERGY", 7: "PLAY",
       8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT", 13: "ATTACK", 14: "END"}


def his_deck(d, seat):
    for s in d["steps"][:6]:
        if isinstance(s, list) and len(s) > seat and isinstance(s[seat], dict):
            a = s[seat].get("action")
            if isinstance(a, list) and len(a) == 60:
                return list(a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--ref", default="80723114.json")
    args = ap.parse_args()
    import glob
    refd = his_deck(json.load(open(ROOT / "data" / "external" / "replays" / args.ref, encoding="utf-8")), 0)
    refc = Counter(refd)
    # collect his single-pick decisions
    decisions = []
    for fn in glob.glob(str(ROOT / "data" / "external" / "replays" / "*.json")):
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("steps"):
            continue
        seat = next((s for s in range(2) if his_deck(d, s) and sum((Counter(his_deck(d, s)) & refc).values()) >= 55), None)
        if seat is None:
            continue
        deck = his_deck(d, seat)
        for step in d["steps"]:
            if not isinstance(step, list) or len(step) <= seat:
                continue
            ag = step[seat]
            if not isinstance(ag, dict):
                continue
            obs = ag.get("observation") or {}
            sel = obs.get("select") or {}
            act = ag.get("action")
            if (sel.get("maxCount") or 0) == 1 and len(sel.get("option") or []) >= 2 and isinstance(act, list) and act:
                decisions.append((obs, deck, act[0]))
    random.seed(7)
    random.shuffle(decisions)
    decisions = decisions[:args.sample]

    agree_plain = agree_hoard = 0
    by_type = defaultdict(lambda: [0, 0, 0])   # type -> [n, agree_plain, agree_hoard]
    examples = []
    base = None
    for i, (obs, deck, his) in enumerate(decisions):
        cur = obs.get("current") or {}
        me = cur.get("yourIndex", 0)
        opts = obs["select"]["option"]
        if his >= len(opts):
            continue   # replay action index doesn't cleanly index this select (multi-pick / data mismatch)
        ht = OPT.get(opts[his].get("type"), str(opts[his].get("type")))
        try:
            EV.W_POWERFUL_HAND = 0.0
            op = SS.mk_deck(deck)(obs)
            EV.W_POWERFUL_HAND = 15.0
            oh = SS.mk_deck(deck)(obs)
            EV.W_POWERFUL_HAND = 0.0
        except Exception:
            continue
        op0 = op[0] if isinstance(op, list) and op else -1
        oh0 = oh[0] if isinstance(oh, list) and oh else -1
        ap_ = int(op0 == his)
        ah_ = int(oh0 == his)
        agree_plain += ap_
        agree_hoard += ah_
        by_type[ht][0] += 1
        by_type[ht][1] += ap_
        by_type[ht][2] += ah_
        if not ap_ and len(examples) < 12:
            me_p = (cur.get("players") or [{}])[me]
            hc = me_p.get("handCount", "?")
            examples.append({"decision_type": ht, "hand": hc,
                             "his_choice": f"{OPT.get(opts[his].get('type'),'?')} {name(SCH.card_identity(opts[his], me_p))}",
                             "our_choice": f"{OPT.get(opts[op0].get('type'),'?')} {name(SCH.card_identity(opts[op0], me_p))}" if 0 <= op0 < len(opts) else str(op0)})
    n = sum(v[0] for v in by_type.values())
    out = {"decisions_compared": n,
           "agreement_plain": round(agree_plain / max(1, n), 3),
           "agreement_hoard": round(agree_hoard / max(1, n), 3),
           "by_decision_type": {t: {"n": v[0], "agree_plain": round(v[1] / v[0], 3), "agree_hoard": round(v[2] / v[0], 3)}
                                for t, v in sorted(by_type.items(), key=lambda kv: -kv[1][0]) if v[0] >= 3},
           "divergence_examples": examples}
    json.dump(out, open(ROOT / "data" / "manifests" / "compare_choices_v1.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out[k] for k in ("decisions_compared", "agreement_plain", "agreement_hoard", "by_decision_type")}, indent=1))
    print("\nWHERE WE DIVERGE (his choice vs ours, on decisions we disagreed):")
    for e in examples[:10]:
        print(f"  [{e['decision_type']}, hand {e['hand']}] he: {e['his_choice']}  |  we: {e['our_choice']}")


if __name__ == "__main__":
    main()
