"""Label-quality audit for the imitation dataset (the other model's step 1: validate targets before
training more). For the chosen deck's WINNER decisions, answer:
  1. One policy or many? distinct winner players piloting this exact deck (mixed labels if many).
  2. Are option labels semantically unique? fraction of decisions where the chosen option has an
     EQUIVALENT sibling (same type+card+target) -> exact top-1 mislabels a strategically-equal pick.
  3. How many decisions are genuinely STRATEGIC vs forced/target-resolution? by SelectContext, and by
     "distinct option types > 1".
  4. option-0 rate overall vs on the clean strategic subset (the real bar to beat).

    python tools/audit_action_labels.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A_HAND = 2
CTX = {0: "main", 1: "setup-active", 2: "setup-bench", 3: "switch/select", 4: "to-active", 5: "to-bench",
       7: "to-hand", 8: "discard/target", 9: "target2", 18: "evolve-from", 19: "evolve-to",
       21: "attach-from", 22: "attach-to", 35: "attack", 37: "evolve", 41: "go-first", 42: "mulligan"}


def winner_deck(d, win):
    for s in d.get("steps", []):
        if win < len(s) and isinstance(s[win], dict):
            a = s[win].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def opt_key(o, me, cur):
    """A canonical key: two options with the same key are strategically equivalent (same move)."""
    cid = None
    if o.get("area") == A_HAND and isinstance(o.get("index"), int):
        hand = ((cur.get("players") or [{}])[me]).get("hand") or []
        if 0 <= o["index"] < len(hand):
            c = hand[o["index"]]
            cid = c.get("id") if isinstance(c, dict) else c
    return (o.get("type"), cid, o.get("attackId"), o.get("inPlayArea"), o.get("inPlayIndex"))


def main():
    files = glob.glob(str(ROOT / "data" / "external" / "replays" / "*.json"))
    # pass 1: pick deck with most winner-decisions
    stat = defaultdict(lambda: {"wdec": 0, "deck": None})
    games = []
    for fp in files:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rw = d.get("rewards") or []
        if len(rw) != 2 or rw[0] == rw[1] or None in rw:
            continue
        win = 0 if rw[0] > rw[1] else 1
        deck = winner_deck(d, win)
        if not deck:
            continue
        sig = tuple(sorted(deck))
        nm = [a.get("Name") for a in d.get("info", {}).get("Agents", [])]
        wname = nm[win] if win < len(nm) else "?"
        nd = 0
        for s in d.get("steps", []):
            if win >= len(s) or not isinstance(s[win], dict):
                continue
            sel = (s[win].get("observation") or {}).get("select") or {}
            if (sel.get("maxCount") or 0) == 1 and len(sel.get("option") or []) >= 2 and len(s[win].get("action") or []) == 1:
                nd += 1
        stat[sig]["wdec"] += nd; stat[sig]["deck"] = deck
        games.append((sig, win, wname, d))
    target = max(stat.values(), key=lambda x: x["wdec"])
    tsig = tuple(sorted(target["deck"]))

    # pass 2: audit the target deck's winner decisions
    players = Counter()
    ctx_count = Counter()
    n_dec = n_equiv_chosen = n_strategic = 0
    opt0_all = opt0_strat = strat_total = 0
    for sig, win, wname, d in games:
        if sig != tsig:
            continue
        players[wname] += 1
        for s in d.get("steps", []):
            if win >= len(s) or not isinstance(s[win], dict):
                continue
            obs = s[win].get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            act = s[win].get("action") or []
            cur = obs.get("current") or {}
            if (sel.get("maxCount") or 0) != 1 or len(opts) < 2 or len(act) != 1 or not cur.get("players"):
                continue
            me = cur.get("yourIndex", win)
            chosen = act[0]
            if not (isinstance(chosen, int) and 0 <= chosen < len(opts)):
                continue
            n_dec += 1
            ctx_count[CTX.get(sel.get("context"), str(sel.get("context")))] += 1
            keys = [opt_key(o, me, cur) for o in opts if isinstance(o, dict)]
            distinct = len(set(keys))
            chosen_key = keys[chosen] if chosen < len(keys) else None
            chosen_dupes = keys.count(chosen_key)
            if chosen_dupes > 1:
                n_equiv_chosen += 1
            # "strategic" = the options are not all the same type AND > 1 distinct move
            types = {o.get("type") for o in opts if isinstance(o, dict)}
            strategic = (len(types) > 1) and (distinct > 1)
            if strategic:
                n_strategic += 1
                strat_total += 1
                if chosen == 0:
                    opt0_strat += 1
            if chosen == 0:
                opt0_all += 1

    print(f"TARGET DECK winner-decisions: {n_dec}  (deck sig has {target['wdec']} via pass-1 count)")
    print(f"\n1. POLICY MIXTURE -- distinct winner players piloting this exact deck: {len(players)}")
    for nm, c in players.most_common(8):
        print(f"   {c:>3} win-games  {nm}")
    print(f"   (if >1, the imitation labels mix multiple policies on the same states)")
    print(f"\n2. LABEL AMBIGUITY -- decisions where the chosen option has an EQUIVALENT sibling "
          f"(same type+card+target): {n_equiv_chosen}/{n_dec} = {n_equiv_chosen/max(1,n_dec):.1%}")
    print(f"   (exact top-1 scores these as wrong even when the alternative is identical)")
    print(f"\n3. DECISION TYPE mix (SelectContext):")
    for k, c in ctx_count.most_common(10):
        print(f"   {c:>5}  {k}")
    print(f"\n4. STRATEGIC subset (>1 option type AND >1 distinct move): {n_strategic}/{n_dec} = "
          f"{n_strategic/max(1,n_dec):.1%}")
    print(f"   option-0 rate: ALL {opt0_all/max(1,n_dec):.3f} | STRATEGIC-only {opt0_strat/max(1,strat_total):.3f}")
    print(f"\nRead: high player-count = mixed policies (clean by player). High equivalence% = top-1 is the")
    print(f"wrong metric. Low strategic% = most 'decisions' are forced/target prompts. The real imitation")
    print(f"task + baseline is the STRATEGIC subset, ideally from one coherent player.")


if __name__ == "__main__":
    main()
