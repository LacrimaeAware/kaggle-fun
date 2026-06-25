"""Dump REAL cabt observations while piloting the Starmie deck, so heuristics map to the true option
schema (setup placement, energy attach, attack menu, target/search prompts) instead of guesses.

Runs one cabt game with a recorder agent on the Starmie deck. For each decision it logs the select
context, min/max count, the histogram of option types, and one full example option per (type) -- plus
the full option list for the first time each distinct (context,maxCount,types) signature appears. The
recorder returns a fast legal default so the game advances quickly (no search).

  python tools/dump_starmie_obs_v1.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))

STARMIE_DECK = (
    [3] * 9 + [17] * 4 + [666] * 4 + [1030] * 3 + [1031] * 3 + [1086] * 4 + [1097] * 2 + [1120] * 4
    + [1121] * 1 + [1122] * 4 + [1145] * 4 + [1159] * 1 + [1182] * 1 + [1189] * 4 + [1223] * 2
    + [1225] * 2 + [1227] * 4 + [1229] * 4
)

TYPE_NAMES = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD", 6: "ENERGY",
              7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT",
              13: "ATTACK", 14: "END"}

seen_sigs: set = set()
records: list = []


@contextlib.contextmanager
def _quiet_import():
    old = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(dn)
        os.close(old)


def _default(sel):
    opts = sel.get("option") or []
    k = sel.get("maxCount") or 0
    mn = sel.get("minCount") or 0
    n = len(opts)
    if n == 0 or k <= 0:
        return []
    return list(range(max(min(k, n), min(mn, n))))


def _cardname(cid, CDB):
    return (CDB.get(str(cid), {}) or {}).get("n", f"#{cid}")


def make_recorder(CDB, ATK):
    def agent(obs):
        sel = obs.get("select")
        if sel is None:
            return list(STARMIE_DECK)
        opts = sel.get("option") or []
        types = Counter(o.get("type") for o in opts)
        ctx = sel.get("context")
        sig = (ctx, sel.get("maxCount"), tuple(sorted(types)))
        type_str = {TYPE_NAMES.get(t, t): c for t, c in types.items()}
        rec = {"ctx": ctx, "min": sel.get("minCount"), "max": sel.get("maxCount"),
               "n_opt": len(opts), "types": type_str}
        # attach attackId/name detail when attacks present
        atks = []
        for o in opts:
            if o.get("type") == 13:
                aid = o.get("attackId")
                arow = ATK.get(str(aid), {})
                atks.append({"attackId": aid, "name": arow.get("n"), "dmg": arow.get("d")})
        if atks:
            rec["attacks"] = atks
        records.append(rec)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            # full dump of one example for each distinct signature
            example_opts = []
            for o in opts[:8]:
                example_opts.append({k: v for k, v in o.items()})
            rec["EXAMPLE_OPTIONS"] = example_opts
            # also dump my board summary
            cur = obs.get("current") or {}
            players = cur.get("players") or []
            yi = cur.get("yourIndex", 0)
            me = players[yi] if yi < len(players) else {}
            act = (me.get("active") or [None])[0]
            rec["my_active"] = (_cardname((act or {}).get("id"), CDB) if act else None)
            rec["my_bench"] = [_cardname((b or {}).get("id"), CDB) for b in (me.get("bench") or [])]
            rec["my_handcount"] = me.get("handCount")
        return _default(sel)
    return agent


def main():
    with _quiet_import(), contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make
    CDB = json.load(open(ROOT / "agent" / "card_stats.json", encoding="utf-8"))
    ATK = json.load(open(ROOT / "agent" / "attack_stats.json", encoding="utf-8"))
    rec = make_recorder(CDB, ATK)
    print("running 1 cabt game (Starmie mirror, recorder pilot)...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        env = make("cabt")
        env.run([rec, rec])
    print(f"captured {len(records)} decisions\n", flush=True)
    out = ROOT / "data" / "starmie_obs_dump.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    # print distinct signatures with their first example
    print("=== distinct prompt signatures (first example each) ===", flush=True)
    for r in records:
        if "EXAMPLE_OPTIONS" in r:
            print(json.dumps(r, indent=2), flush=True)
            print("-" * 60, flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
