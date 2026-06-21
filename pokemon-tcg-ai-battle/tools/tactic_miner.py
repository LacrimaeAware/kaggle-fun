"""Branch A / Tactic Miner V1.

Mines strong-player (game-winner) replay decisions from the FROZEN snapshot. For each decision it compares
the CHOSEN action against all legal SIBLING actions from the same root, classifies each into the tactic
ontology, and buckets by discrete context. It surfaces (context, tactic) patterns where strong players
systematically pick a tactic above its base rate (lift), plus short within-turn tactic sequences (macros).

Output: data/manifests/mined_tactics_v1.json (ranked candidate tactics with support, choose-rate, lift,
contexts, coverage, leverage, mechanics confidence) + a printed summary.

    python tools/tactic_miner.py --snapshot replays_20260618.json --split replays_20260618_split.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import state_action_schema_v2 as SCH   # noqa: E402
import tactics_ontology as TO          # noqa: E402

REPLAY_DIR = ROOT / "data" / "external" / "replays"

# coarse mechanics confidence: how well-understood/clean is acting on this tactic (draw burned us once)
MECH_CONF = {"ko": 0.95, "attack": 0.9, "gust": 0.9, "retreat": 0.7, "evolve": 0.75, "accelerate": 0.7,
             "tutor": 0.7, "develop_board": 0.7, "attach": 0.7, "heal": 0.7, "disruption": 0.7,
             "ability_unlock": 0.6, "draw": 0.4, "switch": 0.6, "select": 0.4, "play": 0.4, "end": 0.5}


def _winner(d: dict):
    rw = d.get("rewards") or []
    if len(rw) != 2 or None in rw or rw[0] == rw[1]:
        return None
    return 0 if rw[0] > rw[1] else 1


def mine(snapshot: str, split_name: str, max_games: int):
    split = json.load(open(ROOT / "data" / "splits" / split_name, encoding="utf-8"))
    files = split["train"][:max_games] if max_games else split["train"]

    tac_all = defaultdict(lambda: [0, 0])          # tactic -> [available, chosen]
    ctx_tac = defaultdict(lambda: [0, 0])          # (ctx_key, ctx_val, tactic) -> [available, chosen]
    bigrams = Counter()                            # (tacticA -> tacticB) within a winner's turn
    decks = set(); players = Counter()
    n_dec = 0

    for fn in files:
        fp = REPLAY_DIR / fn
        if not fp.exists():
            continue
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        w = _winner(d)
        if w is None:
            continue
        nm = [a.get("Name") for a in (d.get("info", {}).get("Agents") or [])]
        if w < len(nm):
            players[nm[w]] += 1
        turn_seq = []
        for s in d.get("steps", []):
            if w >= len(s) or not isinstance(s[w], dict):
                continue
            obs = s[w].get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            act = s[w].get("action") or []
            cur = obs.get("current") or {}
            if (sel.get("maxCount") or 0) != 1 or len(opts) < 2 or not cur.get("players") or len(act) != 1:
                continue
            ci = act[0]
            if not (isinstance(ci, int) and 0 <= ci < len(opts)):
                continue
            me = cur.get("yourIndex", w)
            mp = (cur.get("players") or [{}])[me]
            chosen = TO.classify_tactic(opts[ci], obs, mp)
            sib = {TO.classify_tactic(o, obs, mp) for o in opts if isinstance(o, dict)}
            ctx = TO.context_features(obs)
            n_dec += 1
            if len(act) == 60:
                pass
            for T in sib:
                tac_all[T][0] += 1
                if T == chosen:
                    tac_all[T][1] += 1
                for ck, cv in ctx.items():
                    k = (ck, cv, T)
                    ctx_tac[k][0] += 1
                    if T == chosen:
                        ctx_tac[k][1] += 1
            # within-turn macro sequence (reset on END / turn change)
            if chosen == "end":
                for a, b in zip(turn_seq, turn_seq[1:]):
                    bigrams[(a, b)] += 1
                turn_seq = []
            else:
                turn_seq.append(chosen)

    # ---- base rates + context lifts ----
    base = {T: (c / a if a else 0.0) for T, (a, c) in tac_all.items()}
    patterns = []
    for (ck, cv, T), (a, c) in ctx_tac.items():
        if a < 30 or T in ("other", "select", "play", "end"):
            continue
        rate = c / a
        br = base.get(T, 0.0)
        if br <= 0:
            continue
        lift = rate / br
        leverage = a * abs(rate - br)              # support x effect size
        patterns.append({
            "tactic": T, "context": f"{ck}={cv}", "support": a, "chosen_in_context": c,
            "choose_rate": round(rate, 3), "base_rate": round(br, 3), "lift": round(lift, 2),
            "leverage": round(leverage, 1), "mechanics_confidence": MECH_CONF.get(T, 0.5),
        })
    patterns.sort(key=lambda p: -p["leverage"])

    tactics_summary = sorted(
        ({"tactic": T, "available": a, "chosen": c, "choose_rate": round(c / a, 3) if a else 0,
          "mechanics_confidence": MECH_CONF.get(T, 0.5)} for T, (a, c) in tac_all.items()),
        key=lambda x: -x["available"])

    macros = [{"sequence": f"{a} -> {b}", "count": n} for (a, b), n in bigrams.most_common(15)]

    art = {
        "miner": "tactic_miner_v1", "snapshot": snapshot, "strong_proxy": "game winners",
        "n_winner_decisions": n_dec, "n_decks_or_players": len(players),
        "top_players": [p for p, _ in players.most_common(8)],
        "tactic_base_rates": tactics_summary,
        "context_patterns_ranked": patterns[:40],
        "macros_within_turn": macros,
    }
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="replays_20260618.json")
    ap.add_argument("--split", default="replays_20260618_split.json")
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()
    art = mine(args.snapshot, args.split, args.max_games)
    out = ROOT / "data" / "manifests" / "mined_tactics_v1.json"
    out.write_text(json.dumps(art, indent=1), encoding="utf-8")
    print(f"mined {art['n_winner_decisions']} winner decisions from {args.snapshot}")
    print("\n=== tactic base rates (chosen / available among strong players) ===")
    for t in art["tactic_base_rates"]:
        if t["available"] >= 50:
            print(f"  {t['tactic']:16} avail={t['available']:6} choose_rate={t['choose_rate']:.3f} "
                  f"mech_conf={t['mechanics_confidence']}")
    print("\n=== TOP context patterns by leverage (strong players choose T more/less in context) ===")
    for p in art["context_patterns_ranked"][:14]:
        print(f"  {p['tactic']:14} | {p['context']:26} | rate {p['choose_rate']:.2f} vs base {p['base_rate']:.2f}"
              f" (lift {p['lift']}) | support {p['support']} | lev {p['leverage']} | conf {p['mechanics_confidence']}")
    print("\n=== top within-turn macros ===")
    for m in art["macros_within_turn"][:8]:
        print(f"  {m['sequence']:34} x{m['count']}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
