"""Heuristic Search V2 harness. Runs an isolated v2 agent (fail-closed proposals + search validator)
against the PRODUCTION baseline, same pilot deck, with full instrumentation proving search stayed
authoritative. Candidates are toggled, never combined silently.

  python tools/ab_heuristic_search_v2.py --candidate a0   --games 16   # control, expect ~0.5
  python tools/ab_heuristic_search_v2.py --candidate m0   --games 10   # mechanical fixes only
  python tools/ab_heuristic_search_v2.py --candidate m0_poffin --games 10

Candidates: a0 (all off = control), s32 (N=32 sampling), m0 (mechanical fixes), m0_poffin, s32_m0, ...
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_candidate_v1 as AB
AB.CAND_SRC = ROOT / "agent"
AB.MODS = ["main", "search", "eval", "features", "deck_policy_v2"]

OPT = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE",
       10: "ABILITY", 11: "DISCARD", 12: "RETREAT", 13: "ATTACK", 14: "END"}

CANDIDATES = {
    "a0":        {},
    "s32":       {"s32": True},
    "m0":        {"m0": True},
    "m0_poffin": {"m0": True, "resolvers": {"poffin"}},
    "s32_m0":    {"s32": True, "m0": True},
    "s32_m0_poffin": {"s32": True, "m0": True, "resolvers": {"poffin"}},
    "phfix":        {"forced": "phaware"},
    "phfix_poffin": {"forced": "phaware", "resolvers": {"poffin"}},
}


def build_v2_agent(v2main, S, DP2, toggles, C):
    s32 = bool(toggles.get("s32"))
    n_determ = 32 if s32 else 8
    budget = 15.0 if s32 else 0.6
    S.N_DETERM = n_determ
    resolvers = toggles.get("resolvers")
    forced_mode = toggles.get("forced") or ("m0" if toggles.get("m0") else "baseline")

    def _forced(obs):
        if forced_mode == "m0":
            return DP2.forced_move_m0(obs)
        if forced_mode == "phaware":
            return DP2.forced_move_ko_phaware(obs)
        return v2main._forced_move(obs)

    def agent(obs):
        if obs.get("select") is None:
            return list(v2main.DECK)
        C["decisions"] += 1
        sel = obs.get("select")
        single = (sel.get("maxCount") or 0) == 1 and len(sel.get("option") or []) >= 2

        def record(ret):
            opts = sel.get("option") or []
            if single and isinstance(ret, list) and ret and 0 <= ret[0] < len(opts):
                t = opts[ret[0]].get("type")
                C["choice_" + OPT.get(t, str(t))] += 1
            return ret

        # 1. forced floor (baseline / m0=final-prize-only / phaware=auto-KO with PH damage)
        try:
            mv = _forced(obs)
        except Exception:
            C["swallowed_exc"] += 1
            mv = None
        if mv is not None:
            C["forced"] += 1
            return record(mv)

        # 2. heuristic PROPOSE + search VALIDATE (fail-closed)
        if resolvers:
            try:
                cands = DP2.propose(obs, enabled=resolvers)
                if cands:
                    C["proposal_calls"] += 1
                    default = DP2.default_selection(obs)
                    chosen, st = DP2.compare_selections(obs, v2main.DECK, cands, default, S,
                                                        time_budget=budget, n_determ=n_determ)
                    C["accepted"] += int(st["accepted"])
                    C["tiebreak"] += int(st["tiebreak"])
                    C["incomplete"] += int(st["incomplete"])
                    C["val_error"] += int(st["error"])
                    if chosen is not None and list(chosen) != list(default) and (st["accepted"] or st["tiebreak"]):
                        C["heuristic_used"] += 1
                        return record(chosen)
                    C["heuristic_rejected"] += 1
            except Exception:
                C["swallowed_exc"] += 1

        # 3. search (authoritative)
        try:
            if single:
                C["search_applicable"] += 1
            mv = S.best_option(obs, v2main.DECK, leaf_mode="hand", time_budget=budget)
            if mv is not None:
                C["search_used"] += 1
                return record(mv)
        except Exception:
            C["swallowed_exc"] += 1

        # 4. fallback (baseline heuristic)
        C["fallback"] += 1
        return record(v2main.agent(obs))

    return agent, {"n_determ": n_determ, "budget": budget, "resolvers": sorted(resolvers) if resolvers else [], "forced": forced_mode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--progress", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "docs" / "workstreams" / "heuristic_search_v2_results.json"))
    args = ap.parse_args()

    AB.build_candidate_pkg()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "agent"))
    base_prod = importlib.import_module("main")       # production baseline opponent (N=8, 0.6s)
    v2main = importlib.import_module("_candv1.main")
    S = importlib.import_module("_candv1.search")
    DP2 = importlib.import_module("_candv1.deck_policy_v2")
    PILOT = AB.pilot_deck()
    base_prod.DECK = PILOT
    v2main.DECK = PILOT

    C = Counter()
    toggles = CANDIDATES[args.candidate]
    v2agent, cfg = build_v2_agent(v2main, S, DP2, toggles, C)

    print(f"V2 candidate '{args.candidate}' {cfg} vs production baseline, pilot deck, {args.games} games")
    r = AB.run(args.games, v2agent, base_prod.agent_search, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = AB.wilson(r["wins_a"], dec)
    wr = r["wins_a"] / dec if dec else 0.0

    inst = dict(C)
    print(f"\n=> {args.candidate}: {wr:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  "
          f"({r['wins_a']}-{r['wins_b']}, {r['draws']}d, {r['errors']}e, {r['seconds']}s)")
    print("--- instrumentation (search-authoritative proof) ---")
    for k in sorted(inst):
        print(f"   {k}: {inst[k]}")

    res = {"candidate": args.candidate, "config": cfg, "games": args.games,
           "win_rate": round(wr, 3), "wilson95": [round(lo, 3), round(hi, 3)],
           "wins": r["wins_a"], "losses": r["wins_b"], "draws": r["draws"], "errors": r["errors"],
           "s_per_game": r.get("seconds", 0) / max(1, args.games), "instrumentation": inst}
    out = Path(args.out)
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing[args.candidate] = res
    out.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
