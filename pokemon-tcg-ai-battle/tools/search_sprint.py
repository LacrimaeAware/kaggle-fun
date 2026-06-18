"""Bounded search-strengthening sprint (the plan's performance lane). Controlled A/Bs of agent_search
knobs, head-to-head, same deck, seat-swapped, small n for a directional read. Freeze the winner as the
teacher; do not open-endedly tune.

Experiment selector:
  determ : N_DETERM budget (8 vs 4, 16 vs 4) -- does averaging more hidden worlds help move choice?

    python tools/search_sprint.py determ --games 40
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
import cabt_arena as A  # noqa: E402
import main as M  # noqa: E402
import search as S  # noqa: E402


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def mk_ndet(n):
    """agent_search with N_DETERM = n (set per call; single-threaded, so safe in the arena loop)."""
    def ag(obs):
        if obs.get("select") is None:
            return list(M.DECK)
        S.N_DETERM = n
        return M.agent_search(obs)
    return ag


def load_meta_deck():
    import json
    decks = json.load(open(Path(__file__).resolve().parent.parent / "data" / "replay_db" / "decks.json", encoding="utf-8"))
    return max(decks, key=lambda d: d.get("n_games", 0))["deck"]


def mk_our(opp_prior):
    """Our search on M.DECK; opp_prior fills the opponent's hidden zones (belief) or None (same-deck)."""
    def ag(obs):
        if obs.get("select") is None:
            return list(M.DECK)
        try:
            mv = M._forced_move(obs)
            if mv is not None:
                return mv
            mv = S.best_option(obs, M.DECK, opp_prior=opp_prior)
            if mv is not None:
                return mv
        except Exception:
            pass
        return M.agent(obs)
    return ag


def mk_meta_opp(meta):
    """A meta-deck opponent: pilots `meta` with the engine's first_agent."""
    def ag(obs):
        return list(meta) if obs.get("select") is None else A.cabt.first_agent(obs)
    return ag


def ab(label, a, b, games, progress):
    r = A.run(games, a, b, label=label, progress=progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = wilson(r["wins_a"], dec)
    verdict = "A better" if lo > 0.5 else ("B better" if hi < 0.5 else "tie (CI spans 0.5)")
    print(f"=> {label}: A {r['a_win_rate_decided']:.3f}  Wilson95 [{lo:.3f},{hi:.3f}]  {verdict}"
          f"  ({r['wins_a']}-{r['wins_b']}, {r['s_per_game']}s/g)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["determ", "belief"])
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--progress", type=int, default=10)
    args = ap.parse_args()
    if args.experiment == "determ":
        print(f"determinization-budget A/B, {args.games} games each (default N_DETERM=4)\n", flush=True)
        ab("N=8 vs N=4", mk_ndet(8), mk_ndet(4), args.games, args.progress)
        ab("N=16 vs N=4", mk_ndet(16), mk_ndet(4), args.games, args.progress)
        print("\nRead: if more determinizations clear 0.50 (Wilson lower bound > 0.5), bump N_DETERM; "
              "else keep 4 (cheaper) and move to the next knob (rollout / belief / continuation).", flush=True)
    elif args.experiment == "belief":
        meta = load_meta_deck()
        opp = mk_meta_opp(meta)
        print(f"belief-determinization A/B: our search (M.DECK) vs a META-deck opponent (first_agent on the\n"
              f"top replay deck). best-case belief (opp hidden zones filled from the META deck) vs same-deck\n"
              f"(filled from our deck). {args.games} games each.\n", flush=True)
        ab("belief: our-search(opp_prior=meta) vs meta-opp", mk_our(meta), opp, args.games, args.progress)
        ab("same-deck: our-search(opp_prior=None) vs meta-opp", mk_our(None), opp, args.games, args.progress)
        print("\nRead: if belief's win-rate vs the meta opponent clears same-deck's by more than noise,\n"
              "knowing the opponent's deck helps the search -> build a meta/archetype belief prior.", flush=True)


if __name__ == "__main__":
    main()
