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


def load_meta_deck(exclude_deck=None, min_games=5):
    """Most-played corpus deck. exclude_deck (a card-id list) skips decks multiset-equal to it, so the
    belief test uses a DIFFERENT archetype than ours (else opp_prior==our deck and the test is a no-op)."""
    import json
    from collections import Counter
    decks = json.load(open(Path(__file__).resolve().parent.parent / "data" / "replay_db" / "decks.json", encoding="utf-8"))
    ex = Counter(exclude_deck) if exclude_deck is not None else None
    cand = [d for d in decks if d.get("n_games", 0) >= min_games and (ex is None or Counter(d["deck"]) != ex)]
    if not cand:
        cand = decks
    return max(cand, key=lambda d: d.get("n_games", 0))["deck"]


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


# Praxel (leaderboard #1 as of 2026-06-18): Mega Lucario ex Fighting deck (10 basics, 13 energy).
PRAXEL_DECK = ([6] * 13 + [678] * 4 + [1102] * 4 + [1141] * 4 + [1142] * 4 + [1152] * 4 + [1192] * 4 +
               [1227] * 4 + [676] * 3 + [677] * 3 + [673] * 2 + [674] * 2 + [675] * 2 + [1123] * 2 +
               [1182] * 2 + [1252] * 2 + [1159])


def mk_deck(deck):
    """Our search piloting an arbitrary deck (deck-select returns `deck`; decisions use best_option)."""
    def ag(obs):
        if obs.get("select") is None:
            return list(deck)
        try:
            mv = M._forced_move(obs)
            if mv is not None:
                return mv
            mv = S.best_option(obs, deck)
            if mv is not None:
                return mv
        except Exception:
            pass
        return M.agent(obs)
    return ag


def mk_cont(policy, determ=None, budget=None):
    """agent_search with MY_CONT = policy. determ/budget (when set) FIX the determinization count and use
    a generous time cap so both arms complete the SAME number of worlds -- isolating continuation QUALITY
    from setup's slower (fewer-worlds-under-the-cap) rollout. None = the real-budget agent_search."""
    def ag(obs):
        if obs.get("select") is None:
            return list(M.DECK)
        S.MY_CONT = policy
        if determ is not None:
            S.N_DETERM = determ
        if budget is not None:
            try:
                mv = M._forced_move(obs)
                if mv is not None:
                    return mv
                mv = S.best_option(obs, M.DECK, time_budget=budget)
                if mv is not None:
                    return mv
            except Exception:
                pass
            return M.agent(obs)
        return M.agent_search(obs)
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
    ap.add_argument("experiment", choices=["determ", "belief", "continuation", "cont-clean", "deck"])
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
        meta = load_meta_deck(exclude_deck=M.DECK)   # a DIFFERENT archetype, else opp_prior==our deck (no-op)
        opp = mk_meta_opp(meta)
        print(f"belief-determinization A/B: our search (M.DECK) vs a META-deck opponent (first_agent on the\n"
              f"top replay deck). best-case belief (opp hidden zones filled from the META deck) vs same-deck\n"
              f"(filled from our deck). {args.games} games each.\n", flush=True)
        ab("belief: our-search(opp_prior=meta) vs meta-opp", mk_our(meta), opp, args.games, args.progress)
        ab("same-deck: our-search(opp_prior=None) vs meta-opp", mk_our(None), opp, args.games, args.progress)
        print("\nRead: if belief's win-rate vs the meta opponent clears same-deck's by more than noise,\n"
              "knowing the opponent's deck helps the search -> build a meta/archetype belief prior.", flush=True)
    elif args.experiment == "continuation":
        print(f"continuation-policy A/B (the my-turn rollout finish), head-to-head, {args.games} games.\n"
              f"setup = develop the board (play/attach/evolve/ability) before attacking; aggro = attack on\n"
              f"the first legal attack (current). Opponent rollout stays aggressive in both.\n", flush=True)
        ab("setup vs aggro (A=setup)", mk_cont("setup"), mk_cont("aggro"), args.games, args.progress)
        print("\nRead: if setup's Wilson lower bound > 0.5 it finishes my turn more realistically -> set\n"
              "agent/search.py MY_CONT='setup'; if it loses, the aggressive finish was already fine.", flush=True)
    elif args.experiment == "cont-clean":
        print(f"continuation QUALITY A/B, determinization count FIXED at 4 (generous 10s cap so both arms\n"
              f"complete all 4 worlds) -- isolates continuation quality from setup's slower rollout.\n"
              f"setup vs aggro, head-to-head, {args.games} games.\n", flush=True)
        ab("setup vs aggro @ fixed N=4 (A=setup)", mk_cont("setup", 4, 10.0), mk_cont("aggro", 4, 10.0),
           args.games, args.progress)
        print("\nRead: this removes the determinization-budget confound. If setup still leads, the develop-\n"
              "first continuation is genuinely better and worth adopting (then address its match-time cost).", flush=True)
    elif args.experiment == "deck":
        print(f"DECK A/B: our SAME search piloting Praxel's Mega-Lucario-ex deck (leaderboard #1) vs our\n"
              f"DENPA92 deck, head-to-head, {args.games} games. Tests whether adopting the #1 deck (under\n"
              f"OUR search) is better -- deck swaps have moved our LB score the most.\n", flush=True)
        ab("Praxel-deck vs DENPA92 (A=Praxel)", mk_deck(PRAXEL_DECK), mk_deck(M.DECK), args.games, args.progress)
        print("\nRead: if Praxel-deck's Wilson lower bound > 0.5, our search plays it better -> swap DECK and\n"
              "submit. (Deck value is policy-coupled; this is our-search-on-both, the relevant comparison.)", flush=True)


if __name__ == "__main__":
    main()
