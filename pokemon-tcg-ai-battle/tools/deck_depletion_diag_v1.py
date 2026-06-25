"""Does the deck-out term ever fire? Before spending a big A/B on it, measure how often MY deck is low
(<= 5) at the actual search LEAVES the deckout hinge reads. If it almost never fires, the term is INERT
(neutral-because-it-never-bites), which is a different verdict from neutral-because-proxied.

Monkeypatches eval.evaluate_deck_v3 to log the leaf deckCount, plays phaware_search_deckout vs phaware_search
single-process so the counters aggregate, and reports the leaf deck distribution + gate-fire rate.
  python tools/deck_depletion_diag_v1.py --games 30
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW_DECK = Path(r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/decks/current_deck.csv")
sys.path.insert(0, str(ROOT / "agent"))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kaggle_environments import make
    import eval as EV
    import deck_policy_v3 as DP3
    import search_v3 as S
    import main as M

S.USE_DYNAMIC_ATTACKS = True
DECK = [int(x) for x in NEW_DECK.read_text(encoding="utf-8").split() if x.strip()]
M.DECK = DECK

STATS = {"calls": 0, "fire": 0, "decks": []}
_orig = EV.evaluate_deck_v3


def _wrapped(cur, me, **kw):
    try:
        players = cur.get("players") or []
        if len(players) >= 2:
            d = float((players[me] or {}).get("deckCount", 0) or 0)
            STATS["calls"] += 1
            STATS["decks"].append(d)
            if kw.get("deckout_weight", 0) and d <= 5:
                STATS["fire"] += 1
    except Exception:
        pass
    return _orig(cur, me, **kw)


EV.evaluate_deck_v3 = _wrapped


def _phaware_search(mode):
    def agent(obs):
        if obs.get("select") is None:
            return list(DECK)
        try:
            ko = DP3.best_ko_attack(obs)
            if ko is not None:
                return [ko[0]]
            mv = S.best_option(obs, DECK, leaf_mode=mode)
            if mv is not None:
                return mv
        except Exception:
            pass
        return M.agent(obs)
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    args = ap.parse_args()

    A = _phaware_search("deckout")     # leaves routed through evaluate_deck_v3 with deckout_weight on
    B = _phaware_search("hand")        # control; its leaves use evaluate_obs (not logged)

    wa = wb = dr = 0
    end_decks = []
    for i in range(args.games):
        a_seat = i % 2
        pair = [A, B] if a_seat == 0 else [B, A]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = make("cabt")
            env.run(pair)
        last = env.steps[-1]
        r0, r1 = last[0].get("reward"), last[1].get("reward")
        # final deck counts (both players) for context
        try:
            cur = last[0]["observation"]["current"]
            for p in cur.get("players") or []:
                end_decks.append(float(p.get("deckCount", 0) or 0))
        except Exception:
            pass
        if r0 is None or r1 is None or r0 == r1:
            dr += 1
        else:
            w = 0 if r0 > r1 else 1
            if w == a_seat:
                wa += 1
            else:
                wb += 1
        print(f"  game {i+1}/{args.games}  leaf_evals={STATS['calls']} fires={STATS['fire']}", flush=True)

    decks = STATS["decks"]
    n = len(decks) or 1
    import statistics as st
    buckets = {"<=2": 0, "3-5": 0, "6-8": 0, "9-12": 0, "13+": 0}
    for d in decks:
        if d <= 2:
            buckets["<=2"] += 1
        elif d <= 5:
            buckets["3-5"] += 1
        elif d <= 8:
            buckets["6-8"] += 1
        elif d <= 12:
            buckets["9-12"] += 1
        else:
            buckets["13+"] += 1
    print("\n=== deck-out gate diagnostic ===")
    print(f"games={args.games}  deckout_agent winrate (n small, directional only) = {wa/(wa+wb):.3f} ({wa}-{wb}, {dr}d)")
    print(f"leaf evals on deckout agent = {STATS['calls']}")
    print(f"gate fires (leaf deck<=5)   = {STATS['fire']}  =>  fire rate = {STATS['fire']/n:.4f}")
    print(f"leaf deckCount: mean={st.mean(decks):.1f} median={st.median(decks):.1f} min={min(decks):.0f} max={max(decks):.0f}")
    print("leaf deckCount distribution:")
    for k, v in buckets.items():
        print(f"  {k:>5s}: {v:6d}  ({100*v/n:5.1f}%)")
    if end_decks:
        print(f"game-end deckCount (both players): mean={st.mean(end_decks):.1f} min={min(end_decks):.0f} "
              f"frac<=5={sum(1 for d in end_decks if d<=5)/len(end_decks):.3f}")
    verdict = ("INERT (gate almost never fires; deckout cannot move win rate -> skip its A/B)"
               if STATS["fire"] / n < 0.02 else
               "LIVE (gate fires enough to matter -> run the A/B)")
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
