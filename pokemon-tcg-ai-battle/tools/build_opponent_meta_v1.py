"""Build the opponent meta deck distribution from the replay corpus, for search determinization.

The forward-model search fills the opponent's hidden cards from OUR deck, which is wrong. This extracts
the actual decks opponents play (replay corpus in the pokemon-ai-agent repo, READ-ONLY), canonicalizes
and counts them, and saves the distribution so search can sample a realistic opponent deck per
determinization world instead of assuming the opponent runs our list.

  python tools/build_opponent_meta_v1.py
Output: data/opponent_meta_v1.json (this repo). Reads pokemon-ai-agent/data/external/replays (read-only).
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
OUT = Path(__file__).resolve().parent.parent / "data" / "opponent_meta_v1.json"


def deck_of(d, seat):
    for step in d.get("steps", [])[:8]:
        if isinstance(step, list) and len(step) > seat and isinstance(step[seat], dict):
            a = step[seat].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def main():
    decks = Counter()
    n_games = 0
    files = sorted(glob.glob(REPLAYS + "/*.json"))
    for i, fn in enumerate(files):
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("steps"):
            continue
        n_games += 1
        for s in (0, 1):
            dk = deck_of(d, s)
            if dk and len(dk) == 60:
                decks[tuple(sorted(dk))] += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(files)} files, {len(decks)} distinct decks so far", flush=True)

    total = sum(decks.values())
    ordered = decks.most_common()
    meta = {
        "n_games": n_games,
        "n_deck_instances": total,
        "n_distinct": len(decks),
        "decks": [{"count": c, "freq": round(c / total, 5), "deck": list(dk)} for dk, c in ordered],
    }
    OUT.write_text(json.dumps(meta), encoding="utf-8")

    print(f"\ngames {n_games}  deck instances {total}  distinct decks {len(decks)}")
    cum = 0.0
    print("top 12 opponent archetypes (count, share, cumulative, a few signature card ids):")
    for c, (dk, cnt) in enumerate(ordered[:12]):
        cum += cnt / total
        cnts = Counter(dk)
        sig = [cid for cid, _ in cnts.most_common(6)]
        print(f"  #{c + 1:2d}  n={cnt:4d}  {cnt / total:5.1%}  cum={cum:5.1%}  cards~{sig}")
    # concentration: how many distinct decks cover 80% of games
    cov = 0.0
    k80 = 0
    for cnt in (c for _, c in ordered):
        cov += cnt / total
        k80 += 1
        if cov >= 0.80:
            break
    print(f"\n{k80} distinct decks cover 80% of opponent appearances (of {len(decks)} total)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
