"""Which decks actually WIN in the replay corpus (not just which are common).

Scans every replay, and for each game credits a win/loss to each seat's 60-card deck signature. Reports the
best-performing decks by win rate (with a minimum game count so small samples don't top the list), labeled by
their signature Pokemon. This is the human pilots' win rate with each deck -- the right pool to draw "strong
archetypes" from for the fair arena, instead of the most-frequent decks.

  python tools/deck_winrate_v1.py --min-games 40 --top 25
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
CARDS = Path(r"C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/pokemon-tcg-ai-battle/agent/card_stats.json")
OUT = Path(__file__).resolve().parent.parent / "data" / "deck_winrate_v1.json"


def _deck_of(d, seat):
    for step in (d.get("steps") or [])[:8]:
        if isinstance(step, list) and len(step) > seat and isinstance(step[seat], dict):
            a = step[seat].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def _winner(d):
    r = d.get("rewards") or []
    if len(r) < 2 or r[0] is None or r[1] is None or r[0] == r[1]:
        return None
    return 0 if r[0] > r[1] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-games", type=int, default=40)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    card = json.loads(CARDS.read_text(encoding="utf-8"))
    names = {int(k): (v.get("n") or f"#{k}") for k, v in card.items()}
    # Pokemon = cards with hp; used to label a deck by its signature mons.
    is_mon = {int(k) for k, v in card.items() if isinstance(v, dict) and v.get("hp")}

    games = Counter()
    wins = Counter()
    files = sorted(glob.glob(REPLAYS + "/*.json"))
    print(f"scanning {len(files)} replays...", flush=True)
    for i, fn in enumerate(files):
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        w = _winner(d)
        if w is None:
            continue
        for seat in (0, 1):
            dk = _deck_of(d, seat)
            if not dk or len(dk) != 60:
                continue
            sig = tuple(sorted(dk))
            games[sig] += 1
            if seat == w:
                wins[sig] += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)} files, {len(games)} distinct decks", flush=True)

    rows = []
    for sig, n in games.items():
        if n < args.min_games:
            continue
        wr = wins[sig] / n
        mons = [cid for cid in Counter(sig) if cid in is_mon]
        label = ", ".join(f"{names.get(c, c)}" for c in sorted(set(mons), key=lambda c: -Counter(sig)[c])[:4])
        rows.append((wr, n, label, list(sig)))
    rows.sort(reverse=True)

    OUT.write_text(json.dumps({"min_games": args.min_games,
                               "decks": [{"winrate": round(wr, 4), "games": n, "label": lab, "deck": dk}
                                         for wr, n, lab, dk in rows]}, indent=2), encoding="utf-8")
    print(f"\nbest-performing decks (min {args.min_games} games), top {args.top}:", flush=True)
    print(f"  {'winrate':>7} {'games':>6}  archetype (signature mons)", flush=True)
    for wr, n, lab, _ in rows[:args.top]:
        print(f"  {wr*100:6.1f}% {n:6d}  {lab}", flush=True)
    print(f"\nwrote {OUT}  ({len(rows)} decks with >= {args.min_games} games)", flush=True)


if __name__ == "__main__":
    main()
