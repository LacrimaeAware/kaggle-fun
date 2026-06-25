"""Archetype-merged deck win rates, with per-sublist game counts and small-sample handling.

Scans the replay corpus. Groups exact 60-card lists into an ARCHETYPE = the set of Pokemon it runs (so lists
that differ only in trainers/energy merge; different Pokemon lineups stay separate). For each archetype it
reports: total games (= seats = appearances, NOT pilots), wins, win rate, number of distinct exact lists, and
each sublist's own game count + win rate. A 100%-on-2-games list is therefore obvious. Sorted by win rate,
archetypes below --min-games are dropped from the headline (still counted).

  python tools/deck_winrate_v2.py --min-games 50 --top 20
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
CARDS = Path(r"C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/pokemon-tcg-ai-battle/agent/card_stats.json")
OUT = Path(__file__).resolve().parent.parent / "data" / "deck_winrate_v2.json"


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
    ap.add_argument("--min-games", type=int, default=50)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    card = json.loads(CARDS.read_text(encoding="utf-8"))
    names = {int(k): (v.get("n") or f"#{k}") for k, v in card.items()}
    is_mon = {int(k) for k, v in card.items() if isinstance(v, dict) and v.get("hp")}

    exact_g, exact_w = Counter(), Counter()
    exact_pilots = defaultdict(set)        # exact sig -> set of distinct team names that piloted it
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
        team_names = (d.get("info") or {}).get("TeamNames") or []
        for seat in (0, 1):
            dk = _deck_of(d, seat)
            if dk and len(dk) == 60:
                sig = tuple(sorted(dk))
                exact_g[sig] += 1
                if seat == w:
                    exact_w[sig] += 1
                if seat < len(team_names) and team_names[seat]:
                    exact_pilots[sig].add(str(team_names[seat]))
        if (i + 1) % 1500 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    # archetype = frozenset of Pokemon ids in the list
    arch_g, arch_w = Counter(), Counter()
    arch_subs = defaultdict(list)
    arch_pilots = defaultdict(set)
    for sig, g in exact_g.items():
        mons = frozenset(c for c in set(sig) if c in is_mon)
        arch_g[mons] += g
        arch_w[mons] += exact_w[sig]
        arch_pilots[mons] |= exact_pilots[sig]
        arch_subs[mons].append((g, exact_w[sig], len(exact_pilots[sig]), sig))

    rows = []
    for mons, g in arch_g.items():
        wr = arch_w[mons] / g if g else 0.0
        label = ", ".join(names.get(c, c) for c in sorted(mons, key=lambda c: -sum(Counter(s).get(c, 0) for _, _, _, s in arch_subs[mons]))[:4])
        rows.append((wr, g, arch_w[mons], len(arch_pilots[mons]), label, mons))
    rows = [r for r in rows if r[1] >= args.min_games]
    rows.sort(reverse=True)

    payload = []
    print(f"\narchetype win rates (>= {args.min_games} games), top {args.top}.", flush=True)
    print("  games = seats = appearances (NOT pilots); pilots = distinct teams that ran it.\n", flush=True)
    print(f"  {'winrate':>7} {'games':>6} {'pilots':>6} {'lists':>5}  archetype", flush=True)
    for wr, g, w, npil, label, mons in rows[:args.top]:
        subs = sorted(arch_subs[mons], reverse=True)
        sub_str = "; ".join(f"{sg}g/{spil}p={sw/sg*100:.0f}%" for sg, sw, spil, _ in subs[:4])
        print(f"  {wr*100:6.1f}% {g:6d} {npil:6d} {len(subs):5d}  {label}", flush=True)
        print(f"           top sublists (games/pilots=winrate): {sub_str}", flush=True)
        payload.append({"winrate": round(wr, 4), "games": g, "wins": w, "n_pilots": npil, "n_lists": len(subs),
                        "label": label,
                        "sublists": [{"games": sg, "wins": sw, "pilots": spil, "winrate": round(sw / sg, 4)}
                                     for sg, sw, spil, _ in subs]})
    OUT.write_text(json.dumps({"min_games": args.min_games, "archetypes": payload}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
