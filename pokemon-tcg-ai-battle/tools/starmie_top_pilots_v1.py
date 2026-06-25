"""Rank the top-performing pilots of the Cinderace / Mega-Starmie archetype in the replay corpus, and
index their WINNING games -- the imitation targets for the decision-divergence analysis.

A seat counts as the archetype if its 60-card deck contains Mega Starmie ex (1031). For each such seat
we credit a win/loss to its TeamName. Pilots are ranked by win rate (with a min-games floor). We then
list each top pilot's winning episode ids + the seat they played, so the gap analyzer can feed those
games' observations to our agent.

  python tools/starmie_top_pilots_v1.py --min-games 15 --top-pilots 12
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
OUT = Path(__file__).resolve().parent.parent / "data" / "starmie_top_pilots.json"

MEGA_STARMIE = 1031
CINDERACE = 666


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
    ap.add_argument("--min-games", type=int, default=15)
    ap.add_argument("--top-pilots", type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob.glob(REPLAYS + "/*.json"))
    print(f"scanning {len(files)} replays for the Mega-Starmie archetype...", flush=True)

    pilot_g = defaultdict(int)
    pilot_w = defaultdict(int)
    # (pilot) -> list of (episode_id, seat) it WON with the archetype
    pilot_wins = defaultdict(list)

    n_arche = 0
    for i, fn in enumerate(files):
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        w = _winner(d)
        if w is None:
            continue
        team_names = (d.get("info") or {}).get("TeamNames") or []
        epid = (d.get("info") or {}).get("EpisodeId") or Path(fn).stem
        for seat in (0, 1):
            dk = _deck_of(d, seat)
            if not dk or MEGA_STARMIE not in dk:
                continue
            n_arche += 1
            name = str(team_names[seat]) if seat < len(team_names) and team_names[seat] else f"seat{seat}"
            pilot_g[name] += 1
            if seat == w:
                pilot_w[name] += 1
                pilot_wins[name].append([epid, seat])
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)} scanned, {n_arche} archetype seats so far", flush=True)

    rows = []
    for name, g in pilot_g.items():
        if g < args.min_games:
            continue
        wr = pilot_w[name] / g
        rows.append((wr, g, pilot_w[name], name))
    rows.sort(reverse=True)

    print(f"\ntop Mega-Starmie pilots (>= {args.min_games} games):", flush=True)
    print(f"  {'winrate':>7} {'games':>6} {'wins':>5}  pilot", flush=True)
    top = rows[: args.top_pilots]
    for wr, g, w, name in top:
        print(f"  {wr*100:6.1f}% {g:6d} {w:5d}  {name}", flush=True)

    payload = {
        "min_games": args.min_games,
        "n_archetype_seats": n_arche,
        "pilots": [{"name": name, "games": g, "wins": w, "winrate": round(wr, 4),
                    "winning_episodes": pilot_wins[name]} for wr, g, w, name in top],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    total_win_games = sum(len(pilot_wins[name]) for _, _, _, name in top)
    print(f"\nwrote {OUT}  ({len(top)} top pilots, {total_win_games} winning games indexed)", flush=True)


if __name__ == "__main__":
    main()
