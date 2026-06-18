"""Parse downloaded cabt replays into a stats store the viewer reads (registry/replays.json).

For each full replay (a cabt episode JSON with "steps"), extract: id, date (file mtime not
trusted; left blank for the user to fill or for a future submission-id->date map), both
players' decks (card ids), the winner, game length, whether it is a self-play game, and a
per-player count of cards actually PLAYED (LogType.PLAY=10 events in obs.logs). The viewer
aggregates these into card play-rate and per-deck win-rate, filterable.

Right now we have only a couple of self-play replays, so the aggregates are thin. The point
is the pipeline: as real ranked-match replays are downloaded into data/external/replays/,
re-run this and the stats fill in.

Run: python tools/build_stats.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAY_DIR = ROOT / "data" / "external" / "replays"
OUT = ROOT / "registry" / "replays.json"
PLAY_LOG = 10  # LogType.PLAY


def parse_replay(path: Path) -> dict | None:
    rep = json.loads(path.read_text(encoding="utf-8"))
    steps = rep.get("steps")
    if not steps:
        return None
    decks = {0: None, 1: None}
    played = {0: Counter(), 1: Counter()}
    for s in steps:
        for ai, agent in enumerate(s):
            if not isinstance(agent, dict):
                continue
            act = agent.get("action")
            if isinstance(act, list) and len(act) == 60 and decks[ai] is None:
                decks[ai] = act
            obs = agent.get("observation") or {}
            for log in (obs.get("logs") or []):
                if log.get("type") == PLAY_LOG and log.get("cardId"):
                    played[ai][log["cardId"]] += 1
    rewards = rep.get("rewards") or []
    winner = None
    if len(rewards) == 2 and rewards[0] != rewards[1]:
        winner = 0 if rewards[0] > rewards[1] else 1
    self_play = decks[0] and decks[1] and sorted(decks[0]) == sorted(decks[1])
    return {
        "id": rep.get("id") or path.stem,
        "file": path.name,
        "date": "",
        "steps": len(steps),
        "winner": winner,
        "self_play": bool(self_play),
        "deck0": decks[0] or [],
        "deck1": decks[1] or [],
        "played0": dict(played[0]),
        "played1": dict(played[1]),
    }


def main() -> None:
    replays = []
    for p in sorted(REPLAY_DIR.glob("*.json")):
        try:
            r = parse_replay(p)
        except Exception:
            r = None
        if r:
            replays.append(r)
    OUT.write_text(json.dumps(replays, indent=2), encoding="utf-8")
    real = sum(1 for r in replays if not r["self_play"])
    print(f"wrote {len(replays)} replays ({real} real, {len(replays)-real} self-play) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
