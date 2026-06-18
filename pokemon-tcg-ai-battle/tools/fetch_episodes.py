"""Download Kaggle cabt episode replays by episode id.

The full replay JSON is on Kaggle's public CDN, no auth needed (verified 2026-06-17):
  https://www.kaggleusercontent.com/episodes/<EpisodeId>.json
It contains configuration (incl. the game seed), every step with BOTH agents' observations and
actions, statuses, and rewards -- i.e. the complete game, usable for opponent modelling, building
real-pool stats, or evaluating against actual ladder opponents (not just self-play).

Saves to data/external/replays/<id>.json (gitignored). Be polite: a sleep between requests, and
respect Kaggle's rate limits / terms. Episode ids come from the leaderboard URL
(...leaderboard?submissionId=<S>&episodeId=<E>); bulk id discovery (ListEpisodes) needs auth and
is not done here -- pass ids explicitly or via --file.

    python tools/fetch_episodes.py 80411394 80408508
    python tools/fetch_episodes.py --file episode_ids.txt
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external" / "replays"
CDN = "https://www.kaggleusercontent.com/episodes/{}.json"


def fetch(eid: int, out_dir: Path, overwrite: bool = False) -> str:
    path = out_dir / f"{eid}.json"
    if path.exists() and not overwrite:
        return "skip (exists)"
    req = urllib.request.Request(CDN.format(eid), headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    obj = json.loads(data)                       # validate it parses
    n_steps = len(obj.get("steps", []))
    seed = obj.get("configuration", {}).get("seed")
    path.write_bytes(data)
    return f"{len(data) // 1024} KB, {n_steps} steps, seed {seed}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", type=int, help="episode ids to download")
    ap.add_argument("--file", help="text file of episode ids, one per line")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests (rate-limit politeness)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    ids = list(args.ids)
    if args.file:
        ids += [int(x) for x in Path(args.file).read_text().split() if x.strip()]
    if not ids:
        ap.error("give episode ids or --file")
    OUT.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for i, eid in enumerate(ids):
        try:
            print(f"  {eid}: {fetch(eid, OUT, args.overwrite)}", flush=True)
            ok += 1
        except Exception as e:
            print(f"  {eid}: FAILED {e!r}", flush=True)
            fail += 1
        if args.sleep and i < len(ids) - 1:
            time.sleep(args.sleep)
    print(f"done: {ok} fetched, {fail} failed -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
