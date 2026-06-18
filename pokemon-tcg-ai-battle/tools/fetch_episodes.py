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

    python tools/fetch_episodes.py 80411394 80408508            # by id (CDN, no auth)
    python tools/fetch_episodes.py --file episode_ids.txt
    python tools/fetch_episodes.py --submission 53794404        # our episodes (needs ~/.kaggle/kaggle.json)
    python tools/fetch_episodes.py --team <leaderboard-team-id> # an opponent team's episodes (needs auth)
    python tools/fetch_episodes.py --top-teams 40 --max 500     # AUTO: top 40 leaderboard teams, no manual linking

Auth for discovery: kaggle.com -> Settings -> API -> Create New Token -> save kaggle.json to
~/.kaggle/kaggle.json. Download-by-id needs no auth. Kaggle will also post a daily top-episode
export (for BC/RL/IL) in the competition forums.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external" / "replays"
CDN = "https://www.kaggleusercontent.com/episodes/{}.json"
COMP = "pokemon-tcg-ai-battle"


def _load_env() -> None:
    """Load Kaggle creds from the gitignored .env (KAGGLE_KEY / KAGGLE_USERNAME, or a KGAT token)
    so the CLI subprocess can auth without the user sourcing anything. Never printed or committed."""
    envf = ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
    tok = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_TOKEN")
    if tok and not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = tok


def _kaggle_ids(subargs: list, col: str = "id") -> list:
    """Run a kaggle CLI command with -v (CSV) and return the `col` column as ints. Discovery needs
    ~/.kaggle/kaggle.json (the API token from kaggle.com/settings -> Create New Token). Download
    (the CDN) does NOT need auth -- only this id-discovery does."""
    cmd = [sys.executable, "-m", "kaggle", "competitions", *subargs, "-v"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"kaggle CLI failed ({' '.join(subargs)}): {(out.stderr or out.stdout).strip()[:200]}")
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and not ln.startswith("Warning:")]
    start = next((i for i, ln in enumerate(lines) if ln.lower().split(",")[0] == col), 0)
    return [int(r[col]) for r in csv.DictReader(lines[start:]) if str(r.get(col, "")).isdigit()]


def episodes_for_submission(subid: int) -> list:
    return _kaggle_ids(["episodes", str(subid)])


def submissions_for_team(teamid: int) -> list:
    return _kaggle_ids(["team-submissions", str(teamid)])


def leaderboard_team_ids(n: int) -> list:
    """Top-n team ids straight off the public leaderboard (no manual linking) via the Kaggle Python
    API. The CLI only shows a few rows; this paginates up to 500 in one call. Needs auth (.env)."""
    import kaggle
    api = kaggle.KaggleApi()
    api.authenticate()
    lb = api.competition_leaderboard_view(COMP, page_size=min(max(n, 1), 500))
    return [e.team_id for e in lb[:n] if getattr(e, "team_id", None)]


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
    ap.add_argument("ids", nargs="*", type=int, help="episode ids to download (CDN, no auth)")
    ap.add_argument("--file", help="text file of episode ids, one per line")
    ap.add_argument("--submission", nargs="*", type=int, default=[],
                    help="submission id(s): discover their episodes via the kaggle CLI (needs kaggle.json)")
    ap.add_argument("--team", nargs="*", type=int, default=[],
                    help="team id(s) from the leaderboard: discover their submissions' episodes (needs kaggle.json)")
    ap.add_argument("--top-teams", type=int, default=0, metavar="N",
                    help="auto-discover the top N teams off the public leaderboard and pull their episodes (no manual linking)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests (rate-limit politeness)")
    ap.add_argument("--max", type=int, default=0, help="cap total episodes downloaded (0 = no cap)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.submission or args.team or args.top_teams:
        _load_env()                              # auth only needed for discovery
    if args.top_teams:
        discovered = leaderboard_team_ids(args.top_teams)
        print(f"  leaderboard top {args.top_teams}: {len(discovered)} team ids")
        args.team = list(dict.fromkeys(discovered + list(args.team)))
    ids = list(args.ids)
    if args.file:
        ids += [int(x) for x in Path(args.file).read_text().split() if x.strip()]
    for sub in args.submission:
        eps = episodes_for_submission(sub)
        print(f"  submission {sub}: {len(eps)} episodes")
        ids += eps
    for team in args.team:
        for sub in submissions_for_team(team):
            eps = episodes_for_submission(sub)
            print(f"  team {team} submission {sub}: {len(eps)} episodes")
            ids += eps
    ids = list(dict.fromkeys(ids))               # dedup, preserve order
    if args.max and len(ids) > args.max:
        print(f"  capping {len(ids)} -> {args.max} episodes (--max)")
        ids = ids[:args.max]
    if not ids:
        ap.error("give episode ids, --file, --submission, or --team")
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
