"""Daily top-team replay refresh -- grow the ROLLING RAW CORPUS with the newest games of the highest-
ranked teams, then (optionally) write a NEW named snapshot. Never mutates SPLIT_BASE_V2.

Design (per the corpus rule): raw replays keep growing; frozen snapshots are explicit and named. This
script ADDS new replays to data/external/replays/ (gitignored) and writes a new dated manifest/split.

Filter correctness (the important bit): dedup is per-EPISODE-ID, not per-team. Every run re-discovers
each top team and pulls their NEWEST episodes first (episode ids are monotonic), skipping only the
specific games already on disk -- so re-scraping a team always catches its fresh matches and never
re-downloads old ones. Priority is rank-major (highest-ranked team first), recency-within.

    python tools/refresh_top20.py --top 20 --cap-gb 2 --snapshot              # one-off refresh + snapshot
    python tools/refresh_top20.py --top 20 --cap-gb 2 --snapshot --commit     # + commit manifest/split (no push)
    python tools/refresh_top20.py --top 5 --subs-per-team 1 --max-discover 8 --cap-gb 0.05  # quick test

Discovery needs Kaggle auth (loaded BOM-safe from .env); download-by-id is CDN/no-auth. Replays stay
gitignored; only the small named manifest/split are committable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fetch_episodes as FE  # noqa: E402  (reuse discovery + CDN fetch helpers)


def load_env_robust() -> None:
    """Load .env BOM-safe (utf-8-sig) so KAGGLE_* creds set even if the file has a BOM. Mirrors a
    token into KAGGLE_KEY if needed. Values are never printed."""
    envf = ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)
    if not os.environ.get("KAGGLE_KEY"):
        tok = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_TOKEN")
        if tok:
            os.environ["KAGGLE_KEY"] = tok


def discover_ordered(top: int, subs_per_team: int, sleep: float) -> list:
    """Rank-major, recency-within episode ids: highest-ranked team's newest submissions' newest
    episodes first. No team-level skip -- re-discovered every run so NEW matches are always found."""
    teams = FE.leaderboard_team_ids(top)
    print(f"  top-{top} team ids: {teams}", flush=True)
    ordered, seen = [], set()
    for rank, team in enumerate(teams):
        try:
            subs = sorted(FE.submissions_for_team(team), reverse=True)[:subs_per_team]  # newest bots first
        except Exception as e:
            print(f"  rank {rank+1} team {team}: submissions FAILED {e!r}", flush=True)
            continue
        for sub in subs:
            try:
                eps = sorted(FE.episodes_for_submission(sub), reverse=True)  # newest games first
            except Exception as e:
                print(f"    sub {sub}: episodes FAILED {e!r}", flush=True)
                continue
            added = sum(1 for e in eps if e not in seen)
            for e in eps:
                if e not in seen:
                    seen.add(e)
                    ordered.append(e)
            print(f"  rank {rank+1} team {team} sub {sub}: {len(eps)} eps (+{added} to queue)", flush=True)
            if sleep:
                time.sleep(min(sleep, 0.5))
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--subs-per-team", type=int, default=2, help="newest N submissions per team (recency)")
    ap.add_argument("--cap-gb", type=float, default=2.0, help="approx GB of NEW replays to download this run")
    ap.add_argument("--sleep", type=float, default=0.8, help="seconds between downloads (politeness)")
    ap.add_argument("--max-discover", type=int, default=0, help="cap the discovery queue length (0=all; for testing)")
    ap.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    ap.add_argument("--snapshot", action="store_true", help="write a new named snapshot after fetching")
    ap.add_argument("--commit", action="store_true", help="git add+commit the new manifest/split (no push)")
    args = ap.parse_args()

    load_env_robust()
    OUT = FE.OUT
    OUT.mkdir(parents=True, exist_ok=True)
    cap_bytes = int(args.cap_gb * 1e9)

    print(f"[refresh_top20] discovering top-{args.top} (newest {args.subs_per_team} subs/team)...", flush=True)
    ordered = discover_ordered(args.top, args.subs_per_team, args.sleep)
    if args.max_discover:
        ordered = ordered[:args.max_discover]
    print(f"[refresh_top20] {len(ordered)} candidate episodes (rank-major, newest-first); cap {args.cap_gb} GB", flush=True)

    new = skipped = fail = 0
    got = 0
    t0 = time.time()
    for eid in ordered:
        path = OUT / f"{eid}.json"
        if path.exists():
            skipped += 1
            continue
        if got >= cap_bytes:
            print(f"[refresh_top20] reached cap (~{got/1e9:.2f} GB); stopping", flush=True)
            break
        try:
            FE.fetch(eid, OUT)
            got += path.stat().st_size
            new += 1
        except Exception as e:
            fail += 1
            print(f"  {eid}: FAILED {e!r}", flush=True)
            continue
        if new % 25 == 0:
            print(f"  +{new} new ({got/1e6:.0f} MB), {skipped} had, {fail} fail, {time.time()-t0:.0f}s", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    import glob
    corpus_n = len(glob.glob(str(OUT / "*.json")))
    corpus_gb = sum(os.path.getsize(f) for f in glob.glob(str(OUT / "*.json"))) / 1e9
    print(f"[refresh_top20] FETCH done: +{new} new ({got/1e6:.0f} MB), {skipped} already-had, {fail} failed "
          f"in {time.time()-t0:.0f}s | corpus now {corpus_n} games, {corpus_gb:.1f} GB", flush=True)

    stamp = f"{args.stamp}_top20_refresh"
    if args.snapshot:
        print(f"[refresh_top20] writing snapshot {stamp} ...", flush=True)
        rc = subprocess.run([sys.executable, str(ROOT / "tools" / "snapshot_replays_v2.py"),
                             "--stamp", stamp], cwd=str(ROOT)).returncode
        if args.commit and rc == 0:
            repo = ROOT.parent
            rel = "pokemon-tcg-ai-battle/data"
            man = f"{rel}/manifests/replays_{stamp}.json"
            spl = f"{rel}/splits/replays_{stamp}_split.json"
            subprocess.run(["git", "-C", str(repo), "add", man, spl])
            subprocess.run(["git", "-C", str(repo), "commit", "-m",
                            f"pokemon-tcg: top-20 replay refresh snapshot {stamp} (+{new} games; rolling corpus)"])
            print(f"[refresh_top20] committed {man} + split (not pushed)", flush=True)
    print(f"[refresh_top20] done. snapshot stamp: {stamp}", flush=True)


if __name__ == "__main__":
    main()
