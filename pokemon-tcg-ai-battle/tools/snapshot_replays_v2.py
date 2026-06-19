"""SPLIT_BASE_V2 / P1 -- freeze an IMMUTABLE dated snapshot of the replay corpus + fixed splits.

The rolling downloader may keep appending raw files, but no experiment may silently consume a moving
corpus. This tool hashes the current replays and writes two COMMITTED artifacts (the raw replays stay
gitignored):

    data/manifests/replays_<stamp>.json        -- per-file hash + identity metadata + skip reasons
    data/splits/replays_<stamp>_split.json      -- fixed train/val/test by chronology, plus
                                                   held-out-player and held-out-deck eval subsets

Chronology note: cabt replays carry NO timestamp. info.EpisodeId is a monotonically increasing Kaggle
id, so it is used as the chronological key (documented proxy for date). Splits are at the GAME level,
so candidate rows from the same DECISION never cross the train/test boundary (P5 requirement).

    python tools/snapshot_replays_v2.py [--stamp YYYYMMDD_HHMM] [--train 0.70 --val 0.15]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import state_action_schema_v2 as SCH  # noqa: E402  (deck_signature, is_single_pick_decision)

PARSER_VERSION = "split_base_v2.snapshot.1"
REPLAY_DIR = ROOT / "data" / "external" / "replays"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _player_decks(d: dict) -> dict:
    """Each player's 60-card deck-selection action -> deck_signature, by player index."""
    out = {}
    for s in d.get("steps", []):
        for ai, agent in enumerate(s):
            if ai in out or not isinstance(agent, dict):
                continue
            act = agent.get("action")
            if isinstance(act, list) and len(act) == 60:
                out[ai] = SCH.deck_signature(act)
        if len(out) >= 2:
            break
    return out


def _count_decisions(d: dict) -> dict:
    """Single-pick decisions (the unit experiments use) per player index."""
    n = Counter()
    for s in d.get("steps", []):
        for ai, agent in enumerate(s):
            if not isinstance(agent, dict):
                continue
            obs = agent.get("observation") or {}
            if SCH.is_single_pick_decision(obs):
                n[ai] += 1
    return dict(n)


def _record(path: str) -> dict:
    fn = os.path.basename(path)
    rec = {"file": fn, "sha256": None, "episode_id": None, "players": None,
           "decks": None, "result": None, "n_steps": 0, "n_decisions": {}, "skipped_reason": None}
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        rec["skipped_reason"] = f"unreadable:{type(e).__name__}"
        return rec
    if not isinstance(d, dict):
        rec["skipped_reason"] = "not_a_dict"
        return rec
    rec["sha256"] = _sha256(path)
    info = d.get("info") or {}
    rec["episode_id"] = info.get("EpisodeId") or os.path.splitext(fn)[0]
    rec["players"] = [a.get("Name") for a in (info.get("Agents") or [])] or (info.get("TeamNames") or [])
    rec["decks"] = {str(k): v["hash"] for k, v in _player_decks(d).items()}
    rec["n_steps"] = len(d.get("steps") or [])
    rec["n_decisions"] = {str(k): v for k, v in _count_decisions(d).items()}
    rw = d.get("rewards") or []
    if len(rw) != 2 or None in rw:
        rec["skipped_reason"] = "no_result"        # kept in manifest, excluded from splits
    elif rw[0] == rw[1]:
        rec["result"] = "draw"
        rec["skipped_reason"] = "draw"
    else:
        rec["result"] = {"rewards": rw, "winner": 0 if rw[0] > rw[1] else 1}
    return rec


def _epoch_key(rec: dict):
    try:
        return (0, int(rec["episode_id"]))
    except (TypeError, ValueError):
        return (1, str(rec["episode_id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPLAY_DIR / "*.json")))
    records = [_record(fp) for fp in files]
    included = [r for r in records if r["skipped_reason"] is None]
    skipped = [r for r in records if r["skipped_reason"] is not None]

    manifest = {
        "snapshot_id": args.stamp,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parser_version": PARSER_VERSION,
        "schema_version": SCH.SCHEMA_VERSION,
        "source_dir": str(REPLAY_DIR.relative_to(ROOT)),
        "chronology_key": "info.EpisodeId (proxy for date; replays carry no timestamp)",
        "n_files": len(records),
        "n_included": len(included),
        "n_skipped": len(skipped),
        "skip_reason_counts": dict(Counter(r["skipped_reason"] for r in skipped)),
        "corpus_sha256": hashlib.sha256(
            "".join(sorted(r["sha256"] for r in records if r["sha256"])).encode()).hexdigest()[:16],
        "files": records,
    }

    # ---- chronological game-level split (test = future slice) ----
    inc = sorted(included, key=_epoch_key)
    n = len(inc)
    n_tr = int(n * args.train)
    n_va = int(n * args.val)
    train, val, test = inc[:n_tr], inc[n_tr:n_tr + n_va], inc[n_tr + n_va:]

    def ids(rs):
        return [r["file"] for r in rs]

    def deck_set(rs):
        return {h for r in rs for h in (r["decks"] or {}).values()}

    def player_set(rs):
        return {p for r in rs for p in (r["players"] or []) if p}

    train_players, train_decks = player_set(train), deck_set(train)
    held_out_players = sorted(player_set(val + test) - train_players)
    held_out_decks = sorted(deck_set(test) - train_decks)

    split = {
        "snapshot_id": args.stamp,
        "manifest": f"replays_{args.stamp}.json",
        "strategy": "chronological by EpisodeId at the GAME level (decisions never cross the boundary)",
        "fractions": {"train": args.train, "val": args.val, "test": round(1 - args.train - args.val, 4)},
        "n": {"train": len(train), "val": len(val), "test": len(test)},
        "train": ids(train),
        "val": ids(val),
        "test": ids(test),
        "held_out_players": held_out_players,        # players absent from train (player generalization)
        "held_out_decks": held_out_decks,            # deck sigs in test absent from train (deck transfer)
    }

    (ROOT / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "splits").mkdir(parents=True, exist_ok=True)
    mpath = ROOT / "data" / "manifests" / f"replays_{args.stamp}.json"
    spath = ROOT / "data" / "splits" / f"replays_{args.stamp}_split.json"
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    spath.write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(f"snapshot {args.stamp}: {len(included)} included / {len(skipped)} skipped of {len(records)}")
    print(f"  skip reasons: {manifest['skip_reason_counts']}")
    print(f"  corpus_sha256: {manifest['corpus_sha256']}")
    print(f"  split train/val/test = {len(train)}/{len(val)}/{len(test)}"
          f"  held_out_players={len(held_out_players)} held_out_decks={len(held_out_decks)}")
    print(f"  wrote {mpath.relative_to(ROOT)}  +  {spath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
