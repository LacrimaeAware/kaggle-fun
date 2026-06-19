"""Branch A -- label EXACTLY the states Model B requests (its failure/test decisions), in the same
self-contained Teacher V2 format. Consumes B's request file; does not invent a generic batch.

Each request entry may provide the root observation directly (preferred -- fully self-contained), or just
source identifiers (file/step [+obs_hash]) to recover it from the replay. Deck for determinization is taken
from the entry, else recovered as the deciding player's deck from the source replay, else falls back to the
production deck (flagged). Requested states are labeled REGARDLESS of criticality (B chose them).

    python tools/label_requested_states.py --request data/manifests/teacher_v2_label_request_for_A.json

Request schema (list, or {"requests":[...]}); all fields optional except a way to get the obs:
    {"decision_id": "<file>:<step>:<player>", "obs_hash": "<12hex>",
     "observation": {...root obs...}, "deck": [60 ints],
     "source": {"file": "<x>.json", "step": <int>, "player": <0|1>}, "reason": "<why>"}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import teacher_api_v2 as T2   # noqa: E402
import main as M              # noqa: E402

REPLAY_DIR = ROOT / "data" / "external" / "replays"
MAN = ROOT / "data" / "manifests"


def _hash(obs):
    return hashlib.sha1(json.dumps(obs, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _recover_from_source(src, obs_hash, cache):
    """Return (observation, deck, player) recovered from the source replay step, or (None,None,None)."""
    fn, step = (src or {}).get("file"), (src or {}).get("step")
    if fn is None or step is None:
        return None, None, None
    if fn not in cache:
        try:
            cache[fn] = json.load(open(REPLAY_DIR / fn, encoding="utf-8"))
        except Exception:
            cache[fn] = None
    d = cache[fn]
    if not d or step >= len(d.get("steps", [])):
        return None, None, None
    row = d["steps"][step]
    obs, player = None, src.get("player")
    for ai in (0, 1):
        ag = row[ai] if ai < len(row) and isinstance(row[ai], dict) else None
        cand = (ag or {}).get("observation") or {}
        if cand and (obs_hash is None or _hash(cand) == obs_hash):
            obs, player = cand, ai
            if obs_hash is not None:
                break
    deck = None
    if player is not None:
        for s in d.get("steps", []):
            a = s[player].get("action") if player < len(s) and isinstance(s[player], dict) else None
            if isinstance(a, list) and len(a) == 60:
                deck = a
                break
    return obs, deck, player


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", default="data/manifests/teacher_v2_label_request_for_A.json")
    ap.add_argument("--out", default="teacher_v2_labels_for_B_failures")
    ap.add_argument("--n-determ", type=int, default=32)
    ap.add_argument("--k-outcome", type=int, default=16)
    args = ap.parse_args()

    req_path = ROOT / args.request if not Path(args.request).is_absolute() else Path(args.request)
    raw = json.load(open(req_path, encoding="utf-8"))
    entries = raw["requests"] if isinstance(raw, dict) and "requests" in raw else raw
    print(f"[label-req] {len(entries)} requested states from {req_path.name}", flush=True)

    cache, labels, failed = {}, [], []
    for e in entries:
        obs = e.get("observation")
        deck = e.get("deck")
        src = e.get("source") or {}
        obs_hash = e.get("obs_hash")
        if obs is None or deck is None:
            r_obs, r_deck, _ = _recover_from_source(src, obs_hash, cache)
            obs = obs or r_obs
            deck = deck or r_deck
        if obs is None:
            failed.append({"decision_id": e.get("decision_id"), "reason": "obs not reconstructable"})
            continue
        deck_src = "request/source"
        if not deck:
            deck, deck_src = list(M.DECK), "FALLBACK_PRODUCTION_DECK"
        lab = T2.query_v2(obs, deck, n_determ=args.n_determ, k_outcome=args.k_outcome,
                          crit_threshold=0.0, seed=1234)   # label regardless of criticality (B chose it)
        if not lab.get("evaluated"):
            failed.append({"decision_id": e.get("decision_id"), "reason": "search not applicable"})
            continue
        lab["decision_id"] = e.get("decision_id") or (f"{src.get('file')}:{src.get('step')}" if src else _hash(obs))
        lab["obs_hash"] = _hash(obs)
        lab["observation"] = obs
        lab["legal_options"] = (obs.get("select") or {}).get("option") or []
        lab["source"] = src
        lab["deck_source"] = deck_src
        lab["request_reason"] = e.get("reason")
        labels.append(lab)
        print(f"  labeled {lab['decision_id']} (crit {lab['criticality']['score']}, deck={deck_src})", flush=True)

    out = MAN / f"{args.out}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")
    print(f"\n[label-req] labeled {len(labels)}/{len(entries)} requested states -> {out.relative_to(ROOT)}")
    if failed:
        print(f"  could NOT label {len(failed)}: {failed[:5]}")
    return len(labels), failed


if __name__ == "__main__":
    main()
