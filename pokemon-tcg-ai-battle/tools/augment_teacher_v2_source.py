"""Branch A -- augment a Teacher V2 label artifact with the recoverable ROOT OBSERVATION + legal sibling
list + decision_id, so Model B can featurize the decisions directly (Path B). Recovers each obs from its
source replay step (no expensive re-run) and verifies it against the stored obs_hash before attaching.

    python tools/augment_teacher_v2_source.py --artifact teacher_v2_labels_scaled.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAY_DIR = ROOT / "data" / "external" / "replays"
MAN = ROOT / "data" / "manifests"


def _hash(obs):
    return hashlib.sha1(json.dumps(obs, sort_keys=True, default=str).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="teacher_v2_labels_scaled.jsonl")
    args = ap.parse_args()
    path = MAN / args.artifact
    labels = [json.loads(line) for line in open(path, encoding="utf-8")]
    recovered = mismatch = notfound = 0
    cache = {}
    for lab in labels:
        src = lab.get("source") or {}
        fn = src.get("file")
        step = src.get("step")
        if fn is None or step is None:
            notfound += 1
            continue
        if fn not in cache:
            try:
                cache[fn] = json.load(open(REPLAY_DIR / fn, encoding="utf-8"))
            except Exception:
                cache[fn] = None
        d = cache[fn]
        if not d or step >= len(d.get("steps", [])):
            notfound += 1
            continue
        obs = None
        for ai in (0, 1):
            row = d["steps"][step]
            ag = row[ai] if ai < len(row) and isinstance(row[ai], dict) else None
            cand = (ag or {}).get("observation") or {}
            if cand and _hash(cand) == lab.get("obs_hash"):
                obs = cand
                break
        if obs is None:
            mismatch += 1
            continue
        recovered += 1
        lab["decision_id"] = lab.get("decision_id") or f"{fn}:{step}"
        lab["observation"] = obs
        lab["legal_options"] = (obs.get("select") or {}).get("option") or []

    with open(path, "w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    sz = path.stat().st_size
    print(f"augmented {path.name}: recovered obs for {recovered}/{len(labels)} labels "
          f"(hash-mismatch {mismatch}, source-missing {notfound}); now {sz//1024} KB")
    if labels and "observation" in labels[0]:
        o0 = labels[0]
        print(f"  each label now carries: decision_id, observation (root state), legal_options "
              f"({len(o0.get('legal_options', []))} siblings), plus the existing label fields")


if __name__ == "__main__":
    main()
