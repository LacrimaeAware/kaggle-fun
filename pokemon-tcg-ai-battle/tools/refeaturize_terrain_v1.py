"""Continuous Terrain V1 -- recompute the A5 semantic vectors in the finalized dataset with the improved
category-aware coverage + overrides, WITHOUT re-running the engine: the forward-model option-deltas (d_*)
are already baked, so we recompute identity/meta/effects/context/coverage and preserve the d_* fields.

    python tools/refeaturize_terrain_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import features as FT                  # noqa: E402
import action_semantics_v1 as AS       # noqa: E402

MAN = ROOT / "data" / "manifests"
PATH = MAN / "continuous_terrain_v1.jsonl"

recs = [json.loads(l) for l in open(PATH, encoding="utf-8")]
changed = 0
for r in recs:
    obs = r["observation"]
    cur = obs.get("current") or {}
    me = r.get("me", cur.get("yourIndex", 0))
    feats = FT.encode_state(obs)
    for o in r["options"]:
        old = o.get("semantic_vector") or {}
        new = AS.semantic_vector(obs, o["index"], None, feats, cur, me)   # d_* = 0 (no engine)
        for k, v in old.items():
            if k.startswith("d_"):
                new[k] = v                                                # preserve baked forward-model deltas
        if new.get("semantic_coverage") != old.get("semantic_coverage"):
            changed += 1
        o["semantic_vector"] = new

with open(PATH, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print(f"re-featurized {len(recs)} decisions; {changed} options changed coverage tier.")
