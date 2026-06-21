"""Parallel V3 ablation: each V3 entry point (isolated copy of the V3 package) vs the CURRENT
production agent_search, same pilot deck, seats swapped. Per-cell Wilson CIs.

V3 source is staged at tools/_v3_src (deck_policy_v3 + search_v3 + the additive main/eval/features).
Each worker builds its OWN isolated package `_v3_<tag>` so 16 engine processes never collide.
Side A = the V3 entry point; side B = pure current production (agent/main.agent_search).

The 'control' cell is V3's inherited agent_search vs production agent_search: if it is ~0.5, the
features.py drift is inert and every other cell's delta is real V3 logic.

    python tools/ab_v3_ablate_v1.py --cells control,phfix,v3,v3_poffin,v3_boss --games 20 --shards 3

Cells -> entry point on the V3 main:
  control   agent_search            (inherited baseline; expect ~0.5)
  phfix     agent_search_phfix      (legacy search + W/R-correct dynamic-PH auto-KO)
  v3        agent_search_v3         (phfix + turn-aware KO window, no resolver)
  v3_poffin agent_search_v3_poffin
  v3_boss   agent_search_v3_boss
  v3_n32    agent_search_v3_n32     (slow: N=32)
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tools" / "_v3_src"
MODS = ["main", "search", "search_v3", "eval", "features", "deck_policy_v3"]
DATA = ["card_stats.json", "card_features.json", "attack_stats.json", "card_effects.json"]

ENTRY = {
    "selfbase":  "agent_search",   # production vs itself (harness-fairness probe; side A is production)
    "control":   "agent_search",
    "phfix":     "agent_search_phfix",
    "v3":        "agent_search_v3",
    "v3_poffin": "agent_search_v3_poffin",
    "v3_boss":   "agent_search_v3_boss",
    "v3_n32":    "agent_search_v3_n32",
    # the pro's decomposition: PH as visibility vs forcing, on N=8 and the N=32 sampling axis
    "ph_vis":            "agent_search_ph_vis",            # PH_DAMAGE_FIX (visibility, no PH force)
    "ph_vis_final":      "agent_search_ph_vis_final",      # + PH_FORCE_FINAL (force PH only on game win)
    "s32":               "agent_search_s32_plain",         # S32 sampling axis, no PH
    "s32_ph_vis":        "agent_search_s32_ph_vis",         # S32 + PH_DAMAGE_FIX
    "s32_ph_vis_final":  "agent_search_s32_ph_vis_final",   # S32 + PH_DAMAGE_FIX + PH_FORCE_FINAL
}


def _build_pkg(tag: str) -> str:
    pkg_name = f"_v3_{tag}"
    pkg = ROOT / pkg_name
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for m in MODS:
        # Use PRODUCTION features.py in the candidate so the inherited baseline is byte-identical to
        # production (kills the features.py drift confound -> control becomes a true 0.5). The V3 logic
        # (deck_policy_v3 / search_v3 / entry points) does not depend on the new feature keys.
        src_path = (ROOT / "agent" / "features.py") if m == "features" else (SRC / f"{m}.py")
        src = src_path.read_text(encoding="utf-8")
        for x in MODS:
            src = re.sub(rf"^(\s*)import {x} as (\w+)", rf"\1import {pkg_name}.{x} as \2", src, flags=re.M)
            src = re.sub(rf"^(\s*)import {x}\s*$", rf"\1import {pkg_name}.{x} as {x}", src, flags=re.M)
        (pkg / f"{m}.py").write_text(src, encoding="utf-8")
    for j in DATA:
        shutil.copy(SRC / j, pkg / j)
    return pkg_name


def run_shard(task):
    cell, tag, games = task
    sys.path.insert(0, str(ROOT / "tools"))
    import ab_candidate_v1 as AB                  # run / wilson / pilot_deck (engine import)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "agent"))
    base = importlib.import_module("main")               # pure current production
    PILOT = AB.pilot_deck()
    base.DECK = PILOT
    pkg_name = None
    try:
        if cell == "selfbase":
            side_a = base.agent_search               # production vs ITSELF: harness-fairness probe
        else:
            pkg_name = _build_pkg(tag)
            v3main = importlib.import_module(f"{pkg_name}.main")  # V3 candidate
            v3main.DECK = PILOT
            side_a = getattr(v3main, ENTRY[cell])
        r = AB.run(games, side_a, base.agent_search, progress=0)
        out = {"cell": cell, "tag": tag, "wins": r["wins_a"], "losses": r["wins_b"],
               "draws": r["draws"], "errors": r["errors"], "seconds": r["seconds"]}
        (ROOT / "tools" / f"_v3ab_{tag}.json").write_text(json.dumps(out), encoding="utf-8")
        return out
    finally:
        if pkg_name:
            shutil.rmtree(ROOT / pkg_name, ignore_errors=True)


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="control,phfix,v3,v3_poffin,v3_boss")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "docs" / "workstreams" / "v3_ablation_results.json"))
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip() in ENTRY]
    tasks = []
    for cell in cells:
        for s in range(args.shards):
            tasks.append((cell, f"{cell}_{s}", args.games))
    print(f"V3 ablation: cells={cells} x {args.shards} shards x {args.games} games "
          f"= {len(tasks) * args.games} games on {args.workers} workers", flush=True)

    t0 = time.time()
    agg = {c: {"wins": 0, "losses": 0, "draws": 0, "errors": 0} for c in cells}
    done = 0
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for res in pool.imap_unordered(run_shard, tasks):
            c = res["cell"]
            for k in ("wins", "losses", "draws", "errors"):
                agg[c][k] += res[k]
            done += 1
            a = agg[c]; n = a["wins"] + a["losses"]; wr = a["wins"] / n if n else 0.0
            print(f"  [{done}/{len(tasks)}] {res['tag']} done ({res['wins']}-{res['losses']}, "
                  f"{res['seconds']:.0f}s) | {c}: {a['wins']}-{a['losses']} ({wr:.3f})", flush=True)

    el = time.time() - t0
    print(f"\n=== V3 ABLATION ({el:.0f}s wall) ===", flush=True)
    table = {}
    for c in cells:
        a = agg[c]; n = a["wins"] + a["losses"]; wr = a["wins"] / n if n else 0.0
        lo, hi = wilson(a["wins"], n)
        table[c] = {"win_rate": round(wr, 3), "wilson95": [round(lo, 3), round(hi, 3)],
                    "wins": a["wins"], "losses": a["losses"], "draws": a["draws"], "errors": a["errors"], "n": n}
        print(f"  {c:10s} {wr:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  ({a['wins']}-{a['losses']}, "
              f"{a['draws']}d {a['errors']}e, n={n})", flush=True)
    Path(args.out).write_text(json.dumps({"table": table, "wall_seconds": round(el, 1),
                                          "games_per_shard": args.games, "shards": args.shards}, indent=1),
                              encoding="utf-8")
    print(f"\nwrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
