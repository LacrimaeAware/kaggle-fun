"""Parallel factorial A/B: the 2x2 design {PH-fix off/on} x {N=8 / N=32} vs the SAME frozen
production baseline (agent_search, N=8, 0.6s, no PH-fix), same pilot deck, seats swapped.

Why: single n=20 A/Bs cannot resolve a ~0.55-0.65 edge from 0.5 (a +/-0.20 swing is pure noise).
This runs every cell at high n in PARALLEL (each worker builds its OWN isolated agent package so
16 engine processes never collide on disk), then reports per-cell Wilson CIs, the two main effects,
and the interaction. The combined cell (phfix_s32) is the actual ship candidate.

    python tools/ab_factorial_v1.py --games 20 --shards 4     # 4 cells x 4 shards x 20 = 320 games

Cells: a0 (control, expect ~0.5), phfix (PH on, N8), s32 (PH off, N32), phfix_s32 (PH on, N32).
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent"
MODS = ["main", "search", "eval", "features", "deck_policy_v2"]
DATA = ["card_stats.json", "card_features.json", "attack_stats.json", "card_effects.json"]

# the 2x2 (factor PH = forced-move PH-awareness; factor N = determinization width)
CELLS = {
    "a0":        {},                                    # PH off, N=8   -> control, expect ~0.5
    "phfix":     {"forced": "phaware"},                 # PH on,  N=8
    "s32":       {"s32": True},                         # PH off, N=32
    "phfix_s32": {"forced": "phaware", "s32": True},    # PH on,  N=32  -> ship candidate
}


def _build_pkg(tag: str) -> str:
    """Copy agent/ modules into an isolated package `_cand_<tag>` with cross-imports rewritten."""
    pkg_name = f"_cand_{tag}"
    pkg = ROOT / pkg_name
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for m in MODS:
        src = (AGENT / f"{m}.py").read_text(encoding="utf-8")
        for x in MODS:
            src = re.sub(rf"^(\s*)import {x} as (\w+)", rf"\1import {pkg_name}.{x} as \2", src, flags=re.M)
            src = re.sub(rf"^(\s*)import {x}\s*$", rf"\1import {pkg_name}.{x} as {x}", src, flags=re.M)
        (pkg / f"{m}.py").write_text(src, encoding="utf-8")
    for j in DATA:
        shutil.copy(AGENT / j, pkg / j)
    return pkg_name


def run_shard(task):
    """One isolated shard: build a private package, construct the candidate agent, play `games`
    seat-swapped vs the frozen baseline. Returns (cell, wins, losses, draws, errors, seconds)."""
    cell, tag, games = task
    sys.path.insert(0, str(ROOT / "tools"))
    import ab_candidate_v1 as AB                 # run / wilson / pilot_deck (engine imported here)
    import ab_heuristic_search_v2 as H2          # build_v2_agent (pure)
    pkg_name = _build_pkg(tag)
    try:
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "agent"))
        base_prod = importlib.import_module("main")                  # frozen baseline (N=8, no PH)
        v2main = importlib.import_module(f"{pkg_name}.main")
        S = importlib.import_module(f"{pkg_name}.search")
        DP2 = importlib.import_module(f"{pkg_name}.deck_policy_v2")
        PILOT = AB.pilot_deck()
        base_prod.DECK = PILOT
        v2main.DECK = PILOT
        C = Counter()
        v2agent, cfg = H2.build_v2_agent(v2main, S, DP2, CELLS[cell], C)
        r = AB.run(games, v2agent, base_prod.agent_search, progress=0)
        return {"cell": cell, "tag": tag, "wins": r["wins_a"], "losses": r["wins_b"],
                "draws": r["draws"], "errors": r["errors"], "seconds": r["seconds"], "cfg": cfg,
                "inst": dict(C)}
    finally:
        shutil.rmtree(ROOT / pkg_name, ignore_errors=True)


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    import math
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="games per shard")
    ap.add_argument("--shards", type=int, default=4, help="shards per cell")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "docs" / "workstreams" / "factorial_v1_results.json"))
    args = ap.parse_args()

    tasks = []
    for cell in CELLS:
        for s in range(args.shards):
            tasks.append((cell, f"{cell}_{s}", args.games))
    print(f"factorial: {len(CELLS)} cells x {args.shards} shards x {args.games} games "
          f"= {len(tasks) * args.games} games on {args.workers} workers", flush=True)

    t0 = time.time()
    agg = {c: {"wins": 0, "losses": 0, "draws": 0, "errors": 0} for c in CELLS}
    inst = {c: Counter() for c in CELLS}
    done = 0
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for res in pool.imap_unordered(run_shard, tasks):
            c = res["cell"]
            for k in ("wins", "losses", "draws", "errors"):
                agg[c][k] += res[k]
            inst[c].update(res.get("inst", {}))
            done += 1
            a = agg[c]
            n = a["wins"] + a["losses"]
            wr = a["wins"] / n if n else 0.0
            print(f"  [{done}/{len(tasks)}] shard {res['tag']} done ({res['wins']}-{res['losses']}, "
                  f"{res['seconds']:.0f}s) | {c} running: {a['wins']}-{a['losses']} ({wr:.3f})",
                  flush=True)

    el = time.time() - t0
    print(f"\n=== FACTORIAL RESULTS ({el:.0f}s wall, {sum(t[2] for t in tasks)} games) ===")
    table = {}
    for c in CELLS:
        a = agg[c]
        n = a["wins"] + a["losses"]
        wr = a["wins"] / n if n else 0.0
        lo, hi = wilson(a["wins"], n)
        table[c] = {"win_rate": round(wr, 3), "wilson95": [round(lo, 3), round(hi, 3)],
                    "wins": a["wins"], "losses": a["losses"], "draws": a["draws"],
                    "errors": a["errors"], "n": n}
        print(f"  {c:12s} {wr:.3f}  Wilson [{lo:.3f}, {hi:.3f}]  ({a['wins']}-{a['losses']}, "
              f"{a['draws']}d {a['errors']}e, n={n})")

    # main effects + interaction (point estimates on win rate)
    def wr(c):
        a = agg[c]; n = a["wins"] + a["losses"]; return a["wins"] / n if n else 0.0
    ph_at_n8 = wr("phfix") - wr("a0")
    ph_at_n32 = wr("phfix_s32") - wr("s32")
    n_at_phoff = wr("s32") - wr("a0")
    n_at_phon = wr("phfix_s32") - wr("phfix")
    interaction = wr("phfix_s32") - wr("s32") - wr("phfix") + wr("a0")
    effects = {
        "PH_effect_at_N8": round(ph_at_n8, 3), "PH_effect_at_N32": round(ph_at_n32, 3),
        "N32_effect_at_PHoff": round(n_at_phoff, 3), "N32_effect_at_PHon": round(n_at_phon, 3),
        "interaction": round(interaction, 3),
    }
    print("\n  --- effects (win-rate deltas) ---")
    for k, v in effects.items():
        print(f"    {k:22s} {v:+.3f}")

    out = {"table": table, "effects": effects, "games_per_shard": args.games, "shards": args.shards,
           "wall_seconds": round(el, 1), "instrumentation": {c: dict(inst[c]) for c in CELLS}}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
