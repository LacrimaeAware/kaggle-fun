"""Head-to-head A/B: heuristic_search_candidate_v1 vs the current baseline agent_search.

The candidate ships its own main/search/eval/features/deck_policy. To run it against the
baseline in ONE process we copy it into an isolated package `_candv1` (rewriting its internal
imports to `_candv1.*`) so the two codebases never collide -- including lazy imports.

Both sides are forced to the SAME pilot deck (read from the restored replay 80723114.json), so
any win-rate gap is PURE POLICY, not the cards. Seats are swapped every game. Reports Wilson 95%.

    python tools/ab_candidate_v1.py --games 6          # smoke
    python tools/ab_candidate_v1.py --games 60          # real A/B
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # pokemon-tcg-ai-battle
CAND_SRC = ROOT / "dropoff" / "inbox" / "heuristic_search_candidate_v1" / "heuristic_search_candidate_v1"
PKG = ROOT / "_candv1"
MODS = ["main", "search", "eval", "features", "deck_policy"]
DATA = ["card_stats.json", "card_features.json", "attack_stats.json", "card_effects.json"]

logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt          # noqa: F401
    from kaggle_environments import make


def build_candidate_pkg() -> None:
    """Copy the candidate into an isolated package, rewriting cross-imports to _candv1.*."""
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir()
    (PKG / "__init__.py").write_text("", encoding="utf-8")
    for m in MODS:
        src = (CAND_SRC / f"{m}.py").read_text(encoding="utf-8")
        for x in MODS:
            src = re.sub(rf"^(\s*)import {x} as (\w+)", rf"\1import _candv1.{x} as \2", src, flags=re.M)
            src = re.sub(rf"^(\s*)import {x}\s*$", rf"\1import _candv1.{x} as {x}", src, flags=re.M)
        (PKG / f"{m}.py").write_text(src, encoding="utf-8")
    for j in DATA:
        shutil.copy(ROOT / "agent" / j, PKG / j)


def pilot_deck() -> list:
    d = json.load(open(ROOT / "data" / "external" / "replays" / "80723114.json", encoding="utf-8"))
    for step in d["steps"][:6]:
        if isinstance(step, list) and step and isinstance(step[0], dict):
            act = step[0].get("action")
            if isinstance(act, list) and len(act) == 60:
                return list(act)
    raise SystemExit("pilot deck not found in 80723114.json")


def wilson(w: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def winner_of(env):
    last = env.steps[-1]
    r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def run(games: int, a, b, progress: int = 5) -> dict:
    wins_a = wins_b = draws = errors = 0
    t0 = time.time()
    for g in range(games):
        a_seat = g % 2
        agents = [a, b] if a_seat == 0 else [b, a]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env = make("cabt")
                env.run(agents)
            w = winner_of(env)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR game {g+1}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        if w is None:
            draws += 1
        elif w == a_seat:
            wins_a += 1
        else:
            wins_b += 1
        done = g + 1
        if progress and (done % progress == 0 or done == games):
            dec = wins_a + wins_b
            wr = wins_a / dec if dec else 0.0
            el = time.time() - t0
            eta = el / done * (games - done)
            print(f"  {done}/{games} | cand {wr:.3f} ({wins_a}-{wins_b}, {draws}d {errors}e) | "
                  f"{el:.0f}s, ~{eta:.0f}s left", flush=True)
    return dict(games=games, wins_a=wins_a, wins_b=wins_b, draws=draws, errors=errors,
                seconds=round(time.time() - t0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--progress", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "tools" / "_ab_candidate_v1.json"))
    args = ap.parse_args()

    build_candidate_pkg()
    sys.path.insert(0, str(ROOT))               # for `import _candv1.*`
    sys.path.insert(0, str(ROOT / "agent"))     # baseline modules
    base_main = importlib.import_module("main")
    cand_main = importlib.import_module("_candv1.main")

    PILOT = pilot_deck()
    base_main.DECK = PILOT
    cand_main.DECK = PILOT

    print(f"candidate vs baseline, pilot deck both sides, {args.games} games seat-swapped", flush=True)
    r = run(args.games, cand_main.agent_search, base_main.agent_search, progress=args.progress)
    dec = r["wins_a"] + r["wins_b"]
    lo, hi = wilson(r["wins_a"], dec)
    res = {
        "games": r["games"], "decided": dec,
        "candidate_win_rate": round(r["wins_a"] / dec, 3) if dec else None,
        "wilson95": [round(lo, 3), round(hi, 3)],
        "cand_wins": r["wins_a"], "base_wins": r["wins_b"], "draws": r["draws"],
        "errors": r["errors"], "seconds": r["seconds"],
    }
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\n=> candidate vs baseline: win rate {res['candidate_win_rate']} "
          f"Wilson {res['wilson95']} ({res['cand_wins']}-{res['base_wins']}, "
          f"{res['draws']}d, {res['errors']}e) over {dec} decided games", flush=True)


if __name__ == "__main__":
    main()
