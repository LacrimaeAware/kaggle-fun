"""SPLIT_BASE_V2 / P0 -- freeze the production baseline into a machine-readable manifest.

Records the EXACT deployed agent_search configuration (so it is never re-inferred from mutable code
later) plus the established win-rate anchors with provenance, and -- with --measure N -- a fresh
seat-swapped arena measurement at this config. Branch A's A1 reproduces the baseline from this file.

    python tools/freeze_baseline_v2.py --stamp 20260618                  # config + anchors only (instant)
    python tools/freeze_baseline_v2.py --stamp 20260618 --measure 20     # + a fresh search-vs-first run

Note: the cabt engine's rollout RNG is not Python-seedable (see teacher_api_v1), so arena win rates are
Monte Carlo estimates with Wilson CIs, not bit-reproducible. The frozen CONFIG is the reproducible part.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import search as S          # noqa: E402
import main as M            # noqa: E402
import state_action_schema_v2 as SCH  # noqa: E402


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - m) / d, 3), round((c + m) / d, 3))


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return None


# established anchors (measured earlier; provenance recorded). A1 reproduces these from CONFIG.
ANCHORS = [
    {"matchup": "agent_search vs first_agent", "win_rate": 0.585, "n": 800, "wilson95": [0.551, 0.619],
     "meaning": "search beats the contest baseline", "source": "dropoff/outbox/2026-06-18-CONSENSUS-and-way-forward.md"},
    {"matchup": "agent_search vs heuristic", "win_rate": 0.833, "record": "50-10", "n": 60,
     "meaning": "this-turn anchor (DENPA92)", "source": "docs/CURRENT.md"},
    {"matchup": "agent_search vs heuristic (DENPA92)", "win_rate": 0.86, "n": 50,
     "meaning": "deck-dependent; not comparable across decks", "source": "CONSENSUS table"},
]
PUBLIC_LB = {
    "latest_complete": {"score": 697.7, "config": "N_DETERM=4", "source": "docs/SUBMISSIONS.md"},
    "pending": {"config": "N_DETERM=8 (first such submission)", "note": "public score not yet recorded"},
    "caveat": "compare OUR subs to each other; do not over-read absolute LB number (aggregation caveat).",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))
    ap.add_argument("--measure", type=int, default=0, help="run a fresh search-vs-first arena of N games (0=skip)")
    ap.add_argument("--also-heuristic", action="store_true", help="also measure search vs heuristic")
    args = ap.parse_args()

    manifest = {
        "baseline_id": args.stamp,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCH.SCHEMA_VERSION,
        "base_commit": _git_head(),
        "submission_agent": "agent_search (main.agent_search)",
        "config": {
            "deck": {"name": "DENPA92", "signature": SCH.deck_signature(M.DECK), "cards": list(M.DECK)},
            "search": {
                "N_DETERM": S.N_DETERM,
                "per_decision_budget_s": S.DEFAULT_BUDGET,
                "DEPTH_CAP": S.DEPTH_CAP,
                "continuation_policy_MY_CONT": S.MY_CONT,
                "leaf_mode": "hand",
                "plies": 1, "opp_k": 0, "opp_prior": None,
            },
            "forced_move_floor": "lethal/KO attack (attack_value>=8000) else go-first (IS_FIRST) -- main._forced_move",
            "fallback_policy": "main.agent (KO/energy heuristic + safe default order); never raises",
            "match_time_logic": "per-decision budget cap only; the agent manages no global match pool",
            "legality": "every agent returns a legal selection and never raises (mirror forfeits on exception)",
        },
        "evaluation_protocol": {
            "engine": "kaggle_environments cabt (real ruleset)",
            "seat_alternation": "per game (g % 2), to cancel first-player advantage",
            "seeding": "kaggle make('cabt') default; engine rollout RNG is NOT Python-seedable",
            "ci": "Wilson 95% on decided games",
        },
        "anchored_results": ANCHORS,
        "public_leaderboard": PUBLIC_LB,
        "fresh_measurement": None,
    }

    if args.measure > 0:
        import cabt_arena as A
        pairs = [("search", "first")] + ([("search", "heuristic")] if args.also_heuristic else [])
        fresh = []
        for a, b in pairs:
            print(f"measuring {a} vs {b}, {args.measure} games (seat-swapped)...", flush=True)
            r = A.run(args.measure, A.AGENTS[a], A.AGENTS[b], label=f"{a} vs {b}", progress=10)
            dec = r["wins_a"] + r["wins_b"]
            r["wilson95_decided"] = list(wilson(r["wins_a"], dec))
            r["matchup"] = f"{a} vs {b}"
            fresh.append(r)
        manifest["fresh_measurement"] = {"games_each": args.measure, "results": fresh,
                                         "note": "small preflight confirmation; A1 runs the full reproduction"}

    (ROOT / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    path = ROOT / "data" / "manifests" / f"baseline_v2_{args.stamp}.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"froze baseline -> {path.relative_to(ROOT)}")
    print(f"  base_commit={manifest['base_commit'][:10] if manifest['base_commit'] else None}"
          f"  deck={manifest['config']['deck']['signature']['hash']}  N_DETERM={S.N_DETERM}"
          f"  budget={S.DEFAULT_BUDGET}s  cont={S.MY_CONT}")
    if manifest["fresh_measurement"]:
        for r in manifest["fresh_measurement"]["results"]:
            print(f"  fresh: {r['matchup']} {r['a_win_rate_decided']:.3f} "
                  f"Wilson95 {r['wilson95_decided']} ({r['wins_a']}-{r['wins_b']}, {r['errors']}e, {r['s_per_game']}s/g)")


if __name__ == "__main__":
    main()
