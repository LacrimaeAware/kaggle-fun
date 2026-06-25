"""The heuristic lab's testable UNIT: A/B two heuristic CONFIGS of the 916 agent (sub_heuristic).

A "config" is kwargs to hiroingk_alakazam_heuristics (toggles + params), e.g. {"enable_gust_threat": false}
or {"max_energy_per_alakazam_line": 2}. Side A = baseline config + the idea's changes; side B = baseline.
Heuristics-first + the same search fallback as the deployed agent. Parallel, seat-alternated, durable JSON.
Win rate is A's; >0.5 means the idea helped. Prints CI; reports errors (a broken config shows as errors).

  python tools/config_ab_v1.py --a '{"gust_enable_defensive_stall": true}' --games 100
  python tools/config_ab_v1.py --a-file idea.json --games 100 --out data/heuristic_lab/runs/idea.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if not os.environ.get("LAB_DEBUG"):
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submissions" / "sub_heuristic"
sys.path.insert(0, str(SUB))


def _agent_for(cfg: dict):
    """Build the 916 heuristics-first agent with the registry overrides in cfg."""
    from pokemon_ai_agent.policy.heuristics.registry import hiroingk_alakazam_heuristics
    from pokemon_ai_agent.policy.coordinator import first_heuristic_choice
    from pokemon_ai_agent.policy.fallback import legal_fallback
    import agent_impl as AI

    H = hiroingk_alakazam_heuristics(card_data=AI._CARD, attack_stats=AI._ATK, replay_scores=None, **cfg)

    def agent(obs):
        if obs.get("select") is None:
            return list(AI.DECK)
        try:
            hc = first_heuristic_choice(obs, H)
            if hc is not None:
                return list(hc[1])
            mv = AI._search_fallback(obs)
            if mv:
                return list(mv)
        except Exception:
            pass
        try:
            return legal_fallback(obs, reason="lab")
        except Exception:
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            k = sel.get("minCount") or 1
            return list(range(min(max(k, 1), len(opts)))) if opts else []
    return agent


def run_chunk(task):
    cfg_a, cfg_b, n, seat0 = task
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    A, B = _agent_for(cfg_a), _agent_for(cfg_b)
    wa = wb = dr = er = 0
    for i in range(n):
        a_seat = (seat0 + i) % 2
        pair = [A, B] if a_seat == 0 else [B, A]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt")
                env.run(pair)
            last = env.steps[-1]
            r0, r1 = last[0].get("reward"), last[1].get("reward")
        except Exception:
            er += 1
            continue
        if r0 is None or r1 is None or r0 == r1:
            dr += 1
        else:
            w = 0 if r0 > r1 else 1
            (wa, wb) = (wa + 1, wb) if w == a_seat else (wa, wb + 1)
    return (wa, wb, dr, er)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None, help="config A as JSON (the idea); default {} = baseline")
    ap.add_argument("--a-file", default=None)
    ap.add_argument("--b", default="{}", help="config B as JSON; default {} = baseline registry")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="idea")
    args = ap.parse_args()

    cfg_a = json.loads(Path(args.a_file).read_text(encoding="utf-8")) if args.a_file else json.loads(args.a or "{}")
    cfg_b = json.loads(args.b)
    print(f"config A/B: A={cfg_a}  vs  B={cfg_b}  n={args.games}", flush=True)

    tasks, rem, seat = [], args.games, 0
    while rem > 0:
        k = min(args.chunk, rem)
        tasks.append((cfg_a, cfg_b, k, seat))
        seat = (seat + k) % 2
        rem -= k

    wa = wb = dr = er = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
            cwa, cwb, cdr, cer = f.result()
            wa += cwa; wb += cwb; dr += cdr; er += cer
    dec = wa + wb
    p = wa / dec if dec else 0.0
    hw = 1.96 * math.sqrt(p * (1 - p) / dec) if dec else 0.0
    lo, hi = p - hw, p + hw
    verdict = (">=60 escalate" if p >= 0.60 else "55-59 escalate-if-clean" if p >= 0.55
               else "<=47 reject" if p <= 0.47 else "48-54 noise/refine")
    res = {"label": args.label, "cfg_a": cfg_a, "cfg_b": cfg_b, "games": args.games, "wins_a": wa,
           "wins_b": wb, "draws": dr, "errors": er, "winrate_a": round(p, 4),
           "ci95": [round(lo, 4), round(hi, 4)], "gate": verdict}
    print(f"RESULT {args.label}: A={p:.3f} [{lo:.3f},{hi:.3f}] ({wa}-{wb}, {dr}d, {er}e) -> {verdict}", flush=True)
    out = Path(args.out) if args.out else (ROOT / "data" / "heuristic_lab" / "runs" / f"{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
