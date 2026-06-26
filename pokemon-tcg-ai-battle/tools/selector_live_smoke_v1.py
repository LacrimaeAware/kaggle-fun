"""Tiny live smoke for the learned selector wiring: run the heavy Starmie agent in three modes
S0=off (baseline), S1=top1_gate, S2=top3_selector against 5 mode-insensitive opponents, N games per matchup.

Each mode is set via STARMIE_SELECTOR_MODE in the worker (re-read at decision time). Opponents never use the
Starmie heuristic agent, so they are unaffected by the env var. This is a SAFETY/DIRECTION smoke, NOT a ranking:
local self-play does not predict the ladder (proven repeatedly). It answers: does the selector run cleanly live,
stay legal, and not obviously regress vs off?

  PYTHONIOENCODING=utf-8 python tools/selector_live_smoke_v1.py --games 20 --budget 0.2 --workers 6
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_selector_live_smoke_v1"

ALAKAZAM = ([5]*3 + [13] + [19]*4 + [66]*3 + [305]*4 + [741]*4 + [742]*4 + [743]*4 + [1079]*4 + [1081]*4
            + [1086]*4 + [1097]*3 + [1129] + [1152]*4 + [1182]*3 + [1184] + [1225]*4 + [1231]*4 + [1264])
DENPA92 = ([5]*3 + [19]*4 + [65]*4 + [66]*4 + [741]*4 + [742]*4 + [743]*3 + [1079]*3 + [1081]*3 + [1086]*4
           + [1097] + [1129] + [1146] + [1152]*4 + [1159] + [1182]*3 + [1184] + [1225]*4 + [1231]*4 + [1264]*4)
MODES = {"S0": "off", "S1": "top1_gate", "S2": "top3_selector"}
OPPONENTS = ["deployed", "alakazam", "denpa92", "first", "random"]


@contextlib.contextmanager
def _quiet_import():
    old = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(dn)
        os.close(old)


_G: dict = {}


def _winit(budget):
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet_import(), contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make  # noqa: F401
        import deck_policy_v3 as DP
        import search_v3 as S
        import starmie_heuristics as SH
        import main as M
    S.USE_DYNAMIC_ATTACKS = True
    with contextlib.suppress(Exception):
        S.DEFAULT_BUDGET = budget
    _G.update(make=make, DP=DP, S=S, SH=SH, M=M)


def _field_pilot(deck):
    DP, S = _G["DP"], _G["S"]

    def pilot(obs):
        if obs.get("select") is None:
            return list(deck)
        try:
            ko = DP.best_ko_attack(obs)
            if ko is not None:
                return [ko[0]]
            mv = S.best_option(obs, deck, leaf_mode="hand")
            if mv:
                return list(mv)
        except Exception:
            pass
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        k = sel.get("minCount") or 1
        return list(range(min(max(k, 1), len(opts)))) if opts else []
    return pilot


def _opponent(name):
    if name == "deployed":
        return _G["M"].agent_starmie
    if name == "alakazam":
        return _field_pilot(ALAKAZAM)
    if name == "denpa92":
        return _field_pilot(DENPA92)
    return name  # "first" / "random" -> cabt built-in agent string


def _winner(env):
    last = env.steps[-1]
    r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def run_chunk(task):
    mode_key, opp_name, n, seat0 = task
    os.environ["STARMIE_SELECTOR_MODE"] = MODES[mode_key]
    make, SH = _G["make"], _G["SH"]
    test = SH.agent
    opp = _opponent(opp_name)
    wt = wo = dr = err = 0
    for i in range(n):
        t_seat = (seat0 + i) % 2
        pair = [test, opp] if t_seat == 0 else [opp, test]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt")
                env.run(pair)
            w = _winner(env)
        except Exception:
            err += 1
            continue
        if w is None:
            dr += 1
        elif w == t_seat:
            wt += 1
        else:
            wo += 1
    return (mode_key, opp_name, wt, wo, dr, err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=2)
    args = ap.parse_args()
    args.games = min(args.games, 50)  # hard cap per the smoke spec

    tasks = []
    for mk in MODES:
        for opp in OPPONENTS:
            rem, seat = args.games, 0
            while rem > 0:
                k = min(args.chunk, rem)
                tasks.append((mk, opp, k, seat))
                seat = (seat + k) % 2
                rem -= k
    print(f"Selector live smoke: modes {list(MODES.items())} vs {OPPONENTS}, n={args.games}/matchup, "
          f"budget={args.budget}s", flush=True)
    print("CAVEAT: local self-play does NOT predict the ladder; this is a safety/direction smoke.\n", flush=True)

    agg = Counter()
    done, total = 0, len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit, initargs=(args.budget,)) as ex:
        for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
            done += 1
            try:
                mk, opp, wt, wo, dr, err = f.result()
            except Exception as exc:
                print(f"  [{done}/{total}] ERROR {exc!r}", flush=True)
                continue
            agg[(mk, opp, "wt")] += wt
            agg[(mk, opp, "wo")] += wo
            agg[(mk, opp, "dr")] += dr
            agg[(mk, opp, "err")] += err
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] chunks done", flush=True)

    report = {"games_per_matchup": args.games, "budget": args.budget, "modes": MODES, "opponents": OPPONENTS,
              "results": {}}
    print("\n=== heavy Starmie win rate by mode x opponent ===", flush=True)
    for mk, mname in MODES.items():
        row = {}
        for opp in OPPONENTS:
            wt, wo, dr, err = (agg[(mk, opp, k)] for k in ("wt", "wo", "dr", "err"))
            tot = wt + wo
            wr = round(100 * wt / tot, 1) if tot else None
            row[opp] = {"win": wt, "loss": wo, "draw": dr, "err": err, "win_pct": wr}
            print(f"  {mk}({mname:13s}) vs {opp:9s}: {wt}-{wo}  ({wr}% win, {dr} draw, {err} err)", flush=True)
        report["results"][mk] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'live_smoke_report.json'}", flush=True)


if __name__ == "__main__":
    main()
