"""Local A/B for the heavy-heuristic Starmie pilot (sub_starmie2 = starmie_heuristics.agent) vs the deployed
Starmie agent (agent_starmie = KO floor + search), and vs field decks under a generic pilot.

CAVEAT (proven repeatedly): local mirror self-play does NOT predict the ladder (sub_starmie went 33-7 locally
but 0.480 on the ladder). So this is a CATASTROPHE check ("is heavy badly broken / forfeiting / far worse?"),
not a ranking. The imitation-gap agreement vs top pilots (tools/imitation_gap_v1.py) is the real signal.

  python tools/starmie_ab_v1.py --games 30 --budget 0.3 --workers 6
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ALAKAZAM = ([5]*3 + [13] + [19]*4 + [66]*3 + [305]*4 + [741]*4 + [742]*4 + [743]*4 + [1079]*4 + [1081]*4
            + [1086]*4 + [1097]*3 + [1129] + [1152]*4 + [1182]*3 + [1184] + [1225]*4 + [1231]*4 + [1264])
DENPA92 = ([5]*3 + [19]*4 + [65]*4 + [66]*4 + [741]*4 + [742]*4 + [743]*3 + [1079]*3 + [1081]*3 + [1086]*4
           + [1097] + [1129] + [1146] + [1152]*4 + [1159] + [1182]*3 + [1184] + [1225]*4 + [1231]*4 + [1264]*4)


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


_G = {}


def _winit(budget):
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet_import(), contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make  # noqa: F401
        import deck_policy_v3 as DP
        import search_v3 as S
        import starmie_heuristics as SH
        import main as M
    S.USE_DYNAMIC_ATTACKS = True
    try:
        S.DEFAULT_BUDGET = budget
    except Exception:
        pass
    _G.update(make=make, DP=DP, S=S, SH=SH, M=M)


def _make_agent(name):
    DP, S, SH, M = _G["DP"], _G["S"], _G["SH"], _G["M"]
    if name == "heavy":
        return SH.agent
    if name == "deployed":
        return M.agent_starmie
    # generic field pilot (KO floor + search hand-leaf), bound to a deck
    deck = {"alakazam": ALAKAZAM, "denpa92": DENPA92}[name]

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


def _winner(env):
    last = env.steps[-1]
    r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def run_chunk(task):
    a_name, b_name, n, seat0 = task
    make = _G["make"]
    A, B = _make_agent(a_name), _make_agent(b_name)
    wa = wb = dr = 0
    for i in range(n):
        a_seat = (seat0 + i) % 2
        pair = [A, B] if a_seat == 0 else [B, A]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt")
                env.run(pair)
            w = _winner(env)
        except Exception:
            dr += 1
            continue
        if w is None:
            dr += 1
        elif w == a_seat:
            wa += 1
        else:
            wb += 1
    return (a_name, b_name, wa, wb, dr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--budget", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=3)
    args = ap.parse_args()

    matchups = [("heavy", "deployed"), ("heavy", "alakazam"), ("heavy", "denpa92")]
    tasks = []
    for a, b in matchups:
        rem, seat = args.games, 0
        while rem > 0:
            k = min(args.chunk, rem)
            tasks.append((a, b, k, seat))
            seat = (seat + k) % 2
            rem -= k
    print(f"Starmie A/B: heavy vs [deployed, alakazam, denpa92], n={args.games}/matchup, budget={args.budget}s",
          flush=True)
    print("CAVEAT: local mirror play does NOT predict the ladder; this is a catastrophe check.\n", flush=True)
    agg = Counter()
    done, total = 0, len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit, initargs=(args.budget,)) as ex:
        for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
            done += 1
            try:
                a, b, wa, wb, dr = f.result()
            except Exception as exc:
                print(f"  [{done}/{total}] ERROR {exc!r}", flush=True)
                continue
            agg[(a, b, "wa")] += wa
            agg[(a, b, "wb")] += wb
            agg[(a, b, "dr")] += dr
            print(f"  [{done}/{total}] {a} vs {b}: +{wa}-{wb} (={agg[(a,b,'wa')]}-{agg[(a,b,'wb')]}"
                  f" draws {agg[(a,b,'dr')]})", flush=True)
    print("\n=== heavy Starmie win rate ===", flush=True)
    for a, b in matchups:
        wa, wb, dr = agg[(a, b, "wa")], agg[(a, b, "wb")], agg[(a, b, "dr")]
        tot = wa + wb
        wr = wa / tot if tot else 0.0
        print(f"  heavy vs {b:9s}: {wa}-{wb}  ({wr*100:.0f}% win, {dr} draws)", flush=True)


if __name__ == "__main__":
    main()
