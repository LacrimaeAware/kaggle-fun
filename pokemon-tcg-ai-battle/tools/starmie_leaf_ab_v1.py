"""STARMIE TACTICAL-LEAF V1 -- Section 7: EXPLORATORY A/B for ATTACKER_CONTINUITY_V1.

The cleanest ISOLATED test of a SHARED-search leaf term is a head-to-head MIRROR where only ONE side has the
term: heavy_continuity vs heavy_baseline (identical deck + heuristics + search; only eval.ATTACKER_CONTINUITY_ON
differs). Play is sequential per process, so we toggle the module flag before each agent's move -> each side's
leaf evals use its own setting, no cross-contamination. We also report each side vs the frozen field
(alakazam/denpa92), whose pilots use leaf_mode="hand" and are UNAFFECTED by the term, to check for regression.

  python tools/starmie_leaf_ab_v1.py --games 120 --budget 0.4

Paired: seat0 alternates (swapped seats); same opponent decks; identical budget.
"""
from __future__ import annotations
import argparse, contextlib, io, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALAKAZAM = ([5]*3+[13]+[19]*4+[66]*3+[305]*4+[741]*4+[742]*4+[743]*4+[1079]*4+[1081]*4+[1086]*4+[1097]*3+[1129]+[1152]*4+[1182]*3+[1184]+[1225]*4+[1231]*4+[1264])
DENPA92 = ([5]*3+[19]*4+[65]*4+[66]*4+[741]*4+[742]*4+[743]*3+[1079]*3+[1081]*3+[1086]*4+[1097]+[1129]+[1146]+[1152]*4+[1159]+[1182]*3+[1184]+[1225]*4+[1231]*4+[1264]*4)
_G = {}


@contextlib.contextmanager
def _quiet():
    old = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2); yield
    finally:
        os.dup2(old, 2); os.close(dn); os.close(old)


def _winit(budget):
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet(), contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make
        import deck_policy_v3 as DP, search_v3 as S, starmie_heuristics as SH, eval as EV
    S.USE_DYNAMIC_ATTACKS = True
    try: S.DEFAULT_BUDGET = budget
    except Exception: pass
    EV.ATTACKER_CONTINUITY_ON = False  # we set it per-move explicitly below
    _G.update(make=make, DP=DP, S=S, SH=SH, EV=EV)


def _heavy(continuity):
    SH, EV = _G["SH"], _G["EV"]
    def agent(obs):
        EV.ATTACKER_CONTINUITY_ON = continuity   # this move's leaf evals use this setting (sequential play)
        return SH.agent(obs)
    return agent


def _field(deck):
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
        sel = obs.get("select") or {}; o = sel.get("option") or []; k = sel.get("minCount") or 1
        return list(range(min(max(k, 1), len(o)))) if o else []
    return pilot


def _winner(env):
    last = env.steps[-1]; r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def run_chunk(task):
    kind, n, seat0, budget = task
    make = _G["make"]
    wa = wb = dr = 0
    for i in range(n):
        a_seat = (seat0 + i) % 2
        if kind == "mirror":
            A, B = _heavy(True), _heavy(False)       # A = continuity, B = baseline
        elif kind == "cont_vs_alakazam":
            A, B = _heavy(True), _field(ALAKAZAM)
        elif kind == "base_vs_alakazam":
            A, B = _heavy(False), _field(ALAKAZAM)
        elif kind == "cont_vs_denpa92":
            A, B = _heavy(True), _field(DENPA92)
        else:  # base_vs_denpa92
            A, B = _heavy(False), _field(DENPA92)
        pair = [A, B] if a_seat == 0 else [B, A]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt"); env.run(pair)
            w = _winner(env)
        except Exception:
            dr += 1; continue
        if w is None:
            dr += 1
        elif w == a_seat:
            wa += 1
        else:
            wb += 1
    return (kind, wa, wb, dr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=3)
    a = ap.parse_args()
    kinds = ["mirror", "cont_vs_alakazam", "base_vs_alakazam", "cont_vs_denpa92", "base_vs_denpa92"]
    tasks = []
    for kind in kinds:
        rem, seat = a.games, 0
        while rem > 0:
            k = min(a.chunk, rem); tasks.append((kind, k, seat, a.budget)); seat = (seat + k) % 2; rem -= k
    print(f"leaf A/B: continuity vs baseline | n={a.games}/matchup budget={a.budget}s", flush=True)
    agg = {k: [0, 0, 0] for k in kinds}
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_winit, initargs=(a.budget,)) as ex:
        for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
            done += 1
            kind, wa, wb, dr = f.result()
            agg[kind][0] += wa; agg[kind][1] += wb; agg[kind][2] += dr
            if done % 10 == 0:
                print(f"  [{done}/{len(tasks)}] {kind}: +{agg[kind][0]}-{agg[kind][1]}", flush=True)
    print("\n=== RESULTS (A = first-named) ===")
    def rate(w, l):
        return round(100 * w / max(1, w + l), 1)
    m = agg["mirror"]
    print(f"  MIRROR  continuity vs baseline : {m[0]}-{m[1]}  ({rate(m[0],m[1])}% for continuity, draws {m[2]})")
    for opp in ("alakazam", "denpa92"):
        c, b = agg[f"cont_vs_{opp}"], agg[f"base_vs_{opp}"]
        print(f"  vs {opp:9}: continuity {c[0]}-{c[1]} ({rate(c[0],c[1])}%) | baseline {b[0]}-{b[1]} ({rate(b[0],b[1])}%)  delta {rate(c[0],c[1])-rate(b[0],b[1]):+.1f}")


if __name__ == "__main__":
    main()
