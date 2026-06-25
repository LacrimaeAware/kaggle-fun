"""Find a heuristic that improves local win rate, on the new deck, in this repo.

The deck is an evolution engine; the plain heuristic plays option-0 for development (tutor/evolve/play),
which cannot pilot it. This A/Bs the kaggle-fun agent variants to see which heuristic layer actually helps:
  choose  = main.agent      (lethal KO + energy + go-first, else default option order)
  eff     = main.agent_eff  (adds effect-aware development: values search/draw/evolve/energy-accel plays)
  search  = main.agent_search (forward-model search)
  first   = legal fallback baseline

New deck read from pokemon-ai-agent (read-only). Seat-alternated, multiprocessed, CI + two-sided p.
  python tools/run_heuristic_ab_v1.py --games 300 --matchups eff_vs_choose,eff_vs_first,choose_vs_first
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW_DECK = Path(r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/decks/current_deck.csv")
VARIANTS = ("first", "choose", "eff", "search", "phaware", "phaware_search", "phaware_search_ca",
            "phaware_search_deckout", "phaware_search_ph", "phaware_search_v3", "phaware_search_dev",
            "phaware_search_planner", "heuristic_first")
# phaware_search_<x> variants: same PH-aware KO floor + search, differing only in the leaf eval used for
# development. _LEAF_MODE maps the variant suffix to the search leaf_mode.
_LEAF_MODE = {"phaware_search_deckout": "deckout", "phaware_search_ph": "ph", "phaware_search_v3": "deck"}
_D: dict = {}


def _load() -> dict:
    if _D:
        return _D
    sys.path.insert(0, str(ROOT / "agent"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make  # noqa
        import main as M  # noqa
        import deck_policy_v3 as DP3  # noqa  (PH-aware best_ko_attack)
        import search_v3 as S  # noqa  (PH-aware forward-model search)
    S.USE_DYNAMIC_ATTACKS = True
    deck = [int(x) for x in NEW_DECK.read_text(encoding="utf-8").split() if x.strip()]
    M.DECK = deck
    _D.update(make=make, M=M, DP3=DP3, S=S, deck=deck)
    return _D


def _legal_first(deck):
    def agent(obs):
        sel = obs.get("select")
        if sel is None:
            return list(deck)
        opts = sel.get("option") or []
        need = (sel.get("minCount") or 1)
        return list(range(min(max(need, 1), len(opts)))) if opts else []
    return agent


def _make_agent(name: str, D: dict):
    M, deck = D["M"], D["deck"]
    if name == "first":
        return _legal_first(deck)
    if name == "choose":
        return M.agent
    if name == "eff":
        return M.agent_eff
    if name == "search":
        return M.agent_search
    if name == "phaware":
        DP3 = D["DP3"]
        def agent(obs):
            if obs.get("select") is None:
                return list(deck)
            try:
                ko = DP3.best_ko_attack(obs)   # PH-aware: sees Powerful Hand KOs that _choose misses
                if ko is not None:
                    return [ko[0]]
            except Exception:
                pass
            return M.agent(obs)                 # else fall to the plain heuristic (energy/go-first/default)
        return agent
    if name == "phaware_search":
        DP3, S = D["DP3"], D["S"]
        def agent(obs):
            if obs.get("select") is None:
                return list(deck)
            try:
                ko = DP3.best_ko_attack(obs)    # heuristic takes the KO (PH-aware)
                if ko is not None:
                    return [ko[0]]
                mv = S.best_option(obs, deck, leaf_mode="hand")   # search picks the developmental move
                if mv is not None:
                    return mv
            except Exception:
                pass
            return M.agent(obs)
        return agent
    if name == "phaware_search_ca":
        DP3, S = D["DP3"], D["S"]
        def agent(obs):
            if obs.get("select") is None:
                return list(deck)
            try:
                ko = DP3.best_ko_attack(obs)
                if ko is not None:
                    return [ko[0]]
                mv = S.best_option(obs, deck, leaf_mode="ca")     # search maximizes CARD ADVANTAGE
                if mv is not None:
                    return mv
            except Exception:
                pass
            return M.agent(obs)
        return agent
    if name in ("phaware_search_dev", "phaware_search_planner"):
        DP3, S = D["DP3"], D["S"]
        # dev = develop rollout with the plain hand leaf (isolates the rollout); planner = the full config,
        # develop rollout + the validated deck-out leaf (the all-findings candidate to deploy).
        leaf = "deckout" if name == "phaware_search_planner" else "hand"
        def agent(obs):
            if obs.get("select") is None:
                return list(deck)
            try:
                ko = DP3.best_ko_attack(obs)
                if ko is not None:
                    return [ko[0]]
                mv = S.best_option(obs, deck, leaf_mode=leaf, rollout_mode="develop")
                if mv is not None:
                    return mv
            except Exception:
                pass
            return M.agent(obs)
        return agent
    if name == "heuristic_first":
        # the vendored heuristic-first submission agent (hiroingk registry + search_v3 fallback + legal)
        import sys as _sys
        sub = str(ROOT / "submissions" / "sub_heuristic")
        if sub not in _sys.path:
            _sys.path.insert(0, sub)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import agent_impl as AI
        return AI.agent
    if name in _LEAF_MODE:
        DP3, S = D["DP3"], D["S"]
        mode = _LEAF_MODE[name]
        def agent(obs):
            if obs.get("select") is None:
                return list(deck)
            try:
                ko = DP3.best_ko_attack(obs)
                if ko is not None:
                    return [ko[0]]
                mv = S.best_option(obs, deck, leaf_mode=mode)   # deck-aware leaf (deckout / ph / both)
                if mv is not None:
                    return mv
            except Exception:
                pass
            return M.agent(obs)
        return agent
    raise ValueError(name)


def _winner(env):
    last = env.steps[-1]
    r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def run_chunk(task):
    a_name, b_name, n, seat0 = task
    D = _load()
    make = D["make"]
    A, B = _make_agent(a_name, D), _make_agent(b_name, D)
    wa = wb = dr = er = 0
    for i in range(n):
        a_seat = (seat0 + i) % 2
        pair = [A, B] if a_seat == 0 else [B, A]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env = make("cabt")
                env.run(pair)
            w = _winner(env)
        except Exception:
            er += 1
            continue
        if w is None:
            dr += 1
        elif w == a_seat:
            wa += 1
        else:
            wb += 1
    return (wa, wb, dr, er)


def _ci(p, n):
    if n == 0:
        return (0.0, 0.0)
    se = math.sqrt(p * (1 - p) / n)
    return (p - 1.96 * se, p + 1.96 * se)


def _pval(wa, n):
    if n == 0:
        return 1.0
    z = (wa / n - 0.5) / math.sqrt(0.25 / n)
    return math.erfc(abs(z) / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--matchups", default="eff_vs_choose,eff_vs_first,choose_vs_first")
    ap.add_argument("--out", default=str(ROOT / "data" / "heuristic_ab_v1.json"))
    args = ap.parse_args()

    matchups = []
    for m in args.matchups.split(","):
        a, b = m.split("_vs_", 1)
        assert a in VARIANTS and b in VARIANTS, f"unknown variant in {m}"
        matchups.append((a, b))

    per = max(2, ((args.games // args.workers) // 2) * 2)
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for (a, b) in matchups:
            remaining = args.games
            chunks = []
            while remaining > 0:
                k = min(per, remaining)
                if k % 2 == 1:
                    k += 1
                chunks.append(k)
                remaining -= k
            futures[(a, b)] = [ex.submit(run_chunk, (a, b, k, 0)) for k in chunks]
        for (a, b), futs in futures.items():
            wa = wb = dr = er = 0
            for f in futs:
                cwa, cwb, cdr, cer = f.result()
                wa += cwa; wb += cwb; dr += cdr; er += cer
            decided = wa + wb
            p = wa / decided if decided else 0.0
            lo, hi = _ci(p, decided)
            pv = _pval(wa, decided)
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
            results[f"{a}_vs_{b}"] = dict(a=a, b=b, wins_a=wa, wins_b=wb, draws=dr, errors=er,
                                         decided=decided, a_winrate=round(p, 4),
                                         ci95=[round(lo, 4), round(hi, 4)], p_value=round(pv, 4), sig=sig)
            print(f"{a}_vs_{b}: A={p:.3f} [{lo:.3f},{hi:.3f}] ({wa}-{wb}, {dr}d {er}e) p={pv:.4f} {sig}", flush=True)

    payload = dict(games_per_matchup=args.games, workers=args.workers, seconds=round(time.time() - t0, 1), matchups=results)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({payload['seconds']}s, {args.workers}w)", flush=True)


if __name__ == "__main__":
    main()
