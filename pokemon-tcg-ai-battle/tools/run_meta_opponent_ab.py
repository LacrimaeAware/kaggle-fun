"""Does a real opponent model make depth pay off? A/B over the opponent belief and search depth.

All variants are the same PH-aware search-driven agent (force game-winner + go-first, then search), on
the same new deck. They differ in two knobs only:
  - opponent belief: our-own-deck (the current wrong assumption) vs sampled from the replay meta.
  - depth: 1-ply (opp_k=0) vs 2-ply (opp_k=2).

Hypothesis: 2-ply lost before because the opponent model was garbage (hidden cards drawn from our deck).
With a real meta belief, depth should stop hurting and may help.

Reads pokemon-ai-agent (new deck) and this repo's data/opponent_meta_v1.json, both read-only.
  python tools/run_meta_opponent_ab.py --games 120 --matchups sd_meta_vs_sd_ourdeck,sd_meta2_vs_sd_meta,sd_ourdeck2_vs_sd_ourdeck
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
META = ROOT / "data" / "opponent_meta_v1.json"
TOP_K = 25
YES, IS_FIRST = 1, 41

VARIANTS = {
    "sd_ourdeck":  dict(meta=False, opp_k=0),
    "sd_meta":     dict(meta=True, opp_k=0),
    "sd_ourdeck2": dict(meta=False, opp_k=2),
    "sd_meta2":    dict(meta=True, opp_k=2),
}
_D: dict = {}


def _load() -> dict:
    if _D:
        return _D
    sys.path.insert(0, str(ROOT / "agent"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make  # noqa
        import search_v3 as S  # noqa
        import deck_policy_v3 as DP3  # noqa
    S.USE_DYNAMIC_ATTACKS = True
    deck = [int(x) for x in NEW_DECK.read_text(encoding="utf-8").split() if x.strip()]
    meta = json.loads(META.read_text(encoding="utf-8"))
    opp_decks = [d["deck"] for d in meta["decks"][:TOP_K]]
    opp_weights = [d["count"] for d in meta["decks"][:TOP_K]]
    _D.update(make=make, S=S, DP3=DP3, deck=deck, opp_decks=opp_decks, opp_weights=opp_weights)
    return _D


def _legal(sel):
    opts = sel.get("option") or []
    k = sel.get("maxCount") or 0
    mn = sel.get("minCount") or 0
    n = len(opts)
    return list(range(max(min(k, n), min(mn, n)))) if (n and k > 0) else []


def _make_agent(name: str, D: dict):
    S, DP3, deck = D["S"], D["DP3"], D["deck"]
    cfg = VARIANTS[name]
    opp_decks = D["opp_decks"] if cfg["meta"] else None
    opp_weights = D["opp_weights"] if cfg["meta"] else None
    opp_k = cfg["opp_k"]

    def agent(obs):
        if obs.get("select") is None:
            return list(deck)
        sel = obs.get("select") or {}
        try:
            if sel.get("context") == IS_FIRST:
                for i, o in enumerate(sel.get("option") or []):
                    if isinstance(o, dict) and o.get("type") == YES:
                        return [i]
            ko = DP3.best_ko_attack(obs)
            if ko is not None and ko[1].get("game_win"):
                return [ko[0]]
            mv = S.best_option(obs, deck, leaf_mode="hand", opp_k=opp_k,
                               opp_decks=opp_decks, opp_weights=opp_weights)
            if mv is not None:
                return mv
        except Exception:
            pass
        return _legal(sel)
    return agent


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
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--matchups", default="sd_meta_vs_sd_ourdeck,sd_meta2_vs_sd_meta,sd_ourdeck2_vs_sd_ourdeck")
    ap.add_argument("--out", default=str(ROOT / "data" / "meta_opponent_ab.json"))
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

    payload = dict(games_per_matchup=args.games, workers=args.workers, top_k=TOP_K,
                   seconds=round(time.time() - t0, 1), matchups=results)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({payload['seconds']}s, {args.workers}w)", flush=True)


if __name__ == "__main__":
    main()
