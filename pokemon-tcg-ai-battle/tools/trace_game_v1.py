"""Trace ONE head-to-head game to see whether the candidate's search path actually runs or
silently falls back. The candidate's _agent_search wraps its whole new path in `except: pass`,
so a throw in root_option_priors / choose_subdecision / best_option would make it play the weak
heuristic every turn while the baseline runs full search. We re-raise from inside (after logging)
so the normal fallback still happens, but we capture the first few tracebacks."""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_candidate_v1 as AB


def main():
    AB.build_candidate_pkg()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "agent"))
    base_main = importlib.import_module("main")
    cand_main = importlib.import_module("_candv1.main")
    cDP = importlib.import_module("_candv1.deck_policy")
    cS = importlib.import_module("_candv1.search")
    bS = importlib.import_module("search")
    PILOT = AB.pilot_deck()
    base_main.DECK = PILOT
    cand_main.DECK = PILOT

    C = Counter()
    exc = []

    def log_exc(where, e):
        C[where + "_exc"] += 1
        if len(exc) < 4:
            exc.append(f"{where} -> {e!r}\n" + traceback.format_exc()[-1100:])

    o_priors = cDP.root_option_priors
    def priors_t(obs, perspective=None):
        try:
            return o_priors(obs, perspective)
        except Exception as e:
            log_exc("cand_priors", e); raise
    cDP.root_option_priors = priors_t

    o_sub = cDP.choose_subdecision
    def sub_t(obs, perspective=None):
        try:
            r = o_sub(obs, perspective)
            if r is not None:
                C["cand_subdecision"] += 1
            return r
        except Exception as e:
            log_exc("cand_subdecision", e); raise
    cDP.choose_subdecision = sub_t

    o_cbest = cS.best_option
    def cbest_t(*a, **k):
        C["cand_search_call"] += 1
        try:
            r = o_cbest(*a, **k)
            if r is None:
                C["cand_search_none"] += 1
            return r
        except Exception as e:
            log_exc("cand_best_option", e); raise
    cS.best_option = cbest_t

    o_cag = cand_main.agent
    def cag(obs):
        if obs.get("select") is not None:
            C["cand_fallback"] += 1
        return o_cag(obs)
    cand_main.agent = cag

    o_bbest = bS.best_option
    def bbest_t(*a, **k):
        C["base_search_call"] += 1
        r = o_bbest(*a, **k)
        if r is None:
            C["base_search_none"] += 1
        return r
    bS.best_option = bbest_t

    o_bag = base_main.agent
    def bag(obs):
        if obs.get("select") is not None:
            C["base_fallback"] += 1
        return o_bag(obs)
    base_main.agent = bag

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = AB.make("cabt")
        env.run([cand_main.agent_search, base_main.agent_search])
    w = AB.winner_of(env)

    print(f"winner seat: {w}  (0 = candidate)")
    print("counters:")
    for k in sorted(C):
        print(f"  {k}: {C[k]}")
    print("\n--- first candidate-path exceptions (swallowed by _agent_search in real play) ---")
    if not exc:
        print("  (none -> candidate search path ran clean; bug is elsewhere)")
    for e in exc:
        print(e)
        print("-" * 50)


if __name__ == "__main__":
    main()
