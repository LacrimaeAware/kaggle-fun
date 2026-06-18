"""Search-bootstrapped value data (the research's #1 fix for poor local leaf ranking).

Instead of labeling a state with the eventual game outcome (Monte-Carlo, which ranks nearby
moves poorly), label each start-of-turn state with the SEARCH's own backed-up value at that state
(A0GB-style: max over options of the determinization-averaged leaf value). The value model then
learns to predict what the search thinks a state is worth, which inherently encodes the local
sibling ranking search needs. Iterating (retrain -> search with the new value -> regenerate)
is expert iteration; this produces iteration 1 from the hand-eval search.

Rows (JSONL): {"feat":[47], "y":<logit target = search_value/BLEND_SCALE>, "turn", "seat", "gid"}.
y is a LOGIT (value_model applies the sigmoid), so train with tools/train_value.py --target value.

    python tools/datagen_bootstrap.py --games 150 --out data/selfplay/boot.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import sys

logging.disable(logging.CRITICAL)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt
    from kaggle_environments import make

import features as FT
import main as M
import search as S
import eval as EV

SCALE = EV.BLEND_SCALE     # share the squash scale so the logit target matches the blend's


def _target_logit(val: float, leaf: str) -> float:
    """Convert the search's backed-up value to a logit target. For leaf="hand" the value is in
    hand-eval scale (matches the blend's sigmoid(val/SCALE)); for leaf="learned"/"blend" the
    value is already a probability in [0,1], so take its logit."""
    if leaf == "hand":
        return float(val) / SCALE
    p = min(1.0 - 1e-4, max(1e-4, float(val)))
    return math.log(p / (1.0 - p))


def _logged_search(seat: int, sink: list, deck: list, leaf: str, policy):
    """Play with `policy`; at the first decision of each real turn, log the search's backed-up
    value (as a logit target) for that state. Pass 1 uses leaf="hand" + the hand search; later
    expert-iteration passes use leaf="learned"/"blend" + the current learned agent."""
    state = {"last_turn": None}

    def f(obs):
        sel = obs.get("select")
        cur = obs.get("current")
        if sel is not None and cur is not None:
            t = cur.get("turn", 0)
            players = cur.get("players") or []
            yi = cur.get("yourIndex", 0)
            me = players[yi] if yi < len(players) else {}
            act = me.get("active") or []
            if t >= 1 and t != state["last_turn"] and act and act[0]:
                try:
                    mv, val = S.best_option_value(obs, deck, leaf_mode=leaf)
                    if val is not None:
                        state["last_turn"] = t
                        sink.append({"feat": FT.vectorize(FT.encode_state(obs)),
                                     "y": _target_logit(val, leaf), "turn": t, "seat": seat})
                        if mv is not None:
                            return mv
                except Exception:
                    pass
        return policy(obs)
    return f


def generate(games: int, out_path: str, deck: list, leaf: str = "hand", policy=None) -> dict:
    policy = policy or M.agent_search
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for g in range(games):
            rows: list = []
            wrapped = [_logged_search(0, rows, deck, leaf, policy), _logged_search(1, rows, deck, leaf, policy)]
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    env = make("cabt")
                    env.run(wrapped)
            except Exception:
                continue
            for r in rows:
                r["gid"] = g
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
                n_rows += 1
    with open(out_path + ".keys.json", "w", encoding="utf-8") as kf:
        json.dump(FT.FEATURE_KEYS, kf)
    return {"games": games, "rows": n_rows, "out": out_path}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "selfplay", "boot.jsonl"))
    ap.add_argument("--leaf", choices=["hand", "learned", "blend"], default="hand",
                    help="leaf eval the search uses for the bootstrap target (pass 1 = hand)")
    ap.add_argument("--policy", choices=["search", "search_v", "combine"], default="search",
                    help="data-generating agent (later passes use the current learned agent)")
    args = ap.parse_args()
    pol = {"search": M.agent_search, "search_v": M.agent_search_v, "combine": M.agent_combine}[args.policy]
    r = generate(args.games, args.out, list(M.DECK), leaf=args.leaf, policy=pol)
    print(f"datagen_bootstrap: {r['games']} games, {r['rows']} rows -> {r['out']} (leaf={args.leaf}, policy={args.policy})")


if __name__ == "__main__":
    main()
