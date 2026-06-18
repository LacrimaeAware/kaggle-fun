"""Self-play data logger (REAL cabt engine): (state-features -> outcome) rows for the learner.

This is the data source for L2 (the learned value). For every decision a logged agent makes we
record the L1 feature vector of that state, tagged with which seat made it and the turn. When
the game ends we label every row with that seat's result (win=1, loss=0, draw=0.5). That is a
Monte-Carlo value target: "from this position, did the player to move go on to win?"

Both seats are logged, each labelled from its own side, so one game yields ~2x rows, balanced.
(Distinct from the legacy mock-based selfplay.py, and from cabt_arena.py which only scores.)

    python agent/datagen.py --games 200 --out ../data/selfplay/v0.jsonl

Rows (JSONL): {"feat":[47 floats], "y":1.0|0.0|0.5, "turn":int, "seat":0|1}. Feature order is
features.FEATURE_KEYS, also written to a sidecar <out>.keys.json for the trainer.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import random
import sys

logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt
    from kaggle_environments import make

import features as FT
import main as M


def _winner(env):
    last = env.steps[-1]
    r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def _logged(base_agent, seat: int, sink: list):
    """Wrap an agent so it appends ONE feature row at the start of each real turn -- the first
    decision of a new turn with a dealt board and an active Pokemon. That is the same state class
    the search leaf evaluates (control just returned to me). Logging every mid-turn sub-decision
    instead would train the value on a different distribution than search queries, and logging
    turn-0 setup states (no active) would label garbage with the eventual outcome."""
    state = {"last_turn": None}

    def f(obs):
        cur = obs.get("current")
        if obs.get("select") is not None and cur is not None:
            t = cur.get("turn", 0)
            if t >= 1 and t != state["last_turn"]:       # first decision of a new turn
                players = cur.get("players") or []
                yi = cur.get("yourIndex", 0)
                me = players[yi] if yi < len(players) else {}
                act = me.get("active") or []
                if act and act[0]:                       # real position, not setup
                    state["last_turn"] = t
                    try:
                        sink.append({"feat": FT.vectorize(FT.encode_state(obs)),
                                     "turn": t, "seat": seat})
                    except Exception:
                        pass
        return base_agent(obs)
    return f


def _explore(agent, eps: float):
    """With probability eps, play a random legal move instead of `agent` -- adds state variety so
    winning and losing positions separate (mirror-skilled self-play stays near 50/50 and the value
    cannot learn from states that all look like coin flips)."""
    def f(obs):
        if eps > 0 and obs.get("select") is not None and random.random() < eps:
            return cabt.random_agent(obs)
        return agent(obs)
    return f


def generate(games: int, out_path: str, agent_a=None, agent_b=None,
             eps: float = 0.0, append: bool = False, gid_offset: int = 0) -> dict:
    a = _explore(agent_a or M.agent, eps)
    b = _explore(agent_b or M.agent, eps)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n_rows = decided = 0
    with open(out_path, "a" if append else "w", encoding="utf-8") as fh:
        for g in range(games):
            rows: list = []
            # swap which agent sits in seat 0 each game so the corpus is not skewed toward
            # first-player positions (seat 0 nearly always elects to go first). The seat label
            # passed to _logged follows the actual board seat, so labels stay correct.
            wrapped = ([_logged(a, 0, rows), _logged(b, 1, rows)] if g % 2 == 0
                       else [_logged(b, 0, rows), _logged(a, 1, rows)])
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    env = make("cabt")
                    env.run(wrapped)
                w = _winner(env)
            except Exception:
                continue
            if w is not None:
                decided += 1
            for r in rows:
                r["y"] = 0.5 if w is None else (1.0 if w == r["seat"] else 0.0)
                r["gid"] = gid_offset + g                 # globally-unique game id for the split
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
                n_rows += 1
    with open(out_path + ".keys.json", "w", encoding="utf-8") as kf:
        json.dump(FT.FEATURE_KEYS, kf)
    return {"games": games, "decided": decided, "rows": n_rows, "out": out_path}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "selfplay", "v0.jsonl"))
    choices = ["heuristic", "first", "random", "search", "search_v"]
    ap.add_argument("--a", default="heuristic", choices=choices)
    ap.add_argument("--b", default="heuristic", choices=choices)
    ap.add_argument("--eps", type=float, default=0.0, help="exploration: prob of a random legal move")
    ap.add_argument("--append", action="store_true", help="append to --out (build a mixed corpus)")
    ap.add_argument("--gid-offset", type=int, default=0, help="add to game ids so appended runs stay unique")
    args = ap.parse_args()
    pick = {"heuristic": M.agent, "first": cabt.first_agent, "random": cabt.random_agent,
            "search": M.agent_search, "search_v": M.agent_search_v}
    r = generate(args.games, args.out, pick[args.a], pick[args.b],
                 eps=args.eps, append=args.append, gid_offset=args.gid_offset)
    print(f"datagen: {r['games']} games ({r['decided']} decided), {r['rows']} rows -> {r['out']} (append={args.append})")


if __name__ == "__main__":
    main()
