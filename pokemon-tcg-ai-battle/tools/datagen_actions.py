"""Candidate-ACTION data: per decision, log EVERY legal option's leaf features + value.

The diagnosis (docs/RESEARCH.md): we only ever trained a STATE value (rank positions), so it ranks
NEARBY sibling moves poorly. This logs the sibling leaves of each decision (search.option_evals),
so a model can be trained on within-decision differences (the action-ranking objective, H024).

Each option of a multi-option decision becomes one row, grouped by a decision id (gid) so a
game-wise split holds whole decisions together (no sibling leakage):
  {"feat":[47 leaf features], "y":<leaf value / BLEND_SCALE logit>, "gid":<decision id>, "turn", "seat"}

    python tools/datagen_actions.py --games 150 --out data/selfplay/actions.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys

logging.disable(logging.CRITICAL)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt
    from kaggle_environments import make

import main as M
import search as S
import eval as EV
import features as FT

SCALE = EV.BLEND_SCALE


def _logged(seat: int, sink: list, deck: list, did: list):
    def f(obs):
        sel, cur = obs.get("select"), obs.get("current")
        if sel is not None and cur is not None and (sel.get("maxCount") or 0) == 1 and len(sel.get("option") or []) >= 2:
            t = cur.get("turn", 0)
            players = cur.get("players") or []
            yi = cur.get("yourIndex", 0)
            me = players[yi] if yi < len(players) else {}
            act = me.get("active") or []
            if t >= 1 and act and act[0]:                # every real multi-option decision (leaves are start-of-next-turn)
                try:
                    evs = S.option_evals(obs, deck, leaf_mode="hand")   # per-option (value, features)
                    if evs:
                        did[0] += 1
                        best_i, best_v = None, None
                        for i, e in enumerate(evs):
                            if e is None:
                                continue
                            val, feats = e
                            sink.append({"feat": feats, "y": float(val) / SCALE,
                                         "gid": did[0], "turn": t, "seat": seat})
                            if best_v is None or val > best_v:
                                best_v, best_i = val, i
                        if best_i is not None:
                            return [best_i]
                except Exception:
                    pass
        return M.agent_search(obs)
    return f


def generate(games: int, out_path: str, deck: list) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n_rows = 0
    did = [0]
    with open(out_path, "w", encoding="utf-8") as fh:
        for _ in range(games):
            rows: list = []
            wrapped = [_logged(0, rows, deck, did), _logged(1, rows, deck, did)]
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    env = make("cabt")
                    env.run(wrapped)
            except Exception:
                continue
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
                n_rows += 1
    with open(out_path + ".keys.json", "w", encoding="utf-8") as kf:
        json.dump(FT.FEATURE_KEYS, kf)
    return {"games": games, "rows": n_rows, "decisions": did[0], "out": out_path}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "selfplay", "actions.jsonl"))
    args = ap.parse_args()
    r = generate(args.games, args.out, list(M.DECK))
    print(f"datagen_actions: {r['games']} games, {r['decisions']} decisions, {r['rows']} option-rows -> {r['out']}")


if __name__ == "__main__":
    main()
