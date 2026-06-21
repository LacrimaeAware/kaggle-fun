"""Log what each agent actually CHOOSES in one head-to-head game. Reveals behavioral asymmetry:
e.g. if the candidate rarely attacks or rarely evolves, the deck_policy intercept is making it
passive. Per agent: histogram of chosen option type, and attack-availability vs attack-taken."""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_candidate_v1 as AB

OPT = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL", 5: "ENERGY_CARD", 6: "ENERGY",
       7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT",
       13: "ATTACK", 14: "END"}


def make_logger(label, fn, stats):
    def ag(obs):
        ret = fn(obs)
        try:
            sel = obs.get("select")
            if sel is not None:
                opts = sel.get("option") or []
                k = sel.get("maxCount") or 0
                if k == 1 and isinstance(ret, list) and ret and 0 <= ret[0] < len(opts):
                    ch = opts[ret[0]].get("type")
                    stats[label][OPT.get(ch, ch)] += 1
                    types = {o.get("type") for o in opts}
                    if 13 in types:
                        stats[label + "_atkavail"]["available"] += 1
                        if ch == 13:
                            stats[label + "_atkavail"]["took_attack"] += 1
                    if 9 in types:
                        stats[label + "_evoavail"]["available"] += 1
                        if ch == 9:
                            stats[label + "_evoavail"]["took_evolve"] += 1
                else:
                    stats[label]["MULTI_or_setup"] += 1
        except Exception:
            pass
        return ret
    return ag


def main():
    AB.build_candidate_pkg()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "agent"))
    base_main = importlib.import_module("main")
    cand_main = importlib.import_module("_candv1.main")
    PILOT = AB.pilot_deck()
    base_main.DECK = PILOT
    cand_main.DECK = PILOT

    stats = defaultdict(Counter)
    cand = make_logger("cand", cand_main.agent_search, stats)
    base = make_logger("base", base_main.agent_search, stats)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = AB.make("cabt")
        env.run([cand, base])   # candidate seat 0
    w = AB.winner_of(env)

    print(f"winner seat: {w}  (0 = candidate)\n")
    for label in ("cand", "base"):
        print(f"=== {label} chosen-option histogram ===")
        for t, n in stats[label].most_common():
            print(f"  {t:16s} {n}")
        av = stats[label + "_atkavail"]
        ev = stats[label + "_evoavail"]
        print(f"  ATTACK available {av['available']}, took {av['took_attack']}")
        print(f"  EVOLVE available {ev['available']}, took {ev['took_evolve']}")
        print()


if __name__ == "__main__":
    main()
