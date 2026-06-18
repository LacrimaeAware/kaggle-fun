"""Diagnostic: tally which option types each agent actually chooses, to see why the
heuristic underperforms first_agent. Not a permanent tool."""
from __future__ import annotations
import contextlib, io, logging
from collections import Counter
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt
    from kaggle_environments import make
import main as M

TYPE = {0:"NUMBER",1:"YES",2:"NO",3:"CARD",4:"TOOL",5:"ECARD",6:"ENERGY",7:"PLAY",8:"ATTACH",
        9:"EVOLVE",10:"ABILITY",11:"DISCARD",12:"RETREAT",13:"ATTACK",14:"END",15:"SKILL",16:"SPC"}

tally = {"heur": Counter(), "first": Counter()}
ndec = {"heur": 0, "first": 0}

def wrap(name, fn):
    def g(obs):
        r = fn(obs)
        sel = obs.get("select")
        if sel is not None:
            opts = sel.get("option") or []
            for i in (r or []):
                if 0 <= i < len(opts):
                    tally[name][TYPE.get(opts[i].get("type"), opts[i].get("type"))] += 1
            ndec[name] += 1
        return r
    return g

def first_sd(obs):
    return list(M.DECK) if obs.get("select") is None else cabt.first_agent(obs)

H = wrap("heur", M.agent)
F = wrap("first", first_sd)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    for g in range(30):
        e = make("cabt"); e.run([H, F] if g % 2 == 0 else [F, H])
for who in ("heur", "first"):
    tot = sum(tally[who].values()) or 1
    print(f"\n{who}: {ndec[who]} decisions, {tot} picks")
    for t, n in tally[who].most_common():
        print(f"  {t:8s} {n:5d}  {100*n/tot:4.1f}%")
