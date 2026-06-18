"""Probe the real cabt engine: confirm it runs locally, dump the observation schema, and
test whether a forward model is reachable from inside an agent (registry H1).

Run with the venv that has kaggle-environments:
    python explore_cabt.py
Writes a short schema summary to stdout. This is exploration, not the agent.
"""
from __future__ import annotations

import json
import traceback

import kaggle_environments.envs.cabt.cabt as cabt
from kaggle_environments import make


def banner(s: str) -> None:
    print("\n" + "=" * 8 + " " + s + " " + "=" * 8)


banner("cabt module public API")
api = [a for a in dir(cabt) if not a.startswith("_")]
print(api)
print("has module-level 'deck':", hasattr(cabt, "deck"))
for name in ("search_begin", "search_step", "all_card_data", "all_attack",
             "battle_start", "battle_select", "random_agent", "first_agent"):
    print(f"  {name}: {'yes' if hasattr(cabt, name) else 'no'}")

# Try the card data API (used to build a deck and the heuristic features)
banner("card data")
try:
    cards = cabt.all_card_data()
    print("all_card_data() -> type", type(cards).__name__, "len", len(cards) if hasattr(cards, "__len__") else "?")
    sample = cards[0] if isinstance(cards, (list, tuple)) else next(iter(cards))
    print("first card:", json.dumps(sample, default=str)[:400])
except Exception as e:
    print("all_card_data() failed:", e)

# A probe agent that records the observation schema for the first few decisions.
captured: list[dict] = []


def probe(obs):
    rec = {"keys": sorted(obs.keys())}
    sel = obs.get("select")
    if sel is None:
        rec["phase"] = "deck"
    else:
        rec["phase"] = "play"
        rec["maxCount"] = sel.get("maxCount")
        opt = sel.get("option")
        rec["n_options"] = len(opt) if opt else 0
        rec["option_0"] = opt[0] if opt else None
        rec["option_1"] = opt[1] if opt and len(opt) > 1 else None
    cur = obs.get("current")
    rec["current_type"] = type(cur).__name__
    if isinstance(cur, dict):
        rec["current_keys"] = sorted(cur.keys())
    if len(captured) < 12:
        captured.append(rec)
    # behave like first_agent, with a valid deck in the deck phase
    if sel is None:
        return getattr(cabt, "deck", list(range(60)))
    return list(range(sel["maxCount"]))


banner("run a real match: probe vs random_agent")
try:
    env = make("cabt", debug=True)
    out = env.run([probe, cabt.random_agent])
    final = out[-1]
    print("match ran. final-step rewards:",
          [s.get("reward") for s in final] if isinstance(final, list) else final)
    print("captured", len(captured), "observation snapshots")
    for r in captured:
        print(json.dumps(r, default=str)[:600])
except Exception:
    print("match run FAILED:")
    traceback.print_exc()

# Forward-model test (registry H1): can an agent call the search API to look ahead?
banner("forward-model / search API reachability (H1)")
for name in ("search_begin", "search_step"):
    fn = getattr(cabt, name, None)
    print(f"{name}: {'callable' if callable(fn) else 'absent'}")
print("Interpretation: if search_begin/search_step are callable from inside an agent and "
      "operate on the agent's observation, a forward model exists for planning. If they "
      "are interpreter-internal only, the agent is opponent-stepped and must roll its own "
      "next-state model. Confirm by trying to call them from an agent next.")
