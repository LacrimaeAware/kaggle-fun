"""Parse a downloaded cabt replay JSON into a readable summary.

A Kaggle cabt replay is one episode: top-level `steps` is a list of per-step pairs
[agent0, agent1], each with `action` (the indices that agent returned), `observation`
(that agent's view, including `select` with the options offered), `reward`, `status`.

What this extracts, and why it matters:
  - Both players' DECKS (the 60-card deck-selection action), as card-id -> count. For a real
    opponent match this is the opponent's deck in the engine's own card IDs, which is the
    bridge between the real-game decklists we downloaded and the contest's card pool.
  - The result and game length.
  - A per-player histogram of what kinds of actions were taken (decoded option types:
    13 attack, 14 end turn, 8 target in-play, 3 select card, 7 pick, 1/2 menu).
  - Whether it is a self-play game (both decks identical), e.g. the submission self-validation.

Usage: python tools/parse_replay.py <path-to-replay.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter

TYPE_NAME = {13: "attack", 14: "end-turn", 8: "target-inplay", 3: "select-card",
             7: "pick", 9: "target2", 0: "number", 1: "menu1", 2: "menu2"}


def deck_counts(action: list[int]) -> Counter:
    return Counter(action)


def parse(path: str) -> None:
    rep = json.load(open(path, encoding="utf-8"))
    print(f"replay: {path}")
    print(f"  env: {rep.get('name')}  id: {rep.get('id')}")
    print(f"  result rewards: {rep.get('rewards')}  statuses: {rep.get('statuses')}")
    steps = rep.get("steps") or []
    print(f"  steps: {len(steps)}")

    decks = {0: None, 1: None}
    action_types = {0: Counter(), 1: Counter()}
    n_decisions = {0: 0, 1: 0}

    for s in steps:
        for ai, agent in enumerate(s):
            if not isinstance(agent, dict):
                continue
            act = agent.get("action")
            if not isinstance(act, list) or not act:
                continue
            if len(act) == 60 and decks[ai] is None:        # deck-selection
                decks[ai] = act
                continue
            n_decisions[ai] += 1
            obs = agent.get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            for idx in act:
                if isinstance(idx, int) and 0 <= idx < len(opts):
                    action_types[ai][opts[idx].get("type")] += 1

    won = None
    rewards = rep.get("rewards") or []
    if len(rewards) == 2 and rewards[0] != rewards[1]:
        won = 0 if rewards[0] > rewards[1] else 1

    for ai in (0, 1):
        d = decks[ai]
        tag = "  <-- WON" if won == ai else ""
        print(f"\nagent {ai}{tag}: {n_decisions[ai]} decisions")
        if d:
            c = deck_counts(d)
            print(f"  deck: {len(d)} cards, {len(c)} distinct ids")
            print("  composition (id x count):", ", ".join(f"{k}x{v}" for k, v in sorted(c.items(), key=lambda x: -x[1])))
        if action_types[ai]:
            hist = ", ".join(f"{TYPE_NAME.get(t, t)}:{n}" for t, n in action_types[ai].most_common())
            print("  actions taken:", hist)

    if decks[0] and decks[1]:
        same = sorted(decks[0]) == sorted(decks[1])
        print(f"\nboth decks identical: {same}"
              + ("  (this is a self-play / self-validation game, not a real opponent)" if same else
                 "  (real opponent: agent 1's deck above is the opponent's deck in cabt ids)"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/parse_replay.py <replay.json>")
        sys.exit(1)
    parse(sys.argv[1])
