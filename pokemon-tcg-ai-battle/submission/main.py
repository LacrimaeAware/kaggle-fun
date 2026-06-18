"""cabt agent submission entry point: `def agent(obs) -> list[int]`.

STALE BASELINE (2026-06-16): this submission/ folder is the OLD day-one heuristic-only agent
and is NOT what to ship. The current agents live in ../agent/ (heuristic, agent_search,
agent_search_v); the strongest is agent_search (0.585 vs first_agent). A forward model IS
available (cg/api.py search_begin/search_step; registry H001 SUPPORTED) - the claim below that
search is "blocked" is WRONG and kept only as a historical record. See ../agent/README.md for
the real packaging + the file set the agent needs.

This file is the robust, always-legal heuristic (../AGENTS.md section 7: a legal mediocre bot
beats a clever bot that forfeits). [HISTORICAL/INCORRECT: "search is blocked, the engine
exposes no clonable forward model, see registry H1" - superseded by H001 supported.]

How the engine talks to us (verified by reading cabt.py / cg/game.py and by running the
engine locally, see explore_cabt.py):
  - Deck-selection phase: obs["select"] is None. Return a 60-int deck list (card IDs).
  - Play phase: obs["select"] = {"option": [...], "maxCount": k}. Each option is a typed
    dict like {"type": 8, "inPlayArea": 4, ...}. Return up to k indices into "option".
    Only legal options are ever offered, so any subset of indices is legal.

The heuristic: among the offered options, prefer productive actions and defer the
turn-ending option (observed as type 14, which appears alone or as the alternative to a
real action). Measured in the real engine (cabt_arena.py, 200 games each):
  - this agent vs random_agent: win rate 0.835
  - first_agent  vs random_agent: win rate 0.830
  - this agent vs first_agent:    win rate 0.515 (within the n=200 noise band)
So the deferral adds NO measurable edge over just taking the first options; the real jump
is consistency over random. This is recorded honestly in the registry (H003 supported,
H016 the deferral idea refuted). The agent ships because it is always legal and beats
random decisively; the genuine next lever is a board-aware evaluation that reads
current.players (HP, prizes, attached energy), which is what would beat first_agent.
The TYPE_PRIORITY table is the scaffold for that, not a finished heuristic.
"""
from __future__ import annotations

# The engine's default 60-card deck (card IDs), embedded so this file is self-contained.
# Building a custom deck is a later lever (see the registry, deck-selection hypotheses).
DECK = (
    [721] * 2 + [722] * 4 + [723] * 4 + [1092] + [1121] * 2 + [1145] * 2
    + [1163] * 2 + [1219] * 4 + [1227] * 4 + [1262] * 2 + [3] * 33
)
assert len(DECK) == 60

# Option-type preference. Higher is chosen first. The turn-ender (observed as type 14) is
# pushed last so the agent acts before it passes. Unknown types default to neutral (0).
# This is a hypothesis-bearing table, tuned against cabt_arena.py, not a verified mapping.
TYPE_PRIORITY: dict[int, float] = {
    14: -10.0,   # looks like end-turn / pass / done: defer it
}
DEFAULT_PRIORITY = 0.0


def _score(option: dict) -> float:
    return TYPE_PRIORITY.get(option.get("type"), DEFAULT_PRIORITY)


def _choose(obs: dict) -> list[int]:
    sel = obs.get("select")
    if sel is None:
        return list(DECK)                       # deck-selection phase
    options = sel.get("option") or []
    k = sel.get("maxCount") or 0
    n = len(options)
    if n == 0 or k <= 0:
        return []
    # Rank option indices by preference (stable), take the top k. Any subset is legal.
    order = sorted(range(n), key=lambda i: _score(options[i]), reverse=True)
    return sorted(order[:k])


def agent(obs: dict) -> list[int]:
    """Always returns a legal selection. On any internal error, falls back to the first
    maxCount indices (the first_agent behavior), which is always legal."""
    try:
        return _choose(obs)
    except Exception:
        sel = obs.get("select") or {}
        if obs.get("select") is None:
            return list(DECK)
        n = len(sel.get("option") or [])
        k = sel.get("maxCount") or 0
        return list(range(min(k, n)))
