# Pokemon TCG AI Battle

## Question

Build an AI agent for the Pokemon TCG AI Battle (Simulation track, Kaggle slug
`pokemon-tcg-ai-battle`). The agent is a function `agent(obs) -> list[int]` that plays the
organizer's `cabt` engine; the metric is a skill rating earned from bot-vs-bot episodes on
the ladder. One submission is one agent. A linked Strategy track carries the cash and is
judged partly on this agent's Simulation rating (`docs/COMPETITION.md`).

## Method

A robust, always-legal heuristic. During deck selection the agent returns a fixed 60-card
deck (the engine default); during play it receives `obs["select"]["option"]` (typed action
dicts) with a `maxCount` and returns up to that many indices. Only legal options are
offered, so any subset is legal. The heuristic defers the turn-ending option (observed as
option type 14) so the agent acts before it passes, with a safe first-`maxCount` fallback
on any error. Decisions are O(n log n) over a handful of options, microseconds each.
Everything was measured in the real engine via `agent/cabt_arena.py` (200 games per
matchup, seats swapped), not against a proxy.

## Result

| Matchup (200 real cabt games) | Win rate |
| --- | --- |
| heuristic vs random_agent | 0.835 |
| first_agent vs random_agent | 0.830 |
| heuristic vs first_agent | 0.515 |

Across 600+ games the agent produced 0 illegal-move or timeout forfeits (registry H2,
supported). It beats random decisively (H3, supported, Wilson lower bound about 0.78).

## Caveat

The 0.515 against `first_agent` is inside the n=200 noise band, so the type-14 deferral
adds no measurable edge over taking the first options (registry H16, refuted): the real
gain is consistency over random, not this heuristic. The engine exposes no in-match
clonable forward model (H1, refuted by reading `cg/sim.py`), so tree search is not
available in-match this version; offline self-play does work. Several competition facts
(exact prize total, GPU and internet limits, per-move vs per-match time, Strategy judging
weights) are unconfirmed and need the live Kaggle pages (`docs/COMPETITION.md`). No
submission has been made.

## Lesson

Consistency beats random by a wide margin here, and a plausible "play more, pass less"
heuristic did not beat the dumb take-first baseline once measured. The lever that would
actually beat `first_agent` is a board-aware evaluation that reads `current.players` (HP,
prizes, attached energy), which is the next step. The durable output of day one is the
infrastructure: a verified engine harness that runs locally, and a hypothesis registry
that recorded the forward-model finding and tombstoned the refuted heuristic so neither is
rediscovered. Full plan in `docs/STRATEGY.md`; live and dead hypotheses in
`registry/BELIEFS.md` and `registry/GRAVEYARD.md`.
