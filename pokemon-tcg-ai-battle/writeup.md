# Pokemon TCG AI Battle

## Question

Build an AI agent for the Pokemon TCG AI Battle (Simulation track, Kaggle slug
`pokemon-tcg-ai-battle`). The agent is a function `agent(obs) -> list[int]` that plays the
organizer's `cabt` engine; the metric is a skill rating earned from bot-vs-bot episodes on
the ladder. One submission is one agent. A linked Strategy track carries the cash and is
judged partly on this agent's Simulation rating (`docs/COMPETITION.md`).

## Method

Three agents, all always-legal and crash-safe (`agent/main.py`):
- **heuristic**: rules on the current options (take a listed lethal, go first, attach energy
  to an energy-short active), else the engine default order. No lookahead.
- **agent_search**: 1-ply forward-model search. The official `cg/api.py` exposes
  `search_begin`/`search_step` on `cg.dll`, so for each option we simulate the line (my turn
  plus the opponent's reply), average over several determinizations of the hidden cards, and
  score the resulting board with a hand formula (prizes dominate, then board HP/bodies/energy).
- **agent_search_v**: identical search, but the leaf is scored by a learned gradient-boosted
  tree predicting P(win) over a 47-feature state encoding (`features.py` -> `train_value.py`
  -> pure-numpy `value_model.py`).

Everything is measured in the real engine via `agent/cabt_arena.py`, seats swapped, with
Wilson confidence intervals.

## Result

| Matchup (real cabt games) | Win rate | 95% CI |
| --- | --- | --- |
| agent_search vs first_agent | 0.585 (n=800) | [0.551, 0.619] |
| agent_search vs heuristic | 0.543 (n=300) | [0.487, 0.599] |
| heuristic vs first_agent | 0.513 (n=300) | inside noise |
| agent_combine vs first_agent | 0.557 (n=400) | [0.509, 0.605] |
| agent_search_v vs heuristic | 0.517 (n=400) | [0.469, 0.566] |
| agent_search_v vs heuristic (MC-outcome value, superseded) | 0.427 (n=400) | [0.380, 0.476] |
| heuristic vs random_agent | 0.875 (n=200) | — |

Learned-value progression vs the heuristic (search_v): 0.427 (MC-outcome value) -> 0.517
(search-bootstrapped value, pass 1). All learned-value intervals include 0.5 (parity).
0 illegal-move/timeout forfeits across thousands of games (registry H2). The forward model
runs locally with no reentrancy crash (registry H001, supported).

## Caveat

The strongest agent is **agent_search** (forward search + hand eval). The learned value started
worse: as an MC-outcome-trained leaf eval it LOST to the heuristic (0.427), because it had good
GLOBAL win/loss accuracy (AUC ~0.74) but ranked NEARBY candidate leaves poorly, which is what
1-ply search needs (registry H023). Training it on SEARCH-bootstrapped targets (the search's own
backed-up value) moved it to parity (0.517 vs heuristic); pass 1 imitates the hand search, so it
matches rather than exceeds it. Intervals are wide at these sample sizes, so the learned-value
differences are not separated from noise. Three adversarial-review/research workflows drove fixes
(incl. a scale bug that had inflated search_v). Competition facts (prize total, GPU/internet
limits, per-move vs per-match time) are unconfirmed (`docs/COMPETITION.md`). No submission made.

## Lesson

A board-aware hand search beats the dumb baseline; a learned Monte-Carlo value does not beat
the hand eval as a 1-ply leaf yet. A deep-research pass (`docs/RESEARCH.md`) gives the
evidence-backed next steps: search-bootstrapped value targets (the proven fix for the
good-AUC/poor-local-ranking symptom), ensemble determinization, and contrastive
magnitude-aware embeddings, toward expert iteration / RL. The durable output is the
infrastructure: a verified local engine harness, the full learned-value loop, and an
anti-rehash registry so refuted ideas are not rediscovered. Live/dead hypotheses in
`registry/BELIEFS.md` and `registry/GRAVEYARD.md`; current plan in `docs/LEARNING_PLAN.md`.
