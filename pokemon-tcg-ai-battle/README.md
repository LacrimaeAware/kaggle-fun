# Pokemon TCG AI Battle

Work on The Pokemon Company's Pokemon TCG AI Battle Challenge on Kaggle. The contest has
two Kaggle pages that are two tracks of one competition, so this is one folder, not two.
One agent serves both tracks.

| Track | Kaggle slug | Window | Reward | What you submit |
| --- | --- | --- | --- | --- |
| Simulation | `pokemon-tcg-ai-battle` | Jun 16 - Aug 17, 2026 | Knowledge (no cash) | an AI agent that plays automated matches |
| Strategy | `pokemon-tcg-ai-battle-challenge-strategy` | Jun 16 - Sep 14, 2026 | cash (see `docs/COMPETITION.md`) | a report on the agent's strategy, ranked partly on Simulation performance |

The Simulation track is the engineering leaderboard: your agent plays Pokemon TCG matches
against other agents under Standard-format rules and a per-match time budget. The Strategy
track is where the prize money sits: you submit a written explanation of your agent and
deck design, judged together with how the agent performed in Simulation. They are the same
build seen from two sides. `docs/COMPETITION.md` holds the verified fact ledger and marks
what still needs the live Kaggle pages.

This is an agent competition, not a prediction competition. The deliverable is a bot that
plays legal moves well, not a model that outputs a CSV.

## Layout

- `AGENTS.md` - operating contract. Read it first, every session. It encodes the rules
  that keep this from turning into the UMUD rehash loop.
- `docs/` - curated, committed docs:
  - `PLAN.md` - the plain, flat, what-we-do-next plan. Start here for the way forward.
  - `LANDSCAPE.md` - the deeper map: what the game is, the constraints, and the options.
  - `COMPETITION.md` - the verified fact ledger for the contest (the ground truth).
  - `STRATEGY.md` - the longer strategy memo.
  - `conventions.md` - writing and experiment conventions for this folder.
- `registry/` - the hypothesis and experiment ledger (JSONL canon + SQLite/FTS5 search +
  generated `BELIEFS.md` / `GRAVEYARD.md`). This is how a conclusion is recorded so it is
  not rehashed. Search it before proposing anything.
- `IDEAS.md` - the idea bank: raw directions we might pursue, before they become tested
  hypotheses. The "park a thought and find it again" surface.
- `research/` - cannibalized external research (competition, game model, methods,
  cross-repo transfers) plus the `sources.md` URL ledger.
- `journal/` - daily brain-dump, one file per day. The only place unlimited new files are
  correct. Not authoritative.
- `agent/` - the agent code (the first attempt and what follows).
- `tools/` - helper scripts, e.g. `fetch_data.py` (downloads the reference data).
- `data/` - local data, gitignored. `data/external/` holds the downloaded card database and
  1630 real competitive decklists (regenerate with `python tools/fetch_data.py`).

## How a session runs

1. Read `AGENTS.md`, `docs/COMPETITION.md`, `registry/BELIEFS.md`, `registry/GRAVEYARD.md`,
   `docs/STRATEGY.md`.
2. Before testing an idea: `python registry/registry.py search "<idea>"`. Do not redo a
   refuted hypothesis unless its re-open gate has changed.
3. Do the work. Record runs and results in the registry, not in new prose files.
4. Brain-dump to today's `journal/` file. Promote conclusions into hypotheses or `docs/`.
5. No submission without the user's explicit say-so.

## Status

Day one (2026-06-16) done: folder, registry (15 hypotheses, 3 resolved with real data),
operating contract, research, and a first-attempt agent that runs on the real engine (beats
random 0.835, ties first_agent). Engine audited, reference data downloaded. The way forward
is in `docs/PLAN.md`. Next decision point: does a board-aware bot beat first_agent. See
`journal/2026-06-16.md`.
