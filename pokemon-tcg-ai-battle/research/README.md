# research/

Cannibalized external research, kept so it can be referenced and synthesized later
instead of re-derived. Raw findings land here; the curated conclusions live in
`../docs/STRATEGY.md` and in the registry.

Layout:

- `competition/` - facts about the contest: prize structure, rules, submission
  mechanics, the two-track relationship. The verified subset is promoted to
  `../docs/COMPETITION.md`.
- `game-model/` - Pokemon TCG rules as a formal game, the state and action space, and
  the landscape of open-source simulators, engines, card databases, and datasets.
- `methods/` - game-AI methods for imperfect-information card games (heuristic search,
  determinized MCTS, Information-Set MCTS, CFR family, RL self-play, LLM agents) and how
  they fit this contest's constraints.
- `cross-repo/` - methods from the user's other repos that transfer here, with honest
  caveats about whether they buy win rate or only understanding.
- `sources.md` - the URL ledger. Every external claim used in a committed doc traces to
  a row here.

Provenance discipline (see `../AGENTS.md` section 2): a file here marks each claim as
verified, claim, inference, or unconfirmed. Kaggle competition pages are JavaScript
rendered and often return only a title to automated fetchers, so several competition
facts here are sourced to news coverage and remain `unconfirmed` until checked against
the live Data / Rules / Code tabs.
