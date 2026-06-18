# Competition research (raw landing notes)

Curated version is in `../../docs/COMPETITION.md`. This is the raw research with sources.
Provenance tiers: `verified` (primary source), `claim` (press release / FAQ / one outlet),
`unconfirmed` (needs live Kaggle pages).

## The engine is public (verified, primary source)

The biggest finding. The organizers ship the simulator as the `cabt` ("Card Battle")
`kaggle_environments` env, in the official repo `Kaggle/kaggle-environments` at
`kaggle_environments/envs/cabt/` (files: `cabt.json`, `cabt.py`, `cg/`, `visualizer/`).

- `cabt.json` (read verbatim): agents 2, reward `Lost:-1, Won:1, Draw:0`, `episodeSteps`
  10000, `actTimeout` 0, `runTimeout` 3000, observation `remainingOverageTime` 600.
- Agent API (read from `cabt.py`): `def agent(obs: dict) -> list[int]`.
  - Deck phase: `obs["select"] is None`, return a 60-int deck list.
  - Play phase: `obs["select"] = {"option": [...], "maxCount": k}`, return up to k indices.
  - `obs["current"]`: board state with `yourIndex`, `result`. `obs["logs"]`: event log.
  - Only legal options are offered; you do not check legality yourself.
- Baselines to copy: `random_agent`, `first_agent` (in `cabt.py`).
- Docs `matsuoinstitute.github.io/cabt`: `all_card_data()`, `all_attack()`, and a
  forward-search API `search_begin` / `search_step` (planning agents are the intended
  design, not LLMs).
- Run a match: `from kaggle_environments import make; make("cabt").run([agent, agent])`.

## Two tracks, one contest (claim/verified)

Simulation (`pokemon-tcg-ai-battle`, Jun 16 - Aug 17): no-cash ladder, the agent plays.
Strategy (`...-challenge-strategy`, Jun 16 - Sep 14): the report + cash, ranked partly on
Simulation rating. The cash sits on Strategy; Simulation's Kaggle reward type is "Knowledge".

## Prizes (claim, press release + FAQ; live Rules page authoritative)

Round 1 Strategy top 8 teams: $30,000 each = $240,000 (this is the "$240k" the page shows).
Round 2 finals: champion $50,000, runner-up $30,000, plus $3,000 Google Cloud credits per
finalist. Defensible total about $320,000 cash + credits. Outlets disagree on the headline
total ($50k Dexerto, $300k+ PokeBeach, an unreliable $424k); treat the exact total as
unconfirmed.

## Submission and rules (claim / unconfirmed)

`.tar.gz` with `main.py` at top + `deck.csv`; 5 submissions/team/day; team size up to 5;
Standard-format-based on an organizer card list; deck-building allowed, not required. GPU
availability, internet access, external-weights permission, and per-move-vs-per-match time
granularity are unconfirmed and need the live Rules tab.

## Sources

See `../sources.md`. Primary: the GitHub `cabt.py` / `cabt.json` raw files and the GitHub
API directory listing. Secondary: Dexerto, PokeBeach, Shacknews, the HEROZ release, the
matsuoinstitute.github.io/cabt docs. The PokeAgent Challenge (pokeagent.github.io) is a
different competition, do not conflate.
