# Competition fact ledger

The verified ground truth about the contest. Every load-bearing fact is marked with its
provenance tier, because the sources are not equally trustworthy:

- `verified` - read from primary source: the engine code on GitHub (`cabt.py`, `cabt.json`)
  or the official cabt docs. High confidence.
- `claim` - from a press release, an organizer FAQ, or one news outlet. Plausible, not
  primary, sometimes contradicted by other outlets.
- `unconfirmed` - needs the live Kaggle Data / Rules / Code tabs. Kaggle pages are
  JavaScript-rendered and return only a title to automated fetchers, so several
  submission-flow and prize facts circulated as "search snippets" of pages no one
  actually read. Do not promote an `unconfirmed` item to fact by repeating it.

Last updated: 2026-06-16. Sources in `../research/sources.md`. The big finding this day:
the game engine is public, so most of the "how does the agent work" uncertainty is gone;
the remaining real unknowns are prize totals, compute limits, and the Strategy judging
formula.

## The two tracks are one contest

The Pokemon Company is running one AI competition with Kaggle, on the same engine and the
same agent, presented as two Kaggle pages.

| | Simulation track | Strategy track |
| --- | --- | --- |
| Kaggle slug | `pokemon-tcg-ai-battle` | `pokemon-tcg-ai-battle-challenge-strategy` |
| Window | Jun 16 - Aug 17, 2026 | Jun 16 - Sep 14, 2026 |
| You submit | a code agent that plays matches | a written report on the same agent and its decks |
| Kaggle reward type | Knowledge (no cash) | cash prizes |
| Judged on | head-to-head ladder rating | agent stability, deck design, and Simulation rating |

Provenance: dates and the two-track split are `verified`/`claim` (Dexerto plus the two
Kaggle slugs). The Strategy track reuses Simulation performance in its ranking, which is
why one build serves both. The exact judging weights are `unconfirmed`.

## Why one shows a dollar figure and the other shows "Knowledge"

This answers "why is one about $240,000 and the other just knowledge." On Kaggle every
competition page carries a reward type. The Simulation track awards no money, so its
reward type is literally "Knowledge" (leaderboard standing, points, medals). All the cash
is attached to the Strategy track, so its page shows a prize figure. Same contest, two
pages, the money sits on the Strategy half. The Simulation track is the measurement engine:
it produces your ladder rating, which then feeds the Strategy judging.

## Prizes

The $240,000 is real and has a specific explanation. Per the HEROZ press release and the
organizer FAQ (`claim`, not yet read off the live Kaggle Rules page):

| Stage | Award | Provenance |
| --- | --- | --- |
| Round 1 (Strategy), top 8 teams | $30,000 each = $240,000 | claim (HEROZ release + FAQ) |
| Round 2 finals, champion | $50,000 | claim |
| Round 2 finals, runner-up | $30,000 | claim |
| Round 2 finalists | $3,000 Google Cloud credits per participant | claim |
| Simulation track | $0 (Knowledge) | verified |

So the "$240,000" the Strategy page shows is the Round-1 pool: eight teams at $30,000. A
defensible total is about $320,000 cash plus Google Cloud credits ($240k + $50k + $30k).
The headline totals disagree across outlets ($50k per Dexerto, $300k+ per PokeBeach, an
unreliable $424k from one auto-read that double-counted), so the exact total is
`unconfirmed`: the live Strategy Rules page is authoritative. What is settled enough to
plan on: the cash is concentrated on the Strategy track, and reaching the Round-1 top 8 is
the gate to it.

## The engine is public (the key finding)

The organizers ship the game simulator and it is already in the open-source Kaggle
environments package. This is `verified` from primary source:

- The env is `cabt` ("Card Battle"), a Python `kaggle_environments` environment, in the
  official repo at `kaggle_environments/envs/cabt/` (`cabt.json`, `cabt.py`, `cg/`,
  `visualizer/`). Confirmed via the GitHub API directory listing and by reading the files.
- `cabt.json` (read verbatim): 2 agents (1v1), reward `Lost:-1, Won:1, Draw:0`,
  `episodeSteps` 10000, `actTimeout` 0, `runTimeout` 3000, observation
  `remainingOverageTime` 600.
- Agent API (read from `cabt.py`): `def agent(obs: dict) -> list[int]`. During deck
  selection `obs["select"]` is `None` and the agent returns its 60-card deck list. During
  play `obs["select"]` is `{"option": [...legal choices...], "maxCount": k}` and the agent
  returns up to `k` indices into `option`. `obs["current"]` holds the board state
  (including `yourIndex` and `result`); `obs["logs"]` is the event log. You never check
  legality yourself; only legal options are offered.
- Built-in baseline agents exist to copy: `random_agent` (samples `maxCount` random option
  indices) and `first_agent` (takes the first `maxCount`).
- A match runs locally and reproducibly: `from kaggle_environments import make;
  env = make("cabt"); env.run([agent, agent])`. Confirmed working off-Kaggle.

## Update 2026-06-17: the official Data tab (major corrections)

The competition download (`pokemon-tcg-ai-battle.zip`) was opened. It contains
`EN_Card_Data.csv`, `JP_Card_Data.csv`, the card-image PDFs, and a `sample_submission/` with
the REAL `cg/api.py`. This corrects several earlier claims. All `verified` from these files.

- Forward-model SEARCH API EXISTS. `sample_submission/cg/api.py` exposes `search_begin`,
  `search_step`, `search_end`, `search_release`. `search_begin` takes your PREDICTED
  opponent deck / hand / prizes / active Pokemon, i.e. determinized forward search built for
  ISMCTS. So in-match tree search is the intended design. The earlier "no forward model"
  finding (registry H1, since corrected) was measured against the STRIPPED installed
  `kaggle_environments` cabt module, not the real competition `cg` module. Use the
  `sample_submission/cg/` module for development.
- Full card data. `EN_Card_Data.csv` lists 1267 distinct cards (id, name, expansion, stage,
  category, HP, type, weakness, resistance, retreat, and per-attack name/cost/damage/effect).
  Also `all_card_data()` and `all_attack()` in `cg/api.py`. The card pool is fully known.
  Helper: `tools/cards.py`.
- Option-type legend (authoritative, from `OptionType` in api.py): 0 NUMBER, 1 YES, 2 NO,
  3 CARD, 4 TOOL_CARD, 5 ENERGY_CARD, 6 ENERGY, 7 PLAY, 8 ATTACH, 9 EVOLVE, 10 ABILITY,
  11 DISCARD, 12 RETREAT, 13 ATTACK, 14 END, 15 SKILL, 16 SPECIAL_CONDITION.
- Prize rules matter for evaluation: a Pokemon ex Knock Out gives the opponent 2 prizes; a
  Mega Evolution ex gives 3; Tera Pokemon take no damage on the Bench; ACE SPEC max 1 per
  deck. (A Mega-ex attacker is a 3-prize liability; the current deck is DENPA92's Dudunsparce/Alakazam.)
- Deck-building is allowed, `verified` by Kaggle staff (Addison Howard): "Participants can
  build their own decks based on the available cards listed on the Data tab." Not required.
- Rules differences (host, engine is authoritative): a few edge-case attacks are
  un-selectable rather than fizzling (no bench space, 0-card draw, empty-hand interaction);
  Mega Zygarde ex coin order is auto left-to-right; simultaneous-KO prize order differs but
  both-take-all is a draw. "Simulator behavior will be treated as the correct behavior."
- Four sample decks ship as rule-based starter agents: Iono, Dragapult ex, Mega Abomasnow
  ex, Mega Lucario ex, plus a Reinforcement Learning & MCTS sample. A third-place competitor
  (ISAKA) measured them: Mega Lucario 60.4%, Dragapult 55.6%, Iono 43.8%, Mega Abomasnow
  40.2% over ~15k games, RPS-like (registry H21, third-party). Our agent previously ran Mega
  Abomasnow; the current deck is DENPA92's Dudunsparce/Alakazam, adopted by measurement (agent/main.py).

## Format and mechanics

| Fact | Value | Provenance |
| --- | --- | --- |
| Game | Pokemon TCG, Standard-format-based, organizer card list | claim |
| Deck | 60 card IDs (from `all_card_data()`), building your own is allowed, not required | claim (FAQ) + verified (deck.csv shape) |
| Match budget | `remainingOverageTime` 600; read as ~10 min per player per match | verified (the 600) + inference (the 10-min/forfeit semantics) |
| Submission | `.tar.gz` with `main.py` at top level plus `deck.csv` | claim (Kaggle snippet) |
| Submissions/day | 5 per team | claim |
| Team size | up to 5 | claim |
| Partners | Matsuo Institute, HEROZ (named in the engine copyright line) | verified |
| Sponsors | Google, Google Cloud, Nvidia | claim (single outlet; absent from engine copyright) |

## Open questions (need the live Kaggle pages, do not guess)

RESOLVED by the 2026-06-17 Data tab: the forward-model question (yes, `search_begin`/
`search_step` exist), the card pool (1267 cards in `EN_Card_Data.csv`), and deck-building
(allowed, build from the Data tab card list). Still open, ranked by impact:

2. Compute limits. CPU-only or GPU at evaluation? Internet access and external pretrained
   weights allowed inside the submitted agent? The "no GPU, large models impractical"
   claim is community analysis, not a quoted rule.
3. Time budget granularity. Is the budget per-move or per-match? If per-match, expensive
   early search starves late turns.
4. Exact total prize pool, verbatim from the Rules page.
5. Strategy judging criteria and weights, and confirmation that stability, deck design, and
   Simulation rating are exactly the official components.
6. Whether the Data tab ships a downloadable card-pool CSV or an official starter notebook
   (only the in-engine `all_card_data()` is confirmed).

When one is answered, record it here with its source and flip its provenance. The matching
hypotheses in the registry depend on these answers.
