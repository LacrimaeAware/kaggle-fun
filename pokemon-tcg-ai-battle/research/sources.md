# Sources

The URL ledger. Every external claim used in a committed doc traces to a row here. Add
the date you read it and whether it was reachable by an automated fetch (Kaggle pages
usually are not).

| Source | URL | Used for | Read | Fetchable |
| --- | --- | --- | --- | --- |
| Kaggle - Simulation track | https://www.kaggle.com/competitions/pokemon-tcg-ai-battle | the agent/match track | 2026-06-16 | no (JS, title only) |
| Kaggle - Strategy track | https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy | the report/cash track | 2026-06-16 | no (JS, title only) |
| Dexerto | https://www.dexerto.com/pokemon/pokemon-tcg-launches-ai-battle-challenge-with-50000-prize-pool-3376414/ | dates, prizes, format, 10 min/match, partners | 2026-06-16 | yes |
| PokeBeach | https://www.pokebeach.com/forums/threads/the-pokemon-company-launches-ai-competition-to-build-the-strongest-pokemon-tcg-player-featuring-300-000-in-prizes.156876/ | "$300,000+" prize headline, partners | 2026-06-16 | no (403) |
| Shacknews | https://www.shacknews.com/article/149677/pokemon-tcg-ai-battle-challenge | the two divisions described | 2026-06-16 | yes |

Note: PokéAgent Challenge (pokeagent.github.io) is a different, academic competition
(Pokemon video game and Showdown). Do not conflate it with this contest.

The deep-research workflow run on 2026-06-16 adds rows for the game-model, simulator, and
methods sources into `game-model/` and `methods/`. Merge those source URLs here so this
stays the single URL ledger.
