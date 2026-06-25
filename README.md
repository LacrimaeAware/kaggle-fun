# kaggle-fun

Kaggle competition work. One folder per competition, each treated as a self-contained experiment with a fixed-shape writeup. This file is the index over those folders.

## Competitions

| Competition | Task | Metric | Status | Lesson |
| --- | --- | --- | --- | --- |
| [Predicting Stellar Class](predicting-stellar-class/) | Multiclass classification (GALAXY, QSO, STAR) | Balanced accuracy | Stack submitted | Stack public 0.96659 (baseline 0.96523); at the optical-plus-redshift ceiling, top cluster noise-limited |
| [UMUD Muscle Architecture](umud-muscle-architecture/) | Image regression (pennation angle, fascicle length, muscle thickness) | UMUD Score (tolerance-normalized MAE) | Active, methodology reset after band-fix gains | Best public 0.46041; global multipliers are spent; next is segmentation-vs-measurement error decomposition plus a test-distribution gate |
| [Pokemon TCG AI Battle](pokemon-tcg-ai-battle/) | Game-playing agent (Pokemon TCG, `cabt` engine), two linked tracks | Skill rating from bot-vs-bot episodes | Active, day-one agent + system | Engine is the public `kaggle_environments` cabt env; always-legal heuristic beats random 0.835 but ties first_agent; in-match search blocked (no forward model) |
| [Spaceship Titanic](spaceship-titanic/) | Binary classification | Accuracy | Parked (closed 2026-02-28) | Pending |

Digit Recognizer (MNIST) was completed before this repository and is not tracked here.

## Repository layout

- [docs/conventions.md](docs/conventions.md): writing and documentation conventions for this repository.
- [docs/lessons-learned.md](docs/lessons-learned.md): rules that carry across competitions, each with its evidence.
- [predicting-stellar-class/](predicting-stellar-class/): active competition (Kaggle Playground Series Season 6 Episode 6, slug `playground-series-s6e6`). Holds the analysis code, a gitignored `data/` directory, and [writeup.md](predicting-stellar-class/writeup.md).
- [umud-muscle-architecture/](umud-muscle-architecture/): active UMUD Challenge work. Start with [umud-muscle-architecture/docs/CURRENT_STATE.md](umud-muscle-architecture/docs/CURRENT_STATE.md), then [umud-muscle-architecture/docs/DOC_INDEX.md](umud-muscle-architecture/docs/DOC_INDEX.md). Holds a gitignored `data/` directory and generated `results/`.
- [pokemon-tcg-ai-battle/](pokemon-tcg-ai-battle/): active Pokemon TCG AI Battle agent work (two linked Kaggle tracks). This folder uses an expanded structure to avoid the documentation sprawl seen elsewhere: start with [pokemon-tcg-ai-battle/AGENTS.md](pokemon-tcg-ai-battle/AGENTS.md) (the operating contract), then [docs/COMPETITION.md](pokemon-tcg-ai-battle/docs/COMPETITION.md) and [docs/STRATEGY.md](pokemon-tcg-ai-battle/docs/STRATEGY.md). Conclusions live in a hypothesis registry ([registry/](pokemon-tcg-ai-battle/registry/), generated `BELIEFS.md`/`GRAVEYARD.md`), not in scattered prose; raw daily notes in `journal/`; cannibalized research in `research/`; the agent in `agent/`.
- [spaceship-titanic/](spaceship-titanic/): parked competition (closed 2026-02-28). Holds the analysis code, a gitignored `data/` directory, and [writeup.md](spaceship-titanic/writeup.md).

Each competition folder holds the code, a local `data/` directory (not committed), and a `writeup.md`. Every `writeup.md` uses the same five sections in the same order: Question, Method, Result, Caveat, Lesson.

## Environment

Python 3.13. Dependencies are listed in [requirements.txt](requirements.txt).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Data is downloaded per competition with the Kaggle API into that competition's `data/` directory, which is gitignored.

## AI usage

AI assistance was used for scaffolding and debugging. The design, the validation scheme, and the reported claims were reviewed by the author.

## Updating these documents

- When a result changes, update the relevant `writeup.md` first, then the Status and Lesson cells in the Competitions table above.
- When a mistake recurs or a rule proves out, add it to [docs/lessons-learned.md](docs/lessons-learned.md) with the evidence, so it is read before the next competition.
- Public documents stand alone. They do not reference private notes or external conversations.
