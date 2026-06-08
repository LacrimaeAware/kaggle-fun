# kaggle-fun

Kaggle competition work. One folder per competition, each treated as a self-contained experiment with a fixed-shape writeup. This file is the index over those folders.

## Competitions

| Competition | Task | Metric | Status | Lesson |
| --- | --- | --- | --- | --- |
| [Predicting Stellar Class](predicting-stellar-class/) | Multiclass classification (GALAXY, QSO, STAR) | Balanced accuracy | Stack submitted | Stack public 0.96659 (baseline 0.96523); at the optical-plus-redshift ceiling, top cluster noise-limited |
| [UMUD Muscle Architecture](umud-muscle-architecture/) | Image regression (pennation angle, fascicle length, muscle thickness) | UMUD Score (tolerance-normalized MAE) | Selected, planning | Deadline 2026-11-14; segment-then-measure; GPU-gated |
| [Spaceship Titanic](spaceship-titanic/) | Binary classification | Accuracy | Parked (closed 2026-02-28) | Pending |

Digit Recognizer (MNIST) was completed before this repository and is not tracked here.

## Repository layout

- [docs/conventions.md](docs/conventions.md): writing and documentation conventions for this repository.
- [docs/lessons-learned.md](docs/lessons-learned.md): rules that carry across competitions, each with its evidence.
- [predicting-stellar-class/](predicting-stellar-class/): active competition (Kaggle Playground Series Season 6 Episode 6, slug `playground-series-s6e6`). Holds the analysis code, a gitignored `data/` directory, and [writeup.md](predicting-stellar-class/writeup.md).
- [umud-muscle-architecture/](umud-muscle-architecture/): next competition (UMUD Challenge, muscle architecture from ultrasound, deadline 2026-11-14). Holds [plan.md](umud-muscle-architecture/plan.md), [writeup.md](umud-muscle-architecture/writeup.md), and a gitignored `data/` directory.
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
