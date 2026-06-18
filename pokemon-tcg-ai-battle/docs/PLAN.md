# Plan (plain version)

The short, flat plan. No jargon, no deep nesting. Full reasoning in `LANDSCAPE.md`, the
notebook digest in `research/notebooks/SUMMARY.md`.

## Where we are (2026-06-17)

- The competition gives us their simulator (the `cabt` engine), installed and running here.
- It has a forward-model SEARCH API (`search_begin`/`search_step`), so in-match tree search
  IS possible. The full card pool (1267 cards) and all card art are now downloaded.
- We have a basic always-legal agent (beats random 0.835, ties the dumb baseline). It runs
  the worst sample deck (Mega Abomasnow ex).
- We downloaded 1667 human decklists, 3746 tournament match results, 21 competitor notebooks
  (digested), and the official card data + sample submission.
- A card / deck / stats / replay viewer is built (`viewer.html`).

## The plan of attack (3 stages, heuristics first)

### Stage 1: a real heuristic agent (the floor, do this first)
- Wrap the agent in a crash-safe layer (it forfeits on any exception). Non-negotiable.
- Build a board-aware evaluation that reads the board (prizes, HP, energy, lethal). Reuse the
  eval pieces the digest found (prize math, weakness/resistance, one-turn attack planner).
- Swap off the worst sample deck to a better or less-contested one.
- Win condition for this stage: clearly beat the dumb `first_agent` baseline in self-play.
  That is the registry gate. If a real eval cannot beat take-first, stop and rethink.

### Stage 2: shallow search + knowing the opponent
- Add a 1-ply forward search with `search_begin` on top of the heuristic (copy the one
  working pattern from the notebooks, not the broken ones).
- Seed the search's hidden-info guess from a real deck prior instead of placeholders. Every
  competitor notebook skips this; it is the clearest edge (registry H22).
- Win condition: the search+opponent-model version out-rates the Stage 1 heuristic on the
  ladder beyond the noise band.

### Stage 3: try a learned model, only if Stage 2 plateaus
- Train a learned evaluation or policy from self-play (imitation first, then self-improve) and
  swap it in for the hand-built eval behind the same search. This is the RL/AlphaZero-style
  lever. It needs a lot of self-play and is the most expensive, lowest-certainty step, so it
  is last, not first. Compare it head-to-head against the Stage 1-2 heuristic; keep whichever
  wins. Do not assume it wins.

That is the whole plan. We will not plan past Stage 3 until we see Stage 1-2 results.

## Build status (honest)

Done and usable:
- [x] Scrape all 1267 card images from the official PDF (`tools/scrape_card_images.py`)
- [x] Card lookup from the official CSV (`tools/cards.py`, 1267 cards)
- [x] Deck store (`registry/decks.json`, `tools/build_decks.py`)
- [x] Deck save / load / clone / rename / import (`tools/decks.py`)
- [x] Replay parser + stats store (`tools/parse_replay.py`, `tools/build_stats.py`)
- [x] HTML viewer: Cards, Decks, Stats, Replays, sidebar nav (`viewer.html`)
- [x] Digest of all 21 notebooks (`research/notebooks/SUMMARY.md`)
- [x] Docs corrected (forward model exists, card data, deck-building) and registry updated

Built but data-limited (not a code gap, a data gap):
- [data] Stats view and Replays view work, but we have only ONE replay (the self-play
  validation game). They fill in as real ranked-match replays are downloaded into
  `data/external/replays/` and `tools/build_stats.py` + `tools/build_viewer.py` are re-run.

Tried and measured, NOT shipped as an improvement:
- [measured] A hand-scored heuristic battler. The aggressive version lost to first_agent
  (0.44); a conservative version ties it (0.500); both beat random (~0.90). So hand-scoring
  does not beat the baseline. `agent/main.py` is left as the safe conservative baseline. The
  real battler approach is undecided and is the thing to agree on before building more.

Parked:
- The card-embedding / hidden-structure idea (`IDEAS.md`), with its caveats. Later diagnostic
  experiment, not a near-term win.

## Regenerating the viewer

```
python tools/scrape_card_images.py   # once (needs the PDF + pymupdf); images are gitignored
python tools/build_stats.py
python tools/build_decks.py
python tools/build_viewer.py          # writes viewer.html
```
Open `viewer.html` from inside this folder so the image paths resolve.
