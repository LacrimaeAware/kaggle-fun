# Conventions for this folder

These extend the repository conventions in `../../docs/conventions.md` (tone, no em
dashes, numbers first, the five-section writeup). What follows is specific to an agent
competition with a hypothesis registry.

## Where things are recorded

- A belief or conclusion is a hypothesis in `registry/`, not a paragraph in a doc. If you
  want a future session to trust it, it has a hypothesis id and evidence.
- An experiment and its result go in the registry (`experiment`, `result`), not in a new
  markdown file. Do not add the 31st note on one topic; update the hypothesis.
- `BELIEFS.md` and `GRAVEYARD.md` are generated. Never hand-edit them. Edit the JSONL and
  run `python registry/registry.py render`.
- Raw daily notes go in `journal/`, which is not authoritative.
- A new committed doc is justified only by a genuinely new category, not by accumulation.

## Reporting a result

- Numbers first, with `n` and the baseline. "win rate 0.54 vs 0.50, n=400, local-sim."
- Name the provenance: `local-sim`, `kaggle-LB`, or `manual`. A local-sim number and a
  Kaggle-LB number are different measurements; do not treat one as the other.
- A win rate against a fixed opponent pool is one aggregate number. Do not narrate
  per-matchup mechanism from it unless you measured the split. See `AGENTS.md` section 3.

## Code

- The agent must always produce a legal move and stay within the time budget. A change
  that risks an illegal move or a timeout is a regression regardless of its win rate.
- Keep the policy decoupled from the Kaggle harness behind a thin adapter, so the same
  policy runs against the local mock environment and against the hosted engine.
- Commit messages: what changed and why, plainly. No Co-Authored-By trailer.
