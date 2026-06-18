# Operating instructions (read this first, every session)

This file is the contract for any model or person working in this folder. It exists
because a sibling project (UMUD) kept failing the same way: a session would reach a
conclusion, write it into a doc, and the next session would read the discredited
conclusion and rehash it. The diagnosis was that the failure is **enactment, not
knowledge**. The docs stated the correct understanding and the work violated it. The
rules below are built to make that hard to do.

Start a session by reading, in order: this file, `docs/COMPETITION.md` (what is
actually true about the contest), `registry/BELIEFS.md` (current live hypotheses) and
`registry/GRAVEYARD.md` (what is already refuted, do not redo it), then `docs/STRATEGY.md`.

## 1. The enactment rule

The hard part is not writing down the right thing. It is acting on it later. Two
concrete obligations:

- Before you propose, build, or re-test any idea, search the ledger:
  `python registry/registry.py search "<the idea>"`. If it is `REFUTED`, you may not
  redo it unless its re-open gate condition has measurably changed. If you think the
  refutation was wrong, change the hypothesis status with new evidence; do not quietly
  re-run the dead experiment.
- A claim that lives only in prose is not trusted. Load-bearing beliefs live in the
  registry with their evidence. `BELIEFS.md` and `GRAVEYARD.md` are generated from it,
  so a stale paragraph in some other doc has no authority. If two docs disagree, the
  registry and `docs/COMPETITION.md` win.

## 2. Provenance: mark verified vs claimed, always

The standing failure mode is relaying a claim as fact when it came from a doc, a single
news outlet, or your own inference. Mark every load-bearing statement:

- `verified` (you ran it, read the code, or saw the official source) vs `claim`
  (a doc, a forum post, one article) vs `inference` (you reasoned to it).
- Competition facts that can only be settled on the live Kaggle Data / Rules / Code
  tabs are marked `unconfirmed` in `docs/COMPETITION.md` until someone confirms them.
  Do not promote an `unconfirmed` item to fact by repetition.

## 3. The win-rate is a global probe (the aggregation fallacy)

A win rate against a fixed opponent pool measures one thing: the net result moved in
that direction over those games. It does not decompose into per-matchup mechanism. So
"agent B beats pool P 54%" does **not** license "B is better against aggro" or "the
energy change fixed the mirror" unless you measured that split directly with enough
games. The same trap sank UMUD repeatedly with leaderboard shifts. Before writing any
mechanism conclusion, ask: is this just the net direction, or am I claiming per-matchup
structure I did not measure? If you did not split it, say "net win rate moved, mechanism
unknown."

Related: a single experiment fitting better is not a general or causal result. State
each link of an inference chain as its own falsifiable hypothesis in the registry, not
as a settled fact in prose.

## 4. Recording work

- A new idea becomes a hypothesis: `registry.py add` with a falsifiable `--statement`,
  a `--test`, and a `--refute` condition. No untestable hypotheses.
- A run becomes an experiment: `registry.py experiment --hyp Hxxx`.
- A measurement becomes a result: `registry.py result --exp Exxx --hyp Hxxx --metric ...
  --value ... --baseline ... --n ... --verdict supports|refutes|inconclusive
  --provenance local-sim|kaggle-LB|manual`. Always record `n` and the baseline.
- A status change carries `--evidence`; `refuted` and `parked` also carry `--gate`.
- Do not create a new markdown file to hold an experiment result. The registry holds
  results. New docs are for genuinely new categories, not for the 31st note on one topic.

## 5. The journal is the exception

`journal/` may have unlimited files, one per day (`YYYY-MM-DD.md`). It is the raw
brain-dump: what was tried, what was seen, loose ideas, dead ends. It is allowed to be
messy and is not authoritative. Curated conclusions graduate out of the journal into a
hypothesis (registry) or into `docs/`. Nothing reads the journal as truth.

## 6. Tone and writing (committed docs)

From the house conventions (`../docs/conventions.md`) and the prompt-engineering library:

- Plain, factual, numbers first. "win rate 0.54 over 400 games" not "a strong result."
- No spin, no hype words (powerful, elegant, exciting, transformative), no morale
  framing, no "honest"/"to be clear"/"great question" throat-clearing.
- No em dashes. Periods, commas, parentheses, colons; hyphens only for ranges.
- Claims proportional to evidence. State limitations in the section where they apply.
- Do the work, do not narrate the act of doing it. (This instruction file is itself an
  instruction document, so it is allowed to talk about process. Normal docs are not.)
- Each competition writeup uses the five fixed sections: Question, Method, Result,
  Caveat, Lesson.

## 7. Competition invariants

- This is an agent competition, not a prediction competition. The deliverable is a bot
  that plays legal Pokemon TCG moves under a time budget. Correctness first: an agent
  that never makes an illegal move and never times out beats a clever agent that
  crashes. Every change keeps the agent always-legal and within the per-match clock.
- The agent must NEVER raise an exception. On submit, Kaggle validates it by playing a copy
  of itself; any uncaught exception or timeout forfeits and the submission fails. Wrap the
  policy in a crash-safe envelope with a legal fallback (the first `minCount` distinct
  in-range indices), before adding strategy. Pure downside protection (see
  `research/notebooks/SUMMARY.md`).
- The forward-model search API (`cg.api` `search_begin`/`search_step`) IS available and is
  the intended design; in-match determinized search / ISMCTS is on the table (registry H1).
- There are two tracks of one contest (`docs/COMPETITION.md`). One agent serves both.
- Do not make a submission without the user's explicit say-so. This is a standing rule
  across the kaggle-fun repo.
- Commits in this repo state what changed and why, plainly, with no Co-Authored-By
  trailer.

## 8. The pre-commit checklist

Run this before writing any conclusion to a committed doc or changing a hypothesis to
`supported`:

1. Searched the registry for prior/refuted versions of this idea.
2. Marked provenance (verified / claim / inference / unconfirmed) on every load-bearing
   statement.
3. For any win-rate claim: is it the net aggregate, or did I measure the split I am
   claiming? Reported `n` and baseline.
4. Not re-asserting something already flagged unproven or refuted.
5. Not inventing per-matchup or per-game-state structure that the data cannot show.
6. Numbers first, no hype, no em dashes, limitation stated.
