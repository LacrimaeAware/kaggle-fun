# Documentation conventions

These rules apply to all committed documents in this repository: the README, the files in `docs/`, and each competition `writeup.md`.

## Tone

- Plain and factual, in the manner of a technical report. State what is the case. No spin, no overclaiming.
- No promotional or emotional words. No meta-commentary about the writing itself.
- Impersonal and standalone. A reader needs no outside context. Do not reference private notes, conversations, or other repositories.

## Punctuation

- No em dashes. Use periods, commas, parentheses, and colons. Hyphens are for ranges only.

## Numbers and claims

- Numbers first. Write "local CV 0.803, public 0.801", not "a strong score".
- Be explicit about what a metric measures.
- State limitations plainly, in the section where they apply. Claims are proportional to the evidence.

## Structure

- Headings are nouns.
- Each competition `writeup.md` uses five sections in this order:
  - Question: the target, the metric, and what one row represents, in one sentence.
  - Method: the approach, model, features, and validation scheme, named concretely.
  - Result: local cross-validation score and leaderboard score in a table, numbers first.
  - Caveat: what the result does not establish.
  - Lesson: what moved the score, what failed, and what to try next.

## Commits

- State what changed and why, plainly. No Co-Authored-By trailer in this repository.
