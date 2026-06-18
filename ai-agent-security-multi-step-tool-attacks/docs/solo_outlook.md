# Solo outlook

A calibrated answer to "do I have a chance as a solo undergrad, or do I lose to teams
automatically." Short version: no, you do not lose automatically, and this is one of the
more solo-friendly Kaggle code competitions. The honest ceiling for a first-timer is a
top-half finish with a credible shot at a bronze medal and at a 2,500 USD Working Note
award, not the main prize.

## Why solo is viable here

This format rewards a few good ideas plus disciplined iteration far more than headcount.

- The deliverable is one algorithm, not a portfolio of findings. You submit one
  `attack.py`. A team of six still ships one `attack.py`. The marginal value of the
  fourth through sixth teammate is low. This is the opposite of the predecessor gpt-oss
  red-teaming hackathon, which was human-judged on many written findings and genuinely
  rewarded headcount (the winning teams were multi-person organizations).
- Scoring is automated and reproducible. There is no judge to impress and no presentation
  to polish, so the team advantage of dividing report-writing labor does not exist.
- The winning artifact is small and idea-dense. The math is in the search design and the
  scoring analysis, not in raw volume.

Where teams still have a real edge: parallel GPU quota. Each Kaggle account gets its own
weekly GPU allowance, so a three-person team has three times the iteration budget. That is
the single biggest team advantage in this competition, and it is real. But it is a quota
advantage, not a brainpower advantage, and it caps out: past a point everyone is
bottlenecked on ideas worth spending a run on, not on quota.

## The binding constraint: GPU iteration

The deterministic local agent scores zero, so attack quality is only observable against
the real GPT-OSS-20B and Gemma-4 weights, which need a GPU. On free Kaggle quota
(roughly 30 hours per week, sessions capped around 9 to 12 hours), and with each model
getting a 9,000-second budget across two models, a single full scored run costs about 5
hours of GPU wall-clock. That is roughly five to six full two-model runs per week.

This is a brutally low iteration count compared with a normal tabular competition, and it
is deliberately so: the zero-scoring local agent forces you to do most design offline and
analytically. The discipline that follows is the actual skill that decides your result:

1. Build a free CPU instrumentation layer. You cannot use the real reward locally, but
   you can enumerate reachable tool-call cells and firable predicates against the
   deterministic env and the fixtures, dedupe candidate traces, and prune anything that
   cannot fire a predicate. Spend GPU only on candidates that survive offline filtering.
2. Batch ruthlessly. Never spend a GPU run on a single hypothesis. Each run tests a family
   of variations whose offline structure you have already validated.
3. Separate smoke tests from scored runs. Use the smallest budget to confirm plumbing,
   and a full-budget run only when the algorithm is stable.
4. Cache and grow a library. Replay is deterministic at seed 123, so keep every candidate
   that ever scored and extend coverage rather than rediscovering it.
5. Pre-register each submission. Treat each submission as a measurement of the hidden
   public guardrail, and design experiments so each answers one question.

The net effect compresses the field toward whoever extracts the most signal per GPU-hour,
which favors a disciplined analyst over a brute-forcer with a big cluster.

## Outcome ladder

Kaggle medal cutoffs scale with team count: for 250 to 1,000 teams, bronze is top 100,
silver is top 50, gold is top 10. The live team count for this competition was not
verifiable (the page is JavaScript-rendered and the competition was two days old at
research time), so the tiers below are conditioned on a small-to-medium field, which a
security-flavored code competition with a zero-score trap and a GPU bottleneck tends to
produce as casual entrants drop out.

- Any non-zero score: a focused engineering task of a few days. Many entrants sit at zero
  for a while because of the deterministic-agent trap, so clearing it puts you ahead of
  the lurkers. Realistic.
- Beat the random and Go-Explore baselines: needs an archive-based search that revisits
  promising cells rather than sampling at random. Realistic in two to three weeks once
  plumbing works.
- Top half: mostly "a competent, non-broken pipeline that iterated a bit." Given how many
  entrants stall at zero or at a single trivial predicate, this is a realistic center-of-
  mass target.
- Bronze medal: genuinely in reach if the active field stays small to medium. The
  zero-score trap and the GPU bottleneck cull casual entrants, which compresses the field
  in your favor.
- Silver medal: requires a genuinely better search than the median serious competitor.
  Doable solo, but a stretch.
- Gold or top-10 or main prize: this is where solo plus free GPU starts to bite. Unlikely
  for a first-timer, and not worth burning three months chasing.
- Working Note award (2,500 USD, two awarded): the realistic prize target. It rewards the
  written method and findings, which is exactly the clear, methodical analytical writing a
  math and statistics person can produce, and it is scored separately from leaderboard
  rank. This is the highest expected-value prize play for a solo first-timer.

## The asymmetric edge for a math and statistics person

This is not motivation, it is structural. The score rewards coverage of a discrete space
(unique cells) and diversity of triggered predicates, discovered by an automated search.
That is a search-and-coverage optimization problem dressed as a security problem. Deep
security expertise helps hand-craft a few clever exploits; it does not help systematically
cover the space. The latter is most of the score, and it is a "map the state space,
characterize which regimes trigger predicates, allocate exploration budget" task.

Where the mindset converts directly to score:

- Exploration versus exploitation under a hard budget. About five GPU runs per week and a
  9,000-second cap per model is a bandit and coverage-maximization problem. Most
  security-background entrants will under-formalize it.
- Reachability structure offline. Map which tool sequences are reachable and which
  predicates are firable from which states, and build the archive that links each cell to
  the best trajectory that reaches it. This is state-space analysis, free on CPU.
- Severity-weight convexity. A severity-5 hit is 16; a cell is 2. Do the marginal-value-
  per-GPU-second math on whether the next run is better spent on one more severity-5
  trigger or eight more cells. Add the duplication question from [strategy.md](strategy.md)
  Lever 1, which may change the answer entirely.
- A small designed experiment. Four score rows (two models, two guardrails). Decompose
  them, find the weakest, and target runs there instead of optimizing the aggregate blind.
- The Working Note literally rewards characterizing the regime: how agent failures arise,
  how to measure them, how they might be defended. That is boundary-mapping and diagnosis.

The honest caveat: you still have to implement a non-trivial search loop in a constrained
sandbox and debug it against opaque GPU runs. The analytical framing is the edge; the
engineering is the cost of entry.

## Bottom line

Best realistic target over the remaining roughly eleven weeks, solo, on free GPU: clear
zero comfortably, beat the baselines, finish in the top half, and seriously contest a
Working Note award, with a bronze medal as a credible stretch if the field stays small to
medium. Silver is the upper stretch. Do not spend the runway chasing the top of the
leaderboard. Spend it on a defensible, finished, well-documented result, which is also the
thing that wins the Working Note and the thing that matters for a quant-research portfolio.
