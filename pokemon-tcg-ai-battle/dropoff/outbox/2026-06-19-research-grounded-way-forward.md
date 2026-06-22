# Research-grounded way forward (2026-06-19)

Source: the deep-research digest `dropoff/inbox/2026-06-19-deep-research-beyond-heuristics.md` (two
verified runs: self-play/RL/CFR + poker/chess history/scaling) plus the project audit. A third broad-methods
run is still pending and will be folded in. This is the actionable verdict, not new code.

## Verdict (what the evidence says reliably works)
Across poker, chess, Go, and stochastic card games the reliable levers are, in order of empirical strength:
1. **Consistent real-time SEARCH on an approximate value** (cleanest causal evidence: the Libratus ablation,
   raw blueprint LOST -8 mbb/g, +nested subgame search WON +63 mbb/g). It must be a CONSISTENT /
   equilibrium-aware search (CFR/OOS), NOT naive determinization.
2. **A good learned leaf evaluation** -- necessary (search-alone has a low ceiling), but in imperfect-info it
   is only SOUND over BELIEF STATES (ReBeL), not action/observation history.
3. **Raw compute** -- lawful but only logarithmic in search and it saturates.

## Why everything we tried washed (not an effort/capacity failure)
Our pattern -- every learned addition (distilled ranker, contextual ranker, DAgger, Teacher-V2, risk model)
only MATCHES `agent_search` at equal budget -- is the EXPECTED, documented signature of distilling a
DETERMINIZED (PIMC) search into a net trained on biased + high-variance labels. The unsoundness is formal:
in imperfect-info games an action's value can depend on the probability it is played, so a value keyed on
action/observation history has no unique target (the AlphaZero-style port is provably unsound). A
from-scratch rebuild would hit the same wall. NOTE: the SEARCH side did help (LB 617 -> 640 -> 697; N=4->8);
it is the learning-on-top that washes.

## Highest-value next action: DIAGNOSE variance vs bias (cheap, forks the roadmap)
Before any more learning, measure on the Pokemon TCG game tree:
- leaf correlation, bias, disambiguation factor (the three a-priori predictors of whether PIMC works);
- whether a consistent search's error/exploitability FALLS with budget while the determinized one PLATEAUS;
- whether `agent_search` is past the determinization saturation point (~20 in Dou Di Zhu) or genuinely
  under-simulating.

## The two evidence-backed roads (pick by the diagnostic)
A) Blocker is VARIANCE (curable, cheap, keeps search authoritative):
   - common random numbers / paired evaluation -- compare sibling options on the SAME sampled rollouts so the
     noise cancels (heuristic-search-v2's `compare_selections` already does this; extend it broadly);
   - more determinizations up to saturation; a lower-variance leaf value.
B) Blocker is structural BIAS (strategy fusion / non-locality, incurable by more simulation):
   - move toward a CONSISTENT / belief-state search: a value over PUBLIC BELIEF STATES + regret-minimizing
     look-ahead (CFR / ReBeL-style) -- the proven superhuman imperfect-info recipe (Libratus, ReBeL);
   - open question: tractability under the ~0.6s/move budget for a single fixed deck (card-game belief states
     may be far smaller / more structured than no-limit poker -- worth a small feasibility spike).

## Stop-doing list
- Adding more learned heads trained on determinized-search labels (the research predicts they wash).
- The naive AlphaZero port (distill determinized search into a history-keyed value) -- provably unsound here.

## Highest-ceiling but gated
Self-play Deep Monte-Carlo (DouZero) is the strongest "true AI" path with domain proof, but it is gated on
SIMULATOR THROUGHPUT (our binding constraint). Revisit only if sim throughput is raised (parallelism / a
faster or learned simulator).
