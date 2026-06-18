# Methods research (raw landing notes)

Curated version in `../../docs/STRATEGY.md` section 4. Method ranking for an
imperfect-information, stochastic, sequential card game under a wall-clock agent harness,
adjusted for the verified engine constraints.

## Ranking, after the forward-model finding

The single fact that reorders everything: the installed engine exposes no in-match
clonable forward model (registry H1, refuted). So in-match tree search is not cleanly
available this version. That demotes the search-heavy methods from "the plan" to "blocked
pending a forward model."

| Method | Fit now | Note |
| --- | --- | --- |
| Rule/heuristic policy with a strong eval | high | the floor and the current ceiling we can reach; reads `current.players` |
| Depth-1 expectimax | conditional | needs a next-state model; none in-match, so you would reimplement rules. Blocked. |
| Determinized MCTS / ISMCTS | conditional, blocked | need a clonable fast sim in-match; not available |
| CFR / MCCFR / Deep-CFR | low | needs a fast sim and optimizes toward an unexploitable equilibrium, wrong target for exploiting a weak bot pool |
| RL self-play (PPO/AlphaZero) | low/late | offline self-play via `env.run` is possible, but training needs scale and runway |
| Offline-learned eval/policy, hand-built search-free play | medium/late | the realistic learning path given offline self-play works and in-match search does not |
| LLM-as-agent in-match | very low | latency, CPU-bound harness, empirically weak on this game (unsourced claim) |

## Recommended progression (does not sacrifice score for novelty)

0. Always-legal heuristic. Done. Beats random 0.835, ties first_agent (the deferral idea
   was refuted, H16). Ship-quality floor.
1. Board-aware evaluation: decode the option-type enum and read `current.players` (HP,
   prizes, attached energy, KO/lethal terms) to score options. This is the lever that
   would beat first_agent and the highest-EV next step.
2. Offline self-play to tune the eval weights and, later, an imitation/learned policy
   (the engine supports `env.run` loops). In-match search stays parked behind H1's gate.

## Novel-but-effective angles (each must buy win-rate, see STRATEGY.md section 5)

Opponent modeling for exploitation (a ladder rewards beating weak deterministic bots, not
unexploitability); deck-archetype matchup matrix and a small metagame deck-selection game
(maximin over a payoff matrix, milliseconds, decoupled from in-match cost); risk-aware
play conditioned on the board. The cross-repo honesty note applies (`../cross-repo/`).

## External analogs (UNSOURCED here, motivating only)

LoCM (handcrafted agents competitive), DouZero (RL on DouDizhu), RIS-MCTS (Hanabi),
PokeAgent/PTCG-Bench (LLMs subpar). These came from the research model's memory, not from
a source in this repo. Do not cite them as evidence until a row exists in `../sources.md`.
