"""Seed the registry with the day-one hypotheses (H1-H15 from docs/STRATEGY.md).

One-off. Run once: `python seed.py`. Idempotent enough to re-run on an empty canon; if
the canon already has hypotheses it will append duplicates, so only run on a fresh ledger.
After seeding, the canonical ids are H001.. in creation order; the H1.. labels below map to
them in order. The statements are the falsifiable claims; the tests and refutation
conditions come straight from the strategy memo.
"""
from __future__ import annotations

import registry as R

SEED = [
    dict(title="Forward model is available for search",
         statement="The cabt harness exposes a clonable, steppable forward model usable for search (clone state, step from an arbitrary node).",
         test="Instantiate the env locally; attempt to clone the state and step it forward from a mid-game node.",
         refute_condition="No clone/step is available and the agent only receives one observation per call.",
         confidence="medium", tags=["engine", "gate", "search"]),
    dict(title="Always-legal floor",
         statement="A heuristic agent that returns only select-offered indices with a safe default never emits an illegal move and never times out.",
         test="Run >=2000 self-play games, count illegal-action and timeout forfeits.",
         refute_condition="Any forfeit from an illegal action or a timeout.",
         confidence="high", tags=["agent", "correctness"]),
    dict(title="Heuristic beats random decisively",
         statement="The Step-0 heuristic beats random_agent with a Wilson lower bound well above 0.5.",
         test="Heuristic vs random_agent over n games; Wilson interval on the win rate.",
         refute_condition="Win-rate CI includes or sits near 0.5.",
         confidence="high", tags=["agent", "baseline"]),
    dict(title="Expectimax beats heuristic on the ladder",
         statement="Depth-limited expectimax on the same eval raises the public ladder rating over the Step-0 heuristic beyond the surface noise band.",
         test="Submit both; compare ratings after sigma shrinks; state the band first.",
         refute_condition="The rating delta is inside the noise band.",
         confidence="medium", tags=["agent", "search", "ladder"]),
    dict(title="Time budget is per-move",
         statement="The match time budget is allocated per-move, not as a single per-match pool.",
         test="Read the live Rules tab; if ambiguous, instrument an agent that spends heavily on turn 1 and watch for late-turn starvation.",
         refute_condition="The budget is documented or observed to be a single per-match pool.",
         confidence="low", tags=["engine", "time", "unconfirmed"]),
    dict(title="Within-turn enumeration is tractable",
         statement="After once-per-turn and dominated-line pruning, within-turn action sequences stay small enough for depth-1 expectimax inside the per-decision budget on real decks.",
         test="Instrument max and median sequence count per turn over n self-play games.",
         refute_condition="The budget is exceeded on a non-negligible fraction of turns even after pruning.",
         confidence="medium", tags=["search", "time"]),
    dict(title="Exploitation beats equilibrium on this pool",
         statement="An opponent-adaptive response model out-rates a non-adaptive policy of equal search depth on the live ladder.",
         test="Submit adaptive and non-adaptive variants differing only in the opponent model; compare ratings.",
         refute_condition="The adaptive variant does not exceed the fixed one beyond the noise band.",
         confidence="medium", tags=["opponent-model", "ladder"]),
    dict(title="Archetype-conditioned determinization beats uniform",
         statement="Sampling ISMCTS determinizations from an inferred-archetype belief beats uniform-over-consistent sampling, all else fixed. (Conditional on the forward model.)",
         test="A/B self-play where only the belief sampler differs, plus a ladder check.",
         refute_condition="The archetype sampler does not beat uniform beyond the noise band on the ladder.",
         confidence="medium", tags=["ismcts", "belief"]),
    dict(title="Matchup residual is below the noise floor",
         statement="After deflating the global power-level mode from the deck matchup matrix, residual rock-paper-scissors structure is below the SNR floor at realistic per-matchup game counts.",
         test="Split-half stability of residual matchup edges (recompute on two random halves, correlate).",
         refute_condition="Residual edges are split-half stable above chance at the available n.",
         confidence="medium", tags=["metagame", "cross-repo", "diagnosis"]),
    dict(title="Plain classifier ties latent-archetype model",
         statement="A logistic-regression / gradient-boosting baseline on hand-crafted deck features predicts match outcome at least as well as a latent-archetype embedding model.",
         test="Same train/test split, both models, compare held-out accuracy.",
         refute_condition="The embedding model beats the baseline beyond CI.",
         confidence="medium", tags=["metagame", "cross-repo", "control"]),
    dict(title="Deck-vs-pilot confound is real",
         statement="Ladder win rate by deck does not equal fixed-policy self-play win rate by deck; raw ladder tier lists are partly pilot selection.",
         test="Run the same fixed agent across candidate decks in self-play; compare that deck ranking to the ladder ranking.",
         refute_condition="The two rankings agree within sampling error.",
         confidence="medium", tags=["metagame", "confound"]),
    dict(title="Intervention reveals counter-structure",
         statement="Forcing a deck into the self-play field and measuring the directed win-rate response recovers an 'A suppresses B' relation that symmetric co-occurrence data does not show.",
         test="Compare the forced-flow directed response to the co-occurrence matrix on the same decks.",
         refute_condition="The directed response adds no information beyond the symmetric statistic.",
         confidence="low", tags=["metagame", "cross-repo", "causal"]),
    dict(title="A specific decision is load-bearing",
         statement="A specific agent decision (e.g. tech-card play timing) actually moves win rate, by counterfactual test.",
         test="Re-simulate matches with only that decision randomized, everything else fixed; measure the win-rate flip.",
         refute_condition="Win rate is unchanged within the noise band (the decision was cosmetic).",
         confidence="low", tags=["diagnosis", "counterfactual"]),
    dict(title="Mixed-strategy deck selection beats best single deck",
         statement="Solving the small deck-selection payoff matrix for a maximin/mixed strategy out-rates always bringing the single highest-average deck against a non-stationary field.",
         test="A/B on the ladder: single-deck vs matrix-mixed selection.",
         refute_condition="The mixed strategy does not exceed the single deck beyond the noise band.",
         confidence="medium", tags=["metagame", "game-theory"]),
    dict(title="Heuristic-as-prior cuts playouts 10x",
         statement="ISMCTS with the Step-0 policy as prior plus leaf eval reaches the same strength as vanilla ISMCTS with roughly 10x fewer playouts. (Conditional on the forward model.)",
         test="Fix a strength target (self-play win rate vs a frozen reference); measure playouts to reach it, prior vs vanilla.",
         refute_condition="The playout reduction is far below an order of magnitude.",
         confidence="medium", tags=["ismcts", "search"]),
]


def main() -> None:
    for h in SEED:
        hid = R.add_hypothesis(**h)
        print(f"added {hid}: {h['title']}")
    R.render()
    print("rendered BELIEFS.md and GRAVEYARD.md")


if __name__ == "__main__":
    main()
