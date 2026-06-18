# Research Questions: Pokemon TCG AI Battle — A Flexible Learned Multi-Deck Agent

> For a deep-research model. Goal: tell us where to point effort and what to stop chasing.

## Context (read first)

We are building an agent for a Pokemon Trading Card Game battle simulation competition. The
organizer's real forward-model engine runs locally (we can simulate any legal action one step or roll
out full turns). We have **135 real ladder replays** distilled into a DB: **14,442 outcome-labelled
decisions** across **57 decks**, each recording both seats' full observation (board + the engine's
legal `option` list) and the action taken, plus who won. We also have a card database with structural
fields AND full English effect text for every card (1267 cards, currently undecoded into features).

**The honest empirical state.** Our best agent is a 1-ply forward search with a hand-tuned linear
leaf evaluator (~0.585 vs the baseline opponent). **Every learned variant we have tried clusters at
~0.50 win-rate vs that heuristic.** A separate imitation diagnostic (predict which of ~8 options the
winner chose) gives, after fixing a bug and using the right baseline: random 0.197, **chose-option-0
positional prior 0.587**, heuristic floor 0.553, learned pointwise GBM ~0.50 (and BELOW the option-0
prior on both which-card and mixed strategic slices). So a learned ranker on our current features does
not yet beat "pick the first option the engine offers."

**Known confounds we have already found** (so you don't repeat them): a now-fixed dead card-join; a
strong positional prior (58.7% of winner moves are option 0); a pointwise objective on an inherently
listwise problem; state features that are constant within a decision; no stratification/CIs; and the
fact that winners pilot DIFFERENT decks so "winner's move" is deck/meta-entangled.

**The user's goal (do not lose sight of this).** A flexible learned agent that pilots DIFFERENT decks
at least moderately well — NOT a per-deck heuristic and NOT "master one easy deck." The standing
hypothesis is a "learned heuristics" bridge: richer per-action features + behaviour cloning from
winners, possibly fused with the forward model.

---

## Section 1 — Representation vs objective vs depth: which is binding?

The recurring symptom: a state value separates positions globally (AUC ~0.74) yet cannot separate
sibling MOVES at 1-ply.

**Q1.1** How do we design experiments that isolate representation, objective, and search-depth as
INDEPENDENT axes? We want a small named experiment matrix (hold two axes fixed, vary the third) with
a decision rule per cell. Specifically: a within-decision centered/listwise objective vs the pointwise
loss; state×action interaction features so the constant state vector becomes discriminative;
2-ply/determinization-deepened search with the same leaf. Most valuable single deliverable: a
principled reading of "global AUC high but sibling-ranking flat" — is this a known signature of an
objective mismatch, a coarse-model-class artifact, or a genuine information limit?

**Q1.2** Is the ~0.50 plateau even real, or a positional/aggregation artifact? 58.7% of winner moves
are option 0; many decisions are trivial or pure which-card. We want a recommended stratification +
metric scheme (which sub-populations to score separately, which baselines per stratum, whether
top-3/MRR should accompany top-1, how to bootstrap CIs) — the smallest set of conditioned metrics
that lets us say "features help on stratum X" with confidence.

*Do not waste time on:* generic "tune hyperparameters / try a deeper net" divorced from axis
isolation; anything that requires the broken pointwise-on-aggregate setup as the evaluation.

---

## Section 2 — Imitation / behaviour cloning from winners, done right

**Q2.1** Correct objective and metric for cloning WITHIN-decision choice from a variable-size,
unordered option set (avg ~8.2, one chosen)? This is listwise ranking / discrete choice, not pointwise
classification. Recommend among lambdarank/pairwise, softmax-over-options, or a conditional-logit
(McFadden) formulation, plus the right metric family (top-1/top-k/MRR/NDCG/log-loss) and which to
pre-register. If there is a clean way to encode "multiple options were acceptable" (soft labels,
partial credit from the forward model), we want it.

**Q2.2** How do we avoid the label-circularity / survivorship trap in "the winner's move was a good
move"? Survivorship (winners still misplay), outcome leakage (won by variance/matchup, not per-move
optimality), and deck entanglement (each game is a different real deck). We want practical de-biasing:
margin/outcome-weighted imitation, restricting to high-skill/decisive games, importance weighting, or
forward-model relabeling; AND how to DETECT that a model fit deck/meta priors rather than skill
(leave-one-deck-out, deck-feature ablation). If winner-only BC is fundamentally circular here, say so
plainly with the alternative.

**Q2.3** What makes a cloned policy GENERALISE across decks rather than memorise archetypes? With 57
decks (one appears in 52 games) overfitting to a few archetypes is easy. We want representation
choices known to induce deck-agnostic transfer (functional action featurization vs card identity;
within-decision relative features; deck-conditioning embeddings; domain-randomization over decks) and
any evidence on how much deck diversity transfer actually needs.

*Do not waste time on:* pure imitation pipelines that ignore the forward model (we have the real
engine); generic "fine-tune a big model on the logs" — sample efficiency and circularity dominate at
this scale.

---

## Section 3 — The "learned heuristics" / neuro-symbolic bridge: what could it concretely be?

**Q3.1** Enumerate and rank 2–4 concrete realizations for OUR setting (perfect forward model + card
effect-text DB + outcome-labelled human decisions): (a) learned leaf evaluator inside the existing
1-ply search; (b) forward-model action-delta features into a learned ranker; (c) symbolic effect-text
features feeding a small model; (d) learned weights over hand-designed heuristic primitives. For each:
where it has worked (card games, board games), sample-efficiency/interpretability, and interaction
with a perfect forward model. The decisive question: **given that we can simulate any action one step,
is decoding effect TEXT into symbolic features ever worth it, or does forward-model action-delta
featurization strictly dominate it?**

**Q3.2** Are forward-model action-delta features (immediate post-action state delta per option, no
rollout) the actual bridge, and how to build them? Evidence on whether consequence/delta featurization
is what lifts move-selection in forward-model agents; how to combine cheap one-step deltas with a
bounded opponent-reply rollout without blowing a ~0.6s/move budget; whether our validation logic is
sound (re-run the diagnostic, clear the option-0 / heuristic bar) and what confound it misses.

**Q3.3** Is imitation top-1 even the right yardstick, or should the bridge be validated on win-rate?
Our 1-ply agent loses on imitation top-1 (~0.42) yet wins ~0.585 on win-rate — direct evidence the two
diverge. We want a principled stance on when imitation top-1 is a valid proxy, and a recommended
win-rate A/B protocol (self-play on the real engine, Wilson CIs, n≥400, across 2–3 decks) as the
ground-truth gate every change must pass.

*Do not waste time on:* abstract neuro-symbolic surveys with no card-game grounding; from-scratch deep
RL as the bridge (14k decisions, hours of self-play budget) unless you have specific sample-efficiency
evidence.

---

## Section 4 — How do strong TCG agents represent actions and pilot multiple decks?

**Q4.1** How do the strongest known TCG / card-battler agents represent the ACTION SPACE
(functional/effect-based vs card-identity vs simulated-consequence; with or without a forward model;
action embeddings / factored / pointer-attention over option lists / autoregressive decoding)? We
specifically want to know if "featurize the action by its simulated one-step effect" is an established
technique with a name and known pitfalls. Look at Hearthstone, MTG (Forge), Pokemon TCG bots,
Yu-Gi-Oh, Legends of Code and Magic, and large/variable/parameterized-action-space RL.

**Q4.2** How do these agents achieve (or fail at) MULTI-DECK / cross-archetype play with one model
(deck embeddings, deck-blind representations, mixture-of-experts, self-play across a deck pool vs human
logs)? If the literature's verdict is "single-model multi-deck play is hard and strong systems still
specialize," state that honestly with citations — it would reset the user's expectations.

**Q4.3** For a game with a perfect forward model but hidden information (opponent hand/deck), what is
the established search + learning recipe? Determinized search done well (PIMC pathologies: strategy
fusion, non-locality), Information-Set MCTS, how learned value/policy nets bolt onto imperfect-info
forward-model search; whether to sample opponent decks from the meta (we have the real pool), how many
determinizations matter, and whether a learned policy is best as a search prior (AlphaZero-style) or a
standalone fast policy. A clear PIMC-vs-ISMCTS-vs-policy-prior recommendation at our scale would set
our search architecture.

*Do not waste time on:* perfect-information results (chess/Go) presented as directly applicable; RL
textbook background. Name systems, their action/deck representations, and their honest limitations.

---

## What would most change our direction (priority)

1. A clean way to decide representation vs objective vs depth (Section 1) — gates everything.
2. An honest verdict on whether winner-only BC is salvageable or fundamentally circular (Q2.2).
3. A ranked, card-game-grounded enumeration of the "learned heuristics" bridges (Q3.1), and whether
   the forward model makes effect-text decoding redundant.
4. Whether single-model multi-deck play is an achieved result anywhere and how (Q4.2) — a reality
   check that could reset the target.

Throughout: prefer findings with a falsifiable test we can run on our engine (win-rate A/B with
Wilson CIs is ground truth; imitation top-1 is at best a representation-discovery proxy). State
limitations and negative results plainly.
