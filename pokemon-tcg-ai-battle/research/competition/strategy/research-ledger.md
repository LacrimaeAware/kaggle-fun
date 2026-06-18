# Running Strategy Category research ledger

This is a living one-line ledger for the eventual Kaggle Strategy Category write-up.

The goal is not to claim the project is done. The goal is to preserve the research story while the agent is still being built.

## Current narrative snapshot

As of 2026-06-18, the project is only a few days old. The most coherent story is:

> We started with legal heuristics and forward search, discovered that global state-value prediction did not reliably improve move choice, and shifted toward action-conditioned sibling ranking with decoded card effects, root-to-leaf consequence deltas, and hidden-information-aware search.

## One-line experiment/story ledger

| Lane | One-line finding | Write-up value | Status |
|---|---|---|---|
| Legal baseline | A legal fallback/heuristic agent establishes a playable floor. | Shows basic simulator integration and legality. | Foundation |
| Forward search | Simulating legal actions with the engine became the strongest practical family so far. | Shows dynamic consequence reasoning. | Strong baseline |
| Global value learning | Good prediction metrics did not reliably translate to better gameplay. | Central negative result. | Important lesson |
| Search distillation | Fitting search targets recovered parity-like behavior but did not clearly exceed the teacher. | Shows target circularity / teacher ceiling. | Caution |
| `search_v` | Learned leaf value did not beat plain search across tested decks. | Shows old 47-feature value branch is weak. | Do not over-interpret |
| Card effects decoder | Decoded effects are useful infrastructure, but a file on disk does not affect decisions. | Explains artifact-vs-live-policy distinction. | Needs live integration |
| Hand-weighted effects | Naive effect weights reportedly over-prioritized setup and lost to baseline. | Shows why learned/state-conditioned use is needed. | Negative but useful |
| Action ranking | The intended target is sibling legal-option ranking, not absolute leaf scoring. | Main methodological pivot. | In progress |
| Replays | Public replay JSON can expose decks and decisions. | Supports expert imitation and belief modeling. | Promising source |
| Deck choice | Deck and agent must be co-designed; current best deck may keep changing. | Supports Deck Score. | Live decision |

## Hypotheses to preserve going forward

| ID | Hypothesis | Evidence needed |
|---|---|---|
| W1 | Forward search beats static heuristics because it sees immediate consequences. | Head-to-head results and example tactical positions. |
| W2 | Global value AUC/Pearson is insufficient for move choice. | AUC/Pearson vs win-rate mismatch table. |
| W3 | Sibling-action ranking improves local decisions better than state-value regression. | Within-decision ranking metrics and head-to-head validation. |
| W4 | Decoded card effects help only when connected to state-conditioned action scoring. | Effects-enabled vs effects-zeroed ablation. |
| W5 | Learned card embeddings are useful as residual identity features, not as magic card-text understanding. | Embedding/effect/tag ablations. |
| W6 | Belief-conditioned determinization improves search robustness. | Replay hidden-card likelihood and fixed-budget A/B test. |
| W7 | Critical decisions matter more than average decisions. | Candidate-spread/regret analysis. |

## Future table of contents for final report

1. Problem: dynamic strategy under hidden information.
2. Deck concept: game plan, key cards, and why it creates learnable decisions.
3. Agent architecture: legal fallback, heuristic, search, learned action prior.
4. Experiments: what worked, what failed, and what changed the agent.
5. Key insight: rank sibling actions, not states.
6. Card effects and representation: decoded text, embeddings, and state-context interactions.
7. Hidden information and robustness: determinization, replay priors, seed/seat stability.
8. Final submitted agent and evidence.
9. Limitations and future work.

## Figure and table backlog

| Artifact | Purpose | Source |
|---|---|---|
| Architecture diagram | Explain full decision pipeline. | `writeup/draft-skeleton.md` |
| Experiment summary table | Compress messy research arc into evidence. | Registries + inbox notes |
| Deck strategy diagram | Explain deck concept for Deck Score. | Final deck list |
| Decision-group diagram | Show root state and sibling legal options. | Action dataset / example position |
| AUC-vs-win-rate table | Show prediction/action gap. | Value experiments |
| Effects ablation table | Prove card effects are live and useful or not. | Future action-ranker work |
| Hidden-state diagram | Explain determinization/belief modeling. | Future replay work |

## Process rule

No future result should be entered as a write-up claim unless it records:

- exact agent/version
- exact deck
- opponent
- sample size / seeds / seat handling
- metric and confidence interval where applicable
- interpretation
- whether the submitted agent changed

