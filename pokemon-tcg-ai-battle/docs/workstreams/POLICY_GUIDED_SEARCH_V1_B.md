# Policy-Guided Real-Depth Search V1 — Model B (expert policy / continuation prior)

Branch: `exp/robust-learner-v2`. Goal: a learned deck-specialist continuation prior from strong-pilot
replays, supplied to Model A's search as candidate moves and opponent-reply weights. **Never a standalone
live agent; never the final chooser.** Search evaluates; the prior only proposes/orders.

## B1 — expert dataset (DONE)
`tools/build_expert_policy_dataset_v1.py` -> `data/expert_policy/dataset_v1.jsonl` (gitignored; regenerate from replays).
Grouped-sibling rows: root public-state features (`features.encode_state`), every legal option's
`(type, acting_card_id, target_id)`, the expert's selected option(s), context, player, game, deck tier, outcome.

Result: **165,742 decisions, 1,468 games, 647 players.**
- Tier 1 (near-exact our deck) 14,307 ; Tier 2 (near-identical) 21,572 ; Tier 3 (archetype) 4,423 ; Tier 4 (generic) 125,440.
- **Our-deck (Tier 1-2) = 35,879** decisions for the deck-specialist head.
- Selected-type mix: CARD/tutor 40,952 ; PLAY 38,167 ; multi-select 29,577 ; ATTACH 20,907 ; ATTACK 12,032 ; END 7,399 ; EVOLVE 5,698 ; ABILITY 4,720.

## B2 — deck-specialist behavior policy (NEXT)
`agent/expert_policy_v1.py` + `tools/train_expert_policy_v1.py`. Small grouped sibling-action model:
shared option encoder (type/card/target embeddings + root features + interactions -> MLP score), softmax
per decision, cross-entropy on the expert sibling. **Option-order permutation augmentation** so it can't win
by memorizing option index. Two heads: `our_deck` (Tier 1-2) and `generic_opponent` (all tiers).

## B4 — gates (offline)
Report by held-out game and player: top-1, top-3 recall, MRR, option-0 baseline, per-action-type
performance, calibration. Integrate only if: top-3 recall materially above the option-order baseline;
no option-order leakage; candidate identity alignment > 99%.

## B5/B6 — runtime use
Export `agent/expert_policy_v1.json|npz`; `score_options(obs, legal_options, role) -> list[float]`,
role in {"our_deck","generic_opponent"}; never returns an illegal option; failure -> no prior, search unchanged.
Used as a PRIOR (top-3 candidates + baseline/default + forced tactical candidates), never the chooser.

## B7 — expert iteration (later)
After A's real H2 works: label hard states with higher-compute H2 search, add soft targets, retrain, <=2 bounded rounds. Reuse DAgger machinery; do not restart the old ranker objective unchanged.

## Status (2026-06-21)
- B1 dataset built (above). remainingOverageTime confirmed live (600s, readable) for A's time governor.
- Next: B2 train + B4 metrics. A-side builds search_v2 H1/H2 in parallel on `exp/planner-teacher-v2`.
