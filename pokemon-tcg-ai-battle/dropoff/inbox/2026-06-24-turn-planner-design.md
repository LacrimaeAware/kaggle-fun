# Turn planner: heuristics propose, search scores the whole turn (2026-06-24)

Goal 3 (integrate search with heuristics) and the one lever that addresses the real bottleneck. Status: design,
not built. Build is the next step pending a go.

## Why

1-ply search branches a single decision then finishes the turn with a fixed rollout. So it cannot represent
the value that lives in the SEQUENCE inside a turn: tutor first, draw into the pieces, evolve, attach, attack
last. That is why leaf-eval tweaks keep coming back neutral (card-advantage 0.525; PH-potential trending no).
The fix is not a better leaf number, it is evaluating whole-turn sequences.

## Design

At the start of MY turn:
1. The HEURISTIC proposes a small set of candidate PLANS (ordered intents), using domain knowledge, not
   brute force. Target 3 to 6 plans, e.g.:
   - KO-last: if a KO exists, do all non-endangering development first, take the KO last (the KO-sequencing
     finding: a KO cannot be lost by waiting, only by your own endangering actions, so attack last).
   - Setup: tutors/draw first (Poffin -> Abra, Telepath, Psychic/Run-Away Draw), then evolve the Abra line,
     then attach Psychic, then pass.
   - Develop-attacker: attach + evolve toward an online attacker, modest draw.
   - Deck-safe variants: the same plans with any draw that breaches the deck-out buffer removed.
2. SEARCH executes each plan in the forward model to the end of MY turn (search_begin/step/end), averaged over
   determinizations for hidden info, and scores the end-of-turn state with the leaf eval (now including the
   deck-out term).
3. Pick the plan with the best averaged end-of-turn score. Execute its first action; re-plan next decision.

This beats 1-ply because the tutors-first cascade and the bank-the-KO ordering are actually represented and
compared as end-states, instead of being cut off by a one-decision branch.

## Principles (from the ladder result)

- Search is a CALCULATOR scoped to the heuristic's candidate plans and objective. It never enumerates all
  move orderings (intractable, and enumerate-all search lost on the ladder: 645-687 vs heuristic 736).
- The PH-aware KO floor becomes one candidate plan (KO-last), not a hard override, so search can choose to
  develop first when that scores better.
- The deck-out buffer is both a plan-generation constraint (do not propose draws that breach the buffer) and a
  leaf term (already validated, deck-out 31-19). Near-absolute refusal of self-deck-out, with the two
  exceptions falling out of weight ordering (big prize swing or a this-turn KO override it).

## Reuses

- `search_v3` forward model and determinization machinery.
- `eval.evaluate` plus the deck-out term (`evaluate_deck_v3` with `deckout_weight`).
- `deck_policy_v3` PH-aware attack arithmetic for the KO-last plan.

## The hard part (build risk)

Mapping a PLAN (a sequence of intents like "Poffin for 2 Abra, then evolve, then attach, then attack") onto
the engine's option prompts. The engine presents options at each step; the planner must, at each prompt,
select the option that matches the next intended action in the plan, and fall back gracefully when an intent
is not currently legal. This intent-to-option matcher is the main new code. Start with 2 plans (KO-last vs
setup) to prove the loop before adding more.

## Budget

Self-imposed 0.6s; real limit ~600s/game. A handful of plans times a few determinizations times a short
end-of-turn rollout is feasible. If tight, cut determinizations before cutting plans.

## Test

A/B the turn planner vs `phaware_search` (deployed) with `tools/quick_ab_v1.py` (durable, segmented, small n).
Keep `phaware_search` as the control. The bar: clear 50% on the small read, then confirm. Ladder submission is
the only real arbiter (local self-play does not predict it), so a positive local read warrants a submission.
