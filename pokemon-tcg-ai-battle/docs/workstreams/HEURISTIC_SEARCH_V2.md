# Heuristic Search V2 — fail-closed proposals

**Goal:** improve `agent_search` with deck knowledge **without ever letting heuristics replace search.**
The full `deck_policy` candidate lost ~5-95 because `choose_subdecision`/`rollout_choice` broadly
*replaced* search/default behavior. V2 is the opposite architecture: heuristics **propose**, a paired-world
search **validates**, and the baseline default is always a candidate and wins unless a proposal is clearly
better. Unknown contexts are untouched. Any failure → default. **Fail closed.**

## The contract (do not violate)
1. Baseline/default selection is always available and is the fallback.
2. Heuristics only generate alternative candidate selections (`deck_policy_v2.propose`). They never pick the final move.
3. Candidates are evaluated on **paired hidden worlds** (`deck_policy_v2.compare_selections`).
4. A heuristic candidate is used only when search says it is better, or statistically indistinguishable (then tie-break only).
5. Unknown contexts return `[]` → no change. Exceptions / incomplete coverage / low time → default.
6. No generic target-score blob, no generic NUMBER policy, no generic optional-decline, no global rollout replacement, no live override before search.

## Files (new only; production agent untouched, not merged)
- `agent/deck_policy_v2.py` — `propose(obs, enabled)`, `compare_selections(...)` validator, M0 helpers.
- `tools/ab_heuristic_search_v2.py` — isolated harness, candidate toggles, full instrumentation, control gate.
- `docs/workstreams/heuristic_search_v2_results.json` — per-candidate results + instrumentation.

## Candidates (run/validate one at a time; hold deck, N, budget, eval fixed within a comparison)
- **A0** — exact baseline control (all toggles off). Expect ~0.5 vs production baseline; fail the harness if skewed.
- **S32** — N=32 one-ply sampling only, no time cutoff, read `remainingOverageTime`. Separate axis. (Already measured 0.625, 25-15 over 40 games head-to-head; not evidence for any heuristic.)
- **M0** — mechanical fixes only: (A) dynamic Powerful Hand attack value = `20 * hand size` in static attack selection, KO detection, final-prize detection; (B) force only a confirmed game-ending prize attack (ordinary KOs fall through to search). No other heuristic changes.
- **M0 + Poffin (R1)** — first resolver: on a "put Basic Pokémon onto Bench" prompt, propose a balanced Abra + Dunsparce setup; validated, default always retained.
- Then, one at a time, only if each survives: R2 evolution/tutors (Dawn/Hilda/Poké Pad/Rare Candy, each its own rule), R3 Boss (concrete KO/strand only), R4 recovery (Night Stretcher), R5 Run Away Draw (state-dependent).

## Ablation ladder (per candidate)
- Stage 1: synthetic legality + 2 traced games.
- Stage 2: 10-game smoke. **Stop** if win ≤ 0.30, passive-play pathology, or search coverage collapses.
- Stage 3: 40-game directional confirmation only if smoke survives.

**Auto-reject a candidate if:** search-call coverage drops >10% vs baseline; END choices spike; attack/evolve frequency collapses; fallback or swallowed exceptions rise; the baseline candidate is ever omitted.

## Not in scope yet
- No broad leaf weights during resolver tests (`eval.py` identical to baseline). Future, independently toggled: capped Powerful Hand hand value, evolution-line readiness, attacker continuity, deck-out risk.
- Learned gating later: `contextual_risk_only_v1` predicts general search risk, not "is this proposal better than default." A future gate trains on `target = paired_value(proposal) − paired_value(default)`.

## Status (2026-06-21)
- Framework built; harness smoke clean (search used on 199/201 single-pick decisions, 0 errors).
- Running the first report: A0 control, M0, M0+Poffin. Stop after the first resolver confirmation and request review before adding R2.
