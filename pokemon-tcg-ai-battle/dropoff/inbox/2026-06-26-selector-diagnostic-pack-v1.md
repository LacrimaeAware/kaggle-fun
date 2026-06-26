# Starmie Selector Live-Smoke Diagnostic Pack V1 (2026-06-26)

Model B. Turns the (legal-but-not-promotable) selector live smoke into a clean diagnostic for Model A.
**DIAGNOSTIC_VERDICT = B_FAILURE_MODE_DIRECTIONAL.** No games run, gameplay unchanged, selector stays off.

## Headline
`top3_selector` regresses on the deployed Starmie mirror (20% vs 55% off). The offline mechanism points to
**low-confidence rank-2/3 turn-terminating overrides** (develop -> ATTACK/END), but the causal link to the live
losses is **not measured** — so the finding is directional, not confirmed.

## Important honesty correction (from adversarial verification)
My first pass leaned toward "clear cause = premature terminal overrides." A 4-reviewer adversarial workflow
downgraded it to **directional** and was right to. The offline counts are solid; the *causal attribution* is not:
- The neutral-vs-regressive split lives in one **n=20** mirror cell. top1-vs-top3 (9 vs 4) is Fisher **p~=0.18,
  not significant**; the 300-game aggregate (78-81%) is flat.
- `proposer_rank` is a coarsening of `proposer_prob` (AUC 0.98) — rank and prob are **one** confidence axis, not two.
- `top1_gate` (rank-1 only) **still loses the mirror -10pp** and still emits **128 rank-1 premature terminations**
  (60% of all premature). A pure rank gate cannot reach them.
- A **RETREAT family appears only in top3 (33 picks, 0 in top1)** plus 21 attack re-targets — a non-terminal
  co-driver the terminal framing ignores; untestable without per-game logs.

## Offline mechanism (2564 Starmie single-select replay decisions x 2 modes, all counts reproduce exactly)
| | top1_gate | top3_selector |
|---|---|---|
| override rate | 43.2% | 51.8% |
| terminal overrides | 261 | 388 |
| premature-terminal | 128 | 213 |
| develop->ATTACK / ->END | 205 / 37 | 287 / 81 |
| terminal proposer-prob (rank1 / rank2+) | — | 0.707 / 0.184 |
| RETREAT picks | 0 | 33 |

Field-vs-mirror: top3 **beats** off on 3/5 field decks (+10pp) but **crashes** the mirror (-35pp) — the classic
"selection helps weak policies, hurts the strong mirror" failure (develop-before-attack).

## Data gaps (load-bearing)
1. No per-game outcome linkage — smoke saved aggregate win/loss only; `game_result`/`matchup` are null in every row.
2. Classification on expert-pilot states; the regression is on our-agent mirror states (distribution shift).
These are why the verdict is B not A. Closing them needs an instrumented mirror re-run (Model B's lane).

## What Model A's selector V2 should gate (from the handoff)
- Gate terminal (ATTACK/END) overrides on **stakes, not rank**: block when safe-dev remains AND no guaranteed-KO
  AND no game-win — regardless of rank (reaches the 128 rank-1 premature terminations a rank gate misses).
- Keep KO/gamewin terminals. Decide separately on the 33 top3-only RETREAT picks + attack re-targets.
- Do not ship any gate as outcome-validated; the literal rank-1/KO/no-safe-dev gate blocks only 85 of 213 premature
  terminals, so expect the mirror to land between -10pp and -35pp, not restored to 0.

## Artifacts
`data/generated/starmie_selector_live_smoke_v1/`: `model_a_selector_failure_handoff.md` (the handoff),
`changed_decision_classes.jsonl` (5128 rows), `failure_aggregate_report.json`, `mirror_regression_report.json`,
`selector_smoke_review.html`/`.jsonl` (76 examples), `DIAGNOSTIC_VERDICT.json`. Tools: `selector_diagnostic_*_v1.py`,
`validate_diagnostic_pack_v1.py` (integrity PASS).

## Model B's required follow-up (when greenlit)
Instrument the mirror smoke: log per-game first-changed-decision + rank + terminal flag + result, power to
~60-100 games/mode, and ablate C3 into (rank-1 terminals + rank-2/3 non-terminals) vs full C3 to isolate whether
the extra terminals or the extra non-terminals (RETREAT) drive the loss.
