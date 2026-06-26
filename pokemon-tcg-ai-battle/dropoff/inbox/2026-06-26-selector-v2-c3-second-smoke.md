# Starmie Conservative C3 Selector — Second Smoke (2026-06-26)

Model B. Ran the second live smoke on Model A's conservative C3 (`c3_family_limited`) portable export, with per-game
changed-decision logging. **Verdict adversarially reviewed and downgraded A->B.**

## Verdict: B_C3_SELECTOR_SMOKE_NEUTRAL · Promotion: DO_NOT_PROMOTE / NEEDS_N500

## What is solid (deterministic)
- Gate met; Model A delivered `portable_selector_v2/export`. In-repo parity PASS (selected 100%, terminal-block
  100%, top-k 100%, logit diff 0.0, packer 100%, deterministic, 0 metadata).
- C3 wired (default off, fail-closed). 360 games, **0 errors / 0 illegal**. All 9 suites pass.
- **C3 emitted 0 terminal (ATTACK/END/RETREAT) overrides; blocked 459 live.** The V1 `top3_selector` -35pp
  deployed-mirror crash did NOT recur. Family-limit safety works exactly as designed.

## What I almost overclaimed (caught by adversarial review)
First pass leaned "field-positive, mirror-neutral, clean directional (A)." The review refuted the directional part:
- **Key matchups (deployed+mirror) are FLAT**: off 52.5% = c3 52.5%, and c3 is BELOW top1_gate (57.5%). On deployed
  alone c3 45% < top1 60% < ... actually c3 is worst of the three. mirror +5 / deployed -5 are n=20 noise (p~1.0).
- **100% of C3's +11.6pp aggregate is field-driven** (14/14 extra wins vs weak decks, 0 from deployed+mirror).
- n=20 supports only "deterministic safety + no catastrophe", no win-rate direction.
- Found + fixed 3 logging bugs: blocked decisions weren't logged; game_id non-unique; first_changed_outcomes
  mis-keyed.

## Win rate
| Opp | off | top1 | c3 |
|---|---|---|---|
| deployed | 50 | **60** | 45 |
| mirror | 55 | 55 | 60 |
| alakazam | 65 | 90 | 100 |
| denpa92 | 95 | 80 | 100 |
| first | 60 | 70 | 80 |
| random | 90 | 95 | 100 |
| dep+mirror | 52.5 | **57.5** | 52.5 |
| agg | 69.2 | 75.0 | 80.8 |

## Open question for Model A's next iteration
Does conservative selection *help* the mirror, or only avoid hurting it? On these data it only avoids hurting, and
top1_gate (which still permits terminal overrides) edges it on the key cells — the family-limit may be cutting
useful overrides along with harmful ones. Next: powered A/B (N~500) judged on deployed+mirror, with the fixed
logging, pre-registering "c3 >= off on key cells" as the promotion bar. No submission; selector stays default off.

## Artifacts
`data/generated/starmie_selector_v2_smoke/`: live_smoke_summary.json, changed_decisions.jsonl (1779),
diagnostic_report.json, review_examples.html, selector_v2_parity_report.json, model_a_selector_v2_smoke_handoff.md.
