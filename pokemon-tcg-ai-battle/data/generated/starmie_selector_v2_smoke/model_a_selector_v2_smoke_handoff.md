# Model A Handoff: Conservative C3 Selector — Second Tiny Smoke Result

*(Verdict adversarially reviewed and DOWNGRADED from a first-pass "clean directional" to NEUTRAL. The safety claim
is deterministic and holds; the win-rate claims do not survive the n=20 / key-matchup scrutiny.)*

## Verdict: B_C3_SELECTOR_SMOKE_NEUTRAL — Promotion: DO_NOT_PROMOTE / NEEDS_N500

C3 is **deterministically safe** and **removes the V1 failure surface** (no terminal overrides, no -35pp crash).
But it is **not a directional win**: on the matchups that decide the ladder it is flat vs off and below the simpler
`top1_gate`, and 100% of its aggregate gain is against weak field decks. At n=20 nothing is statistically real.

## What HOLDS (deterministic)
- **In-repo parity PASS:** selected 100%, terminal-override-block 100% (block_ATTACK 46/46, block_END 4/4,
  block_RETREAT 4/4, allow_* correct), top-k 100%, logit diff 0.0, packer 100%, 0 metadata failures, no obs mutation.
- **0 errors / 0 illegal across 360 games.** Off-mode baseline-identical; all 9 test suites pass.
- **C3 emitted ZERO terminal (ATTACK/END/RETREAT) overrides** (full 848-row log) and **blocked 459 live**. The exact
  V1 `top3_selector` failure surface is gone.
- **No catastrophic regression:** the V1 -35pp deployed-mirror crash did NOT recur.

## What is NOT supported (do not promote on this)
- **The key matchups (deployed+mirror) are FLAT, not a win.** Combined: off **52.5%** = c3 **52.5%** (0.0pp), and
  c3 is **below** `top1_gate` (57.5%). On deployed specifically: c3 **45%** vs top1 **60%** vs off **50%** — the
  conservative selector is the *worst* of the three on the single most-relevant opponent. (mirror +5 / deployed -5
  are one-game n=20 noise; Fisher p~=1.0 on every cell.)
- **100% of C3's aggregate gain is field-driven.** Aggregate off 69.2% -> c3 80.8% (+11.7pp), but the decomposition
  is field +17.5pp / key 0.0pp; all 14 extra wins over off come from the 4 weak decks (several already at ceiling),
  **0** from deployed+mirror. This is the same field benefit V1 top3 had — C3 just doesn't crash the mirror.
- **n=20 is underpowered.** Need ~+30pp/cell for p<0.05; every observed direction is noise.

## Win rate by mode x opponent (20 games each, 0 errors)
| Opponent | S0 off | S1 top1_gate | S2 c3_family_limited |
|---|---|---|---|
| deployed | 50% | **60%** | 45% |
| mirror | 55% | 55% | 60% |
| alakazam | 65% | 90% | 100% |
| denpa92 | 95% | 80% | 100% |
| first | 60% | 70% | 80% |
| random | 90% | 95% | 100% |
| **deployed+mirror** | 52.5% | **57.5%** | 52.5% |
| **aggregate** | 69.2% | 75.0% | 80.8% |

## Data-integrity issues found by adversarial review (and fixed for N500)
1. **Trace logger did not record BLOCKED decisions** (only aggregate counters) — so "blocking terminals avoids the
   mirror loss" has zero per-decision outcome evidence in this pack. FIXED: the harness now logs blocked-terminal
   rows with outcome + reason.
2. **`game_id` was a non-unique shard label** (24 distinct for 360 games) — per-game analysis required segmenting on
   `first_changed`. FIXED: game_id now carries a globally unique task index.
3. **`first_changed_outcomes` was keyed by mode-id not mode-value** (reported 0; truth is 233 first-changed rows,
   181W/52L, confounded by field). FIXED in the diagnostic.

## Recommended next step
A **powered A/B (N~500/matchup)** of off vs c3, **judged on the deployed+mirror cells, not the field aggregate**,
with the fixed per-game + blocked-decision logging. Pre-register that promotion requires c3 >= off on deployed+mirror
at power. The honest open question this smoke leaves: does conservative selection *help* the mirror at all, or only
avoid hurting it? On these data it only avoids hurting, and `top1_gate` (which still permits terminal overrides)
edges it on the key cells — so the family-limit may be removing useful overrides along with the harmful ones.

## Artifacts
`pokemon-tcg-ai-battle/data/generated/starmie_selector_v2_smoke/`: `live_smoke_summary.json`,
`changed_decisions.jsonl` (1779 rows), `diagnostic_report.json` (with key_matchup_analysis + aggregate_decomposition
+ what_is_overclaimed), `review_examples.html`, `selector_v2_parity_report.json`, `baseline_manifest.json`.
Vendored: `agent/vendor/portable_selector_v2/`. Wiring: `agent/starmie_heuristics.py` (`c3_family_limited`, default off).

DIAGNOSTIC_VERDICT=B_C3_SELECTOR_SMOKE_NEUTRAL
PROMOTION_STATUS=DO_NOT_PROMOTE / NEEDS_N500
