# Model A Handoff: Starmie Learned-Selector Live-Smoke Failure Forensics

*(Diagnosis adversarially verified by 4 independent reviewers + synthesis; every quantitative claim below was
re-derived from `changed_decision_classes.jsonl` and reproduces exactly. The verdict was downgraded from "clear"
to "directional" by that review — see the load-bearing caveats in section 5.)*

## 1. Summary

A learned single-select selector was wired behind `STARMIE_SELECTOR_MODE` (default off, fail-closed) and
live-smoke tested over 300 games (20 games x 5 opponents x 3 modes, 0 errors). Two selector modes were tested
against the heuristic baseline (`off`):

- **`top1_gate`** — selector may override only when its pick is the proposer's rank-1 candidate.
- **`top3_selector`** — selector may override when its pick is in the proposer's top-3.

**Verdict: do not promote either mode.** `top1_gate` is roughly neutral in aggregate; `top3_selector` regresses
hard against the one opponent that matters — the deployed Starmie mirror. The offline mechanism is directionally
clear (top3 terminates the turn more often, on lower-confidence picks) but the causal link to the live losses is
**not measured** — there is no per-game outcome linkage, and the classification is on the wrong state distribution.

## 2. What passed

- **Infra / fail-closed wiring.** 300 games, 0 errors, 0 illegal actions. `off` is baseline-identical.
- **`top1_gate` aggregate is neutral.** 81% aggregate win vs 79% off-equivalent; +5pp on each of the 4 field opponents.
- **`top3_selector` beats `off` on the field.** alakazam +10pp, first +10pp, random +10pp, denpa92 +0pp. Aggregate 78%.
- **All offline mechanism counts reproduce exactly** from the raw 5128-record JSONL (2564 decisions x 2 modes):
  override counts, terminal/premature counts, per-rank breakdown, and the rank-confidence proposer-prob split.

## 3. What failed — exact regression mode

The failure is **opponent-specific to the deployed Starmie mirror**, the only near-coin-flip matchup (`off` = 55%):

| Mode | Mirror win% | Delta vs off |
|---|---|---|
| off (S0) | 55.0 (11/20) | — |
| top1_gate (S1) | 45.0 (9/20) | **-10pp** |
| top3_selector (S2) | 20.0 (4/20) | **-35pp** |

`top3_selector` collapses to 4/20 on the mirror while improving everywhere else. This is the classic "selection
helps weak policies, hurts the strong mirror" signature: the same added turn-ending aggression that beats
weaker/differently-piloted decks surrenders tempo against a disciplined symmetric opponent — i.e.
**develop-before-attack**, the known piloting error from prior imitation-gap work. Note `top1_gate` also loses the
mirror by -10pp: the damage is reduced but **not removed** by rank-1 gating.

## 4. Suspected cause (rank-confidence, offline)

The top1->top3 difference is concentrated in **terminal (turn-ending ATTACK/END) overrides**, not in non-terminal
sub-choices:

- Override rate: top1 43.2% (1107) -> top3 51.8% (1327).
- Terminal overrides: top1 261 -> top3 388 (+127).
- Premature terminal (terminal AND safe development still available AND no KO AND no game-win): top1 128 -> top3 213 (+85).
- The extra top3 picks are exactly the rank-2/3 candidates the rank-1 gate filters: **127 extra terminal, 85 extra premature**.
- DEVELOP->ATTACK 205->287; DEVELOP->END 37->81. SELECT_CARD sub-changes are ~equal across modes (539 vs 567), so
  they are **not** the differentiator.

The rank-2/3 extras are **low proposer-confidence**: rank-1 terminal proposer prob n=261 mean **0.707**; rank-2/3
terminal proposer prob n=127 mean **0.184**. On expert states, 99.2% of top3 terminal overrides occur with safe
development still available and 54.9% with no KO/game-win on the board.

## 5. Caveats (load-bearing — the verdict must not be read past these)

These are not footnotes; they are why this is directional, not confirmed.

1. **No per-game outcome linkage.** Every JSONL record has `game_result = null` and `matchup = null`. The smoke
   saved aggregate win/loss only. No terminal override is tied to any lost game. The rank-confidence story is
   **confidence-based, never outcome-based** — a plausible correlate, not a measured cause.
2. **Wrong state distribution.** All 2564 classified decisions are **expert-pilot replay states**. The regression
   manifested on **our-agent mirror states**, which develop more and produce different RETREAT/attack-retarget
   frequencies. The expert-state terminal rates are a **lower bound** for the live mirror — but that framing is
   asserted, never measured.
3. **The rank axis is one fact, not two.** `proposer_rank_of_pick` is a coarsening of `proposer_prob_of_pick`
   (AUC 0.98 between them). The 0.707-vs-0.184 separation largely restates "rank-1 is the argmax" — it does not
   independently corroborate the low-confidence claim. The independent levers (`selector_score_margin`, `entropy`)
   were collected but not used in the rank argument.
4. **The outcome split is underpowered.** The neutral-vs-regressive distinction lives only in n=20 mirror cells.
   top1-vs-top3 (9 vs 4) is Fisher p~=0.18 (not significant); off-vs-top3 is barely p~=0.05; the 300-game
   aggregates (78-81%) are flat. The -35pp vs -10pp **ordering** is directionally clear; the point magnitudes carry
   roughly +/-20pp CIs.
5. **Rank-1 gating is necessary-at-best, not sufficient.** `top1_gate` already restricts terminal overrides to
   rank-1, yet still loses the mirror -10pp and still emits **128 rank-1 premature terminations** (mean prob 0.659,
   safe development available, no KO). Premature termination is rank-1-dominated (128 of 213 = 60%), so a pure rank
   gate cannot reach it.
6. **A non-terminal co-driver is not excluded.** Of the 220 top3-exclusive (rank-2/3) overrides, **93 (42%) are
   non-terminal** — including a RETREAT family that exists **only** in top3 (33 picks, 0 in top1_gate; a forced
   active switch costing energy/tempo) and 21 attack re-targets. These are equally low-confidence and the proposed
   terminal-only gate would not touch them. With no outcome linkage, the data cannot rule them out as a co-cause.

## 6. What selector V2 should BLOCK / GATE (with C-mapping)

Candidate-set widths in `agent/vendor/portable_selector_v1/starmie_selector_runtime.py`: **C1** = top1 only;
**C2** = baseline u top1; **C3** = baseline u top3 (this is the `top3_selector` smoke mode); **C5** = baseline u
search u top3. The proposed gate is a **post-selection terminal filter, orthogonal to candidate-set width** — it
applies the same under C3 and C5, and under C5 (wider) it would leak even more rank-2/3 terminals than under C3.

Grounded by the evidence, V2 should:

- **Gate terminal (ATTACK/END) overrides on stakes, not on rank alone.** Block a terminal override when safe
  development remains AND no guaranteed-KO AND no game-winning attack is available — regardless of proposer rank.
  This is the only filter that reaches the 128 rank-1 premature terminations that survive C1/C2 and still lose the
  mirror -10pp. (Maps to **C1 + a development-preserving terminal veto**, i.e. Model A's C1/C5 intent.)
- If a rank gate is kept, note its measured reach: a literal gate (rank-1 OR KO/game-win OR no-safe-dev) over the
  388 top3 terminals **blocks only the 85 rank-2/3 premature terminals (40% of all 213 premature)** and allows 303.
  Expected mirror result therefore lands **between -10pp and -35pp, not restored to 0**. Do not expect the literal
  gate to close the 35pp gap.
- **Do not block the 42 rank-2/3 terminals that pass on guaranteed-KO** — those are correct (this is the C2-style
  "attack only if KO or high-confidence" allowance).
- **Decide separately on the non-terminal rank-2/3 extras**, especially the 33 top3-exclusive RETREAT picks and 21
  attack re-targets; the terminal gate does not touch them and they are an unexcluded co-driver.

What V2 should **not** do on this evidence: ship any gate as outcome-validated, or treat rank and proposer-prob as
two corroborating confidence signals.

## 7. Required next experiment (to move from directional to confirmed)

Re-run the deployed-mirror games **with per-decision instrumentation** on our-agent mirror states: `game_result`
+ first-changed-decision + `proposer_rank` + `terminal_override`. Only then can terminal (and the non-terminal
RETREAT/retarget) overrides be linked to actual losses. To trust the -25pp top1->top3 mirror ordering as real
rather than noise, power the mirror arm up to roughly 60-100 games/mode. Optionally ablate C3 into (rank-1
terminals + rank-2/3 non-terminals) vs full C3 to isolate whether the 127 extra terminals or the 93 extra
non-terminals drive the loss. (This is Model B's lane to instrument and run; Model A builds the conservative
selector offline.)

## 8. Artifacts (paths + counts)

Directory: `pokemon-tcg-ai-battle/data/generated/starmie_selector_live_smoke_v1/`

- `changed_decision_classes.jsonl` — **5128 records** (2564 decisions x 2 modes). `game_result`/`matchup` are
  `null` in every record (the no-outcome-linkage gap).
- `failure_aggregate_report.json` — per-mode override/terminal/premature counts, transition matrices,
  `key_finding_rank_confidence` block; self-flags the data-provenance + distribution-shift gaps.
- `mirror_regression_report.json` — mirror contrast (top3 -35pp mirror vs +0/+10pp field); the field-vs-mirror
  reading; `per_game_first_changed_decision_analysis = UNAVAILABLE`.
- `selector_smoke_review.html` / `.jsonl` — 76 worked examples (premature terminals, develop->END, ATTACH/SELECT,
  safety-veto-blocked) with board/hand/options context.
- `live_smoke_report.json` — raw win/loss per mode x opponent. Mirror: off 11/20, top1 9/20, top3 4/20; 0 errors.

DIAGNOSTIC_VERDICT=B_FAILURE_MODE_DIRECTIONAL
