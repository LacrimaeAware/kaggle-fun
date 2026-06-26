# Starmie tactical-leaf V1 + mirror investigation — closeout (2026-06-26, Model B)

Branch: `exp/starmie-tactical-leaf-v1` (NOT merged to main; main untouched at 6c38e58 = sub_starmie2 lineage).
All work is isolated; no submission built automatically.

## Part 1 — Mirror "skipping-attack" investigation (the user's demand)
- DIAGNOSIS (multiply confirmed): heavy under-attacks in the games it loses (3.07 attacks vs the dumb agent's
  5.02; 64% of armed-Mega-with-attack-available decisions don't attack). The user's guess was right that the
  code was develop-XOR-attack for CHIP (non-KO) attacks: the only turn-ending attack was the KO floor (R9), and
  the develop-rollout digs forever and never reaches the attack.
- FIX BUILT + TESTED + **REFUTED**: `STARMIE_ATTACK_FLOOR` (any attacker attacks at end of turn after free dev).
  A/B n=120: mirror 48%->31%, alakazam 81%->68%, denpa92 88%->80%. It REGRESSED everything. Mechanism: forcing
  the attack traded away the board development the digs were building (thinner board, fewer prizes, swept faster).
  **The under-attacking is a SYMPTOM of a thin/behind board, not the cause of losing.** Candidate reverted.
- Corrected read: the mirror is ~48% at n=120 (near even); the alarming early "42%" was small-n. Not domination.
  The search's develop-vs-attack judgment beats a blunt heuristic floor. The principled lever is the search-leaf
  VALUATION (Part 2), not a forced-attack rule. See 2026-06-26-mirror-misplay-develop-xor-attack.md.

## Part 2 — Tactical-leaf task (S0–S9)
- **S0 baseline frozen** (`baseline_manifest.json`): HEAD 6c38e58, deck sha1, search cfg (budget 0.6, N_DETERM 8,
  leaf deckout, rollout develop), eval weights, opp prior (opponent_meta_v1, 30 decks). All 9 accepted
  correctness fixes verified present (opp-meta prior, Boss/Wally KO gating, Ignition=3 units, tutor zone,
  no-suicide, develop-before-attack). Verdict BASELINE_FROZEN.
- **S1 extractor** (`agent/starmie_tactical_state.py`): public-info-only entity/board features + 5 tactical
  coordinates (RACE/SWEEP/WALL/VALUE/COMMITMENT). Verified Starmie mechanics, Ignition unit-aware, engine menu
  as ground truth where present, uncertainty flags for opponents. No hidden info / outcome / pilot in the payload.
- **S2 dataset** (`starmie_tactical_state_v1.jsonl`, 22,083 rows, gitignored): our-seat non-trivial decisions
  resolved from replays; runtime payload strictly separated from eval-only metadata (leak-checked).
- **S3 leaf audit** (`current_leaf_failure_audit.json`): STRUCTURAL proof the leaf is DECK-BLIND — a Cinderace
  engine and a Mega Starmie attacker with the same 3 energy score IDENTICALLY (W_ENERGY counts active energy
  CARDS regardless of which Pokemon, and counts an Ignition as 1 card). PREVALENCE over 22,083 decisions:
  Cinderace-active-while-Mega-ready 2,339 (768 in disagreement); no-main-continuity 2,753 (939);
  Mega-one-attach-short 6,523 (2,392). The deck-blind patterns are common.
- **S4 term** (`eval.py` ATTACKER_CONTINUITY_V1, env `STARMIE_LEAF_ATTACKER_CONTINUITY=1`, DEFAULT OFF):
  rewards a ready Mega attacker (active/bench) + continuity; penalizes engine overinvestment, redundant energy
  that crosses no threshold, and energy on an exposed 3-prize Mega. Ignition-unit-aware; bounded (max ~tens) <<
  W_PRIZE 1000 so it never flips a KO/win/deck-out. Verified: toggle OFF = identical baseline (424=424);
  toggle ON makes the leaf prefer the Mega attacker (444 > 395).
- **S5 tests**: 10/10 fixed-state tests pass (`tests/test_attacker_continuity_v1.py`); full existing suite still
  green (baseline untouched). Registered in `tests/run_all.py`.
- **S6 offline selection audit** (`offline_selection_audit.json`, 232 triggered+search-decided roots, budget
  0.15): the term changes the search pick on **28%** of triggered roots (mechanism is real), but only ~4.6% of
  those changes move toward the narrow "build/use the Mega line" direction measured, and pilot agreement is FLAT
  (58.2%->57.8%). Weak directional signal (direction classifier is also crude — it doesn't credit
  "stop over-investing in Cinderace" changes).
- **S7 A/B** (n=120/matchup, budget 0.4): head-to-head heavy(+term) vs heavy(baseline) mirror + each vs the
  frozen field (alakazam/denpa92 use leaf_mode="hand" -> UNAFFECTED, isolating the term). RESULT:
  MIRROR continuity vs baseline 65-55 = **54.2%** for continuity; vs alakazam 79.2% vs 78.3% (+0.9); vs denpa92
  89.2% vs 85.8% (+3.4). **Positive direction, NO regression -- the opposite of the refuted attack-floor.** But
  the mirror +4.2 is within noise at n=120 and the offline mechanism signal was weak. Escalation to n=240 running.
- **S8 handoff** (`tactical_feature_schema.json`, `tactical_coordinate_summary.json`,
  `starmie_tactical_state_v1.jsonl`, the audit/A-B reports): feature taxonomy (deck-independent vs Starmie
  semantic-role vs exact-card-id vs public-runtime vs eval-only) so Model A can consume the situation rows.

## Verdicts (S9)
- ATTACKER_CONTINUITY: **B. ATTACKER_CONTINUITY_DIRECTIONAL** (at n=120). Tests pass; the structural deck-blindness
  it targets is real + prevalent; the A/B direction is POSITIVE with NO regression (unlike the attack-floor). But
  the mirror gain is within noise at n=120 and the offline mechanism signal is weak -> not VERIFIED yet. n=240
  escalation running; upgrade to A only if it firms up. Ladder is the only real test.
- TACTICAL_STATE_EXPORT: **A. TACTICAL_STATE_DATA_READY_FOR_MODEL_A** — 22,083 clean rows, all observations
  resolved, runtime/eval separation leak-checked, schema documented.

## Submission decision
- main is UNCHANGED (attack-floor reverted; term default-off on a branch). A submission from main == sub_starmie2
  already on the ladder. The leaf term is a promising NON-REGRESSING candidate (mirror +4.2, field +0.9/+3.4) but
  within noise at n=120 -> await the n=240 escalation. If it firms up clearly >50% mirror with the field holding,
  enabling the term + a submission is defensible (it is the first non-regressing improvement found, with a strong
  structural justification). If it regresses to ~50%, no submission. Do NOT ship on the within-noise n=120 result.
