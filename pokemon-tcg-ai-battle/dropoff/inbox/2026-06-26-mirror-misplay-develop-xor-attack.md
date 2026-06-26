# Mirror misplay found: develop-XOR-attack (heavy under-attacks) — 2026-06-26 (Model B)

The user pushed hard to stop dismissing the local-mirror loss as noise and actually find the misplay
("I'm almost certain a misplay is happening"). This is the diagnosis, multiply confirmed, plus the fix.

## What "the mirror" is
`tools/starmie_ab_v1.py` pits HEAVY (`starmie_heuristics.agent` = full heuristics + search + veto) vs
DEPLOYED (`main.agent_starmie` = explicit KO-floor + the SAME search backend `best_option(leaf_mode="deckout",
rollout_mode="develop")`, with NO heuristics and NO veto). So "heavy loses the mirror" means: **heavy's
heuristic+veto layer is net-negative versus pure KO-floor+search.** Both agents share the identical search
leaf and rollout.

## The deficit is real but modest — NOT catastrophic domination
Mirror win rate (heavy vs deployed), several runs:
- baseline n=40 → 42%, n=60 → 45%, n=80 → 42.5%, n=100 (veto off) → 48%, n=100 (ALL rules off) → 47%.
- Combined ~44-46%. Consistently below 50%, but the earlier alarming "42%" was small-n. No single rule
  isolates it: disabling the dig/bench-dev (R2) → 47%, disabling the whole veto (R13) → 48%, disabling ALL
  rules → 47%. All within noise of baseline. **The deficit is structural, not one bad rule.**

## The mechanism (the actual misplay): heavy UNDER-ATTACKS
`tools/mirror_behavior_v1.py` (n=80, records BOTH agents, unbiased — all games, not a capped loss sample):

| metric (per game)      | HEAVY LOST n=46 |          | HEAVY WON n=34 |          |
|------------------------|-----------------|----------|----------------|----------|
|                        | heavy           | deployed | heavy          | deployed |
| first attack @ decision| **16.0**        | 13.24    | 14.32          | 14.68    |
| attacks made           | **3.07**        | 5.02     | 5.0            | 3.71     |
| prizes taken           | **1.26**        | 2.43     | 2.65           | 1.91     |
| end board size         | 1.43            | 2.07     | 1.91           | 1.32     |
| digs (search/draw)     | 5.07            | 6.28     | 6.09           | 6.0      |

The pattern is self-confirming: **when heavy loses it attacks ~2 fewer times and ~3 decisions (≈1 turn) later
than the dumb agent, taking ~1 fewer prize; when heavy wins, the pattern flips** (it attacks more and earlier).
Digs are ~equal — so it is NOT over-digging. It is UNDER-ATTACKING / attacking too late. In a same-deck race,
attacking later loses.

Decision-level confirmation (`tools/starmie_loss_capture_v1.py` traces): in mirror losses, **64% of the
decisions where heavy has an ARMED Mega Starmie (≥1 energy) with an attack option available, heavy does NOT
attack — it digs/develops instead** (49-55% vs the field too → general, mirror-amplified). One full trace: an
armed Mega Starmie (Jetting Blow 120 available) sat idle for ~4 turns playing Pokégear / Ultra Ball / Harlequin
/ Poffin / Night Stretcher (digging) while the opponent took 3 prizes.

## Root cause: develop-XOR-attack (the user's exact guess) — specifically for CHIP (non-KO) attacks
"Develop THEN attack" was implemented for KOs but collapses to "develop, skip the attack" for chip damage. Two
places:
1. **Heuristic chain (`_main_action`)**: the ONLY turn-ending attack is the KO floor (R9). There is no
   chip-attack floor. With no KO available, heavy does its free development and defers to search instead of
   attacking.
2. **Search rollout (`search_v3.py:_rollout_pick`, develop mode, lines 95-99)**: on my turn it returns the
   FIRST develop action (play/attach/evolve) and only "falls through to attack" when NO develop action remains.
   With a full hand there is always another card to play, so the simulated turn **develops forever and never
   attacks** → the leaf sees attacking as near-valueless → search prefers digging. XOR at the search level.

Develop and attack are NOT mutually exclusive within a turn (free development does not end the turn, the attack
does) — the bug treats them as either/or whenever the attack is chip rather than a KO.

## The fix candidate (STARMIE_ATTACK_FLOOR) -- BUILT, TESTED, AND **REFUTED** by A/B
`STARMIE_ATTACK_FLOOR=1` added a final "if you can attack, attack" floor AFTER the KO floor and the
retreat-to-promote check (any attacker; develop THEN attack). A/B at n=120/matchup, budget 0.3:

| matchup            | baseline (floor OFF) | floor ON | delta |
|--------------------|----------------------|----------|-------|
| mirror (deployed)  | 48% (57-63)          | **31%**  | -17   |
| alakazam           | 81% (97-23)          | **68%**  | -13   |
| denpa92            | 88% (105-15)         | **80%**  | -8    |

The floor REGRESSED every matchup. Mechanism (mirror_behavior, floor ON, losses): heavy dug LESS (4.11 vs
5.07), ended THINNER (board 1.23 vs 1.43), took FEWER prizes (5.49 vs 4.74 left), and got swept FASTER (28.5 vs
34.5 decisions) -- while attack count barely moved. Forcing the attack traded away the board development the
digs were building, collapsing heavy's position faster. **The search's develop-vs-attack judgment was better
than a blunt floor.** Candidate reverted (not promoted, not kept as dead code).

## Corrected conclusion (the honest read)
- The under-attack pattern is REAL but it is a **symptom of a thin/behind board, not the cause of losing**:
  winners attack more BECAUSE they built a board first, not the other way round. The naive causal flip ("attack
  more -> win more") is refuted -- see [[reasoning-pitfalls]] (correlation is not the lever).
- The mirror is ~48% at n=120 (near even); the alarming early "42%" was small-n. Not "domination."
- The right lever is NOT a forced-attack heuristic but the search-leaf VALUATION: make the leaf value a state
  with a ready Mega Starmie attacker online (ATTACKER_CONTINUITY term, the queued tactical-leaf task), which
  shapes WHAT the search prefers while leaving WHEN-to-attack to the search. That term must pass its own A/B;
  do not promote on a hunch (this candidate is exactly why).
- Ladder is still the only real test; mirror A/B is a weak signal for a change that touches the shared search.

Tools added this round: `tools/mirror_behavior_v1.py`. Data: `data/mirror_behavior.json`.
