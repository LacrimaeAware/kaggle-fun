# Heuristics from replay 81555875 (2026-06-23)

Source: user watching replay https://ptcgvis.heroz.jp/Visualizer/Replay/81555875/0 . Ladder context: the
basic heuristic submission scored 736 vs the search submissions 645 to 687, so the heuristic line is the
one to improve. These are prioritized by the user's stated confidence. The game was won despite the
misplays below.

## Card facts (verified from card_stats / card_effects)

- Dudunsparce (66) Run Away Draw is an ABILITY (draw 3). Abilities work from the BENCH. Play Dudunsparce
  early; do not treat it as needing to be active. If code gates it on being active, that is a bug.
- Telepath Energy (19) is a special energy that also tutors 2 basics to the bench (search_to_bench:2). The
  "got two Abra" misplay is this card's fetch choosing badly.
- Kadabra (742) / Alakazam (743) Psychic Draw are abilities (draw 2 / 3) on the evolve step.
- Abra (741) Teleportation Attack switches a Pokemon in (switch_gust).
- Enhanced Hammer (1081) and Battle Cage (1264) have NO decoded effect in card data. The bot cannot reason
  about them. Decode their effects first (Enhanced Hammer = remove a special energy; Battle Cage = stadium).
- Opponent decklist is hidden at game start (determinized by search). Xerosic's Machinations counter-play
  must be reactive, not pre-planned.
- Evolve: the lower stage stays UNDERNEATH the evolution with its energy and damage, not discarded.
- The `stage` field is null for every card in card_stats. Tutor heuristics that need basic / Stage1 / Stage2
  must derive the line from the known evolutions (Abra->Kadabra->Alakazam, Dunsparce->Dudunsparce) or the
  preEvolution field.

## Priority 1 (user: guaranteed or near-guaranteed better, easy)

### P1.1 Sequencing: draw first, irreversible actions last
Misplay (action 66): played Psychic energy on a benched Alakazam, then drew another Alakazam and could not
energize the active. Rule: within a turn, resolve all DRAW first (Psychic Draw on evolve, Run Away Draw, and
the tutors), THEN irreversible plays (attach energy, evolve), THEN attack. Play order: draw > fetch/develop >
attach/evolve > attack. This is the single most repeated blunder class.

### P1.2 Enhanced Hammer discipline
Holding it keeps the hand big for Powerful Hand. Use only when:
1. (guaranteed) NOT if we KO the opponent this turn.
2. (very likely) NOT unless the opponent's main attack is threatening (~50 to 80% of our HP or lethal).
3. (probably) NOT until the opponent is at least 1 energy short of their main attack, so removing one denies it.
4. (rare, tricky) Anti-retreat: if a high-threat-but-not-online attacker is retreating, remove energy from the
   Pokemon they switch TO; otherwise irrelevant.
Blocked on decoding the card's effect first.

### P1.3 Play Dudunsparce to the bench immediately
Run Away Draw works from the bench, costs only a bench slot, enables draw. Play it early most turns.

### P1.4 Do not retreat pointlessly
Misplays (action 11, 62): retreated Abra to bench another Abra; retreated with no energy to attack. Rule: only
retreat if it (a) puts in a Pokemon that can attack or KO this turn, or (b) escapes a lethal threat. Never
retreat to swap for the same or a worse Pokemon.

## Priority 2 (tutor + discard discipline)

### P2.1 Tutor priority: complete the Alakazam line (Alakazam line > Dunsparce line)
For any fetch (Telepath, Dawn, Poffin, Poke Pad, Hilda):
- If a Stage 2 is in hand needing its Stage 1 (Alakazam in hand, no Kadabra), fetch the Stage 1 first.
- Else build the Alakazam line: need Abra get Abra (else Dunsparce); need Kadabra get Kadabra (else
  Dudunsparce). Target about 2 Alakazams.
- Balance Abra / Dunsparce: 2 Abra and 0 Dunsparce -> fetch Dunsparce even if Kadabras are in hand.
- One Psychic energy per Alakazam (split across two, not stacked) as a backup attacker.

### P2.2 Discard / card-advantage discipline (Xerosic counter)
When forced to discard (Xerosic, end-of-hand): discard the lowest card-advantage pieces first (non-line cards,
special energy if not online, cards that do not draw). Keep line pieces, draw engines, and whatever restores
card advantage next turn (Dawn/Lana's Aid value is deck-aware: depends on what is left to draw or recover).
Misplay (action 45): kept all Dudunsparce and nothing else; should keep what restores card advantage + a threat.

## Priority 3 (situational, lower value)
- Battle Cage: only play for its stadium effect, not randomly (decode effect first; it persists as a stadium).
- Crustle: its ability makes ex Pokemon deal 0 to it. Our deck has no ex, so currently moot.

## Implementation note
Where to implement is open: the heuristics live in pokemon-ai-agent (currently read-only) and the active work
is in a separate worktree. P1.1 (sequencing) and P1.4 (retreat) are the highest value and do not need new card
data. P1.2 and P3 are blocked on decoding Enhanced Hammer / Battle Cage effects.
