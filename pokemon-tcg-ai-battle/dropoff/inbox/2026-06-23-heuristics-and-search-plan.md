# Heuristics + non-exclusive search plan (2026-06-23)

Goal: incorporate the replay-derived heuristics, and use search as a TOOL the heuristics call, not a rival
that fights or replaces them. Ladder fact that drives this: the pure-heuristic agent scored 736, the search
agents 645 to 687. Heuristics are the engine. Every time search DROVE the policy it lost on the ladder or
went neutral locally. So the design rule is:

> Search is invoked by a heuristic, scoped to that heuristic's own candidate set and objective, and the
> heuristic keeps a default if search fails. Search never enumerates all legal moves and never overrides the
> strategy. It is a calculator the rules call, not a policy.

That single rule is what stops search fighting heuristics. The rest is where it pays off.

---

## Part A: the heuristics

### Card corrections (verified against card_stats this session)
- Abra 741, Kadabra 742, Alakazam 743 are Psychic. Dunsparce 65/305 and Dudunsparce 66 are Colorless.
- Telepath Energy (19) fetches 2 Basic PSYCHIC. Abra is our ONLY Basic Psychic (Kadabra/Alakazam are
  evolutions). So Telepath grabbing 2 Abra is correct and forced, not a misplay.
- Buddy-Buddy Poffin (1086) fetches any 2 Basics. The Abra-vs-Dunsparce balancing belongs HERE, not on Telepath.
- Dudunsparce Run Away Draw is an Ability usable from the BENCH (draw 3, then shuffle Dudunsparce + its
  attachments, including the Dunsparce underneath, back into the deck). Play it early. Avoid only on: deckout
  risk, it would leave no Active, or the bench body is tactically needed this turn.
- Kadabra/Alakazam Psychic Draw triggers when the card is played from hand to evolve (Kadabra +2, Alakazam +3),
  NOT from being Active. Value Alakazam as evolution-progress + draw-trigger + attacker-readiness, not as static
  board points.
- Colorless cost = any Energy can pay it; Psychic pays Colorless, Colorless does not pay a Psychic-specific cost.
- Evolve: the lower stage stays underneath with its energy and damage.
- Opponent decklist is hidden at game start; Xerosic counter-play is reactive.
- Crustle's ability blocks damage from Pokemon-ex attacks. Our deck runs no ex, so it is moot. Meta-awareness only.

### P1 -- highest value, pure rules, no search needed
1. Sequencing: draw/search FIRST, irreversible actions LAST. This is the single biggest misplay class (the
   Action 66 energy blunder). Per-turn order:
   1. immediate win / forced survival
   2. safe draw/search (Psychic Draw on evolve, Run Away Draw)
   3. resolve tutor targets using the UPDATED hand
   4. evolve draw engines if legal
   5. attach energy
   6. retreat/switch only if it improves attack or survival
   7. optional hand-thinning cards only if they matter (Powerful Hand scales with hand size)
   8. attack / pass
   Exception: Enriching and Telepath are attach-triggered draw/search, so they can come early when the target
   is obvious.
2. Retreat guard: retreat only if the new Active can attack now, OR it escapes a lethal threat, OR the current
   Active is trapped/useless, OR it enables a known attack-switch line. Never retreat for the same or a worse
   Pokemon. Never retreat just because retreat is legal. (Action 11/62 blunders.)
3. Play Dudunsparce to the bench early; remove any code that gates its ability on being Active.

### P2 -- tutor and discard, driven by card advantage
4. Tutor priority, generalized across Dawn, Poffin, Telepath, Poke Pad, Hilda, and any basic/Pokemon tutor:
   Alakazam line > Dunsparce line. If a Stage 2 is in hand needing its Stage 1, fetch the Stage 1 first. Target
   about 2 Alakazams, one Psychic energy each (split, not stacked). Among valid targets, pick the one that
   maximizes card advantage (Part B.1). Telepath is forced to Abra; Poffin does the Abra/Dunsparce balance.
5. Discard-save (Xerosic, end-of-hand): keep by next-turn-playable value, not by card name.
   - keep: KO card / required attack energy, a playable Kadabra or Alakazam (triggers Psychic Draw), a safe draw
     supporter, Psychic for active/backup Alakazam, Rare Candy if the line is live, Dawn/Lana if they restore
     card advantage.
   - discard first: redundant basics, an extra Dudunsparce once the draw engine is online, Enhanced Hammer with
     no target, a stadium with no purpose, special energy if we are not online.
   - note: Lana recovers Pokemon and basic energy, so sometimes discard the recoverable piece and KEEP Lana.

### P3 -- situational, blocked on card-effect decoding
6. Enhanced Hammer (special-energy removal): use only if the target has a special energy AND we are not KOing it
   this turn AND one of (it denies a threatening or lethal attack now, it makes them more than 1 attachment from
   their main attack, it denies a meaningful retreat/pivot). Otherwise hold it (keeps the hand big for Powerful
   Hand). BLOCKED: the bot has no decoded effect for Enhanced Hammer; decode it first.
7. Stadium / Battle Cage and any optional card: if attacking with Powerful Hand this turn, do not play optional
   cards that shrink the hand unless they enable the KO, prevent a loss, or give persistent value. BLOCKED on
   decoding Battle Cage.

---

## Part B: how search HELPS the heuristics (the part you asked for)

Three bounded uses. In all three, a heuristic supplies the candidates and the objective; search only ranks them.

### B.1 Card-advantage sampling for tutor and discard (the bot's real edge)
This is your "sample many draws and see what produces the most card advantage on average" idea, and it is where a
bot genuinely beats a human. When a heuristic has a few valid targets (fetch / keep / discard):
- for each candidate, apply it, then sample our next 1 to 2 turns of draws over K hidden worlds (OUR deck only,
  the opponent barely affects our own card advantage),
- play our development forward greedily,
- score the resulting card advantage = playable cards in hand + threats on board + draw still available.
Pick the best average. It needs no opponent model, so it is cheap and low-regret. Start cheaper: a static
card-advantage value per card (Alakazam = big: +3 draw and a threat; Kadabra/Dudunsparce = +draw; energy =
enabler; redundant basic ~ 0; Enhanced Hammer = 0 unless a target exists). Use the value table as the default and
fall to sampling only to break close calls. You can also literally compute expected draw value per card and skip
the simulation where the math is clear.

### B.2 Survival check via a rules-based opponent model (retreat and defensive development)
The current rollout assumes the opponent "takes the highest-damage attack," which is too dumb to trust for
lookahead. Replace it with your worst-case prep model:
- the opponent's Active gains +1 energy each turn,
- if it can evolve, it evolves next turn,
- if the Active is low-threat and a high-threat sits on their bench, they switch to it and arm it.
Run that 1 to 2 turns forward to answer ONE question: can the opponent KO my Active or a key piece next turn? The
retreat guard (P1.2) and the "do I need a backup attacker or to develop defensively" heuristics call this. Search
here answers "is this line safe," it does not choose the move.

### B.3 Developmental sequencing inside the turn
The draw-before-irreversible order (P1.1) is fixed rules, but among the safe developmental plays the heuristic
already approved, search can pick the ordering that ends the turn with the best board (the bank-the-KO idea),
restricted to heuristic-approved plays so it cannot wander into junk. This is also what would have caught Action
66: simulate "attach to bench now" vs "draw first, then attach," and the first loses the chance to arm the Active.

### Why this never fights heuristics
- Search never sees the full legal move set; it ranks only a heuristic's candidates.
- Search is scored by the heuristic's objective (card advantage, survival), not a generic board eval that can
  disagree with the strategy.
- The heuristic always has a default if search times out or errors.
This is the opposite of the search-driven agent, which enumerated everything and lost on the ladder.

---

## Part C: implementation order
1. P1.1 sequencing and P1.2 retreat guard. Pure rules, biggest misplays, no search.
2. P1.3 play Dudunsparce early; remove any "needs Active" assumption on bench abilities.
3. P2.4 tutor priority with the static card-advantage table.
4. P2.5 discard-save.
5. Decode Enhanced Hammer and Battle Cage effects, then P3.
6. B.1 card-advantage sampling, wired into the tutor/discard heuristics for close calls.
7. B.2 survival check with the rules opponent model, wired into retreat/defensive heuristics.
8. B.3 sequencing search last, bounded to heuristic-approved plays.

Summary: the heuristics are the rules; search is a bounded calculator they call for card-advantage sampling and
survival checks. It never drives. That keeps the 736-on-the-ladder engine in charge and uses search only where it
adds a number a human could not compute by hand.
