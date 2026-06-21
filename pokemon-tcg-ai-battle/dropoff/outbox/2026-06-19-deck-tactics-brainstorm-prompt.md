# Brainstorm prompt: tactical heuristics for the Dudunsparce/Alakazam deck

Paste this to a model that knows the Pokemon TCG. Point it at the card reference file:
`pokemon-tcg-ai-battle/research/BEST_PILOT_DECKLIST.md` (every card in the best pilot's exact list, with
effects, types, weakness/resistance, attacks, and our card-mechanics gotchas).

---

## Your task

You are helping build an agent for the Kaggle "Pokemon TCG AI Battle" competition. We pilot a single fixed
deck: a **Dudunsparce/Alakazam draw-engine toolbox** (Abra -> Kadabra -> Alakazam draw line; Dunsparce ->
Dudunsparce; Alakazam's "Powerful Hand" places 2 damage counters per card in hand; support: Buddy-Buddy
Poffin, Dawn, Hilda, Poke Pad, Rare Candy, Boss's Orders gust, Enhanced Hammer, Night Stretcher recovery).
The full card list with effects is in the reference file above.

Propose **basic, concrete, state-conditional tactical heuristics** for piloting THIS deck well. We want a list
of rules, each in this exact shape:

- **Trigger** (the board condition that must hold), **Action** (what to do), **Rationale** (one line),
  **Priority** (forced / strong-prior / tie-breaker), **How to test** (what measurable outcome confirms it).

## Critical constraints (read before proposing -- these come from our own testing)

1. **The agent already runs a 1-ply forward search** that simulates each legal move and picks the best by a
   board evaluation. Hand heuristics ALONE do not beat it (a board-aware heuristic only ties the take-first
   baseline). So your heuristics must be framed as **priors / tie-breakers / forced-rules that GUIDE the
   search**, not as a standalone policy. The search still decides; heuristics nudge or hard-gate it.
2. **The engine is authoritative.** Code to the simulator's behavior, not paper rules. Some edge cases differ.
3. **The agent must never crash and must always return a legal move** within the time budget. Forced-rules
   (e.g. "take the lethal KO") must be safe and unambiguous.
4. **Win-rate is the only judge.** Every heuristic is a HYPOTHESIS until a seat-swapped win-rate A/B confirms
   it. State, for each, what you would measure. Do not assert a heuristic "works".
5. **Card-mechanics gotchas we already hit** (do not repeat):
   - **Dudunsparce "Run Away Draw" shuffles ITSELF (the 140-HP body) back into the deck.** It is a draw engine
     that removes its own attacker. Do not treat it as a stable wall/attacker after using it.
   - **Alakazam / Kadabra "Psychic Draw" only triggers on the EVOLVE-from-hand step**, not passively each
     turn. Sequencing matters: evolve to draw, do not sit on an un-evolved line.
   - Prize liability: ex KO gives the opponent 2 prizes, Mega ex 3, basic 1. Weakness = 2x damage (often a KO).

## Starter heuristics (Model A's seeds -- expand, correct, and add to these)

These are candidate search-priors/forced-rules grounded in the deck + our lessons. Refine them and add your own:

- **[forced]** If a legal line this turn KOs the opponent's Active and does not lose on the swing-back, take
  it; prefer KOing an ex/Mega-ex (2-3 prizes) over a basic. *Test: win-rate vs not-forcing on KO-available states.*
- **[strong-prior]** Evolve Abra->Kadabra->Alakazam FROM HAND to trigger Psychic Draw (2 then 3 cards) before
  you would otherwise run low on resources; use Rare Candy to skip to Alakazam (draw 3) when you hold the
  Stage-2 and a valid Basic. *Test: cards drawn / setup speed and win-rate.*
- **[strong-prior]** Alakazam's "Powerful Hand" scales with hand size -- build the hand with the draw engine
  BEFORE committing Powerful Hand; do not empty your hand first. *Test: damage output vs naive ordering.*
- **[tie-breaker]** Use Dudunsparce "Run Away Draw" for the draw, but do NOT plan around it surviving as a
  wall; treat its body as temporary. *Test: deck-out / board-stability outcomes.*
- **[strong-prior]** Save Boss's Orders to gust up a benched target you can KO (or a key setup piece), not to
  drag a random body. *Test: prizes taken on Boss turns.*
- **[tie-breaker]** Hold Enhanced Hammer until the opponent has a Special Energy that matters; do not fire it
  blind early. *Test: disruption value.*
- **[strong-prior]** Watch deck-out: this deck draws hard, so late-game prefer lines that do not over-thin the
  deck when deck count is low. *Test: deck-out loss rate.*
- **[strong-prior]** Energy is scarce (8 in the deck): prioritize attaching to the live attacker; avoid
  stranding energy on benched/temporary bodies. *Test: attacker-online rate.*
- **[prior]** Bench management: protect bench slots for the evolution lines (Abra, Dunsparce); avoid clogging
  the bench with dead basics that feed easy prizes.

## What to return

A prioritized list of heuristics in the shape above (Trigger / Action / Rationale / Priority / How-to-test),
grouped by game phase (early setup, mid-game engine, closing/KO race) where useful. Favor a SMALL number of
high-leverage, unambiguous rules over many vague ones. Flag any that need a card effect we may have decoded
wrong (check the reference file's effect text). We will implement the strongest as search priors/forced-rules
and gate each with a seat-swapped win-rate A/B.
