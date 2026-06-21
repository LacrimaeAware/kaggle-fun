# Best-Pilot Deck Reference (for tactical-heuristic brainstorming)

Source: Kaggle Pokemon TCG AI Battle, submission 53802029, episode 80723114, **player 0** (a strong ladder pilot). This is THEIR exact 60-card list, not ours. Card IDs are the cabt engine IDs (what the agent sees). Effect text is from the official EN_Card_Data.csv.

**Deck:** 60 cards, 19 distinct. Archetype: Dudunsparce/Alakazam draw-engine toolbox (near-identical to our DENPA92 list; theirs leans more on the evolution/draw engine + recovery, ours leans more on tools).

## How to read this for heuristics
- **Weakness** = takes 2x damage from that type (huge: a weakness-hit often = a KO). **Resistance** = takes less. These are the 'effectiveness types'.
- **Prize liability:** ex = opponent takes 2 prizes on KO, Mega ex = 3, basic = 1. Avoid feeding multi-prize KOs.
- The engine is authoritative; some paper-rule edge cases differ. Heuristics should be conditional on board state (bench space, energy, prizes, KO availability), not raw card value.

## Pokemon (19 cards)

### 4x Abra  (id 741)
- Basic Pokémon | HP 50 | Type {P} | Weak {D} | Resist {F} | Retreat 1
- **Teleportation Attack** (cost {P}, dmg 10): Switch this Pokémon with 1 of your Benched Pokémon.

### 4x Alakazam  (id 743)
- Stage 2 Pokémon | HP 140 | Type {P} | Weak {D} | Resist {F} | Retreat 1
- **[Ability] Psychic Draw** (cost n/a, dmg n/a): Once during your turn, when you play this Pokémon from your hand to evolve 1 of your Pokémon, you may use this Ability. Draw 3 cards.
- **Powerful Hand** (cost {P}, dmg n/a): Place 2 damage counters on your opponent’s Active Pokémon for each card in your hand.
- **GOTCHA (from our testing):** Psychic Draw (the draw ability) only triggers on EVOLVE-from-hand, not every turn. Sequencing matters: evolve to draw, do not sit on it.

### 4x Dunsparce  (id 305)
- Basic Pokémon | HP 70 | Type {C} | Weak {F} | Resist n/a | Retreat 1
- **Trading Places** (cost ●, dmg n/a): Switch this Pokémon with 1 of your Benched Pokémon.
- **Ram** (cost ●●, dmg 20): n/a
- **GOTCHA (from our testing):** Basic that evolves into Dudunsparce. The deck runs it mainly to access the Dudunsparce draw engine.

### 4x Kadabra  (id 742)
- Stage 1 Pokémon | HP 80 | Type {P} | Weak {D} | Resist {F} | Retreat 1
- **[Ability] Psychic Draw** (cost n/a, dmg n/a): Once during your turn, when you play this Pokémon from your hand to evolve 1 of your Pokémon, you may use this Ability. Draw 2 cards.
- **Super Psy Bolt** (cost {P}, dmg 30): n/a
- **GOTCHA (from our testing):** Same as Alakazam: draw only on the evolve step, not passively.

### 3x Dudunsparce  (id 66)
- Stage 1 Pokémon | HP 140 | Type {C} | Weak {F} | Resist n/a | Retreat 3
- **[Ability] Run Away Draw** (cost n/a, dmg n/a): Once during your turn, you may draw 3 cards. If you drew any cards in this way, shuffle this Pokémon and all attached cards into your deck.
- **Land Crush** (cost ●●●, dmg 90): n/a
- **GOTCHA (from our testing):** Run Away Draw shuffles ITSELF (the 140-HP body) back into the deck when used -- it is a draw engine that removes its own attacker. Do not treat it as a stable wall.
- **GOTCHA (from our testing):** Basic that evolves into Dudunsparce. The deck runs it mainly to access the Dudunsparce draw engine.

## Trainer (33 cards)

### 4x Buddy-Buddy Poffin  (id 1086)
- Item
- Effect: Search your deck for up to 2 Basic Pokémon with 70 HP or less and put them onto your Bench. Then, shuffle your deck.

### 4x Dawn  (id 1231)
- Supporter
- Effect: Search your deck for a Basic Pokémon, a Stage 1 Pokémon, and a Stage 2 Pokémon, reveal them, and put them into your hand. Then, shuffle your deck.

### 4x Enhanced Hammer  (id 1081)
- Item
- Effect: Discard a Special Energy from 1 of your opponent’s Pokémon.

### 4x Hilda  (id 1225)
- Supporter
- Effect: Search your deck for an Evolution Pokémon and an Energy card, reveal them, and put them into your hand. Then, shuffle your deck.

### 4x Poké Pad  (id 1152)
- Item
- Effect: Search your deck for a Pokémon that doesn’t have a Rule Box, reveal it, and put it into your hand. Then, shuffle your deck. (Pokémon {ex}, Pokémon {V}, etc. have Rule Boxes.)

### 4x Rare Candy  (id 1079)
- Item
- Effect: Choose 1 of your Basic Pokémon in play. If you have a Stage 2 card in your hand that evolves from that Pokémon, put that card onto the Basic Pokémon to evolve it, skipping the Stage 1. You can’t use this card during your first turn or on a Basic Pokémon that was put into play this turn.

### 3x Boss’s Orders  (id 1182)
- Supporter
- Effect: Switch in 1 of your opponent’s Benched Pokémon to the Active Spot.

### 3x Night Stretcher  (id 1097)
- Item
- Effect: Put a Pokémon or a Basic Energy card from your discard pile into your hand.

### 1x Battle Cage  (id 1264)
- Stadium
- Effect: Prevent all damage counters from being placed on Benched Pokémon (both yours and your opponent’s) by effects of attacks and Abilities from the opponent’s Pokémon. (Damage from attacks is still taken.)

### 1x Lana’s Aid  (id 1184)
- Supporter
- Effect: Put up to 3 in any combination of Pokémon that don’t have a Rule Box and Basic Energy cards from your discard pile into your hand. (Pokémon {ex}, Pokémon {V}, etc. have Rule Boxes.)

### 1x Sacred Ash  (id 1129)
- Item
- Effect: Shuffle up to 5 Pokémon from your discard pile into your deck.

## Energy (8 cards)

### 4x Telepath Psychic Energy  (id 19)
- Special Energy | Type {P}
- Effect: As long as this card is attached to a Pokémon, it provides {P} Energy.
When you attach this card from your hand to a {P} Pokémon, search your deck for up to 2 Basic {P} Pokémon and put them onto your Bench. Then, shuffle your deck.

### 3x Basic {P} Energy  (id 5)
- Basic Energy | Type {P}

### 1x Enriching Energy  (id 13)
- Special Energy | ACE SPEC | Type {C}
- Effect: As long as this card is attached to a Pokémon, it provides {C} Energy.

When you attach this card from your hand to a Pokémon, draw 4 cards.
