# Mega Starmie ex / Cinderace — card effects (logged) + proposed heuristics (2026-06-25)

Card text read from `data/external/official/card_images/{id}.jpg` (the engine json has types/HP/weakness but
NOT attack costs/effects). Logged here so we don't re-read images. Decklist = the winning Cinderace/Starmie
list (66.7% over 39 pilots; tools/deck_winrate_v2.py).

## Card effects (verified from images)

Pokemon
- [1031] Mega Starmie ex — Stage 1 from Staryu. Water, HP 330, weakness Lightning x2, retreat [C][C].
  MEGA EX RULE: when it is Knocked Out, the opponent takes 3 Prize cards.
  - Jetting Blow [W]: 120, and 50 to 1 of the opponent's Benched Pokemon (no weakness/resistance on bench).
  - Nebula Beam [C][C][C]: 210. Damage NOT affected by weakness/resistance or by effects on the opp's Active.
- [1030] Staryu — Basic. Water, HP 70, weakness Lightning. Filler attack (~20). Evolves into Mega Starmie ex.
- [666] Cinderace — Stage 2 (from Raboot). Fire, HP 160, weakness Water x2.
  - Ability Explosiveness: if in your hand at setup, you may put it face down in the Active Spot.
  - Turbo Flare [C]: 50, then search your deck for up to 3 Basic Energy and attach them to your Benched
    Pokemon any way you like, then shuffle. (This is the ENERGY-ACCELERATION engine.)

Energy
- [3] Basic Water x9.
- [17] Ignition Energy x4 (Special): provides [C] on a Basic, [C][C][C] on an EVOLUTION; discarded at end of
  turn. => attach to Mega Starmie ex (evolution) to fund Nebula Beam (210) in ONE attach, same turn only.

Trainers
- [1145] Mega Signal x4 (Item): search a Mega Evolution ex (Mega Starmie ex) to hand.
- [1086] Buddy-Buddy Poffin x4 (Item): search up to 2 Basic Pokemon to the Bench (Staryu).
- [1121] Ultra Ball x1 (Item): search any card, discard 2 from hand.
- [1097] Night Stretcher x2 (Item): recover a Pokemon or Energy from discard.
- [1120] Crushing Hammer x4 (Item): flip a coin; heads = discard 1 Energy (ANY, not just special) from an
  opponent's Pokemon.
- [1159] Hero's Cape x1 (ACE SPEC Tool): the attached Pokemon gets +100 HP (Mega Starmie -> 430 HP).
- [1122] Pokegear 3.0 x4 (Item): search a Supporter.
- [1189] Salvatore x4, [1225] Hilda x2 (Supporters): search (1 each).
- [1223] Harlequin x2 (Supporter): draw 5. [1227] Lillie's Determination x4 (Supporter): draw 8, shuffle hand.
- [1229] Wally's Compassion x4 (Supporter): heal ALL damage from 1 of your Mega Evolution ex; if you healed,
  put all Energy attached to it into your hand.
- [1182] Boss's Orders x1 (Supporter): gust (switch opp's Benched to Active).
Note: 1 Supporter per turn; Items are free. ~17 supporters total -> very draw/search heavy.

## Deck plan
Open Cinderace face-down active (Explosiveness). Turbo Flare each turn (50 + 3 Basic Water to the Bench) to
power a benched Staryu/Mega Starmie. Find Mega Starmie ex with Mega Signal, evolve Staryu -> Mega Starmie ex,
promote it. Attack with Jetting Blow [W] (120 + 50 bench snipe, repeatable) or Nebula Beam [CCC] (210 flat;
fund instantly with Ignition Energy). Protect Mega Starmie (Hero's Cape +100, Wally's heal+energy-recover) to
deny the 3-prize KO. Crushing Hammer + Boss's Orders disrupt/gust.

## Proposed heuristics (verify each before encoding)

Setup
1. Go first.
2. Start Cinderace in the Active Spot (Explosiveness) when it's in the opening hand — it is the engine.
3. Bench Staryu early (Poffin); Mega Signal -> Mega Starmie ex; evolve Staryu the moment it's developed.

Energy engine
4. Cinderace Turbo Flare: take the 50 and attach all 3 Basic Water to ONE benched line target (the Staryu /
   Mega Starmie you will promote), not spread around.
5. Basic Water -> the Starmie line (Jetting Blow needs 1 W; toward Nebula Beam needs 3).
6. Ignition Energy: attach to Mega Starmie ex (evolution -> 3 colorless) ONLY on a turn you Nebula Beam that
   same turn (discarded end of turn). Never attach it and not use the 3-cost attack.

Attack selection
7. Lethal floor: if Jetting Blow (120) or Nebula Beam (210) KOs the active or a gusted target, take it.
8. Default to Jetting Blow [W] (cheap, repeatable, 120 + 50 bench snipe).
9. Nebula Beam (210) for: high-HP targets Jetting Blow can't KO; opponents with damage-reduction/effects on
   their active; or when their resistance would blunt Jetting Blow (Nebula ignores weakness/resistance/effects).
10. Aim Jetting Blow's 50 bench snipe at one target: a Pokemon you'll gust+KO next turn, a <=50 HP setup piece
    to KO now, or a developing threat to soften. Concentrate snipes; don't spread them.

Protect the 3-prize liability (Mega Starmie ex KO = 3 prizes)
11. Attach Hero's Cape (+100 -> 430 HP) to the active Mega Starmie ex.
12. Wally's Compassion when Mega Starmie ex is damaged and at KO risk next turn: full heal + recover its energy
    to hand (especially vs Lightning). It costs your Supporter that turn -> use when it saves the 3 prizes.
13. Don't field two Mega Starmie ex at once (gives the opponent easy multi-prize KOs).

Disruption / gust
14. Crushing Hammer (free Item, 50% per flip, any energy): hit the opponent's main attacker's energy when it
    denies/delays their attack (their setup turn); chain copies. Skip if you'll KO them first or it's irrelevant.
15. Boss's Orders (1 copy): gust + KO a benched target weak to Water, a low-HP/high-value ex (prizes), or a key
    setup piece; combine with the 50 snipe / Nebula Beam. Save for a swing KO.

Draw/search + deck-out
16. One Supporter/turn: developing -> search (Salvatore/Hilda, Pokegear->supporter); low hand -> draw (Lillie's
    8 / Harlequin 5); Mega Starmie in danger -> Wally's. Pick the one the turn needs.
17. Items freely: Mega Signal, Poffin, Ultra Ball (Night Stretcher recovers the discard), Crushing Hammer, Cape.
18. Don't draw below the deck-out buffer unless it's a winning line.

Safety
19. Mega Starmie ex (330/430 HP) tanks; rarely retreat ([C][C]); don't promote a fragile Staryu into a KO.
20. Never strand your last Pokemon (no-suicide).

Counter-specific (from weaknesses)
21. Vs Lightning (Starmie weak Lightning x2 — e.g. Iono's Voltorb/Wattrel): expect 3-prize KOs on us; lean on
    Hero's Cape + Wally's heal, race with Jetting Blow + snipe, Crushing Hammer their energy.
22. Vs Fire / mirror: Starmie (Water) hits Fire weakness; our own Cinderace (Fire) is weak to Water — keep it
    benched as the engine, don't leave it active into a Water attacker.
23. Nebula Beam (ignores weakness/resistance/effects) is the answer to Water-resistant or damage-prevention
    opponents.
