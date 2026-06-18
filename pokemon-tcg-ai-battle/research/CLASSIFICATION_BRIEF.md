# Card classification brief (hand-off prompt for another AI)

Copy the whole fenced block below into the other AI. It is self-contained (taxonomy, stat
guides, and a verb-to-class guide are inline), so a chat-only model can classify cards you
paste, and a file-capable model can read the repo files named at the end and label everything.

```text
You are helping classify all 1267 cards in The Pokemon Company's "Pokemon TCG AI Battle"
Kaggle competition card pool by FUNCTION (what the card does in play), to build features for
an AI agent that plays the game. These are multi-label FEATURES, not exclusive buckets:
assign every tag that applies (usually 1-4), overlap is expected and fine, and more precise
is better. Classify by what matters for actually playing, using real Pokemon TCG knowledge.

CARD-POOL STATISTICS (use as GUIDES, not hard cutoffs; the pool includes many weak support
Pokemon that drag the medians down):
- Pokemon HP: median 100, mean 122, 75th pct 140, 90th pct 230, max 380.
- Best attack damage per Pokemon: median 50, 75th pct 110, 90th pct 160, max 330.
- Attack Energy cost: median 2, 90th pct 3, max 5.
- 10% of Pokemon have NO damaging attack (pure support/ability Pokemon -> never an attacker).

TAXONOMY (tag -> meaning), grouped:

Attack & win
- main_attacker: the deck's main damage source; a strong attack (guide: best attack ~120+,
  the top ~25% of the pool). NOT a low/no-damage Pokemon.
- tech_attacker: secondary or situational attacker; lower/utility damage, or 1-2 copies for a
  specific matchup.
- snipe_spread: an attack that hits the opponent's BENCHED Pokemon (snipe one, or spread).
- win_condition: wins or combos toward winning beyond raw damage (extra prizes on KO,
  deck-out/mill, an alternate win).

Switching & position control
- gust: switches the opponent's Active Pokémon from Bench/Benched pressure (force active swap).
- switch_pivot: swaps your own Active Pokémon with one of your Benched Pokémon.

Energy
- basic_energy: a basic Energy card.
- special_energy: a special Energy card (carries an extra effect).
- energy_accel: attaches MORE than the one manual Energy per turn (ramp / pseudo-energy).

Card flow
- draw: NET card advantage (draw more than you spend), e.g. Professor's Research, Iono.
- tutor: search the deck for a specific card (ball/search), e.g. Ultra Ball, Nest Ball.
- cycle: draw plus discard / dig and refill without net card gain.
- consistency: smooths setup / makes turns reliable (overlaps draw and tutor; a meta-tag).

Disruption
- hand_disruption: attacks the opponent's HAND (shuffle away or shrink it), e.g. Iono, Judge.
- energy_disruption: removes or denies the opponent's Energy.
- ability_disable: shuts off the opponent's Abilities, OR prevents the EFFECTS of attacks
  (NOT the damage), e.g. Mist Energy, Rock Fighting Energy.
- stall_lock: broad lock or stall (Item lock, Ability lock, deck-out stall).

Defense & recovery
- tanky: durable, can take hits, often a tanky attacker (guide: HP above the ~100 median, 130+).
- wall: built to stall/block; very high HP (top ~10%, ~230+) with little or no offense.
- protection: reduces or prevents DAMAGE to your Pokemon (damage reduction/immunity).
- heal: heal / remove damage from your own Pokemon.
- retrieval: return Pokemon (and basic Energy) from your discard to hand/deck (Super Rod, Night Stretcher).

Setup & engines
- bench_setup: puts EXTRA Pokemon into play onto your Bench beyond the normal flow. If it
  searches the deck, also tag tutor.
- ability_engine: a passive/activated Ability that generates ONGOING value (draw/energy/damage
  engine). NOT for one-shot or purely defensive abilities.

Triggers & mechanics
- on_ko_trigger: an effect that fires when a Pokemon is Knocked Out.
- coin_flip: the outcome depends on a coin flip.
- mill: discards cards from the opponent's deck (a deck-out plan).

Card type / structural
- basic_mon: a Basic Pokemon.   - evolution: a Stage 1 or Stage 2 Pokemon.
- stadium: a Stadium card.       - tool: a Pokemon Tool card.
- ace_spec: an ACE SPEC card (max 1 per deck; a deckbuilding restriction).

VERB-TO-CLASS GUIDE (text patterns -> likely tags; still use judgment):
- "search your deck for ... put into your hand" -> tutor (+ consistency)
- "search your deck for a Pokemon ... onto your Bench" -> tutor + bench_setup
- "draw N cards" (no equal discard) -> draw (+ consistency)
- "shuffle your hand into your deck, then draw" -> cycle (+ draw if net gain, + consistency)
- "attach [Energy] from your deck/discard/hand" (beyond the normal attachment) -> energy_accel
- "switch your Active Pokemon" -> switch_pivot ; "switch your opponent's ... Pokemon" -> gust
- opponent "shuffles/reveals/reduces their hand" -> hand_disruption
- "discard ... your opponent's Energy" -> energy_disruption
- "prevent all effects of attacks, except damage" / opponent "can't use Abilities" -> ability_disable
- "takes N less damage" / "prevent all damage" / "reduce damage" -> protection
- "heal" / "remove N damage counters from your Pokemon" -> heal
- "put a Pokemon from your discard pile into your hand/deck" -> retrieval
- attack "does N damage to ... Benched Pokemon" -> snipe_spread
- "when this Pokemon is Knocked Out" / "when your opponent's Pokemon is Knocked Out" -> on_ko_trigger
- "flip a coin" -> coin_flip ; "discard cards from the top of your opponent's deck" -> mill
- high HP (~230+) with no/low attack -> wall ; durable (HP ~130+) that also attacks -> tanky
- a Pokemon with NO damaging attack is NOT an attacker; tag its support role instead.

IMPORTANT NUANCE (already corrected by the human): "prevents all effects of attacks, except
damage" means the card blocks attack SIDE-EFFECTS (special conditions, energy discard), NOT
the damage. So it is ability_disable plus special_energy, NOT heal/protection.

WHAT TO DO:
1. If you can read files: read pokemon-tcg-ai-battle/data/external/official/cards_full.json
   (id -> name, cardType, hp, skills [ability/effect text], attacks [name/damage/cost]) and
   output labels for ALL cards. Otherwise, classify the cards I paste.
2. Output strict JSON: { "<card_id>": {"tags": ["..."], "why": "<=10 words"}, ... }.
3. Flag cards you are unsure about (low confidence) rather than guessing.
4. Also suggest taxonomy improvements: classes that are missing, redundant, or ambiguous,
   using standard competitive Pokemon TCG terminology.
```

## Files in the repo (for a file-capable model), under `pokemon-tcg-ai-battle/`

- `data/external/official/cards_full.json` : source of truth for what each card does.
- `data/external/official/EN_Card_Data.csv` : the official card table.
- `registry/card_review.json` : the live store of LLM-proposed and human-confirmed labels.
- `tools/review_server.py` : the labeling tool and the exact live taxonomy (edit there as we go).

## Notes for me (not part of the prompt)

The live taxonomy is whatever is in `tools/review_server.py` (I edit it as we go). If I change
classes, regenerate this brief before handing it off again.
