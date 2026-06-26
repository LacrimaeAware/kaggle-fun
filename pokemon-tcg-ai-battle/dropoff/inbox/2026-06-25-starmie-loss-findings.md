# Starmie heavy agent: loss analysis + fixes (2026-06-25)

Method: `tools/starmie_loss_capture_v1.py` runs the heavy agent vs the field (alakazam, denpa92) and the
mirror (deployed sub_starmie), and on every LOSS saves the agent's full decision log (board with current
HP+energy, the option menu, our pick, source heuristic|search, and flags). Output `data/starmie_losses.json`.

## Root cause of losses: we don't build a bench
First capture (15 losses): **0 missed-KO flags** (the KO floor never fails) but **54% of decisions in lost
games had an EMPTY BENCH**. Typical loss: we evolve our only Staryu into Mega Starmie, draw cards with no
board, and either get swept the moment the active is KO'd (no Pokemon to promote = instant loss) or deck out
from over-drawing. We were one KO from losing, all game. Prize-wise we were often "ahead" (our remaining 4-6)
yet lost -- it wasn't a prize race, it was board collapse / deckout.

## Fix
1. `_develop_bench`: when in-play Pokemon <= 2, play Poffin (fetch 2 Staryu to bench) or bench a Staryu from
   hand. FREE (does not end the turn).
2. Reordered `_main_action` to do ALL free development (bench, evolve->Mega, gust/heal/tool, energy, Crushing
   Hammer) BEFORE the turn-ending KO/attack. The KO is preserved (free dev doesn't end the turn), so this is
   the develop-before-attack idea done correctly -- unlike the earlier blunt "stop developing once you can
   attack" gate, which forced generic search/draw and regressed.

## Result (measured)
- Imitation agreement vs top pilots: 55.3% -> **57.9%**, and the `play` category 27% -> **40%** (we now develop
  the turn like pilots instead of attacking into an empty board). evolve 76, select_card 67.
- Empty-bench decisions in losses: 54% -> **46%**; bench-developing plays up.

## Residual (deck-structural -- candidate auditor work)
The deck's ONLY basic Pokemon is Staryu x3 (Cinderace enters via Explosiveness at SETUP only, as the active;
Mega Starmie is Stage 1 from Staryu). So the max board is small and, on bad draws (no Poffin / Staryu drawn),
the bench is unavoidably thin -- ~5 losses still reached max bench 0. Heuristics can't fully fix a structural
basic shortage, but these are worth trying:
- Night Stretcher (1097): when bench is empty and a Staryu is in the discard, recover a Pokemon (its target is
  a sub-prompt; add a rule to recover a Pokemon over energy when board-starved).
- Don't evolve the LAST Staryu into Mega Starmie if it leaves 0 bench and we can't rebuild this turn (keep a
  basic on the bench as insurance) -- needs care vs the value of a tanky 330HP Mega.
- Cinderace Explosiveness at setup (open Cinderace face-down active) -- not explicitly forced; a setup rule may
  improve the early board.
- More aggressive bench target: raise the _develop_bench threshold (in-play <= 3) if win-rate testing supports it.
- Deck question for the owner: is 3 Staryu enough basics, or does the winning list run more / a recovery line?
  (Compare to the highest-winrate sublists in data/starmie_top_pilots.json + deck_winrate_v2.)
