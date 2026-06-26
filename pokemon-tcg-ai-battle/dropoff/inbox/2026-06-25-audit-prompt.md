# Audit prompt (paste into another model, with this folder's files attached)

You are auditing the Pokemon TCG "AI Battle" agent for the Cinderace / Mega-Starmie ex deck. The owner's
recurring finding: the heuristic IDEAS are right, but the IMPLEMENTATIONS are often too naive/restrictive --
worse than the idea. Your job is to find and fix those, plus any logical bugs, plus missing heuristics.

## Files attached
- `starmie_heuristics.py`  -- THE agent to audit. Entry: `choose_action(obs, deck)` = heuristics (`choose`)
  -> forward-model search (search_v3, with a veto on known-bad picks) -> legal default. `choose` runs RULES
  in order: go_first, no_suicide, `_main_action` (forces only high-confidence mechanical moves, defers the
  rest to search), `_opp_bench_target` (gust/snipe), `_tutor_target` (search/fetch).
- `deck_policy_v3.py` -- option/board helpers (option_card_id, option_target_entity, attack_profile,
  best_ko_attack, _attached_count, _prize_value, _bench, _active, ...). Read to understand the obs schema.
- `starmie_deck_and_heuristics.md` -- card texts (real attack/effect text) + the FULL intended heuristic list.
- `starmie-disagreement-moments.json` -- top board states where our agent disagrees with TOP pilots (each has
  the board with current HP+energy, all options labeled, pilot pick vs our pick). This is "seeing the replays".
- `starmie-loss-findings.md` + `starmie_losses.json` -- games our agent LOST, with its full decision log and
  missed-KO flags, plus the owner-side analysis of the failing turns.
- `attack_stats.json`, `card_stats.json` -- card/attack data (advisory; see caveat).

## Obs/option schema you need
Option types: 1 YES, 2 NO, 3 CARD, 7 PLAY, 8 ATTACH, 9 EVOLVE, 10 ABILITY, 11 DISCARD, 12 RETREAT, 13 ATTACK,
14 END. A sub-prompt's `sel.effect.id` is the card whose effect prompted it (e.g. 1145 Mega Signal, 666
Cinderace/Turbo Flare, 1031 Mega Starmie/Jetting snipe, 1182 Boss gust). CARD options resolve via
`sel.deck[index]` (a hidden reveal zone like Pokegear exposes nothing). The engine only offers an attack you
can actually pay for.

## What to deliver
A prioritized list of concrete rule fixes. For each: the bug/naivety, the better rule, and the expected effect.
Specifically interrogate:
1. `_main_action` ordering and gates -- are we forcing anything that should be situational, or deferring
   something we should force? The big soft spot is `play` agreement (~27% vs pilots): pilots play more setup
   before attacking; we under-develop. Find a NON-naive rule (a blunt "stop developing once you can attack"
   gate ALREADY regressed -- don't repeat it).
2. The failing turns in `starmie-loss-findings.md` -- what heuristic change would have avoided each loss?
3. Wally's trigger is an HP proxy (<=50%), not a real "opponent can KO me next turn" check. Improve if safe.
4. Gust/snipe target ranking, Ignition-for-Nebula discipline, energy routing -- correct on the edge cases?
5. attack_stats.json is a flat number: it does NOT capture conditional/stacking damage or "ignores
   weakness/resistance" (Nebula Beam). Any damage rule must not trust it blindly.

## Validation (do not regress)
The metric is AGREEMENT with top pilots (a proxy, not the objective). Current: 57.9% overall (49.4% before
heuristics). Per category: select_card 67, energy_attach 56, evolve 76, attack 71, go_first 86, retreat 60,
play 40. Any change must hold/raise these. Also check it does not raise the loss rate (starmie_loss_capture_v1).
The Kaggle ladder is the real test; mirror self-play is noise. Do NOT rewrite the architecture -- sharpen the rules.

Biggest open problem (from loss analysis): the deck has only Staryu as a basic (Cinderace is setup-only), so the
bench is structurally thin; ~1/3 of losses still reach an empty bench on bad draws. See starmie-loss-findings.md
for candidate fixes (Night Stretcher recovery, don't-evolve-last-Staryu, Cinderace setup, deck-ratio question).
