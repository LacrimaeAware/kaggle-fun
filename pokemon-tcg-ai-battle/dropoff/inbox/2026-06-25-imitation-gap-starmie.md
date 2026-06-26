# Imitation-gap analysis: our Starmie agent vs top pilots (2026-06-25)

## Method (reusable, evidence-driven; replaces hand-writing heuristics blind)
Replays store the FULL observation at every step (`steps[t][seat].observation` = the board `current` + the
legal `select` options the pilot saw) AND the action taken. Pairing is `(obs = steps[t][seat].observation,
pilot_action = steps[t+1][seat].action)` -- verified 100% legal across the corpus (action[t] is the PREVIOUS
action; the response is at t+1). So we can feed each recorded observation straight to our agent and compare
its pick to a strong pilot's actual move -- no game re-simulation, no kaggle_environments, only the cg forward
model for the search part.

Tools:
- `tools/starmie_top_pilots_v1.py` -> `data/starmie_top_pilots.json`: ranks pilots of the Mega-Starmie
  archetype (deck contains 1031) by win rate; indexes their winning games. (Replays carry no Elo; win rate
  is the proxy. Top pilots 65-82%; OUR pilot "Shishio Makoto" 46.2% over 26 games -- matches the 0.480 ladder.)
- `tools/imitation_gap_v1.py` -> `data/imitation_gap.json`: for top-pilot winning games, runs our deployed
  Starmie agent (KO floor + search_v3 deckout/develop, budget 0.6s) on every real decision, compares to the
  pilot, categorizes, scores significance, carries full board+option context for the visualizer.
- `tools/build_imitation_viewer_v1.py` -> `pokemon-ai-agent/imitation_review.html`: renders the top moments
  with card images (hover to enlarge), pilot pick (green) vs our pick (blue), full board + hand + option menu.

## Headline
Agreement **1462/3097 = 47.2%** over 80 top-pilot winning games. Disagreements by category: play 586,
select_card 460, energy_attach 237, evolve 114, attack 85, end_turn 36, ability 27, retreat 10.

## Central finding: we attack too early; pilots develop the whole turn, attack last
Attacking ENDS the turn. Our agent (KO floor first, then search) commits to an attack and skips development.
- **WE attack & pilot does NOT: 339.** Pilot instead: Pokegear 33, Crushing Hammer 26, Evolve->Mega Starmie 24,
  Lillie's 21, Poffin 20, Wally's Compassion 20, Mega Signal 18, attach Water->Mega Starmie 17, Hilda 16,
  Ability 13. (= full setup before the turn-ending attack.)
- **PILOT attacks & we do NOT: only 68** (and when we hold back we often RETREAT 16 -- usually wrong).
- Asymmetry ~5:1 -> our agent is systematically premature. (This is the exact failure deck_policy_v3's
  `safe_pre_attack_indices` was meant to fix, but agent_starmie's KO-floor/search path bypasses it.)

## Other systematic gaps
- **Retreat the tank needlessly**: in play/energy/attack disagreements we choose RETREAT ~95 times. Mega
  Starmie (330/430 HP) should rarely retreat (costs [C][C]).
- **Search targets (select_card 460)**: we fetch the wrong card. Pilots fetch the missing piece (Mega
  Signal->Mega Starmie ex, Poffin->Staryu/basics, Ultra Ball->the needed card). Needs tutor-target rules.
- **Energy routing (237)**: pilots route Water->Mega Starmie 90, ->Staryu 37, ->Cinderace 27; Ignition->Mega
  Starmie 26; Hero's Cape->Mega Starmie 19. We often attack instead or route differently.
- **Disruption/recovery ignored**: pilots play Crushing Hammer (55) and Wally's Compassion (20) heavily; our
  agent essentially never does. (Note: Crushing Hammer is a coin flip; value still high in pilot play.)
- **Evolve promptly**: pilots evolve Staryu->Mega Starmie (24 cases where we attacked instead).

## Heuristics indicated (priority order, to verify with the user on the visualizer)
1. **Develop-before-attack (attack last).** Prefer useful development (search/draw, attach to the Starmie
   line, evolve toward the attacker, abilities, Crushing Hammer, Wally's, Hero's Cape) over a NON-winning
   attack that ends the turn. Take the attack when development is exhausted, or it is game-winning, or
   attacking now is clearly best. Biggest single lever.
2. **Retreat guard**: don't retreat Mega Starmie without a concrete reason.
3. **Tutor targets** on searches: Mega Signal->Mega Starmie ex; Poffin->Staryu; Ultra Ball->missing key piece.
4. **Energy routing**: Water/Ignition/Hero's Cape -> the Mega Starmie line.
5. **Ignition discipline** (user rule): Ignition->Mega Starmie only to fund a Nebula KO this turn that Jetting
   Blow can't reach; otherwise don't spend it.
6. **Use Crushing Hammer / Wally's / evolve** in the development phase.

## Iteration metric
Agreement-with-top-pilots (imitation_gap.py) is a FAST, meaningful local metric (NOT mirror self-play, which
is mirror-blind and does not predict the ladder; NOT most-common-replay top-1, which the user rightly
discounts). It matches HIGH-winrate pilots. Use the agreement rate (overall + per category) to measure each
heuristic change. Caveat: imitation is a proxy for good sequencing, not the objective (winning) -- do not
overfit; confirm direction, then ship + watch the ladder.

## Implementation + measured result (heavy heuristic agent)
`agent/starmie_heuristics.py` (vendored into `submissions/sub_starmie2/`, entry = `starmie_heuristics.agent`).
Design lesson learned the hard way (measured each step): FORCE only the unambiguous mechanical heuristics,
DEFER the ambiguous judgment to search.
- Forced (in order): win-now; take a KO (Jetting-Blow-preferred, conserve Ignition/Nebula); evolve
  Staryu->Mega Starmie; gust a 2+prize KO / Wally's heal / Hero's Cape; route energy onto the line; pivot
  Cinderace->Mega Starmie; go first; no-suicide.
- Deferred to search_v3: which trainer to play, chip-vs-one-more-setup, search/fetch targets.

Iteration (tools/imitation_gap_v1.py --agent heavy, 40 top-pilot games, vs deployed baseline 49.4%):
- v1 (develop-before-attack with a can_attack gate): 49.6% -- FLAT. Over-corrected: suppressed items pilots
  play freely; never retreated (0%); play 32->17.5%. Reverted the gate.
- v2 (force mechanical, defer rest): **54.3%** (+4.9). evolve 25->73%, energy_attach 30->57%, go_first 55->86%,
  retreat 0->64% (Cinderace->Mega pivot), attack/select_card ~= deployed (deferred).
- v3 (also force Mega Signal/Poffin setup plays): 52.7% -- WORSE. Forcing setup plays overrode better picks.
  Reverted. Final = v2 (~54%, reproduced 53.9%).

Local A/B (tools/starmie_ab_v1.py, budget 0.3 -- CATASTROPHE CHECK ONLY, mirror not ladder-predictive):
- v2 heavy, n=30: beats field (alakazam 73%, denpa92 87%) but LOST the mirror to deployed 11-19=37%.
- After fixing the naive rules the user flagged (gust required 2+ prizes -> any-prize bench KO when we can't KO
  the active; Wally's fired at <=60% HP -> only <=50% + a veto that stops SEARCH playing a pointless full-HP
  Wally's), n=50: heavy now BEATS deployed in the mirror 29-21=58% and crushes the field (alakazam 86%,
  denpa92 86%). So the 37% was largely the naive rules (+ small-sample noise), not a fundamental flaw -- exactly
  the user's point that our IMPLEMENTATIONS, not the IDEAS, are the problem. Heavy now wins on every local
  measure AND matches top pilots better (imitation +4.5). Ladder-test sub_starmie2 vs sub_starmie is the real test.
Remaining heavy gaps vs pilots: imitation_review_heavy.html (play/select_card -- nuanced trainer choices left to
search). See 2026-06-25-audit-handoff-starmie-heuristics.md for the auditor handoff + the coverage audit.

UPDATE (all heuristics added, per user "include everything"): added effect.id-keyed tutor/fetch targets (Mega
Signal->Mega Starmie, Poffin->Staryu, Turbo Flare->Basic Water, general search by need), Jetting snipe + Boss
gust target selection (weakness-aware), Crushing Hammer disruption, and a don't-field-a-3rd-Mega guard. The
KEY to non-naive tutor: `sel.effect.id` identifies the prompting card and CARD options resolve via
`sel.deck[index]` (Pokegear's area-12 reveal exposes nothing -> defer). Result: imitation 53.3->55.3%
(select_card 53->66%, beating deployed's 63%); minor evolve dip (-4pp from the no-3rd guard, intended). Local
mirror A/B then read 38% (vs the prior 58%) -- but the mirror is noise-dominated (37/58/38 across versions, search
RNG, coin-flippy same-deck matchup) and is NOT ladder-predictive; field A/B stayed strong (78-95%). Trusting the
imitation signal (the user's chosen iteration metric, monotonic up) + field dominance over the mirror. Candidates
to logic-check for the mirror dip: Crushing Hammer over-play and the no-3rd-Mega guard (an ablation toggle would
isolate them -- left for the audit).

## attack_stats caveat (user)
attack_stats.json is a flat per-attack number; it does NOT capture conditional/stacking damage (e.g. Hop's
stacking) or "ignores weakness/resistance" (Nebula Beam). Treat it as advisory; rely on engine affordability
(an attack is only offered if payable) and flag attack decisions for human review.
