# AUDIT HANDOFF: Starmie heuristic agent (2026-06-25)

Read-first record for an auditing model/human. Goal: win the Kaggle "Pokemon TCG AI Battle" Strategy track.
Current focus: a heuristic-first agent piloting the Cinderace / Mega-Starmie ex deck.

## The one thing to know
The agent's heuristic IDEAS (recognized problems) are mostly correct. The recurring failure is that the
IMPLEMENTATIONS are too naive/restrictive -- worse than the idea itself. Your job: find the naive/buggy/missing
implementations in `agent/starmie_heuristics.py`, propose better ones, and validate against the imitation
metric (must not regress it). Do NOT rewrite the architecture; sharpen the rules.

## Method that grounds everything (imitation gap)
Replays store the full observation at every decision. We feed each recorded observation from a TOP-PILOT
winning game to our agent and compare its pick to the pilot's actual move. Pairing (verified 100% legal):
`obs = steps[t][seat].observation`, `pilot_action = steps[t+1][seat].action`. The metric is AGREEMENT RATE
with top pilots (overall + per category). It is a proxy for good play, NOT the objective -- do not overfit;
confirm direction, then the ladder is the real test. Top Starmie pilots win 65-82%; our deployed pilot 46%.

- `tools/starmie_top_pilots_v1.py` -> `data/starmie_top_pilots.json` (ranks pilots, indexes winning games).
- `tools/imitation_gap_v1.py --agent {deployed|heavy} --max-games 40` -> `data/imitation_gap*.json`
  (agreement + per-category + ranked disagreements WITH full board/option context).
- `tools/build_imitation_viewer_v1.py --in data/imitation_gap_heavy.json --out <html>` -> card-image viewer.
- `dropoff/inbox/2026-06-25-starmie-disagreement-moments.json` = THE COMMITTED MOMENTS FILE (top 40, ~82KB):
  each entry has the board (me/opp active+bench with current HP + energy, my hand, prizes, deck counts), every
  option labeled, the pilot pick and our pick. This is how you "see the replays" without the (gitignored,
  5276-file) corpus. Full regenerable version (all disagreements) = `data/imitation_gap_heavy.json` (gitignored;
  produce with `tools/imitation_gap_v1.py --agent heavy`).
- Card texts + intended heuristic list: `docs/starmie_deck_and_heuristics.md`. Deck list: same doc + the
  `STARMIE_DECK` constant in `agent/starmie_heuristics.py`.

## Result so far
Heavy agent agreement 49.4% (deployed) -> 55.3% over 40 top-pilot winning games. Per-category heavy vs deployed:
evolve 25->69%, energy_attach 30->59%, go_first 55->86%, retreat 0->60%, select_card 63->66% (effect.id tutor);
attack ~= deployed; `play` 32->27% (we still develop less than pilots in nuanced spots -- the known soft spot).
Local A/B (catastrophe check, NOT ladder-predictive -- mirror self-play has repeatedly failed to predict the
ladder): heavy beats field decks (vs alakazam ~73%, vs denpa92 ~87%) but loses the same-deck mirror to the
pure-search deployed agent (~37%, noisy). Submission built+verified: `submissions/sub_starmie2` (6/8 vs first).

## Architecture (agent/starmie_heuristics.py)
`choose_action(obs, deck)` = heuristic-first (`choose`) -> forward-model search (search_v3, with a veto on
known-bad picks) -> legal default. `choose` runs RULES in order: `_go_first`, `_no_suicide`, `_main_action`.
`_main_action` forces ONLY high-confidence mechanical moves, in order, then returns None to DEFER the
ambiguous "which trainer / chip vs one-more-setup / search target" judgment to search:
  win-now -> KO (Jetting-Blow-preferred) -> evolve Staryu->Mega Starmie -> high-value play (gust-KO / Wally's /
  Hero's Cape) -> energy attach to the line -> retreat-pivot Cinderace->Mega Starmie -> None(defer).
LESSON (measured): forcing the ambiguous decisions REGRESSED (a can_attack develop-gate: 49.6%; forcing
Mega Signal/Poffin: 52.7%). Forcing only the unambiguous and deferring the rest: ~54%. Respect this.

## Coverage audit (intended heuristics -> status)  [docs/starmie_deck_and_heuristics.md is the full list]
IMPLEMENTED (forced):
- go first; no-suicide; evolve Staryu->Mega Starmie; game-winning attack; KO floor Jetting-first; energy to the
  line (Water/Ignition/Cape via `_attach_score`); Ignition only to fund a Nebula KO; Hero's Cape on Mega
  Starmie; Wally's only when Mega Starmie <=50% HP (+ a veto that stops search playing a pointless full-HP
  Wally's); Boss gust for ANY-prize bench KO when we can't KO the active; retreat-pivot Cinderace->Mega Starmie;
  Nebula ignores weakness/resistance (`_attack_kos_active`).
- Tutor/fetch TARGETS (`_tutor_target`, NOW ON): keyed on `sel.effect.id` (the card prompting), resolves cards
  via `sel.deck[index]`. Turbo Flare(666)->grab Basic Water; Poffin(1086)->Staryu; Mega Signal/Ultra Ball/
  Salvatore/Hilda->highest-need card (`_need_value`). Raised select_card 53->66% (vs 63% deployed). Pokegear's
  reveal zone (area 12) exposes no cards -> defers.
- Jetting Blow bench-SNIPE target + Boss gust target (`_opp_bench_target`, keyed on effect.id 1031/1182):
  pick the opponent benched Pokemon we can KO this turn (snipe=50, gust=main attack), preferring higher prize ->
  Water-weak (counter-specific) -> lowest HP.
- Crushing Hammer (`_crushing_hammer_play`): play when opp active has energy (free disruption).
- Don't field a 3rd Mega Starmie: evolve->Mega is skipped once 2 are already in play.
DEFERRED to search (intentional): chip attack selection (Jetting vs Nebula when neither KOs); which draw/search
  SUPPORTER to play and the play-vs-develop timing; deck-out buffer (search leaf_mode="deckout"); Pokegear target.
STILL MISSING / WEAK (audit targets):
- `play` agreement only 27% (pilots play more setup; we still under-develop in nuanced spots -- the hardest gap).
- "Don't retreat the tank (Mega Starmie) needlessly" beyond the Cinderace->Mega pivot (search may still retreat it).
- Counter-specific beyond weakness-in-targeting (vs Lightning: protect/race) -- not modeled.
- Wally's trigger is an HP proxy (<=50%), not a real incoming-KO check.

## Suspected weaknesses to investigate (prioritized)
1. `play` agreement 25% (pilots play more setup than we do; we defer to search which under-develops or attacks).
   Is there a non-naive rule for "play a free item that advances the board" that does NOT regress like the
   can_attack gate did? Measure every change with imitation_gap.
2. Mirror loss to no-heuristics (~37%, noisy). RUN AN ABLATION: disable each forced rule group and A/B vs
   deployed to find any rule that HURTS. (No toggle mechanism exists yet -- add one, e.g. an env var read at
   import, gating each `_main_action` sub-rule, then loop tools/starmie_ab_v1.py.) The user expects this probe.
3. Wally's trigger is an HP proxy (<=50%); the real rule is "play only when the opponent can KO us next turn".
   A precise incoming-damage estimate is blocked by attack_stats being unreliable for conditional/stacking
   damage -- improve if you can model opponent damage safely.
4. Gust target selection is deferred to search (we play Boss but search picks the target). Add a target rule
   that picks the KO-able / highest-value benched target (deck_policy_v3._boss_proposals is close but uses the
   MAIN attack damage, wrong for the Jetting snipe target -- do not reuse it blindly).

## attack_stats caveat (from the user)
`attack_stats.json` is a flat per-attack number. It does NOT capture conditional/stacking damage (e.g. Hop's
stacking) or "ignores weakness/resistance" (Nebula Beam). Treat it as advisory; rely on engine affordability
(an attack is offered only if payable). Verify any damage-based rule against this.

## Key files (focus here)
- `agent/starmie_heuristics.py`  -- THE implementation to audit.
- `agent/deck_policy_v3.py`      -- option/board helpers (option_card_id, option_target_entity, attack_profile,
                                    best_ko_attack, _attached_count, _prize_value, _boss_proposals, ...).
- `agent/search_v3.py`           -- forward-model 1-ply search (the fallback). NOTE the opponent-modeling bug:
                                    `_search` defaults the opponent deck to OUR deck if no opp_decks passed.
- `tools/imitation_gap_v1.py`    -- the metric harness (run it to validate any change).
- `data/imitation_gap_heavy.json`-- the moments (board + options + picks) for the CURRENT heavy agent.
- `docs/starmie_deck_and_heuristics.md` -- card texts + the full intended heuristic list + deck plan.
- `submissions/sub_starmie2/`    -- the built submission (gitignored; rebuild via the steps in SUBMISSIONS.md).

## How to validate a change
`python tools/imitation_gap_v1.py --agent heavy --max-games 40 --budget 0.5` -> agreement must not drop
(per-category too). Then `tools/starmie_ab_v1.py` as a catastrophe check. Then the ladder is the real test.
