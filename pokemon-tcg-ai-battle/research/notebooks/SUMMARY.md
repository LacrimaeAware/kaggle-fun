# Pokemon TCG AI Battle: downloaded notebooks digest

Scope: the notebooks in `research/notebooks/` plus the host/competitor discussions, digested
by a parallel read on 2026-06-17. Numbers first.

Hard fact up front: of all the notebooks, **exactly two actually call the cg.api forward
model (`search_begin`/`search_step`)**: the official RL+MCTS sample and
`pokemon-ai-battle-agent-mega-lucario`. Three more (`strong-start-safe-agent-turn-search-lb-860`,
`pokemon-tcg-lucario-v2-strategy-baseline`, `ptcg-ai-battle-simulation-lucario-submission`)
ship a search scaffold that is disabled, broken, or both. Every other "search"/"turn-search"
in a filename is marketing. Treat "lb-860" as a filename label, not a score substantiated
inside any notebook.

## 1. One-line table

| Notebook | Approach | Quality | Reported score |
|---|---|---|---|
| a-sample-agent-raging-bolt-ex-deck | Greedy fixed-priority bot, no search | boilerplate | none |
| a-sample-rule-based-agent-dragapult-ex-deck | Hand-tuned per-option scorer + Phantom Dive KO planner | medium | none |
| a-sample-rule-based-agent-dragapult-ex-deck(1) | Duplicate Dragapult scorer | medium | none |
| a-sample-rule-based-agent-iono-s-deck | Greedy single-ply scorer (Iono Bellibolt) | medium | none |
| a-sample-rule-based-agent-mega-abomasnow-ex-deck | Official greedy sample (Abomasnow) | boilerplate | none |
| a-sample-rule-based-agent-mega-lucario-ex-deck | Greedy 1-ply scorer + attack plan (Lucario) | medium | none |
| battle-resource-overview-ver-1 | Post-hoc log EDA, no agent | low | none |
| battle-simulator-given-2-agents-and-2-decks | Local self-play harness + 2 greedy agents | medium (harness is the value) | none |
| beginner-guide-from-deck-list-to-first-valid-sub | Packaging tutorial, clones Lucario sample | boilerplate | none |
| crustle-wall-mirror-ok | Greedy option-ranker for a Crustle wall deck | medium | none |
| dragapult-v3-tempo-ptcg-ai-battle-agent | Greedy magic-number scoring, no search | medium | none ("860" is a weight + timestamp) |
| notebookbfe9a75755 | MISFILED: UMUD muscle-ultrasound, not Pokemon | n/a | n/a |
| pok-mon-ai-battle-challenge-strategy-tqm | Card-scoring GBR with target leak, never plays | boilerplate | none |
| pokemon-ai-battle-agent-mega-lucario | Rule policy + WORKING 1-ply forward search | medium | none (local sim did not run) |
| pokemon-tcg-lucario-v2-strategy-baseline | Rule policy + search scaffold broken (missing import) | medium | 0.95 vs random; search_fail_rate 1.000 |
| pokemon | Beginner priority-cascade, Abomasnow, no search | low | 10/10 vs random |
| ptcg-abc-deck-builder | Stub: one markdown cell linking a deck builder | boilerplate | none |
| ptcg-ai-battle-simulation-lucario-submission | Cloned Lucario + dead search (wrong arg count) | medium | none |
| reinforcement-learning-and-mcts-sample-code | AlphaZero-lite: Transformer + MCTS over real forward model | medium | self-play vs random 20/50/64/76/76% |
| strong-start-safe-agent-turn-search-lb-860 | Lucario policy + crash-safe wrapper + OFF-by-default search | medium | none in body |
| top-dragapult-ex-tempo-control-agent | Greedy stateless Dragapult scorer | medium | none |
| validated-rule-based-agent-matchup-tests(1) | Hand-tuned Lucario policy + 1-turn planner, no search | medium (one of the better) | self-reported local only |

## 2. The genuinely useful ones, and what to steal

### reinforcement-learning-and-mcts-sample-code (the only correct Search API reference)
- Correct usage: `search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize,
  opponent_hand, opponent_active)` seeds a determinized state; `search_step(searchId, select)`
  applies an action; `search_end()` releases. Lets a rule agent compute the TRUE result of an
  action (exact damage, whether a drawn Item is playable) instead of guessing.
- Determinization recipe: your deck/prize via `random.sample` without replacement; opponent
  deck/prize/hand filled with placeholder IDs. The obvious upgrade is a real opponent prior.
- SparseVector board featurization (EmbeddingBag sum), perspective-relative via `yourIndex`.
- `get_decoder_input` shows exactly how to read OptionType/SelectContext option fields.
- Do not copy blindly: `SEARCH_COUNT=10` sims/move is shallow; opponent filled with constant
  placeholders so the MCTS searches a nonsense opponent; eval is 50 mirror games vs random.

### pokemon-ai-battle-agent-mega-lucario (the only WORKING non-MCTS search)
- 1-ply root re-rank: take the policy's top 6 first-actions; for each, open a determinized
  rollout via `search_begin`, drive forward up to 40 `search_step` calls using the heuristic
  as rollout policy, stop on result/opponent-turn/END, score the leaf with `evaluate_state`,
  pick the best first-action. Soft 1.5s budget, falls back to pure policy on exception with
  guaranteed `search_release`. Cleanest deck-agnostic "bolt search onto a heuristic" template.
- Caveat: its rollout uses the same heuristic for both sides and bails when `yourIndex` flips,
  so it does not model the opponent's turn. Time budget unmeasured.

### strong-start-safe-agent-turn-search-lb-860 (steal the eval + discipline, not the search)
Search hook is `USE_SEARCH=False` and the `search_begin` input format is openly flagged as
guessed/unverified, so the headline feature never ran. Lift verbatim:
- `prize_count(pokemon)`: megaEx 3 / ex 2 / basic 1, minus Legacy Energy and Lillie's Pearl.
- `pokemon_score`: prize*1000 + energies*150 + tools*100 + stage bonus + hp, with per-id tweaks.
- `evaluate_state`: (op.prize - me.prize)*10000 + energies*120 + key-mon bonuses + hp terms.
- Attack-planning loop: attackers x attacks x targets, weakness x2 / resistance -30, lethal
  detection, score=50000 when the KO wins, threaded through every sub-selection.
- Crash-safe envelope + `_legal_fallback` + output clamping. Mirror validation forfeits on any
  exception, so this is pure downside protection.
- Ladder mechanics it documents: 5 subs/day, only latest 2 scored, mu starts 600, W/D/L only
  (margin of victory irrelevant), mirror validation must not error.

### pokemon-tcg-lucario-v2-strategy-baseline (steal the self-audit)
Search dead on arrival (missing `import random`, `search_fail_rate=1.000`). Steal:
- `normalize_selection()`: distinct, in-range indices satisfying minCount, never exceeding
  maxCount, skipping non-positive-score options unless minCount forces a pick.
- `_DIAG`/`fallback_rate` self-audit: counts policy_ok vs fallback per decision; the gate fails
  on high fallback even at 0 engine errors. The anti-self-deception pattern is the best part.
- `validate_submission_bundle()`: asserts main.py+deck.csv at root, 60 int ids, agent present,
  compiles, cg bundled if `from cg.api` used.

### battle-simulator-given-2-agents-and-2-decks (steal the harness wholesale)
- `run_match()` with seat alternation, step cap, per-agent invalid/exception counts.
- `validate_selection()`: list[int], all ints, len in [minCount,maxCount], no dupes, in range.
- `legal_fallback_selection()`: `list(range(min(minCount, n)))` (minimum, not max).
- `infer_winner`/terminal detection probing CABT field paths + prize-zero / no-Pokemon.

### validated-rule-based-agent-matchup-tests(1) (best pure-heuristic reference)
- OptionType x SelectContext scoring dispatcher; `get_card(obs, area, index, player)` AreaType
  resolver; `prize_count`/`target_score`; `_plan_attack` one-turn-commit planner with an
  energy-attach feasibility check that respects `state.energyAttached`; bench index +1 offset
  convention; `_low_deck()` (deckCount<=8) anti-deckout guard; packaging + self-play harness.

### Cross-cutting plumbing worth copying once
- Deck-path fallback: local `deck.csv` else `/kaggle_simulations/agent/deck.csv`.
- Packaging: tar.gz main.py + deck.csv + cg/ at top level, exclude `__pycache__`/`.pyc`.
- Card-counting hidden-info reconstruction: from the known 60-card multiset, subtract
  everything visible (hand/discard/bench/active/stadium/looking), dedupe by `serial`; remainder
  is deck+prizes; subtract `select.deck` on a deck-search prompt to isolate the 6 prized cards.
- Turn-log buffering: accumulate `obs.logs`, flush on `TURN_END`, scan for opponent attack ids
  and KO events to unlock comeback cards.
- Priority-via-magnitude ordering: encode intra-turn sequencing into score bands so a greedy
  scorer sequences a turn without a phase state machine.

## 3. Deck meta from ISAKA's data (third-party, reproduce before trusting)

~15k self-play games, the 4 sample rule-based agents round-robin. Measures deck+baseline-policy
together, not deck ceiling.

| Deck/agent | Win rate | Record |
|---|---|---|
| Mega Lucario ex | 60.4% | 4600-3011 |
| Dragapult ex | 55.6% | 4232-3382 |
| Iono | 43.8% | 3336-4277 |
| Mega Abomasnow ex | 40.2% | 3057-4555 |

Head-to-head (row vs column, % for row):

| | Lucario | Dragapult | Iono | Abomasnow |
|---|---|---|---|---|
| Lucario | - | 50 | 78 | 52 |
| Dragapult | 50 | - | 64 | 53 |
| Iono | 22 | 36 | - | 74 |
| Abomasnow | 48 | 47 | 26 | - |

RPS-like with Iono the swing. Lucario and Dragapult are a coin flip and both crush Iono.
Abomasnow's bad record is partly the Iono matchup (loses 26 vs Iono) while being even vs the
top two. First-player win rate 51.5%. Caveat: "Lucario best" is the sample policy piloting
Lucario in a field of sample policies; Lucario is also the most over-contested archetype here.

## 4. Official rules differences and deck-building

Engine is authoritative: "Simulator behavior will be treated as the correct behavior." Code to
the engine, not paper rules. The four stated divergences (host says none affect outcomes): some
edge-case attacks unselectable rather than fizzling; Mega Zygarde ex coin order auto
left-to-right; simultaneous-KO prize order differs but both-take-all is a draw.

Option-index ruling: returned indices are positions in `select.option`, honor minCount/maxCount,
no duplicates. Deck-building confirmed open by Kaggle staff: build from the full Data-tab pool,
not restricted to the four starters.

## 5. What to do with this

1. Build the local self-play harness first (lift from battle-simulator or the
   `battle_start`/`battle_select`/`battle_finish` loop). Always alternate seats (51.5% first-
   player edge). Prerequisite for everything.
2. Wrap any agent in the crash-safe + legality layer before touching strategy
   (`validate_selection` + `legal_fallback_selection` + `normalize_selection` + try/except).
   Mirror validation forfeits on any exception.
3. Take the working search pattern from `pokemon-ai-battle-agent-mega-lucario` (it runs); use
   the RL sample as ground truth for the correct `search_begin` signature. Ignore the broken
   search in the other three.
4. Fix the opponent model every notebook skips. The single highest-value differentiator is
   seeding `opponent_deck`/`opponent_hand`/`opponent_active` from a real meta prior instead of
   placeholders or a self-mirror. ISAKA's matchup data is a starting prior.
5. Reuse the eval functions (`prize_count`, `pokemon_score`, `evaluate_state`, weakness x2 /
   resistance -30); throw away the per-card magic-number trees (they rot with the meta).
6. Pick a less-contested archetype than Lucario. Deck-building is open; a stronger pilot on a
   deck the field has not tuned against beats out-tuning everyone on the most-cloned deck.
7. Do not trust any in-folder score. No notebook has a real LB number; self-play vs random is
   near-meaningless. Reproduce ISAKA's matrix in our own harness before believing any ordering.
8. Ignore outright: notebookbfe9a75755 (misfiled UMUD), pok-mon...-tqm (target-leaking GBR),
   ptcg-abc-deck-builder (stub), battle-resource-overview (EDA only), and the pure-boilerplate
   samples except as API cheat-sheets.
