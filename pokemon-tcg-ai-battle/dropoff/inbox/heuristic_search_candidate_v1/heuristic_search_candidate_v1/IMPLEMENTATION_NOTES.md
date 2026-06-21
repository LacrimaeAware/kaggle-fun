# Dudunsparce / Alakazam search-policy candidate v1

This folder is a **candidate implementation**, not a validated promotion.
It is based on the uploaded `main(1).py`, `search(1).py`, `eval(1).py`, and
`features(1).py` snapshots.

## Structural change

A new `deck_policy.py` centralizes deck-specific intelligence. The search remains
authoritative at strategic root decisions.

- `main.py`
  - forces only a confirmed final-prize attack, not every KO;
  - recognizes dynamic Alakazam `Powerful Hand` damage;
  - uses `deck_policy.choose_subdecision` for tutor, target, recovery, and
    multi-select prompts;
  - supplies deck-policy priors to search for option ordering and exact ties.
- `search.py`
  - uses the deck policy for simulated tutor/gust/evolution/recovery choices;
  - uses dynamic Powerful Hand damage in simulated attack selection;
  - keeps root search and forward-model evaluation as final authority.
- `eval.py`
  - adds conservative, capped hand/card-advantage terms;
  - adds a modest public-board opponent-threat term;
  - exposes all weights as constants for clean ablation.
- `features.py`
  - adds opponent immediate-threat fields;
  - supports dynamic Powerful Hand damage;
  - fixes Team Rocket Energy as Psychic/Darkness rather than universal wildcard;
  - accepts an explicit evaluation perspective.

## Heuristics implemented

### Root option priors only

These order candidates under the time cap and break exact search ties. They do
not directly replace the search result.

- Powerful Hand post-action hand-budget guard.
- Evolve-from-hand priors for Kadabra and Alakazam.
- Rare Candy only when it completes an immediate Alakazam line.
- Poffin/Dawn/Hilda/Poke Pad priorities based on missing evolution pieces.
- State-dependent Dudunsparce Run Away Draw prior.
- Enriching Energy to Dudunsparce burst prior when Alakazam is online.
- Boss's Orders prior only when an opponent bench target has concrete prize/KO
  or powered-threat value.
- Enhanced Hammer only when a Special Energy is actually attached.
- Night Stretcher/Sacred Ash only when recovery is useful.
- Hold Battle Cage unless a later explicit spread-threat detector supports it.

### Subdecision and multi-select policy

- Complete live Abra -> Kadabra -> Alakazam lines.
- Prefer Alakazam over generic tutor value when Kadabra or Candy+Abra is ready.
- Establish missing Abra and Dunsparce lines with bench-space awareness.
- Recover relevant evolution pieces or Psychic energy.
- Prefer prize-rich / low-HP / highly powered opponent targets on gust prompts.
- Allow optional `minCount == 0` prompts to decline rather than always choosing
  option zero.

### Forced rule

Only a confirmed game-ending prize attack is forced before search. Ordinary KOs
fall through to forward search so the return swing and prize trade can be
considered.

## What still needs real-engine validation

1. Selection-area resolution for every tutor/context. The code is defensive and
   tries all observed replay schemas, but it needs traces from the real engine.
2. Dynamic Powerful Hand attack-name/id mapping. The implementation also falls
   back to active-card id 743 plus zero static damage.
3. Boss target prompts and opponent `playerIndex` across all simulator contexts.
4. Enriching Energy + Run Away Draw sequencing under engine behavior.
5. Weight values in `eval.py`. Test each new term behind independent toggles.

## Suggested clean A/B sequence

Keep deck, search budget, determinizations, and opponent fixed.

1. Baseline uploaded search.
2. Only final-prize forced-KO + dynamic Powerful Hand fix.
3. Add subdecision/continuation policy, with eval unchanged.
4. Add root priors, with eval unchanged.
5. Add opponent-threat term.
6. Add capped hand terms.

The subdecision layer is the highest-priority test because it fixes a structural
option-zero continuation problem rather than relying on tuned weights.
