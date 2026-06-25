# PH-aware heuristic + search: a working local combination (2026-06-23)

Tested in kaggle-fun on the new deck. Tool: `tools/run_heuristic_ab_v1.py`. Data: `data/heuristic_ab_v1.json`.
Seat-alternated, n=220 to 300, two-sided p. Note a ~0.07 side-A harness tilt (self-mirror control ran high);
the winning side is side A below, so correct each number down by roughly that much. The gaps are far larger
than the tilt.

## The major heuristic: PH-aware KO detection
The plain kaggle-fun heuristic reads Powerful Hand as 0 damage, so it cannot take the deck's main attack.
Result of that gap:
- choose (plain heuristic) vs first: 0.483 (tied; PH-blind, cannot pilot the deck)
- eff (effect-aware development) vs first: 0.390 (worse; its CEFF scoring misvalues plays)
- eff vs choose: 0.447

Add only PH-aware KO detection (deck_policy_v3.best_ko_attack):
- phaware vs choose: 0.793 (238-62)
- phaware vs first: 0.760 (228-72)

Seeing Powerful Hand KOs is the single biggest heuristic fix for this deck.

## Combining with search (the non-fighting design)
phaware floor takes the KO; forward-model search (search_v3, PH-aware) picks the developmental move where
the heuristic has no opinion, instead of option-0:
- phaware_search vs phaware: 0.745 (164-56)
- phaware_search vs first: 0.900 (198-22)

So heuristic + search beats heuristic alone by ~0.25. Mechanism: the heuristic supplies the KO tactic the
search eval recognizes unreliably; search supplies the development the heuristic does badly. Each covers the
other's blind spot.

Why this works where the earlier new-repo hybrid was neutral: that hybrid's heuristic floor fired on nearly
every decision, so search never acted. Here the floor is minimal (just the KO), so search drives development.

## Recipe to port
Minimal heuristic floor (PH-aware KO, energy-on-active, go-first) + forward-model search for everything else.
Keep the floor small so search actually acts on development.

## Harness fairness (settled)
Self-mirror at n=600: phaware vs phaware = 0.502 (301-299), choose vs choose = 0.528 (CI includes 0.5). The
harness is fair, no side-A bias. The 0.57 to 0.59 self-mirrors seen earlier at n=120 to 140 were noise. So the
numbers above stand as measured, no "tilt" correction.

## Card-advantage objective: NEUTRAL vs board (2026-06-24)
Added `eval.evaluate_ca` (board + 25 per card in hand) as a `leaf_mode="ca"`; `phaware_search_ca` uses it for
development. Same KO floor and same search as `phaware_search`; only the development objective differs.
- phaware_search_ca vs phaware_search: 0.525 (105-95, p=0.48) -- NEUTRAL.
- phaware_search_ca vs phaware: 0.805; vs first: 0.910 (same as the board version's 0.745 / 0.900).
Why neutral: (1) the board terms (body x30, energy x8) already reward the plays that draw/tutor, so board and
card-advantage pick nearly the same lines; (2) 1-ply search cannot plan the tutors-first cascade -- it branches
one decision then finishes the turn with a fixed rollout. The cascade idea needs a turn-planner that sequences
the whole turn, not a different leaf objective. Bottleneck is sequencing depth, not the objective.

## Caveats
- Local self-play; the ladder is the only truth and has said heuristic (736) beats pure search (687). This
  combination is NOT that pure search: it keeps the PH-aware heuristic floor the losing search lacked. Worth a
  ladder submission to confirm.
- In the new repo, the gain depends on whether its heuristic develops better than option-0. If its development
  is already good, search adds less than the +0.25 seen here.
