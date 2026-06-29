# Archaludon (public notebook bot) vs our fleet — local head-to-head

Date: 2026-06-27
Bot: `submissions/sub_archaludon` (built from the public "Archaludon 75% WR vs my 1300+ Starmie" notebook; rule-based Archaludon ex / Cinderace, Metal deck).
Harness: scratchpad/h2h.py — Kaggle-faithful load (exec, no __file__, last callable), each agent's deck-reset run in its own cwd, seats alternated, separate output file per agent (avoids the Windows non-atomic concurrent-append bug that mangled the first run).
Reads: OUR win rate vs sub_archaludon (Archaludon WR = complement). 0 errors across all matchups.

| Agent | Deck | Pilot | Our WR | Archaludon WR | n |
|---|---|---|---:|---:|---:|
| sub_heuristic | Alakazam (hiroingk) | heuristic | 54% | 46% | 100 |
| sub_heuristic2 | Alakazam | heuristic | 47% | 53% | 100 |
| sub_search | Alakazam | search | 34% | 66% | 50 |
| sub_planner | Alakazam | search | 34% | 66% | 50 |
| sub_phaware | Alakazam | PH heuristic | 20% | 80% | 50 |
| sub_starmie | Starmie | heuristic | 26% | 74% | 100 |
| sub_starmie3 | Starmie | heuristic | 21% | 79% | 100 |
| sub_starmie2 (live 1300+) | Starmie | heuristic | 19% | 81% | 100 |
| sub_combine | other (dead 422 line) | search | 6% | 94% | 50 |

Deck identity confirmed empirically (each agent's own reset): sub_starmie/2/3 -> Starmie list (1030/1031); sub_heuristic/heuristic2/search/planner/phaware -> hiroingk Alakazam list; sub_combine -> a different list.

## Findings
1. Archaludon does NOT beat our best deck. The Alakazam heuristic line is ~47-54% (coinflip). The notebook's advertised 74% was vs the author's FROSLASS Starmie (weak to Metal), explicitly matchup-favorable; vs our Alakazam heuristic it is even.
2. Archaludon is genuine anti-Starmie tech: all Starmie agents lose at 19-26% (Archaludon ~75-81%). Our Starmie builds fold to a Metal/Archaludon deck.
3. Piloting dominates: same Alakazam 60 cards, heuristic ~50% vs search ~34% vs PH-heuristic 20%. Heuristic floor beats search again (re-confirms the project pattern).
4. sub_combine is dead (6%).

## Practical read
Uploading Archaludon ourselves is not a clear upgrade over our Alakazam heuristic (coinflip head-to-head; our heuristic is already a ~770-class agent). Archaludon's value is purely as a Starmie counter. Defensive takeaway: our Starmie line is exploitable by Metal/Archaludon; the Alakazam heuristic is the resilient line.

## Caveats
Local games vs this single opponent, n=50-100 (~±7-10pp). The heuristic 47 vs 54 is sampling noise (~50%). Not ladder win rate. Search-pilot agents (sub_search/planner/phaware/combine) run forward-model search and are slow, hence n=50.

Artifacts: submission built at submissions/sub_archaludon/ and submissions/sub_archaludon.tar.gz (verified Kaggle-style load, 0 errors). Not uploaded — user's call (latest-two-active-submissions slot tradeoff).
