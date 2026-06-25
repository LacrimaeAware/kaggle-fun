# Submission log (canonical)

The real signal is the Kaggle public score. Compare OUR OWN submissions to each other (relative deltas
are informative); do not over-read the absolute number or the global ranking (see the LB-aggregation
caveat in memory). Claude builds the tarball under `submissions/` (gitignored, token rule); the human
uploads it. **Every build that gets uploaded must get a row here, with the commit it was built from.**

## History (06-17 / 06-18)
Times are relative to when this was logged (2026-06-18, ~12:1x local). Scores + status + filenames are
FACTS from the Kaggle submissions page. "Config" is reconstructed from git commit timestamps where it was
not logged at build time -> treat reconstructed configs as best-effort, not certain.

| When (rel) | ~Clock | File | Public score | Status | Config (reconstructed unless noted) | Commit |
|---|---|---|---|---|---|---|
| 27s ago | 06-18 ~12:1x | sub_search.tar.gz | (pending) | Pending | agent_search, DENPA92 deck, **N_DETERM=8**, 1-ply hand-eval, aggro continuation | dc1fba8 (this is the FIRST N=8 submission) |
| 9h ago | 06-18 ~03:00 | sub_search.tar.gz | **697.7** | Complete | agent_search, DENPA92 deck, N_DETERM=4 (pre-N=8 commit 07:13) | ~between 5e71444 and 747d006 |
| 14h ago | 06-17 ~22:00 | sub_search.tar.gz | **640.7** | Complete | agent_search, DENPA92 deck, N_DETERM=4 (deck swap was 21:21) | ~5e71444 |
| 14h ago | 06-17 ~22:00 | sub_combine.tar.gz | **422.1** | Complete | agent_combine (heuristic floor + blended leaf eval) | ~8d82b46 |
| 14h ago | 06-17 ~22:00 | sub_combine.tar.gz | (none) | Error | combine, failed to run | |
| 14h ago | 06-17 ~22:00 | sub_search.tar.gz | (none) | Error | search, failed to run | |
| 1d ago | 06-17 ~12:00 | submission.tar.gz | **617.2** | Complete | earliest upload, pre-DENPA92-deck (old deck) | (pre 5e71444) |

## Reads (relative, our own subs only)
- **combine is much worse than search on the real LB**: 422.1 (combine) vs 640.7 (search) at the same
  time. This matches local self-play (combine loses) -> combine is dead, do not resubmit it.
- **search has been climbing**: 617.2 (old deck) -> 640.7 (DENPA92 deck) -> 697.7. The DENPA92 deck swap
  and later fixes moved it up. Do NOT assert which single change caused each delta (not isolated on the LB).
- **The pending submission (dc1fba8) is the first with N_DETERM=8** (local: N=8 beat N=4 ~0.675 head-to-head).
  Its score vs the 697.7 (N=4) is the real test of whether N=8 helps on the LB. **Record it when it completes.**
- agent_rank (the learned net), belief, and continuation were tested locally this session and did NOT beat
  agent_search, so none were submitted.

## Current standings (2026-06-24, from the Kaggle page; recency in parens)

| File | Public score | Status | Essence of the agent |
|---|---|---|---|
| submission.tar.gz (1d) | **770.9** | Complete | **BEST. No-search basic heuristic, Alakazam.** Pure heuristic piloting the DENPA92 Alakazam draw engine: PH-aware lethal/KO, energy onto the Alakazam line, deck-out denial, go first. This is the pokemon-ai-agent registry heuristic (lethal_ko + alakazam_energy + deck-out denial), the validated heuristic wins. |
| sub_search.tar.gz (6d) | 687.8 / 645.2 / 640.7 | Complete | 1-ply forward-model search, DENPA92 deck, determinization-averaged rollouts (N_DETERM 4 then 8), 4-term hand leaf eval (prize/hp/body/energy), aggressive opp-reply continuation. Early ones PH-blind. |
| submission.tar.gz (7d) | 617.2 | Complete | Earliest upload: day-one always-legal heuristic on the OLD default deck (pre-DENPA92), defer-the-end-turn rule. Beats random, ties first-option. |
| sub_pilot_hoard.tar.gz (5d) | 592.0 | Complete | "Hoard for Powerful Hand" variant: values keeping cards in hand (W_POWERFUL_HAND) on top of the agent. Below plain search, so hoarding-as-an-eval-term did NOT help on the LB. (best-effort ID from tools/hoard_ab.py) |
| sub_combine.tar.gz (6d) | 422.1 | Complete | Worst complete. Search with a BLENDED leaf eval = hand eval + learned gradient-boosted P(win). The learned value tanked it (matches local). DEAD, do not resubmit. |
| sub_combine / sub_search (6d) | (none) | Error | Failed to run. |
| sub_starmie2.tar.gz (2026-06-25) | (not built yet) | Draft | **HEURISTIC-FIRST STARMIE** (the imitation-grounded upgrade of sub_starmie). Same Cinderace/Mega-Starmie deck, but piloted by `starmie_heuristics.agent`: heuristics decide the high-confidence mechanical moves (evolve Staryu->Mega Starmie, route energy onto the line, take KOs Jetting-Blow-first to conserve Ignition/Nebula, gust 2+prize KOs, Wally's heal, Hero's Cape, pivot Cinderace->Mega Starmie, go first, no-suicide); search_v3 fills the ambiguous "which trainer / chip vs setup" decisions; legal-first last. Heuristics derived from the imitation-gap analysis vs top pilots (tools/imitation_gap_v1.py), NOT guessed. Agreement with top pilots 49.4% (deployed sub_starmie) -> ~54% over 40 winning games. Local A/B (catastrophe check, NOT ladder-predictive; n=50, after fixing the naive gust/wally rules): now BEATS the deployed no-heuristics agent in the mirror 29-21=58% (was 37% before the fixes -- the naive 2+prize-gust and full-HP-Wally's rules were actively hurting) and crushes the field (vs alakazam 86%, vs denpa92 86%). Verified loads/plays, 0 errors, fast (heuristic handles most decisions). Ladder-test vs sub_starmie's score is the real comparison. |
| sub_starmie.tar.gz (2026-06-25) | (pending) | Pending | DECK SWITCH: Cinderace/Mega-Starmie ex deck (most common Starmie list in the replay meta) piloted by our GENERIC forward-model search (no deck-specific heuristics; static-damage KO floor + search_v3 develop rollout + deck-out leaf). Built because local pilot A/Bs (tools/deck_pilot_ab_v1.py, n=40) show this deck beats our Alakazam pilot 33-7 (generic) and 28-12 vs heuristic_first (best Alakazam); Starmie is ~70% in human hands and low-complexity. Verified loads/plays, 0 errors. Tests whether switching decks raises the ceiling. |
| sub_heuristic2.tar.gz (2026-06-25) | (pending) | Pending | sub_heuristic + improvements: search fallback now uses the develop-first/attack-last rollout + deck-out leaf (the bracket-winning "dev" config), and determinizes the OPPONENT's hidden cards from the top-30 replay decks (data/opponent_meta_v1.json, 4501 games) instead of assuming the opponent runs our deck. Same hiroingk heuristics-first policy + Dawn fix + no-suicide. Verified 5/6 vs first, 0 errors. Compare vs sub_heuristic to isolate the search-fallback + opponent-modeling changes. |
| sub_heuristic.tar.gz (2026-06-24) | (pending) | Pending | HEURISTIC-FIRST: vendors the proven hiroingk heuristic registry (the 770.9 policy: lethal_ko, alakazam_energy, tutor_targets, setup_search, gust_threat, active_pivot, recovery/discard, draw-engine, retreat_guard; dudunsparce/evolution/resource_guard OFF) and runs it FIRST; kaggle-fun search_v3 fills only decisions no heuristic fires on; legal-first last. Hiroingk deck (Dunsparce 305). Includes the Dawn/tutor fix (fetch the line-completer, never a redundant Dunsparce) patched in the vendored copy. Verified 6/6 vs first, 0 errors; frozen tests pass. Built from e821a8f + uncommitted. This is the "use the heuristics + search as fallback" agent. |
| sub_planner.tar.gz (2026-06-24) | (pending) | Pending | PH-aware KO floor + 1-ply search_v3 (N_DETERM=8) for development, with the deck-out penalty leaf eval + develop-first/attack-last rollout. DENPA92 deck. Submitted as a comparison point (does a PH-aware deck-safe search beat the old PH-blind 640-688 search?). NEUTRAL vs plain phaware_search locally (0.483, 29-31, n=60); expect below the 770.9 heuristic. Built from e821a8f + uncommitted agent edits. Verified 7/8 vs first, 0 errors. |

**Decisive read (three ways agreeing):** the ladder ranks heuristic (770.9) >> search (640-688) >> combine (422).
Every search or learned-value addition has scored BELOW the plain heuristic. Local self-play this session
showed the all-findings planner (deck-out leaf + develop rollout) NEUTRAL vs plain phaware_search (0.483,
n=60; the deck-out/develop findings neither help nor hurt the search), and the combine disaster (422) confirms
learned leaves are harmful. So: the competitive paradigm is HEURISTIC, not
search. The strongest agent we have is the 770.9 no-search Alakazam heuristic (pokemon-ai-agent). The kaggle-fun
search line (sub_search/planner) is a ladder dead-end relative to it.

Local-only agents this session (NOT submitted): `phaware` (PH KO floor + heuristic), `phaware_search` (PH floor +
1-ply search, the local best search), `phaware_search_planner` (PH + deck-out leaf + develop rollout; loses to
phaware_search locally). Tarballs built+verified tonight: sub_phaware (8/8 vs first), sub_planner (7/8 vs first).

## Going-forward protocol
1. On every upload, add a row: time, file, the commit hash it was built from (`git rev-parse --short HEAD`),
   the agent + key config (deck, N_DETERM, leaf eval, continuation), and later the public score + status.
2. Build with `PYTHON=<repo venv python> bash tools/build_submission.sh <variant>` so verify runs.
3. When a score lands, fill it in and add a one-line read vs the previous best.
