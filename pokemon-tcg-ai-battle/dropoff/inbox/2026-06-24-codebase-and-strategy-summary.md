# Codebase map + strategy summary (2026-06-24)

Read-first map of where the logic lives, what each agent is, what the data says, and what to do next.

## 1. Where the code lives (the files that matter)

### The agent (what plays)
- `agent/main.py` — all agent entry points + the heuristic + the deck.
  - `DECK` (~line 61): the DENPA92 Alakazam draw-engine list (Abra 741 -> Kadabra 742 -> Alakazam 743;
    Dunsparce 65 -> Dudunsparce 66; energy/trainers).
  - `_choose` / `agent`: the plain board-aware heuristic (PH-BLIND; ties first-option locally). Weak.
  - `_attack_value`, `_forced_move` (~line 154): the KO floor. `_attack_value` is PH-aware (scores Alakazam
    Powerful Hand = 20*hand as damage counters); `_forced_move` takes a lethal/KO and goes first.
  - `agent_search` (~line 234): legacy 1-ply search via the OLD `search.py`. THIS is what the 640-688 ladder
    `sub_search` runs.
  - `agent_phaware` / `agent_planner` (added 2026-06-24): PH-aware-KO heuristic; and PH KO + `search_v3`
    (deck-out leaf + develop rollout). Entry points for the tonight tarballs.
- `agent/search.py` — OLD 1-ply forward-model search (cg.api determinization rollouts, hand leaf). Used by the
  shipped `agent_search`.
- `agent/search_v3.py` — NEW search used for this session's findings. Key spots:
  - `best_option` / `_search` — the 1-ply determinization-averaged search; `leaf_mode` and `rollout_mode` params.
  - `_simulate` — branches the first decision, finishes the turn with `_rollout_pick`, scores the leaf.
  - `_rollout_pick` (+ `DEV_ROLLOUT_OPT`) — the turn continuation; `mode="develop"` = develop-first/attack-last.
  - `ATTACK_OPT=13`, `DEV_OPT=(7,8,9,10)`, `USE_DYNAMIC_ATTACKS` (must be True for PH-aware arithmetic).
- `agent/eval.py` — the leaf evaluation (the "value" search maximizes).
  - `W_PRIZE=1000, W_HP=1, W_BODY=30, W_ENERGY=8` (~line 23): the 4 ON terms. `evaluate` (~line 66).
  - `evaluate_deck_v3(cur, me, *, ph_weight, backup_weight, deckout_weight)` — the deck-out hinge + PH-potential
    terms (gated OFF by default). `DECKOUT_TEST=100`, `PH_TEST=1.0` are the A/B magnitudes.
  - `evaluate_ca` (card advantage), `evaluate_learned` / `evaluate_blend` (the learned-value leaves -> the 422
    combine).
- `agent/deck_policy_v3.py` — PH-aware attack math: `best_ko_attack` (the KO floor used by phaware/planner),
  `attack_profile` (Powerful Hand value). Loads card_stats/card_features/attack_stats/card_effects.
- `agent/features.py` — `encode_state(obs)` -> 47 `FEATURE_KEYS`. Used by the (failed) learned value and by the
  feature-importance scan.

### The test + build tooling
- `tools/run_heuristic_ab_v1.py` — the VARIANTS registry and `_make_agent`: defines every agent we A/B
  (first, choose, eff, search, phaware, phaware_search, phaware_search_ca/deckout/ph/v3/dev, planner). This is
  the single source of truth for "what is each test agent."
- `tools/par_ab_v1.py` — PARALLEL, streaming, durable A/B (use this). Many matchups across all cores; writes
  `data/ab_runs/<matchup>.json` live per chunk; prints a tally as chunks finish. Self-mirror (x_vs_x) = control.
- `tools/quick_ab_v1.py` — single-process durable A/B (slower; superseded by par_ab for speed).
- `tools/eval_feature_importance_v1.py` — self-play -> which features separate winning states (found deck-out).
- `tools/deck_depletion_diag_v1.py` — deck-out gate-fire diagnostic.
- `tools/build_submission.sh` (legacy search.py bundle) and `tools/build_submission_v3.sh` (search_v3 bundle:
  planner/phaware). `tools/verify_submission.py` execs main.py exactly like Kaggle and plays games.
- `docs/SUBMISSIONS.md` — the canonical submission log (scores + what each was).
- `submissions/sub_*/` + `*.tar.gz` — built bundles (gitignored).

### The other repo (READ-ONLY): pokemon-ai-agent
- `src/pokemon_ai_agent/policy/heuristics/*` + `registry.py` — the 770.9 ladder heuristic (lethal_ko,
  alakazam_energy, active_pivot, gust_threat, dudunsparce/alakazam engines, deck-out denial via deck-safety).
- `src/pokemon_ai_agent/transition_ranker/*` + `features/state_action.py` — the sparse cross-feature replay
  ranker (imitation; not a win signal; not on the ladder).
- `auditor_sandbox/experiments/*.json` — its n=500 heuristic ablations (lethal_ko +0.808, alakazam_energy
  +0.646; deck-out: full 5/500 vs first 91/500).

## 2. What each submitted agent is (by ladder score)

- 770.9  basic heuristic, Alakazam, NO search (pokemon-ai-agent). The best. PH KO + alakazam-line energy +
  deck-out denial + go first.
- 687.8 / 645.2 / 640.7  `sub_search` = `agent_search` (OLD search.py), DENPA92 deck, hand leaf, N_DETERM 4->8.
- 617.2  earliest `submission` = day-one heuristic on the OLD default deck.
- 592.0  `sub_pilot_hoard` = agent + the Powerful-Hand hoard eval term. Below search -> the term hurt.
- 422.0  `sub_combine` = search + learned gradient-boosted value at the leaves. Dead.

## 3. What the data says (do not relitigate)

- LADDER ranking: heuristic (770.9) >> search (640-688) >> combine (422). Three independent confirmations that
  search/learned additions LOSE to the pure heuristic.
- Heuristics DO help search: the PH-aware KO floor makes `phaware_search` beat heuristic-alone locally (0.745).
  That part of the premise holds.
- What does NOT help: the extra leaf/rollout findings on top of search. Deck-out leaf (+0.62 alone, borderline),
  develop rollout (+0.56 alone, borderline), and COMBINED they are NEUTRAL vs plain phaware_search (0.483, n=60;
  the early 0.40 at n=30 was noise). Net: the 1-ply leaf/rollout tweaks are near-neutral; the bottleneck is
  turn-sequencing depth, not the leaf.
- Local self-play does NOT predict the ladder (it said search>heuristic; the ladder says the opposite). Treat
  local A/B as a rough filter only; the ladder is the arbiter.

## 4. The actual path to a stronger agent

The strongest agent is the 770.9 heuristic. Getting stronger means improving THAT (in pokemon-ai-agent), not
the kaggle-fun search line. Candidate improvements to the heuristic, by evidence:
1. KO-last sequencing (develop non-endangering actions first, attack last) IF it is not already in lethal_ko.
   This is the only validated finding that applies to a pure heuristic.
2. Confirm deck-out denial buffer is tuned (it already prevents 86/500 deck-out losses vs first).
3. Search, IF used at all, only as a scoped helper the heuristic calls for a specific hard decision, never as
   the policy driver (search-drives-policy has lost every time on the ladder).

What to STOP doing: shipping search/combine variants; trusting local self-play to rank agents for the ladder;
grinding more 1-ply leaf terms (they are marginal).

## 5. How to run a fast test now

`python tools/par_ab_v1.py --matchups A_vs_B,C_vs_D --games 120 --chunk 8`
Streams a per-chunk tally and writes `data/ab_runs/<matchup>.json` live. Always include a self-mirror control
(e.g. `phaware_vs_phaware`) and read the CI vs 50%, not the point estimate.
