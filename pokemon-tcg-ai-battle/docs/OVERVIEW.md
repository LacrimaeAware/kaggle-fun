# OVERVIEW — Pokemon TCG AI Battle (single-document summary)

Updated 2026-06-19. This is the consolidated entry point: findings, methodologies, experiments, runbooks,
the reports dropped into `dropoff/`, and where everything lives. It summarizes and points to the canonical
docs; it does not replace the live controllers. For running state read `docs/CURRENT.md`; for fresh-session
onboarding `docs/HANDOFF.md`; for competition facts `docs/COMPETITION.md`; for the hypothesis ledger
`registry/` (`BELIEFS.md`/`GRAVEYARD.md` are GENERATED — never hand-edit); the active plan is
`docs/workstreams/BRANCH_PLAN.md`; submissions `docs/SUBMISSIONS.md`.

Numbers are stated with their named baseline, n, and provenance (local-sim / kaggle-LB / manual). A win rate
vs one fixed pool is a single aggregate direction, not a per-matchup mechanism. No em dashes per convention.

---

## 0. Board state in one breath

`agent_search` — a 1-ply forward search over the cabt forward model with a hand-coded leaf evaluator
(N_DETERM=8, DENPA92 deck, aggro opponent-reply rollout, 0.6s/decision) — is the ONLY proven live agent and
the current submission. **Everything learned on top has tied or lost in play.** The strong part is the search;
the open problem is making learning add value. The reframe (Section 4): a model that purely clones a teacher T
cannot beat T, so the teacher must be made stronger than the live agent (slow offline search), and the learned
model should guide search / correct it at the margin, not replace it.

---

## 1. The competition (facts in `docs/COMPETITION.md`, with provenance tiers)

- **Two linked tracks, one engine.** Simulation (slug `pokemon-tcg-ai-battle`, bot ladder, Knowledge/no cash,
  Jun 16–Aug 17 2026) feeds Strategy (slug `…-challenge-strategy`, written report, cash, Jun 16–Sep 14 2026).
  Strategy is judged on agent stability + deck-design concept + Simulation rating (weights unconfirmed). Cash
  is on Strategy; Round-1 top 8 = $30k each = $240k (claim). Exact total prize pool UNCONFIRMED (sources span
  $50k to ~$320k+). Treat unconfirmed items as unconfirmed; do not promote by repetition.
- **Forward-model SEARCH API is real and on the table** (confirmed 2026-06-17 from the Data tab
  `sample_submission/cg/api.py`): `search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize,
  opponent_hand, opponent_active)` → `search_step(searchId, select)` → `search_end()/search_release()`. It is a
  determinized forward search built for ISMCTS; `search_begin` takes your PREDICTED opponent zones. The earlier
  "no forward model" finding (registry H1) was a **false negative** measured on the stripped installed
  `kaggle_environments` cabt module, not the real `cg` module — do not re-inherit it.
- **Reward** (verified `cabt.json`): Lost −1 / Won +1 / Draw 0; 1v1; episodeSteps 10000; remainingOverageTime
  600 (read as ~10 min/player/match, enforcement inferred). Ladder mechanics (competitor-claimed): 5 subs/day,
  only latest 2 scored, μ starts 600, W/D/L only (margin irrelevant), mirror validation must not error.
- **Card pool / rules.** Official `EN_Card_Data.csv` = 1267 cards. OptionType legend (from api.py): 0 NUMBER,
  1 YES, 2 NO, 3 CARD, 4 TOOL_CARD, 5 ENERGY_CARD, 6 ENERGY, 7 PLAY, 8 ATTACH, 9 EVOLVE, 10 ABILITY,
  11 DISCARD, 12 RETREAT, 13 ATTACK, 14 END, 15 SKILL, 16 SPECIAL_CONDITION. Prizes: ex KO = 2, Mega ex = 3,
  Tera takes no bench damage, ACE SPEC max 1/deck. **Engine is authoritative** ("Simulator behavior will be
  treated as the correct behavior") — code to the engine, not paper rules. **Deck-building is OPEN** (Kaggle
  staff confirmed): build from the full pool, not the four starters.
- **The agent must never raise.** Kaggle validates a submission by playing a copy of itself; any uncaught
  exception or timeout forfeits. A crash-safe legality envelope is mandatory before any strategy.
- **Open competition questions** (need live pages): compute limits (CPU/GPU at eval, internet/external
  weights), time granularity (per-move vs per-match pool — if per-match, early expensive search starves late
  turns), exact prize total, Strategy judging weights.

---

## 2. The agent: strongest + the full ledger

**Strongest / submission:** `agent_search` = 1-ply forward search, hand-eval leaves, N_DETERM=8, DENPA92 deck
(Dudunsparce/Alakazam, 8 basics/3 energy, signature `27ed2ff887c1488c`), aggro rollout, DEPTH_CAP=80,
`_forced_move` floor (take lethal/KO ≥8000 else go-first) + heuristic fallback. Built `agent/search.py`
(`option_evals`, `option_deltas`, `best_option`, `_search`).

**Win-rate ledger** (local cabt engine, same deck both sides, seats swapped, Wilson CIs; numbers carry their
deck/opponent context — "vs heuristic" is NOT a fixed strength measure because the heuristic pilots evolution
decks badly):

| matchup | result | n | read |
|---|---|---|---|
| agent_search vs first_agent | **0.585** [0.551,0.619] | 800 | strongest vs day-one baseline |
| agent_search vs heuristic (simple deck) | 0.543 [0.487,0.599] | 300 | beats own no-lookahead policy |
| agent_search vs heuristic (DENPA92) | 0.833 (50-10) / 0.86 | 60 / 50 | deck-dependent, not comparable to above |
| heuristic vs random | 0.835 / 0.875 | 200 | sanity floor only |
| heuristic vs first_agent | 0.513 | 300 | the bare heuristic only TIES take-first |
| N_DETERM 8 vs 4 (head-to-head) | **~0.675** | — | the ONE proven in-search improvement |
| search2 (2-ply opp-min) vs search | 0.35 | 20 | deeper search LOST (over-pessimistic min-leaf) |
| search_v (learned value leaf) vs search | 0.25 | 20 | learned value leaf LOSES |
| combine (search+blend leaf) vs search | 0.367 (22-38) | 60 | LOSES; also LB 422 vs 640 |
| combine vs heuristic | 0.831 [0.766,0.881] | 160 | real but that is mostly the search, not the blend |
| agent_rank (learned net everywhere) vs heuristic | 0.117 | 60 | standalone net FAILS in play |
| agent_rank_hybrid (net on strategic only) vs heuristic | 0.150 | 60 | also fails |
| eff (effect-replacement scorer) vs heuristic | 0.20–0.48/deck | 40/deck | failed (valued setup over attacks) |

**PROVEN:** N_DETERM 4→8 (~0.675). **DEAD as ways to beat agent_search:** learned value at the leaf
(search_v 0.25, combine 0.367, LB 422 vs 640); listwise objective on static features (Gate 1: 0.495/0.327
below the option-0 prior); listwise on forward-model action-deltas (Gate 2: 0.354 mixed below option-0 0.435);
standalone learned net (agent_rank 0.117, offline-faithful but covariate shift; VERIFIED not an inference bug —
feature max-abs-diff 0.0, net fired 60/60, 0 fallbacks); copying the #1 ladder deck (Praxel Mega Lucario =
0.300 under our search). **PARKED, not refuted:** opponent-belief determinization (the 1-ply hand-leaf
evaluates the start of my next turn, near-blind to opp hidden cards, so a tie is the EXPECTED null; the A/B was
also unpaired+unseeded; reopen gate = paired+seeded A/B AND an opponent-sensitive leaf, opp_k>0 / ≥2-ply);
continuation policy setup-vs-aggro (CLOSED, no effect, kept aggro).

**Deck decision:** adopted DENPA92's 8-basic deck (fixed a mulligan outlier in the old 6-basic deck; suits our
basic-attacker policy). **Deck value is policy-coupled** — copying the best ladder deck FAILS (Iono-Bellibolt
0.83 ladder → 0.212 under our heuristic; Praxel Mega Lucario → 0.300 under our search) because our generic
1-ply pilot cannot run an evolution engine. **Freeze DENPA92** as the validated baseline; a deck replaces it
only via a separate predeclared promotion gate; never change deck + methodology in one experiment.

**Public LB scores** (`docs/SUBMISSIONS.md`): old-deck 617.2 → sub_search DENPA92 N=4 640.7 → sub_search 697.7;
sub_combine 422.1 (dead). N=8 submission was pending at last log. Local self-play does NOT predict ladder
placement; only a real submission is ground truth. Claude cannot submit; the human uploads.

---

## 3. The central diagnosis (why learning has not paid off yet)

Named by every reviewer as **"objective slippage":** the project repeatedly implements the easy surrogate
(assign an absolute scalar value to the *resulting state*) instead of the stated hard problem (rank the *legal
sibling moves* of one decision), gets respectable prediction metrics, then finds the surrogate does not
improve decisions.

- **Global value is good; local sibling ranking is the gap.** The learned value predicts the eventual winner at
  AUC ~0.74 (Pearson to the search target rose 0.825 → 0.904 across expert-iteration passes), yet play stayed
  at parity. **Better target fit did NOT lift play** — the bottleneck is local discrimination among nearby
  candidate leaves, not global value or target fit. A piecewise-constant tree ranks neighbors poorly, which is
  exactly what 1-ply search needs.
- **The conversion gap.** The three policies (heuristic / hand-search / learned) disagree on 45–65% of real
  choices, yet outcomes stay ~0.50 — so most single decisions are low-impact, and the leverage is the rare
  high-criticality decisions we have not isolated. Of 400 decisions, 82% are real choices (≥2 options), 18%
  forced — so **the simple deck is NOT the bottleneck;** a complex deck multiplies decisions without fixing the
  value-to-action conversion. Complex-deck switch is DEPRIORITIZED.
- **The option-0 confound.** 58.7% of eventual-winner moves are option index 0 (the engine's option ordering).
  "Choose option-0" scores ~0.587 imitation top-1, beating the hand heuristic (0.553) and every learned ranker
  tried. Random (0.197) must NOT be the imitation baseline; option-0 is the bar. And imitation top-1 DIVERGES
  from win-rate (agent_search ~0.42 imitation but 0.585 win-rate), so **win-rate is the judge; imitation is at
  most a representation probe.**
- **A clone cannot beat its teacher.** Distillation onto hand-eval / 1-ply-search labels is circular (it
  inherits the search's blindness) and buys SPEED, not strength. To exceed the hand-search teacher the target
  must carry information the hand leaf lacks: deeper/longer-horizon search, multiple rollouts, shared hidden
  worlds across candidates, eventual counterfactual outcomes, or stronger-opponent trajectories.

---

## 4. The reframe and the two-branch program

**Reframe:** the mistake was fixing the teacher T = the live N=8 search and concluding learning is capped. T
can instead be a SLOW OFFLINE search (N=32/64, deeper on hard decisions, robust aggregation) far stronger than
the live agent; expert iteration then lets the learned policy guide a stronger planner than either alone. Final
acceptance is NOT "student beats Teacher V2 alone" but "**student-guided live search beats the strongest
standalone live search at equal wall-clock.**" Terminal OUTCOME is an auxiliary critic/validation target only,
not the primary action target (the primary stays repeated counterfactual action advantage from the teacher).

**Structure** (`docs/workstreams/BRANCH_PLAN.md`, authoritative): from a frozen `SPLIT_BASE_V2`, Branch A
(Stronger-Planner/Teacher) + Branch B (Robust-Learner, DAgger/DART distribution-shift), plus a read-only
adversarial auditor / integration lead. Do not place two autonomous implementers on one branch.

**SPLIT_BASE_V2 preflight (landed, commit `2f29e93`):** frozen baseline config (P0); immutable dated replay
snapshot `replays_20260618` (1289/1299 games, GAME-level chronological split 902/193/194 train/val/test, 123
held-out players, 23 held-out decks) with per-file hashes + manifests (P1); semantic state/action schema with
orderless zones as multisets, verified byte-for-byte across 31,146 decisions / 198,423 options, 0 mismatches,
100% PLAY card-id resolution (P2); 130 golden fixtures across 8 action types, 8/8 acceptance gates pass (P3);
Teacher API V1 returning soft policy / advantages / acceptable-set / variance (P4). A daily top-20 replay
refresh tool appends raw files but experiments consume immutable dated snapshots (`tools/refresh_top20.py`).

**Teacher V2 lineage findings (Branch A, all OFFLINE — agent_search remained the live baseline, nothing
promoted, main untouched):**

- **Teacher V1 instability is engine-rollout-RNG-dominated.** Replay states (n=1094, 35,200 queries): 50.5%
  stable / 21.2% near-tie / 28.2% unstable. Self-play states (n=999): 40.2% stable (instability reproduces and
  is slightly worse on-policy). The wobble is NOT which world was sampled — fixing the determinization draw
  (same-seed vs cross-seed) leaves stability essentially unchanged (extra instability −0.011 replay / +0.026
  self-play, ~0). It is the native engine's coin flips / shuffles inside `cg.dll`: ~93% of strategic decisions
  carry engine rollout RNG, the Python seed controls only the world draw, and **each teacher value is a Monte
  Carlo estimate.** A single hard top-1 teacher label is roughly half noise on the non-forced set. The lever
  that reduces this is averaging more rollouts (higher N) and selective computation, NOT world-sampling.
- **Terminal-outcome auxiliary is real-but-weak.** Hand-vs-outcome argmax disagree ~0.52 (scaled n=50) up to
  0.72 (hardest states), so it carries information the hand leaf lacks, but the outcome argmax is only ~50%
  self-stable across two k=32 runs — too unstable for a hard label. Use `hand_norm_advantage` as the primary
  target (weight by criticality, inverse variance, coverage); `outcome_winrate` as an SE-weighted auxiliary
  only. **Scale n (more decisions), not k (>32 not worth it).**
- **Residual + risk reframe (the narrowed target).** The learned model learns a residual correction +
  catastrophic-risk flags ON TOP of agent_search, not a replacement ranker. `delta_to_search` = stronger(N=32)
  − live(N=8) on paired/shared worlds isolates live MC error; it is MEDIAN-0, HEAVY-TAILED (round-1: p50 0,
  p95 +163, a few lethal ±1e6 tails). Residual head: regress `delta_to_search_norm` with clip + Huber/quantile
  loss, down-weight by `value_se`, integrate as `final = search_score + small_gate * predicted_residual`.
- **Risk round-2 (60 labels, supersedes round-1 risk numbers).** c1 = states where agent_search itself selects
  a high_regret option: a 12-shard mine of 3,224 high-criticality states found 16 c1 candidates, 9 in the top
  ~1,100 by criticality and ZERO below rank ~1,350 — so **c1 is a top-tier-only ~0.5% event.** The c1 label is
  only ~53% reproducible (the seed's selected option moved 0→1→3 across three labels, always high_regret).
  Regret MAGNITUDE is RNG noise; the high_regret FLAG is the stable signal. Risk head must CLASSIFY
  `high_regret_flag`/`unacceptable_flag` (never regress raw regret), prefer recall + abstain/extra-search, and
  class-weight (c2 false-positive states dwarf c1 ~5:1). Counts: c1=9 (5 games), c2=49 (11 games), c3=21;
  high_regret 127/588 options, unacceptable 404/588. **Cross-game generalization of the c1 head cannot be
  honestly demonstrated** (9 c1 reduce to ~3-4 independent situations); use `group_id` (=game) for group-held
  splits, hold out `eval_only` seeds. The demonstrable fix is the c2 false-positive head.
- **Tactical floors do not beat the noise.** Draw 0.333, evolve 0.467, gust+evolve 0.471, mined soft-prior
  `agent_search_prior` 0.433 — all washes, because the search's weakness is NOISE (engine variance, ~half of
  decisions near-ties), not systematic tactical wrongness. Card-mechanics lesson: read the card before
  building a heuristic (Dudunsparce "Run Away Draw" shuffles its own 140-HP body away; Kadabra/Alakazam
  "Psychic Draw" only fires on evolve-from-hand). Tactic Miner V1 mined 46,889 winner sibling-decisions
  (gust|can_ko_now lift 1.52, attack|fixing_available=0 lift 2.87).
- **Search-confounder audit (clean):** fake Water Energy padding (id 3) triggers 0% of decisions; the offline
  8s budget gives 8/8 determinizations and 100% per-option coverage; shared hidden worlds across siblings are
  CONFIRMED (paired comparison valid); no public-zone leakage; engine-rollout noise is the dominant confounder,
  averaged down only by higher N.

---

## 5. Methodologies and evaluation discipline (the durable how-we-work)

**Evaluation discipline (the part that generalizes):**
- Every capability is a registry hypothesis with a named eval surface + sample size + stated noise band BEFORE
  the run. Three surfaces are different exams — local self-play, win-rate vs a fixed bot, public ladder — never
  compare numbers across them.
- ACCEPT a gain only at n≥400 with the Wilson lower bound above the band, ideally a 2nd seed. Never REJECT a
  component on one run; require a repeat + implementation sanity-check + eval-surface check, then mark refuted
  WITH a reopen gate. Rejecting one component does not reject the architecture; no component is dead while a
  cheaper untried variant exists.
- A win rate vs one pool is a 1-D global probe; it does not decompose into per-matchup mechanism unless you
  measured the split. Status vocabulary (no more "done" = script ran): specified → implemented →
  data-generated → trained → offline-evaluated → arena-evaluated → accepted | refuted | inconclusive | parked.

**Methods in play:**
- **1-ply forward search** with K-determinization averaging over composition-sampled hidden pools; aggressive
  opponent-reply rollout; hand-eval leaf. Confound control: fix the determinization count + raise the time cap
  to isolate decision quality from "how many determinizations finished."
- **Action-conditioned sibling ranking (H024-v2):** rank the consequential siblings of ONE decision with a
  within-decision pairwise (Bradley-Terry) or listwise (softmax) objective grouped by decision_id, on a
  per-decision-centered advantage target `A_{g,i}=y_{g,i}−mean_j y_{g,j}`, weighted by candidate-spread
  criticality. Inputs = card-id embedding + decoded card_effects + action descriptor + root features +
  forward-model one-step option_deltas. NOT pointwise state-value regression.
- **Forward-model consequence deltas:** `option_deltas` applies exactly ONE `search_step` (no rollout) and
  returns post-minus-root deltas (prizes, opp_ko, dmg, draw, energy, board_dev, hp, deck, discard, ends_turn);
  recovers the played card by serial-diffing the board (join-free). The only sim-touching feature; cache it.
- **Card effects as AFFORDANCES not values:** an effect (e.g. "search 2 basics to bench") is valuable only
  conditioned on state (bench space, deck targets, setup need, no immediate KO). The learned term must be
  state×effect interactions and a conservative RESIDUAL on top of the baseline, never a replacement scorer
  (the eff replacement scorer failed by valuing setup over attacking). Quantify magnitude (draw 8 ≠ draw 2).
- **Belief-conditioned determinization:** infer opponent deck/hand/prizes from revealed cards + meta priors,
  seed `search_begin` with realistic hidden states; validate FIRST on held-out replay hidden-card likelihood,
  then a PAIRED+SEEDED head-to-head; only worth it once the leaf is opponent-sensitive.
- **Expert iteration:** self-play with current best → search-improved targets → retrain → repeat; evaluate by
  head-to-head, not value AUC.
- **DAgger/DART discipline (Branch B):** train on the learner-induced state distribution; Pokemon
  perturbations are generally NOT label-preserving (must go through the engine and be relabeled), only safe
  invariances (option-order/identical-copy permutations, canonical perspective) keep transformed labels.

**Unfakeable gates** before trusting any learned-component number: (1) **wiring proof** — trace one real
decision: legal option → card id → card_effects lookup → score contribution → chosen action, and show one
decision where the policy refuses setup because a KO exists; (2) **enabled-vs-zeroed ablation** — compare the
SAME policy with the component zeroed at inference on common seeds; a win counts only if it beats the zeroed
ablation, not merely a weaker baseline; (3) **multi-deck** evaluation; (4) **frozen seat-swapped win-rate A/B**
with predeclared CIs as the final gate. Keep hard forced rules (lethal, crash-safe fallback) OUTSIDE the
learner so failure is informative.

**Known bugs found + fixed (do not reintroduce):** `evaluate_blend` referenced `VM` without importing it →
NameError silently swallowed → every blend/combine point fell back to hand-only (likely caused sub_combine's
low LB); the imitation diagnostic's card join was dead code (`isinstance(hand[idx], int)` but replay hand
entries are dicts `{'id':…}`) which zeroed the +card feature and INVALIDATED the old "card features add
nothing" claim; `os.path.abspath(__file__)` at module scope NameErrors under Kaggle exec (fixed to the
hardcoded `/kaggle_simulations/agent` path). **Latent issues to fix before deeper search / new decks:**
`_hidden_pool` pads missing cards with id 3 (Water Energy) = fake hidden cards; public Stadium not stripped
from sampled hidden zones; optional `minCount==0` prompts always take an option instead of declining; Team
Rocket Energy (EnergyType 11 = Psychic/Darkness) is wrongly modeled as a universal wildcard (Rainbow=10 is the
real wildcard); search does not read `remainingOverageTime` (do not promote search to default until time
semantics are settled or a budget governor exists).

---

## 6. The competitor field (`research/notebooks/SUMMARY.md`)

The field is heuristic-dominated: of 24 notebooks, only 18% do any search/MCTS and EXACTLY TWO actually call
the `cg.api` forward model (the official RL+MCTS sample and `pokemon-ai-battle-agent-mega-lucario`); three more
ship disabled/broken search. "search"/"turn-search"/"lb-860" in filenames is marketing, not a substantiated
score — no notebook carries a real LB number, self-play-vs-random is near-meaningless.

What to steal: the crash-safe legality envelope (`validate_selection` + `legal_fallback_selection` =
`list(range(min(minCount,n)))` + `normalize_selection` + try/except); the 1-ply root re-rank template (the only
working bolt-search-onto-a-heuristic pattern); the eval primitives (`prize_count` megaEx 3 / ex 2 / basic 1;
weakness ×2 / resistance −30; lethal-KO scored 50000; `evaluate_state` dominated by `(op.prize−me.prize)*10000`)
— and throw away per-card magic-number trees (they rot with the meta); card-counting hidden-info
reconstruction (subtract everything visible from the known 60-card multiset). **The single highest-value
differentiator every notebook skips: a real opponent model** — seed opponent zones from a meta prior instead
of placeholders / self-mirror.

Third-party deck meta (ISAKA, ~15k self-play of the 4 sample agents, measures deck+policy together; reproduce
before trusting): Mega Lucario 60.4%, Dragapult 55.6%, Iono 43.8%, Mega Abomasnow 40.2%; first-player 51.5%;
RPS-like (Lucario/Dragapult coin-flip and both crush Iono; Iono crushes Abomasnow). "Lucario best" is the
sample policy piloting the most over-contested archetype.

---

## 7. Runbooks (step-by-step)

- **Env.** venv `C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/.venv/Scripts/python.exe`, set
  `PYTHONIOENCODING=utf-8`. OpenSpiel "Unknown game" stderr noise is harmless. The cabt forward model runs
  locally; per-decision live budget ~0.6s.
- **Arena A/B.** `tools/run_ab.py --games N a:b …` (agents in `agent/cabt_arena.py` AGENTS: random, first,
  heuristic, search, search2, search_v, combine, eff, rank, rankh). Search-knob sprints:
  `tools/search_sprint.py {determ,belief,continuation,cont-clean,deck}`. Wilson CIs; ~9s/game for search;
  always seat-swap (a_seat = g%2). For a belief A/B, PAIR arms on the same seed
  (`make('cabt', configuration={'seed':g})` + `random.seed(g)`), report McNemar / paired-bootstrap on the
  difference. Early-stop a clearly-losing arm.
- **Submit.** `PYTHON=<venv> bash tools/build_submission.sh search` (the `PYTHON=` is required to actually run
  `verify_submission.py`) → `submissions/sub_search.tar.gz` (gitignored). On every upload add a
  `docs/SUBMISSIONS.md` row (time, file, `git rev-parse --short HEAD`, agent + deck + N_DETERM + leaf + cont);
  fill the score when it lands with a one-line read vs previous best. Claude cannot submit; human uploads.
- **Fetch replays (no auth).** Full episode JSON is on the public CDN
  `https://www.kaggleusercontent.com/episodes/<EpisodeId>.json`; `tools/fetch_episodes.py --top-teams N --max
  M` (auth via `kaggle.json`, NOT `.env` which has a BOM that breaks it). Rebuild the DB:
  `tools/build_replay_db.py` → `data/replay_db/{games,decisions,decks}.jsonl` (gitignored). View any episode:
  HEROZ `https://ptcgvis.heroz.jp/Visualizer/Replay/<EpisodeId>/0`; the Kaggle share link is bugged — replace
  `submissions#` with `leaderboard`.
- **Registry (anti-rehash).** Before testing any idea: `python registry/registry.py search '<topic>'`; if a
  match is REFUTED, read its evidence + reopen gate and do not re-test unless the gate changed. Record:
  `registry.py add … / status H### … --evidence [--gate] / experiment … / result … --verdict
  supports|refutes|inconclusive --provenance` then `registry.py render` (regenerates `BELIEFS.md` /
  `GRAVEYARD.md` — never hand-edit those). Daily notes go in `journal/` (not authoritative).
- **SPLIT_BASE_V2 / Teacher V2 regen.** `tools/freeze_baseline_v2.py`; `tools/snapshot_replays_v2.py`;
  `tests/test_split_base_v2.py` (8 golden gates); `tools/audit_teacher_stability.py`; `tools/query_teacher_v2.py`;
  residual/risk: `tools/label_residual_risk.py` (round-1), then `tools/mine_c1.py` + `tools/label_risk_round2.py`
  + `tools/finalize_round2.py` (round-2); label B's exact requests via `tools/label_requested_states.py`.
- **Viewer.** `tools/scrape_card_images.py` (once, needs PDF + pymupdf) → `build_stats.py` → `build_decks.py`
  → `build_viewer.py` → `viewer.html` (open from inside the folder so image paths resolve).
- **Before trusting any arena result:** verify the thing under test actually fired (agents silently fall back
  to the heuristic on any exception). Before trusting a learned target: assert labels exist (a no-label target
  silently trains nothing and prints a fake 0.000).

---

## 8. Data, artifacts, and code map (gitignored data regenerates)

- **Code:** `agent/search.py` (forward search + teacher primitives), `agent/cabt_arena.py` (AGENTS),
  `agent/main.py` (DECK + heuristic + agent_search), `agent/eval.py`, `agent/features.py` (encode_state, 47
  features), `agent/ranker.py` + `agent/ranker_model.json` (deployed numpy net), `agent/card_effects.json` /
  `card_features.json` / `attack_stats.json` (1556 attackId→damage). Teacher V2 (planner branch):
  `agent/state_action_schema_v2.py`, `agent/teacher_api_v1.py`, `agent/teacher_api_v2.py`.
- **Data:** ~1299 replays in `data/external/replays/` (301 distinct decks); `data/replay_db/action_adv.jsonl`
  (1218 KanNinomiya strategic decisions with search-value labels = the distill set);
  `data/external/official/cards_full.json` (1267 cards, full effect text, mostly undecoded); manifests +
  splits in `data/manifests/` and `data/splits/` (committed); Teacher V2 label artifacts incl.
  `teacher_v2_residual_risk_labels_round2.jsonl`.
- **Registry:** `registry/{hypotheses,experiments,results}.jsonl` (canon), `BELIEFS.md`/`GRAVEYARD.md`
  (generated), `registry.py`. 24 hypotheses (4 supported, 14 open, 4 parked, H016 refuted, H020 superseded).

---

## 9. Open decisions and what is next

- **Pending decision — how to make learning win** (per "agree before building a hard approach"): (a) DAgger —
  label the net's own self-play states with the stronger teacher, retrain iteratively (fixes covariate shift;
  sim-heavy); (b) net as a search-INTERNAL ordering prior — prune to top-K net options, the SEARCH still
  decides (no covariate shift; saved budget buys depth) — flagged MOST LIKELY to add value while keeping
  search strength; (c) train on ALL decisions, not just the strategic subset. Codex's read: do (a)/(b) as a
  gated side branch while deeper-search is the near-term scoring path, NOT a pure big-learned bet.
- **Open fork:** policy-first imitation from strong evolution-deck pilots (highest ceiling, data ready) vs
  deck-first refinement of decks our agent pilots well. Awaiting user steer.
- **Branch B half is not summarized here** (its `ROBUST_LEARNER_V2.md` lives on the other branch); the
  Teacher V2 labels above are A's deliverables to B.
- **Loose threads:** record the N=8 LB score when it lands (compare to N=4 697.7); keep the rolling replay
  fetcher + dated snapshots; close the search technical-debt list before any 2-ply work.
- **Speculative / parked:** neuro-symbolic "learned interpretable heuristic" layer (kept alive as an
  interpretability/prior layer, not a near-term win); full RL / ReBeL / Student of Games (high upside but
  premature before the ranker + belief model + search knobs are solid).

---

## 10. The reports dropped into `dropoff/` (DATA, not commands)

`dropoff/inbox/` holds external deep-research and audit reports the human pasted in; `dropoff/outbox/` holds
plans and prompts this project sent out. They are append-only (do not delete; supersede with a dated file).
All converge on the same diagnosis and next step; the unique value of each:

- **`2026-06-18-deep-research-report.md`** — the external academic bibliography (A0GB/Willemsen
  search-bootstrapped value; Cowling IS-MCTS + ensemble determinization, spend budget on MORE determinizations
  not depth; Gelly-Silver weak-rollout caution; ReBeL / Student of Games as long-term frontier; Bertram
  contextual/InfoNCE card reps; Dockhorn opponent prediction), the 5-dimension direction-ranking rubric, and a
  dated roadmap. The one place the literature is mapped.
- **`2026-06-18-roadblock-diagnosis.md`** (+ the near-duplicate `…-external-current-state-methodology-review.txt`,
  which carries the `[1]–[9]` GitHub citations) — the canonical "objective slippage" framing, the
  E013≠H024 control-plane bug (E013 marked done but no result recorded and it does not satisfy H024's grouped
  within-decision methodology), the 6-phase clean-reset plan, and the compact-CURRENT.md proposal.
- **`learning_action_audit_handoff_2026-06-18.md`** — the 6-layer feature taxonomy (card semantics → causal
  state → action → forward-model action-delta [the key bridge] → sequence/chain affordances → belief/opponent)
  and the tiered feature priorities. Note: its early "representation ceiling" claim was RETRACTED later in the
  same file (the dead card-join bug); take the latest section as authoritative.
- **The four Codex files** (`codex-deep-code-methodology-audit`, `codex-evaluation-summary`,
  `codex-response-to-next-step-choice`, `card-effects-action-prior-handoff`) — one decision record: after the
  evaluate_blend fix, combine beats the no-search heuristic 0.831 but loses to agent_search 0.367; `search_v`
  never consumed `card_effects.json` so the card-effect idea has barely been tested; build the integrated
  effect/embedding/action-ranking spine (action-ranking head live FIRST) but gate every result with wiring
  proofs + enabled-vs-zeroed ablations; the three hard gates (objective-only → immediate action-delta → live
  win-rate); "both in parallel, constrained, deeper-search-first." Constants on record: BLEND_LAMBDA=0.4,
  BLEND_SCALE=2000; distill blockers fixed in commit `f8fce28`.
- **`methodology-compliance-review.md`** — the section-by-section compliance scorecard and the 7 binding rules
  (do not call it card-effect learning unless `card_effects.json` is consumed live; not action ranking without
  grouped within-decision metrics; not embedding learning without trainable card vectors in the scorer; not
  "validated" without the full-vs-ablated comparison; AUC/Pearson are diagnostics only; hand-eval-only labels
  cannot beat hand search; "done" ≠ "script ran").
- **`distill-belief-adversarial-review.md`** — the paired-seeded belief-A/B critique (H008/H022 PARK), the
  distill-buys-speed-not-strength restatement, and the KanNinomiya build context (the deck's most prolific
  winner, 144 win-games / 13,852 decisions; DENPA92 is the archetype name, matched 0 rows as a player).
- **`outbox/2026-06-18-CONSENSUS-and-way-forward.md`** — the outbox single source of truth: the canonical
  results table, the integrated-model plan, the learning-signal spec (outcome-weighted + horizon-discounted
  soft-positives + lethal-KO floor + listwise softmax), the unfakeable benchmark gates. `master-plan.md` and
  `forward-plan.md` are self-marked superseded by it (kept only for the negative-result log).
  `research-questions.md` and `feature-optimization-prompt.md` are outbound asks to external models.

---

## 11. Document map and cleanup status

**Canonical / live (keep, do not fold):** `CURRENT.md` (running state), `HANDOFF.md` (onboarding),
`COMPETITION.md` (facts), `ACTION_RANKER_PLAN.md` (the H024-v2 plan + 7 binding rules), `SUBMISSIONS.md` (LB
log the user expects), `conventions.md` (rules), `workstreams/BRANCH_PLAN.md` + `SPLIT_BASE_V2.md` (the
branch charter + preflight), `registry/` (belief canon), `research/notebooks/SUMMARY.md` (competitor digest),
`AGENTS.md` / `writeup.md` / `README.md`. This `OVERVIEW.md` is the single-document summary over all of them.

**Stale / redundant — flagged for slimming (each now carries a banner pointing here):** `STRATEGY.md`,
`PLAN.md`, `LANDSCAPE.md` (all already self-marked superseded; unique nuggets to keep on slim = the 15 seed
hypotheses, the viewer runbook, the problem characterization + quant-transfer framing), `RESEARCH.md` (a
"living/temporary" doc that grew into a second source of truth; keep the replay-access runbook),
`LEARNING_PLAN.md` (keep the 5-layer architecture skeleton), `MODEL_COMMUNICATION.md` (keep one deduped
code-issues list). On the planner branch, the Teacher V2 execution-log docs can collapse to one
Teacher-V1-stability section + one Teacher-V2-labeling section (round-1 risk numbers are superseded by
round-2). `dropoff/` stays append-only; `BELIEFS.md`/`GRAVEYARD.md` stay generated.

The actual body-slimming of the six stale docs is deferred for sign-off (it would discard their unique
nuggets if done hastily); this OVERVIEW already captures their load-bearing content.
