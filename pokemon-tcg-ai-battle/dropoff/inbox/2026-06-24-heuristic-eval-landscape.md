# Pokemon TCG AI Battle: heuristic and eval landscape (2026-06-24)

Purpose: one read-first map of every heuristic and eval term we run, whether it is on, and the measured
delta where we have one. Written to be handed to other models for ideas and refinement. All win rates are
local same-deck self-play unless stated; the ladder is the only real truth and it disagrees with local
self-play (see Orientation).

## 0. Orientation

Two repos:
- `kaggle-fun/pokemon-tcg-ai-battle` (this one, editable): a SEARCH agent. Deployed = `phaware_search`
  (PH-aware KO floor + 1-ply forward-model search for development).
- `pokemon-ai-agent` (the rebuild, READ-ONLY to this work): a HEURISTIC-REGISTRY agent (`Full` profile),
  plus a separate sparse-feature transition-ranker research line.

Deck (both): DENPA92 Alakazam draw engine. Abra(741) -> Kadabra(742) -> Alakazam(743), all Psychic.
Dunsparce(65/305) -> Dudunsparce(66), Colorless. Alakazam Powerful Hand = 20 damage per card in hand, placed
as DAMAGE COUNTERS, so weakness/resistance do not apply and the card's static damage reads as 0 (naive
heuristics miss PH knockouts). It is a heavy draw engine, so DECK-OUT is a primary loss condition, not a tail.

Ladder truth (user-reported; NOT recorded in either repo's files, so treat as out-of-band fact): pure
heuristic submission ~736 > search-driven submissions ~645-687 > combine ~422. Local self-play does NOT
predict the ladder. Every time search DROVE policy on the ladder it lost or went neutral. Design rule that
follows: search is a bounded calculator the heuristic calls, scoped to the heuristic's candidate set, never an
enumerate-all policy.

Method rules (so deltas are trustworthy):
- Whole-game A/B noise floor is about +/-0.05 at n=300. Always name the baseline. Always run a self-mirror
  control that must land near 0.500. Trust a gap only if it clears the floor while the control is clean.
  (We once chased a phantom "side-A tilt"; an n=600 self-mirror = 0.502 proved the harness fair.)
- For sub-decisions, applicable-state paired-world tests are far more noise-efficient than whole-game A/B.
- Imitation is not optimality: the replay ranker predicts the move a player MADE, not the move that WINS. Its
  top-1 vs always-option-0 is an imitation yardstick skewed by option-0-heavy data, not a quality/win measure;
  do not read "loses to option-0 on top-1" as a failure.
- Distinguish neutral-because-proxied (another term already captures it) from neutral-because-inert (the gate
  never fires). Instrument gate-fire rates before trusting a null.

---

## 1. kaggle-fun heuristics, agent variants, eval terms (with deltas)

Deployed agent = `phaware_search`. Layer key: floor = pre-search heuristic; search-leaf = development objective
inside the 1-ply search; eval-term = a weight inside the leaf eval.

| name | layer | on deployed | what it does | measured delta (vs baseline, n) |
|---|---|---|---|---|
| `best_ko_attack` (PH-aware KO floor) | floor | ON | takes a listed KO attack, correctly scoring Powerful Hand (20*hand, ignores weak/resist) | as `phaware`: 0.793 vs `choose`, 0.760 vs `first` (n=300). Single biggest lever. |
| energy-to-active | floor | ON | attach energy to the active attacker when short | no data (bundled in `choose`) |
| go-first | floor | ON | choose YES to go first | no data (bundled) |
| `choose`/`agent` (plain heuristic) | floor | fallback only | legacy lethal+energy+go-first, else default order; PH-blind | 0.483 vs `first` (tie; cannot pilot the deck) |
| `agent_eff` (effect-aware dev) | floor | OFF | values setup plays via decoded effects | 0.390 vs `first`, 0.447 vs `choose` (worse) |
| **`phaware_search`** | composite | ON | PH KO floor + search develops (`leaf_mode=hand`) | 0.745 vs `phaware`, 0.900 vs `first` (n=220) |
| `leaf_mode=ca` (`phaware_search_ca`) | search-leaf | OFF | development objective = board + 25 per hand card | 0.525 vs `phaware_search` (p=0.48) NEUTRAL |
| `leaf_mode=learned` (`agent_search_v`) | search-leaf | OFF | GBM learned value (AUC 0.735) at leaves | 0.427 vs heuristic LOSES (old deck) |
| `leaf_mode=blend` | search-leaf | OFF | (1-lam)*sigmoid(hand) + lam*P(win), lam=0.4 | no data (same losing family) |
| opp_k=2 (2-ply) | search cfg | OFF | branch on opponent top-k reply, min leaf | 0.575/0.483 (n=120) NEUTRAL |
| opponent meta prior (`sd_meta`) | search cfg | OFF | sample opp deck from replay meta vs assume our deck | 0.517 (p=0.72) NEUTRAL |
| `W_PRIZE=1000` | eval-term | ON | prize differential, dominates | core; never ablated alone |
| `W_HP=1` | eval-term | ON | board-HP race tie-break | core; never ablated alone |
| `W_BODY=30` | eval-term | ON | body count; a KO costs a body | core; never ablated alone (note: rewards emptying hand into bodies) |
| `W_ENERGY=8` | eval-term | ON | energy on my active attacker | core; never ablated alone |
| `W_CA_HAND=25` | eval-term | OFF (ca only) | card advantage per hand card | NEUTRAL (see ca row) |
| `W_POWERFUL_HAND=0` | eval-term | OFF | when Alakazam in play, value each hand card | no data |
| `W_V3_PH_POTENTIAL` (`leaf_mode=ph`) | eval-term | OFF | realized PH damage, Alakazam active+energized, 20/card capped at opp HP | UNDER TEST |
| `W_V3_BACKUP_ATTACKER` | eval-term | OFF | a benched energized Alakazam ready to promote | no data |
| `W_V3_DECKOUT` (`leaf_mode=deckout`) | eval-term | OFF | hinge penalty as my deck approaches empty (deck<=5) | UNDER TEST (see section 3) |

Notes:
- `leaf_mode=deck` (combined PH + deckout, = `phaware_search_v3`) called a missing `evaluate_deck_v3` and
  silently fell back until this session. That function is now defined in `agent/eval.py` (weights default
  OFF, so the deployed `leaf_mode=hand` path is byte-identical). The deck/deckout/ph modes are now live.
- No-data items: the four ON eval weights have never been ablated individually (tuned by intuition). The
  learned-policy agents (`agent_rank`, `agent_rank_hybrid`, `agent_search_ctx`) have no measurements.

---

## 2. pokemon-ai-agent heuristics (READ-ONLY; with deltas)

`Full` = registry default `hiroingk_alakazam_heuristics(...)`. Deltas are `full vs no_<rule>` so a rule that
helps shows `full` winning. Source: `auditor_sandbox/experiments/*.json`, n=500 unless noted.

| heuristic | on in Full | what it does | measured delta |
|---|---|---|---|
| `lethal_ko` | ON | take the best legal KO now, before later choices erase it | full vs no_lethal 0.808 (404-96). Big positive. |
| `alakazam_energy` | ON | attach Psychic to the Alakazam line, capped 1/line | full vs no 0.646 (323-177). Clear positive. |
| `active_pivot` | ON | promote best-threat bench Pokemon; safe Dudunsparce pivot | 0.476 (p=0.28) neutral/slightly negative |
| `alakazam_draw_engine` | ON | Psychic/Run-Away Draw when useful and deck-safe | ~0.476-0.488 NEUTRAL |
| `gust_threat` | ON | Boss's Orders only for KO/forced modes | ~0.470-0.548 roughly NEUTRAL |
| `go_first` | ON | go first | no isolated A/B |
| `gust_hold_guard` | ON | stops burning a Boss card when no gust mode applies | no isolated A/B (fires often) |
| `alakazam_evolution` | OFF | evolve the Abra line | when ON: 0.456 (p=0.049), so OFF is correct |
| `dudunsparce_engine` | OFF | standalone Dunsparce/Dudunsparce setup ordering | when ON: 0.394 strongly NEGATIVE, OFF is correct |
| gust sub: defensive_stall | OFF | gust to stall | when ON 0.466, OFF correct |
| gust sub: future_threat_pull | OFF | pull a future threat | when ON 0.500 exactly neutral |

Deck-out denial (the cross-repo headline):
- `full vs first` deck-out losses = 5/500 vs 91/500; `first` also hit terminal zero-deck 277/500. Game result
  full 414-86 (0.828). Source: `resource_guard_online_evo_deckout_n500.json`, counter in `run_rule_ab.py`.
- The `Full` agent ALREADY avoids deck-out (5/500) through the draw engine's deck-safety checks. So the extra
  `resource_guard` (and its Enhanced-Hammer / special-energy-denial variants) is REDUNDANT there:
  `resource_guard_on vs full` = 248-252 (0.496, p=0.86), deck-out losses 57/57, neutral both ways. These
  guards were measured and NOT shipped into `src/`; they live only in the experiment harness.
- Implication for kaggle-fun: deck-out denial is solved on the pokemon-ai-agent side but MISSING in
  kaggle-fun's `phaware_search` (no deck term at all). That is the gap section 3 closes.

Enhanced Hammer (current pokemon-ai-agent definition, user update 2026-06-24): damage gate removed. It now
asks "does removing this Special Energy change whether an attack is online or one-energy-away?" with no damage
requirement. Plus one-step-ahead next-stage checks for Dunsparce->Dudunsparce and Abra->Kadabra->Alakazam.
Measured as part of resource_guard variants: neutral. Not shipped.

transition_ranker (sparse replay-imitation ranker): trained to predict the replay player's selected option
(imitation, not winning). The metric usually quoted is top-1 imitation accuracy, but that just measures "did
you guess the single most common recorded move," and it is dominated by how often the recorded action is
option-0 (the default/pass), which is ~61-69% of decisions. So always-pick-option-0 is a strong baseline by
construction (top1 0.4744), beating it on top-1 is neither the goal nor a measure of move quality or winning,
and losing it is not a real failure. The learned variants sit below it on raw top-1 (rich_sparse 0.3906,
neural_hash 0.3637, compact 0.2726), but the informative reads are the deviation (nonzero-label) decisions,
top-3, and NLL/calibration, where the features do carry signal. This is early representation work: a rich
feature vocabulary and a candidate search move-prior aimed at eventually informing good moves, not a finished
policy or value head. Not yet on the ladder.

Caveat: the ladder scores in Orientation are NOT in any repo file (no SUBMISSIONS log exists). User-held.

---

## 3. Eval: status, experiments, current plan

Current leaf eval (`agent/eval.py`): four ON terms (`W_PRIZE=1000`, `W_HP=1`, `W_BODY=30`, `W_ENERGY=8`),
terminals +/- 1e6. Everything else gated OFF. The deployed agent runs this pure prize-dominated hand eval.

Done, with verdicts:
- 4-term hand eval + PH KO floor + 1-ply search is the working stack (0.745 vs heuristic, 0.900 vs first).
- Card-advantage leaf (+25/hand-card): NEUTRAL (0.525, p=0.48). Reasons: (a) `W_BODY` and `W_ENERGY` already
  reward the plays that draw/tutor, so board and CA pick nearly the same lines; (b) 1-ply search cannot plan
  the tutors-first cascade. It branches one decision, then finishes the turn with a fixed rollout.
- Learned / blended value leaf: LOST (~0.427). Diagnosis: global P(win) accuracy is a different objective
  from locally ranking the sibling leaves of one decision.
- Search hyperparams settled: N_DETERM=8 (beat 4; 16 lost under the 0.6s budget). Continuation aggro-vs-setup
  is a tie around 0.5.

Stated bottleneck: SEQUENCING DEPTH, not the leaf objective. A different number at the leaf of a one-decision
branch will not buy the tutors-first / bank-the-KO cascade. That needs a turn-planner that sequences the whole
turn, or a sampler the heuristic invokes.

Under test now (this session): the V3 deck-aware terms, each as its own leaf mode against `phaware_search`,
plus the self-mirror control.
- Deck-out gate-fire diagnostic (faithful, monkeypatched the real leaf eval): the deck-out hinge fires on
  10.6% of search leaves (11,894 / 111,748); leaf decks reach 0; 20% of games end with a player at <=5.
  Verdict LIVE (the term bites, it is not inert). n=30 directional win rate for the deck-out agent was 0.700.
- n=300 A/B (deckout / ph / v3=both vs `phaware_search`, control = self-mirror): **PENDING (running).**
  Will fill: `phaware_search_deckout` vs `phaware_search`, `phaware_search_ph` vs `phaware_search`,
  `phaware_search_v3` vs `phaware_search`, and the `phaware_search` self-mirror (must be ~0.500).

Deck-out shaping rules (agreed with user 2026-06-24):
- The literal deck-out is already terminal LOSS in the eval (the leaf is the start of my next turn; a forced
  draw from empty returns LOSS), so that part is covered.
- Keep a BUFFER: deny a draw-3 when the deck is 4, because going to 1 puts you on a clock. In the search
  setup this is emergent (search prefers the line whose resulting deck is higher); the exact draw-amount-aware
  refusal ("draw 3 with 4 left -> deny") belongs in the floor, not the leaf, because the leaf only sees the
  resulting deck, not "this action draws 3."
- Two exceptions self-resolve from weight ordering if the buffer penalty stays in its band (above development
  scale ~tens to low hundreds, below prize scale 1000): a large losing position (big `W_PRIZE` swing
  dominates) and needing a single card to win this turn (the KO/terminal dominates).
- Test weight is 100/card below floor 6 (so -200 at deck 4, -500 at deck 1), inside the band. After the n=300
  read, sweep the weight and try a buffer-shaped variant (mild cushion penalty starting around deck<=9-10 plus
  the steep ramp below 5).

---

## 4. Features: two approaches and the middle path

A. 47 dense state features (`agent/features.py`, `FEATURE_KEYS`). Handcrafted interpretable encoder of one
board: prize race, board counts, HP, status, rules state, type-aware energy affordance, hand resources by
role, engine-truth playability. Honest read: fed a learned VALUE head used as a search leaf, it LOST (~0.427).
The failure is the load-bearing caveat: global P(win) calibration is not the same as ordering the handful of
near-identical post-move states a 1-ply search compares. Correct use is `tools/eval_feature_importance_v1.py`:
ask which features separate winning from losing states (standardized logistic + univariate corr) to decide
which named eval terms to add, not to be a value head. (Top self-play signals were prize_lead, board_hp,
deck_count, hand_size; deckout_risk and opp_bench were omitted-but-correlated, which pointed at deck-out.)

B. pokemon-ai-agent sparse cross-feature transition ranker. Much richer per-OPTION featurization: ~34 state
scalars plus ~70 per-option fields (action type, source/target zone+card+slot, attack/ability ids, and a large
bank of decoded-effect booleans like searches_to_hand, accelerates_energy, places_damage_counters, disrupts),
crossed action-type x state-context into a sparse linear model (up to ~317k-356k crosses). Caveat on metrics:
it is trained to IMITATE replay moves, so its accuracy numbers are imitation, not move quality. The commonly
quoted top-1 (0.39 best vs 0.474 always-option-0) is dominated by the option-0-heavy action ordering (~61-69%
of recorded moves are option-0), so it is not a meaningful quality yardstick and losing it is not a real
failure. Judge this line by deviation-decision accuracy, top-3, NLL, and ultimately by downstream win rate as
a move-prior. Not yet laddered. It is a feature vocabulary and a candidate move-prior, not a policy or value.

The middle path (recommended, and how deck-out was actually found): use feature importance to INFORM
interpretable eval/heuristic terms, keep them as transparent separately-gated weights with prizes dominant,
and keep win rate as the arbiter. This sidesteps both failures (the black-box value's global-vs-local problem
and the ranker's imitation-loses-to-option-0 problem) while still mining the rich feature vocabulary for which
named terms to add and with what sign. Borrow B's decoded-effect features (deckout risk, draw/tutor
availability, hand differential, KO-now, energy shortfall) as candidate term ideas, then A/B each.

(Note on the two failures: A's is a real diagnosis, global P(win) calibration does not order sibling leaves.
B's "failure" is not really one, its top-1-vs-option-0 number is an imitation yardstick skewed by the
option-0-heavy data, not a statement that the features are weak. Both just argue against deploying a learned
head as policy/value, not against mining the features for interpretable terms.)

---

## 5. What we are looking to improve or try (open questions for other models)

1. Turn-level sequencing is the real bottleneck. 1-ply leaf changes keep coming back neutral (CA, and likely
   PH-potential) because the value of the tutors-first cascade lands across the whole turn, not at one
   branch's leaf. A turn-planner that proposes and scores a whole-turn sequence (heuristic proposes, search
   scores) is the highest-leverage open item. How to do this within the ~0.6s budget is the question.
2. Deck-out shaping: confirm the n=300 lift, then sweep the weight and the buffer shape; decide leaf-term vs
   decision-time floor refusal (the draw-amount-aware "draw 3 with 4 -> deny" needs the floor).
3. Contextual card-advantage objective from features (draw/tutor availability, hand differential, deck risk,
   productive hand spending) rather than the raw hand-count term that was neutral.
4. Eval-term ablations: the four ON weights have never been individually measured. Cheap n=100-200 ablations
   would tell us if W_BODY=30 / W_ENERGY=8 are even close to right.
5. Move-ordering prior from the ranker to guide which options search expands first. This is the only learned
   piece that targets sequencing rather than the leaf number.
6. Ladder validation. Local self-play does not predict the ladder. Any change that wins locally still needs a
   ladder submission, and the current best local stack (`phaware_search`) is NOT the pure heuristic that
   scored 736; it keeps the PH KO floor the losing search lacked, so it is worth a submission to confirm.

---

## 6. Files

kaggle-fun: agent/eval.py, agent/search_v3.py, agent/deck_policy_v3.py, agent/features.py,
tools/run_heuristic_ab_v1.py, tools/eval_feature_importance_v1.py, tools/deck_depletion_diag_v1.py,
data/heuristic_ab_v1.json, data/heuristic_ab_deckv3.json, data/meta_opponent_ab.json, dropoff/inbox/*.md.

pokemon-ai-agent (READ-ONLY): src/pokemon_ai_agent/policy/heuristics/*, .../registry.py,
auditor_sandbox/experiments/*.json (deltas), .../run_rule_ab.py (deck-out counters),
src/pokemon_ai_agent/transition_ranker/* (sparse ranker), data/results/current.json.
