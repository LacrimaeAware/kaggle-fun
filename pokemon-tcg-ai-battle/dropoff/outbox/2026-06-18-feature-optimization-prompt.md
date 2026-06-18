# Feature Engineering for a Pokemon TCG Battle Agent: Action / Effect / Interaction Features

> Paste-ready prompt for another strong model. It has repo access but none of our conversation.
> Verified numbers below are from our own re-run after fixing the diagnostic bug (2026-06-18).

## Who you are and what we need

You are a strong ML/RL engineer with full read access to this repository. You have NONE of the prior
conversation that produced this brief, so everything you need is below or in the repo. Your job is to
**analyse our card database and game replays and propose a prioritised set of per-option (action /
effect / interaction) features**, with concrete compute methods, an implementation order, and a
measurement protocol. Treat this as a feature-engineering and diagnostics task, not a "train a big
model" task.

This is a Kaggle simulation-agent competition: agents pilot Pokemon Trading Card Game decks against
each other on the organizer's real engine. The user's explicit goal is **a flexible learned agent
that can pilot DIFFERENT decks at least moderately well — not a hand-tuned heuristic per deck.** The
user believes the bridge is "learned heuristics with RICHER FEATURES so the model can detect what
makes a play good," trained to emulate strong players' winning moves (imitation / behaviour
cloning), possibly fused with the forward model. Do NOT propose "pick one easy deck and master it."

## Repo location and environment

- Repo root: `pokemon-tcg-ai-battle/`
- Python venv (run with `PYTHONIOENCODING=utf-8`): `.venv/Scripts/python.exe`. The engine prints
  noisy OpenSpiel lines to stderr; ignore them.
- The forward-model engine is the real organizer engine (`kaggle_environments` cabt). It runs locally.

### Files to read first

- `agent/features.py` — `encode_state(obs) -> 47 STATE-level features`. The current representation.
  Note the per-option hand-join already done correctly for `draw_playable_now`.
- `agent/eval.py` — prize-dominated hand-eval leaf score. Reusable helpers `_board_hp`, `_n_pokemon`,
  `_active_energy`.
- `agent/search.py` — 1-ply forward-search agent; `option_evals` simulates each option with a FULL
  rollout (DEPTH_CAP=80) and collapses to one scalar. Reuse `_hidden_pool`, the determinization loop.
- `agent/main.py` — `DECK` + hand heuristic; `agent_search` the 1-ply agent.
- `agent/card_features.json`, `agent/attack_stats.json` — per-card structural fields + tags;
  attackId -> damage.
- `data/external/official/cards_full.json` — 1267 cards, FULL English effect text (`skills[]`,
  `atk[]` with rider text). The effect text IS present, just undecoded.
- `data/external/official/sample_submission/cg/api.py` — option/area enums: `OptionType`
  (PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10, RETREAT=12, ATTACK=13, END=14, target types 3–6),
  `AreaType` (HAND=2, ACTIVE=4, BENCH=5, PRIZE=6); `Card.serial`, `Pokemon` (serial, energies,
  preEvolution, hp/maxHp), `PlayerState` (hand, deckCount, discard, prize).
- `data/external/replays/*.json` — 135 real ladder games (both seats' obs + action per step).
- `data/replay_db/{games.jsonl, decks.json, decisions.jsonl}` (built by `tools/build_replay_db.py`)
  — 133 games, 14,442 outcome-labelled decisions, 57 decks. `decks.json` holds real per-game decks
  (use these for forward-model determinization on replays; do NOT use the agent's own `DECK`).
- `tools/diag_action_ceiling.py` — the imitation diagnostic and your measurement harness.
- `tools/build_card_features.py` — emits `agent/card_features.json`; extend for effect decoding.

## The measured finding (verified after a bug fix)

`tools/diag_action_ceiling.py` tests imitation: for every single-select decision the EVENTUAL WINNER
faced (6,953 decisions, avg 8.2 options), train a within-decision ranker to predict the winner's
chosen option; metric = top-1 (model's argmax option == winner's). Stratified, with the right
baselines (our re-run, 2026-06-18):

- random (1/n_options): **0.197**
- **chose-option-0 (positional prior): 0.587** — the engine's option ordering alone reproduces the
  winner 59% of the time; this is the real bar, and it BEATS our heuristic and our learned ranker.
- our hand heuristic re-run on the same obs (a FLOOR, not a claim opponents use it): **0.553**
- learned pointwise GBM ranker, with the card-join FIXED, stratified (ALL | which-card | mixed):
  - type-only: 0.504 | wc 0.818 | mixed 0.336
  - +card class/stage/ex/best_dmg: 0.500 | wc 0.810 | mixed 0.334
  - +attack dmg/cost: 0.493 | wc 0.810 | mixed 0.324
  - +target HP, +KO interaction: ~0.49 | wc ~0.81 | mixed ~0.32

Read it carefully: on which-card decisions the 0.81 is the option-0 prior (type is constant there,
so the GBM ties and argmax picks option 0); **adding card features does NOT beat that prior**. On
mixed strategic decisions the GBM (~0.33) is BELOW the ~0.46 option-0 rate. So **our per-option
features, even with the join fixed, do not beat "pick option 0" under a pointwise objective.**

This does NOT prove a representation ceiling, and it does NOT prove features help. It proves the
diagnostic so far is too confounded to conclude: a strong positional prior dominates the metric, the
objective is a weak pointwise GBM that cannot even match option-0, and the real levers (listwise
loss, card-EFFECT tags, forward-model deltas) are untested. Your job is to remove these confounds and
find out.

## Confounds and bugs already found (fix/respect before trusting any number)

1. **(FIXED) card-identity join was dead code** — it gated on `isinstance(hand[idx], int)` but replay
   hand entries are dicts `{'id':..,'serial':..}`. The reliable join is `AreaType.HAND==2` +
   `index -> hand[idx]['id']` (option.cardId is null in replays). Restrict the hand-join to
   PLAY/ATTACH/EVOLVE; for ATTACK use `attackId -> attack_stats`; for target/ability use
   `inPlayArea/inPlayIndex`. Do NOT join `index->hand` for in-play TARGET option types (3–6).
2. **Positional prior** — ~58.7% of winner moves are option index 0. Always report a chose-option-0
   baseline; it is the bar to beat, not random.
3. **Pointwise objective on a listwise problem** — the ranker is a `GradientBoostingClassifier` with
   per-option binary labels scored by per-group argmax. Use a within-decision LISTWISE/pairwise
   objective (LightGBM `lambdarank` / XGBoost `rank:pairwise` / softmax-over-options cross-entropy)
   with decision id as the group. Test this FIRST: if a listwise loss alone (no new features) beats
   option-0, the prior "ceiling" was an objective artifact.
4. **State features are constant within a decision** — the 47 `encode_state` features are identical
   across siblings, so they only help when CROSSED with per-option action features.
5. **No CIs / no stratification** — report top-1 AND top-3 AND MRR; stratify into which-card vs mixed
   vs trivial (END/YES); give Wilson CIs over multiple game-wise splits.
6. **KO must compare to current `hp`, not `maxHp`** (both fields exist on the in-play dict).

## Your deliverable

A single document (your final message) containing:

### 1. A corrected diagnostic and believable baselines
Fix the objective (listwise/pairwise, decision-id groups), report top-1/top-3/MRR with the
chose-option-0 baseline, stratify (which-card / mixed / trivial), and give Wilson CIs over multiple
splits. The pass bar is **beating chose-option-0 on the MIXED slice** (~0.46), not random.

### 2. A prioritised feature set (the core ask)
For EACH feature: name, one-line semantics, exact compute (file/field/helper, applicable option
types, O(cost)), and whether it needs a forward-model sim or is static. Organise by priority tier.
Cover at least:
- **Static action semantics:** per-option KO-this-target (vs current `hp`), KO prize value, overkill,
  affordability; EVOLVE stage/hp/dmg gain, target-already-powered; ATTACH completes/progresses an
  attack cost; retreat-to-better-attacker. (Gives each sibling its OWN vector.)
- **Card-EFFECT features from `cards_full.json` text** (the untested rung): draw N, search/tutor
  target+count, energy-accel pips+source, gust, evolve-enable (Rare Candy), heal N, self-switch,
  hand-disruption, special-condition, discard cost. Two-layer decode: regex over the stereotyped
  effect text + a small hand-built override table for the ~40 meta cards regex misses. Quantify
  effects (draw HOW MANY), don't just copy presence flags. Do NOT trust `registry/card_review.json`
  tags as ground truth (~10/1267 confirmed).
- **State × option interaction / within-decision RELATIVE features:** cross the constant state vector
  with each option's action features; normalise option attributes within the decision (is-this-the-
  only-KO, is-this-the-highest-damage-affordable-attack), since the question is comparative.
- **Forward-model one-ply DELTA features (the consequence rung):** simulate ONLY that one option
  (one `search_step`, no rollout) and feature the immediate state delta (prizes taken, opp KO, cards
  drawn = Δhand, energy attached, board developed). Recover the played card by serial-diffing the
  board (join-free). Reads effect semantics BEHAVIORALLY without any text DB. State cost in
  engine-steps/decision and a budget argument (live per-decision cap ~0.6s). Use each game's real
  deck from `decks.json` to determinize replays, NOT the agent's `DECK`.

### 3. Implementation order and measurement
Which features to implement FIRST (the minimal set you predict will clear the chose-option-0 bar on
mixed decisions), the exact change to `tools/diag_action_ceiling.py` to wire each tier in, and the
pre-registered pass/fail. State what would CONFIRM "static replay lacks readable action semantics,
forward-model deltas are the bridge" vs REFUTE it. Test the listwise-objective-alone experiment
first (cheapest).

## Do NOT waste time on

- Re-running the old pointwise GBM and re-reporting the flat ~0.50 as informative. Fix first.
- Joining `index->hand` for in-play TARGET option types (3–6). Restrict to PLAY/ATTACH/EVOLVE.
- Comparing KO against `maxHp`. Use current `hp`.
- Trusting `card_review.json` LLM tags as truth; decode from effect text and/or verify behaviorally.
- Single-split point estimates with no CI; unstratified aggregates (they hide that one slice is
  positional-prior-dominated and another is hard).
- Full multi-ply rollouts for ranker features; the action signal is in the IMMEDIATE one-step delta.
- Treating every winner move as good (survivorship/circularity: winners pilot DIFFERENT decks and
  still make weak plays). Consider outcome/margin weighting or high-skill subsets.
- Using the agent's own `DECK` to determinize replays.
- Widening scope to win-rate A/B, deeper search, or new model classes as the primary task (note a
  win-rate A/B as downstream confirmation only; top-1 imitation is not win-rate).
