# CURRENT — execution controller (check every step; the living docs are NOT the controller)

Updated: 2026-06-18

- **Branch:** claude/optimistic-proskuriakova-8800d3
- **Strongest agent:** `agent_search` (1-ply forward search + hand leaf eval, N_DETERM=8). Nothing beats
  it. It is the submission default and the baseline every learned thing must beat. Anchor this turn:
  search 0.833 vs heuristic (50-10, n=60, harness sound).
- **Agents (arena keys in cabt_arena.AGENTS):** random, first, heuristic, search (STRONGEST), search2
  (2-ply, loses), search_v / combine / eff (all lose), `rank` (distilled net, loses), `rankh` (net on
  strategic decisions only, loses). See the distillation result below.
- **Submission ready:** `submissions/sub_search.tar.gz` = agent_search + DENPA92 deck (verified). Upload
  is the user's call.

## What happened this turn (the two lanes the plan called for)

### 0. Search sprint, knob = continuation policy (the other model's #1 target) -> DIRECTIONAL, not adopted
The rollout finished MY turn by attacking on the first legal attack; `MY_CONT="setup"` develops the board
first (play/attach/evolve/ability), attack last. Head-to-head setup vs aggro 0.583 (35-25) [0.457,0.699]
n=60 -- directionally better but the CI spans 0.5, and setup's rollout is ~4x slower so under the 0.6s
match cap it lands fewer determinizations (and N_DETERM=8 > 4). Kept `aggro` (the validated submission
config). Clean re-test should hold the determinization count fixed before adopting setup.

### 1. Search sprint, knob = opponent belief (H022/H008) -> PARKED, not refuted
Belief-conditioned determinization (`opp_prior=meta`, fill the opponent's hidden zones from a DIFFERENT
archetype) did NOT beat the same-deck placeholder (0.800 [0.652,0.895] vs 0.925 [0.801,0.974], n=40).
This is NOT a refutation: an adversarial code review confirmed the 1-ply hand-leaf search evaluates the
START OF MY NEXT TURN, which barely depends on the opponent's hidden cards (search.py's own docstring),
so a tie is the EXPECTED null; and the A/B was unpaired + unseeded. The first run was also an accidental
no-op (the top corpus deck IS our deck; fixed via `load_meta_deck(exclude_deck=M.DECK)`). Reopen gate
(registry H022): a paired+seeded A/B AND an opponent-sensitive leaf (opp_k>0 2-ply, or deeper horizon).
The determinization lever that DID work earlier was N_DETERM 4->8.

### 2. Research lane, the LEARNED model (H024) -> OFFLINE-FAITHFUL but a POOR POLICY
Built the action-CONDITIONED ranker the user asked for since day one: card-id EMBEDDING + decoded card
EFFECTS + action descriptor + root-state features + forward-model one-step DELTAS -> shared MLP ->
listwise softmax per decision. Trained two ways with a non-circular target:
- **distill** (best): reproduce the frozen SEARCH teacher's per-option multi-turn value (S.option_evals).
- **imit**: predict the human winner's move (auxiliary prior). The search and the human are DIFFERENT
  policies (agreement only ~0.18-0.28), so distill and imit are antagonistic; pure distill wins.

**Offline (held-out, canonical eq-class), distill:** DISTILL top-1 all 0.570 / teacher-deviation 0.494
(the slice where option-0 is NOT the teacher's pick, baseline 0.000) / high-crit 0.564. So the net
reproduces the search far above the positional prior. FULL beats no-deltas/no-effects/no-embedding by
~0.03-0.05 on the deviation slice (within small-n noise, n=158). Deltas + effects carry a little; the
embedding now has 16 cards (see the fixed feature bug below).

**Arena (the payoff test) -> it FAILS.** Deployed via `agent/ranker.py` (torch-free numpy, instant):
`agent_rank` (net everywhere) 0.117 vs heuristic / 0.233 vs first; `agent_rank_hybrid` (net on strategic
decisions only, heuristic else) 0.150 vs heuristic / 0.200 vs first. Both BELOW option-0 (anchor:
search 0.833 vs heuristic, 50-10).
**VERIFIED this is NOT an inference bug** -- ranker.predict's features match the trainer's exactly (max
abs diff 0.0 over 285 options, 0 card_id mismatches). Cause: covariate shift + imperfect per-decision
fidelity -- ~0.5 distill top-1 does not compound into a policy, and self-play drifts off the training
distribution (classic behavior-cloning failure). The OFFLINE ranking objective works; turning it into a
winning policy is the open part.

**Fixed a dead-feature bug along the way:** `opt_card_id` only extracted a card id for area==2 options,
so PLAY options (type 7, no area field) had card_id=-1 -> card stats/effects/embedding were dead and
`opt_key` collapsed all PLAY options into one eq-class. Now it extracts the acting-card identity per
option type. Embedding vocab 4 -> 16; card-identity rows 6185 -> 11375 / 13115.

## Decision needed (per "agree before building a new hard approach")
The learned model is real and offline-validated; making it a WINNING policy needs one of:
- (a) **DAgger** -- label the net's OWN self-play states with the search teacher, retrain iteratively
  (fixes covariate shift). Sim-heavy, iterative.
- (b) **Net as a search-INTERNAL ordering prior** -- prune to the top-K net options, the SEARCH still
  decides (no covariate shift); the saved budget buys more determinizations/depth. Most likely to ADD
  arena value while keeping the search's strength.
- (c) Train on ALL decisions, not just the strategic subset (covers the whole game).

## Status vocabulary (no more "done" = script ran)
specified -> implemented -> data-generated -> trained -> offline-evaluated -> arena-evaluated ->
accepted | refuted | inconclusive | parked

## Binding rules (full list in ACTION_RANKER_PLAN.md)
1 card-effect claim needs card_effects.json consumed live. 2 action-ranking needs grouping + ranking
loss + within-decision metrics. 3 embedding claim needs trainable vectors in the scorer. 4 no
"validated" without the ablation. 5 AUC/Pearson are diagnostics only. 6 hand-eval-only labels (and a
hand-search teacher) cap the learner AT the teacher -- distillation buys SPEED, not strength; a swap
still needs a win-rate A/B. 7 the objective is RANK SIBLING ACTIONS, never score a state. 8 offline
within-decision fidelity is NECESSARY but NOT sufficient -- per-decision accuracy does not compound into
a policy (covariate shift); only a frozen seat-swapped win-rate A/B accepts a policy.

## Data
~622 replays in `data/external/replays/` (gitignored). `tools/fetch_episodes.py --top-teams N` pulls
more. `agent/card_effects.json` decoded. Distill set: `tools/build_action_dataset.py --player
KanNinomiya --strategic-only --values` -> `data/replay_db/action_adv.jsonl` (1218 decisions, full
search-value coverage). Deploy model: `agent/ranker_model.json`.
