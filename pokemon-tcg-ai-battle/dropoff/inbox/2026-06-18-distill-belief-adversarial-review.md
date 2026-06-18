# Adversarial review: belief A/B + distillation pipeline (2026-06-18)

Source: an internal multi-agent review (3 reviewers -> verify -> synthesis) run BEFORE the
sim-heavy distill build, to catch design flaws first. 20 findings raised, the confirmed ones below.
All citations are file:line and were verified by a second agent reading the code.

## Belief-determinization A/B (tools/search_sprint.py "belief") -- what the test can and cannot say

The plumbing is CORRECT (verified): `load_meta_deck(exclude_deck=M.DECK)` picks a genuinely different
archetype, and `opp_prior` flows into the opponent's sampled hidden cards. But the A/B as written
cannot answer the hypothesis (H022/H008):

- **Unpaired + unseeded (blocker).** It runs two independent n=40 win-rate samples and reports each
  one's Wilson CI vs 0.5; there is no statistic on the belief-minus-same-deck DIFFERENCE. Nothing is
  seeded (`cabt_arena.py` `make("cabt")` no seed; `search.py:142` unseeded `random.shuffle`), so the
  two arms see different games AND different hidden worlds. A plausible ~0.05-0.08 edge is buried.
  Fix: pair the arms on the same game seeds (`make("cabt", configuration={"seed": g})` + `random.seed(g)`),
  record per-game outcome pairs, report McNemar / paired-bootstrap CI on the difference.
- **The leaf is structurally near-blind to opp hidden cards (major).** The 1-ply hand-leaf search
  evaluates the START OF MY NEXT TURN; `search.py`'s own docstring concedes "Evaluating MY own turn
  barely depends on the opponent's hidden cards." `opp_prior` only changes the opponent's single
  rolled-out reply, gated mostly by their VISIBLE active+energy. So a tie is the EXPECTED null whether
  or not belief matters. Fix: raise sensitivity with `opp_k>0` (2-ply min-leaf branches on the sampled
  opp hand) or a deeper horizon BEFORE concluding anything about belief.
- Rollout-policy mismatch (major): the simulated opp attacks max-damage, but the real meta-opp is
  `first_agent` (option-0). The modelled punish does not match the opponent the win-rate measures.
- Minor: the `opp_prior=None` control strips the opp's META board cards from OUR deck Counter (no-ops,
  pool 60 vs 58); and short pools pad with card id 3, which is in NEITHER deck (stale from the old deck).

**Verdict recorded (E014/R008): inconclusive, hypotheses H008+H022 PARKED, not refuted.** Belief is only
worth pursuing once the search is made opponent-sensitive (2-ply) and the A/B is paired+seeded.

## Distillation pipeline -- blockers fixed before the build

Goal restated (per binding rule 6): distillation buys SPEED, not strength. The net's ceiling is the
hand-search teacher; high distill top-1 means the instant net reproduces the 0.6s/decision search, which
we cannot run inside the match budget. It is NECESSARY-not-sufficient: a swap still needs a win-rate A/B.

Fixed (commit f8fce28):
- **B1**: `train_action_ranker.py --target distill/both` now HARD-FAILS if no `bsrch` labels exist
  (was silently printing a fake `DISTILL top1 0.000` against the label-less imitation file).
- **B2**: `build_action_dataset.py --values` applies `_forced_move` (lethal/KO attack, go-first) BEFORE
  the search argmax when setting `bsrch`, so the label == the move the DEPLOYED teacher (`agent_search`)
  actually makes -- previously wrong precisely on the high-crit/lethal stratum the metric emphasizes.
- Full per-decision option coverage required for a label (drops decisions where any sibling failed to
  simulate; removes a silent centering/argmax selection effect).
- Raised OFFLINE determinizations/budget (`--label-determ 16 --label-budget 4.0`): no match budget
  offline, so average more worlds -> lower label noise (bsrch is a noisy estimate of the teacher).
- Reporting: added a TEACHER-deviation stratum (option-0 != teacher), the only slice that proves the net
  beat the positional prior (option-0-vs-search there is 0 by construction); per-stratum option-0-vs-search
  baselines printed next to each DISTILL number; softened the "can REPLACE search" framing.

Known latent coupling (documented, not active): `option_evals` hardcodes the teacher config to
opp_prior=None / opp_k=0 / leaf="hand", which matches the deployed `agent_search` today. If belief or
2-ply is ever promoted, the label generator must mirror it or the net distills the wrong policy.

## Build context
Distill set built on `--player KanNinomiya --strategic-only --values` (KanNinomiya is the deck's most
prolific winner, 144 win-games / 13852 decisions; DENPA92 is the deck's archetype name but matched 0
rows as a player). KanNinomiya plays the same deck-rank-1 archetype = our deployed deck.
