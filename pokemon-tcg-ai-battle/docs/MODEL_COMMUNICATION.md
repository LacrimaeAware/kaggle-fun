# Model communication notes

> LATEST HANDOFF (2026-06-18): read `dropoff/outbox/2026-06-18-CONSENSUS-and-way-forward.md` first,
> and the inbox handoffs in `dropoff/inbox/`. Fixed since the notes below: `eval.evaluate_blend`
> NameError (blend now runs); the dead card-join in the imitation diagnostic. Confirmed: search_v /
> combine (learned value at the leaf) lose to `agent_search`; 2-ply lost; the hand-weighted effect
> heuristic lost (it was a replacement scorer, not a residual). Active build: integrated offline
> replay-trained action-ranker. Cross-session dropoff lives in `dropoff/` (inbox + outbox).

This file is for handoff between model sessions. It records concrete code issues and audit
targets that should not be lost in prose or chat context. Treat it as a working note, not as
the registry canon for experiment results.

## Status review of the items below (2026-06-17, Opus session) — re-verify, do not trust blindly

Every verdict here should be VERIFIED AT LEAST ONCE by a future model before it is relied on; a
prior model calling something "fixed" or "fine" has been wrong before.
- `zone or [3]` phantom energy in search.py: FIXED this session (empty zones now passed as []).
  Verified the engine accepts empty zones and arena runs error-free. RE-VERIFY.
- features.py Team Rocket Energy (11) = full wildcard: VERIFIED REAL (api.py: TEAM_ROCKET =
  Psychic/Darkness, not all types; Rainbow=10 is the true wildcard). No impact on the current
  Mega-Abomasnow deck (no TR energy); FIX-PENDING and a PREREQUISITE before any TR-energy deck.
- search.py determinization does not strip the public `current.stadium`: VERIFIED REAL,
  FIX-PENDING (low impact unless a Stadium is in play; fix when next touching determinization).
- main.py default-submitted policy: OUTDATED/CORRECTED. The file now defines `agent_search_v`
  LAST, so by the Kaggle last-callable rule the WEAKEST agent would ship. Must pin `agent_search`
  (the strongest, see below) before any submission. See agent/README.md packaging note.
- per-decision time budget: PARTIALLY ADDRESSED (lowered to 0.6s; measured max ~0.14s). Still
  does NOT read `remainingOverageTime`; do not promote search to default until match-time
  semantics are settled or a budget governor is added.
- train_value.py needs sklearn while requirements.txt lists only kaggle-environments: VERIFIED
  (sklearn IS importable in the repo .venv, training works; INFERENCE is sklearn-free, exported
  GBM verified bit-exact vs sklearn). FIX-PENDING: add sklearn+numpy to requirements for repro.
- submission/ is a stale type-14 baseline: VERIFIED REAL. Do not ship it.
- legacy tags wall_tank/disruption: NEEDS-VERIFICATION. The CORRECTED classifier (1267 audited)
  does emit `wall_tank` (seen in the rebuild), so it may now be a sanctioned class; confirm
  against the final taxonomy whether these are legit or stragglers.
- build_stats.py PLAY attribution / list-shaped replay drop, ingest_labels.py tag filtering,
  optional-prompt minCount==0 handling: OPEN, UNVERIFIED by this session (not in the current
  value/search path; revisit if replay-derived stats or optional-effect decks become load-bearing).

- SUBMISSION VALIDATION FAILED 2026-06-17 (FIXED): both submitted agents got ERROR status. Cause:
  the submission main.py used `os.path.abspath(__file__)` at module scope; Kaggle loads via
  `exec(code_object, env)` with no `__file__` -> NameError -> agent never loads. Fixed in
  tools/build_submission.sh (no __file__; hardcoded /kaggle_simulations/agent path) +
  tools/verify_submission.py (exec without __file__, like Kaggle) which the build now runs. Lesson:
  test a submission the way Kaggle loads it, not via `import main`. Raw logs:
  submissions/validation_failures_2026-06-17/.

## Current concrete issues to keep in view

- `agent/search.py` still contains the hidden-zone fallback pattern `zone or [3]` when calling
  `search_begin`. This can inject a fake Water Energy into an actually empty hidden zone.
  Official `cg/api.py` says the predicted hidden-zone lists must match the hidden-zone counts.
  Empty zones should stay empty.
- `tools/build_stats.py` attributes `LogType.PLAY` events to the observation owner index rather
  than `log.playerIndex`, and can double-count repeated logs across observations. Replay-derived
  played-card stats are therefore not trustworthy yet.
- `tools/build_stats.py` silently drops replay files whose top-level JSON shape is a list rather
  than a full episode dict. If those files matter, the parser needs explicit handling. If they do
  not matter, the script should report that they were skipped and why.
- `tools/ingest_labels.py` filters imported tags against `review_server.TAXONOMY` only. User-added
  taxonomy classes saved in `card_review.json` can be dropped on re-ingest unless the saved
  taxonomy is merged into the accepted tag set.
- `agent/main.py` exports the conservative heuristic as `agent`. `agent_search` and
  `agent_search_v` exist, but are not the default submitted policy unless packaging or naming is
  changed deliberately.
- Search currently uses a fixed per-decision budget and does not adapt to `remainingOverageTime`.
  Do not promote it to the default submitted policy until the match/per-move time semantics are
  settled or a budget governor is added.
- `agent/features.py` maps EnergyType 11 (`TEAM_ROCKET`) as a universal wildcard. Official
  `cg/api.py` documents Team Rocket Energy as Psychic/Darkness, not all types. This can
  overstate `active_energy_short`, `active_can_attack_now`, and color matching for future decks
  using Team Rocket's Energy.
- `card_review.json` / `agent/card_features.json` currently contain legacy tags not emitted by
  the current functional classifier: Hippowdon has `wall_tank`, and Milotic has `disruption`.
  These should be mapped to current classes or removed so model features use one taxonomy.
- `tools/train_value.py` depends on `sklearn`, but the project-local `requirements.txt` only lists
  `kaggle-environments`, and the current Python environment could not import `sklearn`. Training is
  therefore not reproducible from the checked dependency file.
- The existing `submission/` folder is an old minimal baseline: `submission/main.py` still says the
  forward model is blocked and ships only the type-14 deferral heuristic. It does not include
  `card_features.json`, `card_stats.json`, `attack_stats.json`, `value_weights.json`, or `cg/`.
  Treat it as a stale baseline unless intentionally submitting that floor agent.
- `agent/search.py` determinization strips active, bench, attached cards, pre-evolutions, hand
  where visible, and discard, but does not strip the public `current.stadium` card from the
  predicted hidden pool. Local replay states do contain a public Stadium, so hidden deck/prize/hand
  sampling can duplicate the in-play Stadium and drop a real hidden card.
- Current policy/search paths do not consider returning an empty selection for optional prompts
  where `minCount == 0` and `maxCount > 0`. Local replay data contains such prompts. Always taking
  one option may force optional searches/keeps/choices that should sometimes be declined.

## Audit focus for next passes

- Classification pipeline: `card_functional_classification.json` -> `card_review.json` ->
  `agent/card_features.json`. Verify tag/schema alignment, user-added classes, and whether audited
  labels are marked in a way downstream tools interpret correctly.
- Feature math: confirm energy-cost letters, wildcard energy semantics, prize values, ex/Mega/Tera
  flags, HP/current-damage interpretation, and legal-option feature extraction.
- Value/RL path: confirm datagen labels match the feature perspective, train/validation split is
  game-wise, exported GBM math exactly matches sklearn, and inference does not silently flip or
  neutralize scores in the wrong seat.
- Search integration: confirm determinizations have exact zone lengths, search branching does not
  mutate the shared root across candidate options, leaf states match the training distribution, and
  broad exception fallbacks are not hiding a permanently broken search/value path.

## Methodology cautions

- The current learned value model is a global state-value predictor: "from this state, did the
  player eventually win under the data-generating policies?" Search needs local sibling-action
  ranking: "among the leaves produced by these legal options right now, which first action is best?"
  Good global AUC can fail to improve play if the model is poorly calibrated or piecewise-constant
  among nearby sibling leaves.
- The current GBM value is trained on start-of-turn Monte Carlo outcomes. That aligns with the
  clean leaf target only when search actually rolls to a clean start-of-my-turn state. Any leaf
  evaluated at another distribution is out-of-distribution and should fall back or be separately
  trained.
- If the goal is better search, the next data product should be candidate-leaf data from search:
  for each root decision, log all candidate first actions, their resulting leaf features, the chosen
  action, and eventual/rollout outcome. Train a Q/advantage/ranking model over siblings, not only a
  state classifier over random self-play states.
- Classifier tags are useful as coarse role features and deck/opponent priors, but they are not
  enough for precise play. They collapse "draw 2" and "draw 7", optional costs, once-per-turn
  constraints, target legality, and card-specific combo text. Keep numeric attack/rule features and
  consider card/entity embeddings alongside tags.
- Embeddings should not be trained only from co-play/co-occurrence if the desired signal is
  counter-structure. Use directed outcomes, forced matchup interventions, or search-generated
  state/action labels. Otherwise embeddings mostly rediscover archetypes/deck membership.
- The highest-leverage learned component before full RL is likely a belief/opponent model for
  determinization: infer opponent deck/hand/prizes from revealed cards and meta priors, then feed
  realistic hidden states to `search_begin`.
- Full RL should come after a strong legal heuristic/search baseline and should be evaluated by
  head-to-head play, not by value AUC. The relevant metric is policy improvement under the same
  deck/opponent distribution, with seat alternation and enough games for the noise band.

## Handoff from the Opus session (2026-06-17): the learner loop + where I want a hard audit

WHAT I BUILT: the full L2 loop. `agent/datagen.py` logs one start-of-turn (features -> eventual
win) row per turn from self-play; `tools/train_value.py` fits a gradient-boosted tree predicting
P(win) and exports it as raw tree arrays (VERIFIED bit-exact vs sklearn); `agent/value_model.py`
runs it in pure numpy; `agent/eval.py:evaluate_learned` + `agent/search.py` use it as the leaf
eval (agent `search_v`). Search was made robust: aggressive opponent reply (not default-order)
and averaging each option over N_DETERM determinizations. Two adversarial-review workflows (34
findings total, all verified) drove the fixes.

KEY RESULT (registry H023, properly powered with Wilson CIs):
- `agent_search` (forward search, HAND eval) is the STRONGEST agent: 0.585 vs first_agent
  (CI [0.551,0.619]), 0.543 vs the heuristic. SHIP THIS one if submitting.
- `agent_search_v` (same search, LEARNED tree value) LOSES: 0.427 vs heuristic (CI [0.380,0.476],
  entirely below 0.5), 0.467 vs hand search. The tree predicts win/loss at AUC 0.735 (good GLOBAL
  classification) but ranks NEARBY candidate leaves poorly (piecewise-constant), which is what
  1-ply search needs. A scale bug (terminal +/-1e6 averaged with P(win)) had inflated it to 0.485;
  fixing it dropped it to 0.427 (the terminal-rate proxy out-ranked the clean value). Not a
  refutation of the architecture; the learned value needs a different FORM or DEEPER search.

WHERE I WANT A STRONG AUDIT (treat as unverified):
1. NON-TRANSITIVITY: search_v does worse vs the heuristic (0.427) than vs hand search (0.467),
   even though hand search BEATS the heuristic. Is this just overlapping noise (CIs do overlap),
   or a real systematic bias the heuristic's go-first/always-lethal style punishes? Re-measure at
   n>=800 and inspect which moves the value picks that the heuristic exploits.
2. THE COMBINE (next build, not yet done): blend leaf eval = hand_eval + lambda*learned_value, and
   value-as-prior to weight options. Audit that the blend keeps ALL leaves on one scale (the scale
   bug must not recur) and is measured at n>=800 with CIs vs hand search AND the heuristic.
3. The value targets are raw Monte-Carlo (SUPERVISED, not RL/bootstrapped). Any future expert-
   iteration ("use loss + search to update the weights") must verify the train/serve distribution
   match and that search-improved targets, not raw game outcomes, are what's regressed.
4. Feature math for OTHER decks before deck expansion: the TR-energy wildcard (real) and the
   stadium determinization (real) are latent now but will bite on new decks.
