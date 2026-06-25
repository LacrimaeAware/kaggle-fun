# Search, card-advantage, and meta-opponent change audit (2026-06-24)

Scope:
- Verified: audited the current working tree in `C:\Users\EcceNihilum\Desktop\GithubRepos\kaggle-fun\pokemon-tcg-ai-battle`.
- Verified: modified files are `README.md`, `agent/eval.py`, and `agent/search_v3.py`.
- Verified: untracked additions include three prior inbox notes and four tools: `build_opponent_meta_v1.py`, `run_heuristic_ab_v1.py`, `run_meta_opponent_ab.py`, and `turn_choice_demo_v1.py`.
- Verified: `python -m py_compile` passed for the touched Python files and the four untracked tools.
- Verified: `python tests/test_heuristics_fixed_state.py` passed 11/11. Worst `agent_search` fixture decision was 0.606s.
- Verified: `python tests/test_split_base_v2.py` passed 8/8.

## Findings

### P2. `leaf_mode="deck"` is still wired to a missing live eval function

Verified: `agent/search_v3.py` calls `EV.evaluate_deck_v3(cur, me)` in both `_simulate` and `_leaf_val`. Verified by import check that live `agent/eval.py` does not define `evaluate_deck_v3`.

Impact: any caller that selects `leaf_mode="deck"` will hit an `AttributeError` inside search. The exception is caught by the search wrapper, so the likely behavior is silent search failure and fallback rather than an obvious test failure. That can make an experiment look like it measured deck-mode search when it actually measured fallback behavior.

Recommendation: either port the intended `evaluate_deck_v3` from `tools/_v3_src/eval.py` into live `agent/eval.py`, or remove/guard the `deck` leaf path so callers cannot select a dead mode.

### P2. A/B tools do not measure whether the intended search path actually executed

Verified: `tools/run_heuristic_ab_v1.py` and `tools/run_meta_opponent_ab.py` catch broad exceptions inside the agent and fall back to the heuristic or legal move path. The reported `errors` counter only catches failures that escape the whole environment run.

Impact: a bad search configuration, meta prior problem, or hidden `opp_decks` sampling issue can be reported as a normal win-rate result. This matters most for the new meta-opponent experiment because the tested difference is inside the swallowed search path.

Recommendation: add counters to each experimental agent for `ko_hit`, `search_hit`, `search_none`, and `search_exception`, then write those counters into the JSON result. Keep the crash-safe fallback, but make fallback rate visible.

### P3. `evaluate_ca` is a raw own-hand-size probe, not a full card-advantage objective

Verified: `agent/eval.py` adds `W_CA_HAND * my_hand` to the board eval. It does not score opponent hand size, hand differential, card quality, deckout risk, or whether cards were spent productively.

Impact: the reported neutral `phaware_search_ca` result is evidence about this exact hand-count probe. It should not be read as evidence that richer card-advantage or tutor-first sequencing is exhausted.

Recommendation: keep `evaluate_ca` as a narrow A/B probe. For the next objective, use existing `features.py`, `search_v3.option_deltas`, or `action_semantics_v1.py` to score contextual card advantage: draw/tutor availability, hand differential, deck risk, payoff cards, and productive hand spending.

## Notes

- Verified: the new opponent meta builder samples both seats from every replay file in the replay directory. This is fine if that directory is a pure external replay corpus. If self-play or submitted-agent replays are mixed in, the file is a both-seat deck distribution rather than a clean opponent-only prior.
- Verified: the local tests do not cover `leaf_mode="ca"`, `leaf_mode="deck"`, or the new `opp_decks` arguments directly.
