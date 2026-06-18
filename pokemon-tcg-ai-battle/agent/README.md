# agent/

The first-attempt agent for the cabt (Card Battle) engine, plus the test harnesses.

## What this is

A robust, always-legal heuristic agent that runs against the real organizer engine. It is
the deliberate first attempt from `../docs/STRATEGY.md`: an agent that always plays a legal
move and never times out, before any search or learning. Correctness first.

## Files

| File | Role |
| --- | --- |
| `main.py` | three agents: `agent` (heuristic), `agent_search` (forward-model search, hand eval), `agent_search_v` (search with the learned value). The submission entry point is the LAST module-level callable (see Submission packaging). Deck embedded. |
| `features.py` | L1 state encoder: `encode_state(obs)` -> 47 named features (needs `card_features.json`) |
| `eval.py` | leaf evals: `evaluate`/`evaluate_obs` (hand), `evaluate_learned` (learned P(win)) |
| `value_model.py` | L2 learned value (gradient-boosted tree), pure-numpy inference from `value_weights.json` |
| `search.py` | L3 1-ply forward-model search (cg.api), determinization-averaged, aggressive rollout |
| `datagen.py` | self-play logger -> `(features, outcome)` rows for training the value |
| `cabt_arena.py` | runs real cabt matches between two agents and reports a win rate |
| `explore_cabt.py` | probe that dumps the real observation schema |
| `ptcg_mock.py`, `policies.py`, `selfplay.py` | the legacy mock environment + its policies/runner (NOT the real game) |

## Measured results (real cabt engine, `local-sim` provenance)

```
python cabt_arena.py --games 200 --a heuristic --b random
```

| Matchup (200 games, seats swapped) | Win rate for A |
| --- | --- |
| heuristic vs random_agent | 0.835 |
| first_agent vs random_agent | 0.830 |
| heuristic vs first_agent | 0.515 (within the n=200 noise band) |

The honest reading: the agent beats random decisively (0.835), but the "defer the
end-turn option" heuristic adds no measurable edge over `first_agent`'s take-the-first
rule. The real jump is consistency over random. This is recorded in the registry: H003
supported (beats random), H016 refuted (the deferral idea does not beat take-first). The
next lever that would actually beat `first_agent` is a board-aware evaluation reading
`current.players` (HP, prizes, attached energy), which is not in this first pass.

About 0.15 to 0.19 s per real match on CPU.

## What the engine looks like (verified by reading it and running it)

- The agent gets `obs` with `select`, `current` (a Struct with `players`, `turn`,
  `result`, `yourIndex`, ...), `logs`, `remainingOverageTime`. During deck selection
  `select` is `None` and you return a 60-card list; during play `select` is
  `{"option": [...typed action dicts...], "maxCount": k}` and you return up to k indices.
- Only legal options are offered, so any subset of indices is legal. That is why an
  always-legal floor is cheap.
- An in-match forward model IS available and runs locally (CORRECTS the earlier claim here):
  `cg/api.py` exposes `search_begin`/`search_step`/`search_end` on the bundled `cg.dll`/`libcg.so`,
  and the cabt obs carries `search_begin_input`. The round-trip was verified end-to-end with no
  reentrancy crash across hundreds of arena games (registry H001 SUPPORTED). `search.py` uses it.

## The mock

`ptcg_mock.py` predates confirming the real engine was installable. It is kept as a fast,
dependency-free unit-test fixture for policy logic (heuristic beats random 0.95 there).
Its numbers have provenance `local-sim-MOCK` and say nothing about the real game. Prefer
`cabt_arena.py` for any real measurement.

## Submission packaging (use `../tools/build_submission.sh <variant>`)

`tools/build_submission.sh search|combine|search_v|agent` assembles a self-contained tarball
under `../submissions/` (gitignored) and self-verifies it. The package contains: a thin `main.py`
entry, `agent_impl.py` (= this main.py), `search.py eval.py value_model.py features.py`, the data
`card_features.json value_weights.json card_stats.json attack_stats.json`, and the `cg/`
forward-model engine. The agent degrades to the heuristic if `cg`/time fails, so it never forfeits.

HARD RULES (a violation cost a failed validation episode on 2026-06-17):
- The submission `main.py` is loaded by Kaggle via `exec(code_object, env)` with NO `__file__` in
  the namespace. DO NOT reference `__file__`/`os.path.abspath(__file__)` at module scope in
  `main.py`. (Imported modules like `features.py` DO have `__file__`, so their path handling is
  fine; only the exec'd entry file does not.) The bug: `sys.path.insert(0, os.path.dirname(
  os.path.abspath(__file__)))` raised `NameError: name '__file__' is not defined` -> ERROR status.
- ENTRY POINT: Kaggle runs the LAST module-level callable. The submission `main.py` defines a
  single `agent` so there is no ambiguity (the repo `agent/main.py` defines four agents; do not
  submit it directly).
- ALWAYS verify with `tools/verify_submission.py` (build_submission.sh runs it): it `exec`s
  `main.py` without `__file__` exactly like Kaggle and plays games. A normal `import main` test
  does NOT catch the `__file__` bug.
Do not submit without the user's say-so (`../AGENTS.md` section 7).
