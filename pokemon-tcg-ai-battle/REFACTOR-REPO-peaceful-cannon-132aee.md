# REFACTOR-REPO-peaceful-cannon-132aee

**Master orientation for the next agent.** Codename `peaceful-cannon-132aee` = the worktree this was written from
(branch `exp/starmie-tactical-leaf-v1`, kaggle-fun/pokemon-tcg-ai-battle). Written 2026-06-27 after a long
two-lane (Model A research / Model B runtime) push on the Pokemon TCG AI Battle competition. The user is about to
**merge several repos + scattered work into one big refactor** and wants a single file that explains the whole
structure and points to the next step. Read this first; then read the dropoff/inbox index (section 9).

---

## 0. TL;DR (read this, then skim the rest)
- **Competition:** Kaggle "Pokemon TCG AI Battle", Strategy track. Engine = `kaggle_environments` `make("cabt")`
  + a native `cg` engine (search via `cg.api.search_begin/search_step`).
- **What actually ships today:** a hand-written **heuristic + 1-ply forward-search** agent. The deployed entry is
  `agent/main.py::agent_starmie` (Starmie deck) / the Alakazam heuristic line. It is a known ~770-LB-class agent.
- **What does NOT ship:** every learned/selector/transplant experiment. They are all **default-OFF and not even
  on the submission path** (see section 3). Nothing learned has beaten the heuristic floor in a powered test.
- **Current honest standing:** heuristic floor is the product; the learned selector is parked (powered-neutral);
  transplant is parked (theory-only); the next mainline step is the **C8 F1 energy-allocation tactic lab**, with a
  **trustworthy local eval harness** now built to judge it.
- **The merge/refactor goal:** consolidate 3+ scattered homes (section 1) into one repo with a clean
  live-agent / research-tools / eval-harness / artifacts separation (section 8).

---

## 1. The three+ homes (cross-repo map) — what the refactor must merge
Per the project's own history there is **ONE project living in several places**:
1. **`kaggle-fun/pokemon-tcg-ai-battle` (THIS repo, the live home).** All Model B runtime/eval/agent work + the
   committed history. Branch `exp/starmie-tactical-leaf-v1`. This is the source of truth for shippable code.
2. **`.codex/worktrees/0557/pokemon-ai-agent` (Model A's offline labs).** Feature/proposer/transplant *modeling*
   work. **Largely UNCOMMITTED** — biggest refactor risk: do not lose it. Its outputs reach this repo only as
   vendored exports (`agent/vendor/portable_selector_v*`) + verdicts quoted in dropoff handoffs.
3. **`pokemon-ai-agent` (dedicated repo) — an empty/near-empty stub.** Intended permanent home; currently unused.
4. **`Downloads/research stuff`** — notebooks, deck lists, strategy docs (some mirrored into `research/`).
5. Adjacent but separate: `mastery-lab` (learning-module system), `Gojomons` (a different game/RL repo). Not part
   of this competition; leave out of the merge unless the user says otherwise.

**Two-lane convention (important):** prompts arrive in PAIRS. **Model A** = the offline modeling agent (Codex, the
.codex worktree). **Model B** = this assistant (runtime/eval/data in this repo). A third **Model C** theory lane
holds the transplant research. Model B never edits Model A's repo and vice-versa; they exchange via vendored
exports + dropoff handoffs.

---

## 2. This repo's structure (what lives where)
- **`agent/`** (31 py) — the live agent + all runtime helpers.
  - `main.py` — submission entry. Functions: `agent_starmie` (deployed Starmie pilot: `deck_policy_v3.best_ko_attack`
    then `search_v3.best_option`), plus `agent`, `agent_search*`, `agent_rank*`, `agent_eff`, etc. **`main.py` does
    NOT import the selector or call `choose_action`.**
  - `starmie_heuristics.py` — the heuristic policy + `choose_action` (which hosts the default-OFF selector wiring:
    `_baseline_pick` + `_selector_override`, `STARMIE_SELECTOR_MODE` env, `selector_trace`). `_attach_score`,
    `_energy_units`, KO/attack helpers live here.
  - `search_v3.py` — 1-ply forward search (`best_option`, `option_deltas` = the engine one-step apply), `eval.py`,
    `features.py`, `deck_policy_v3.py` (option/card resolution, `attack_profile`, `best_ko_attack`),
    `starmie_tactical_state.py` (`semantic_role`, `energy_units`, `_retreat_cost`, `_VERIFIED` attack costs,
    `affordable_attacks`, `entity_features`).
  - `learned_selector_bridge.py` (obs→Feature-V2 payload + `tactical_state_features`/`option_features`),
    `learned_proposer_adapter.py` (compact semantic keys; DISABLED), `turn_context_v0.py` (read-only temporal
    extractor; NOT wired), `cabt_arena.py`.
  - `vendor/portable_selector_v{1,2,3}/` — Model A's exported runtimes + packers + `transplant_support_table.json`.
    Inert unless a selector mode is set.
  - data files: `attack_stats.json` (attackId→{d,c,n}), `card_stats.json`/`card_features.json`, `deck_policy`.
- **`tools/`** (167 py) — research/eval/diagnostic scripts (one-offs + reusable). The reusable crown jewels are in
  section 6. Naming: `*_smoke_*`, `*_diagnostic_*`, `validate_*_parity`, `local_meta_*`, `heuristic_rule_*`, etc.
- **`tests/`** (17 py) — `run_all.py` runs 14 modules as subprocesses (so one module's exit can't abort the rest).
  Suite is green. Engine-backed tests (V5 support, F1, local-meta) included.
- **`submissions/`** — 11 built agents + tarballs: `sub_archaludon` (Metal anti-Starmie tech, ~100-rule bot),
  `sub_starmie{,2,3}`, `sub_heuristic{,2}`, `sub_search`, `sub_planner`, `sub_phaware`, `sub_combine`. Each is a
  self-contained kaggle submission (own `cg/`, `deck.csv`).
- **`data/generated/`** — 19 artifact dirs (mostly gitignored; small manifests/verdicts tracked). One per workstream.
- **`dropoff/inbox/`** — 52 dated handoff docs = the journey log (section 9). Append-only.
- **`research/notebooks/`** — sample/opponent notebooks (Lucario, Abomasnow, Crustle, Dragapult, etc.).
- **`registry/decks.json`** — 5 archetype deck lists (D001 Abomasnow, D002 Koraidon, D003 Mega Lucario, D004/D005
  Abomasnow) used as eval opponents.

---

## 3. Safety posture — what is LIVE vs DEFAULT-OFF vs EXPERIMENTAL (from the stabilization audit)
Verified `NOT_PIPELINE_DIRTY`:
- **LIVE (ships):** `main.agent_starmie` → `deck_policy_v3` + `search_v3`. The heuristic + search floor.
- **DEFAULT-OFF, off the submission path:** the entire selector arc. `STARMIE_SELECTOR_MODE` defaults `off`;
  `_selector_override` returns the baseline unchanged when off; the V3 runtime + transplant table are loaded
  lazily (only when a selector mode is set) so `_SELECTOR_RT_V3` stays `'uninitialised'` by default. `main.py`
  references neither the selector env, the selector runtime, nor `choose_action`. The `ATTACH_MEGA_NOT_ENGINE_V1`
  toggle is also default-off and only affects `choose_action` (not the deployed path).
- **EXPERIMENTAL / keep-branch-only:** vendored `portable_selector_v2/v3` + transplant table, all smoke/powered-AB
  runners, transplant tools. **None earned promotion.**
- **Net:** nothing experimental can affect a submission unless someone both sets an env var AND routes the
  submission through `choose_action` (neither is true today).

---

## 4. The journey (chronological arcs, with verdicts) — "document the journey"
1. **Heuristic + search floor** established as the deployable product (the ~770 line). Recurring measured truth:
   **the heuristic floor beats search-piloted variants; piloting dominates deck.**
2. **Imitation-gap / tactical-leaf work** — replays vs top pilots showed "we attack too early" (develop-before-
   attack). Built tactical-state extractor + ATTACKER_CONTINUITY leaf term.
3. **Learned selector arc** (the big one):
   - **V1 top3_selector:** live-smoke REGRESSIVE (mirror −35pp) — over-disrupted turn structure.
   - **V2 C3 family-limited:** terminal-safe but **NEUTRAL** on key cells (field-driven aggregate, n=20).
   - **V3 transplant-aware:** first INERT (D — structured-vs-compact key mismatch), repaired (compact-key bridge),
     then **powered A/B (n=500/arm): B_V3_POWERED_NEUTRAL** — +2.6pp deployed+mirror, p=0.263, below MDE. The
     n=20 smoke's +15pp shrank to +2.6pp under power = textbook early-stopping bias. **PARK_V3.**
4. **Transplant research:** toy lab (B_AXIS_DELTA_DIRECTIONAL_ONLY — family-conditioning dominant, axis-delta
   second-order), support pack, **V5 state-conditioned** (Model A: `T_ACTION_BASELINE_REMAINS_BEST`). Model B's V5
   runtime-feasibility = FEASIBLE. Net: transplant is a research hypothesis, **parked from implementation**
   (Model C theory lane).
5. **Lane reset → roadmap (C8/C9/C10):** stop bouncing between abstractions. Deliverables this push:
   - **Project stabilization audit** (`A_STABLE_INFRA_READY_TO_MERGE_DEFAULT_OFF`).
   - **Local meta / eval harness V1** (`LOCAL_META_V1_READY`) — the trustworthy evaluation foundation.
   - **C8 F1 energy-allocation runtime prep** (`A_F1_ENERGY_RUNTIME_PREP_READY`).
   - **Heuristic rule reconstruction lab + gap audit** (feature-forensics on the Archaludon bot;
     `B_PARTIAL` → `E_MIXED`: no truly-missing public features).
6. **Side finding (Archaludon h2h sweep):** the Archaludon Metal deck hard-counters Starmie (~19-26% for us) but
   is a coinflip vs the Alakazam heuristic (~47-54%). It's a Starmie-counter, not an upgrade. Defensive signal:
   our Starmie builds fold to Metal.

---

## 5. Current standing per workstream (what's true right now)
- **Heuristic + search floor:** the product. Keep.
- **Selector (V1/V2/V3):** PARKED. Safe, non-inert, but powered-neutral. State-blind T(a) support table is the
  ceiling; real lever would be state-conditioned T(s,a,delta) — which is theory-only.
- **Transplant (V3/V5):** PARKED from implementation (Model C theory lane).
- **Eval harness (local_meta_v1):** READY and reusable — this is how every future tactic gets judged.
- **C8 F1 energy allocation:** runtime prep READY (extractor + feature contract). Waiting on Model A's offline F1
  lab to name the exact candidate mechanism; if it returns READY, the next step is one implementation prompt
  (default-off toggle + fixed-state tests + staged local_meta probe). If not, move to F7 tutor-targets / F3
  draw-deckout.
- **Rule-reconstruction diagnostic:** the public feature layer recovers key rule triggers (evolve-on-metal,
  card-identity picks); gaps are derivable-not-extracted (attack damage, routes) + model-framing (per-decision
  ranking) + trace-support (matchup) + one genuinely hidden (cross-turn opp tracking). Zero truly-missing public
  features.

---

## 6. Reusable infrastructure worth keeping through the refactor (the crown jewels)
- **`tools/local_meta_harness_v1.py`** — staged A/B runner with an extensible opponent roster (6 builtins +
  registry archetypes lucario/koraidon/abomasnow as field pilots), per-opponent counts, full per-game +
  per-decision logging.
- **`tools/local_meta_analyze_v1.py`** — THE trustworthy analyzer: primary deployed+mirror combined, Holm-corrected
  sentinels, field/neg reported separately (never decisive), Wilson CIs, Fisher, MDE, **early-stopping warning**,
  trigger diagnostics. Tested (`tests/test_local_meta_analyze_v1.py`).
- **`tools/selector_v3_powered_ab_v1.py` + `selector_v3_powered_diag_v1.py`** — the powered-A/B + diagnostic that
  produced the honest V3 verdict (reused pattern).
- **`tools/f1_attach_context_extractor_v1.py`** — read-only ATTACH-context extractor (role/energy/shortfall/
  threshold), faithful to the real helpers.
- **`agent/turn_context_v0.py`**, `learned_selector_bridge.py`, the parity validators (`validate_selector_v*_parity`),
  and the `run_all.py` subprocess harness.
- **Method patterns that repeatedly paid off:** adversarial-reviewer pass on every optimistic verdict; judge on
  key cells not field aggregate; two-phase trace→analyze to dodge module conflicts.

---

## 7. Recurring landmines / lessons (things that bit us — do not relearn the hard way)
1. **The deployed agent is `main.agent_starmie`, NOT `choose_action`.** Wiring a feature into `choose_action`
   does nothing for the submission. Know which path your change is on.
2. **Local self-play ≠ ladder.** Every win-rate number here is one matchup, ±7-10pp at n=50-100. Never promote
   on it; it's a safety/direction smoke.
3. **Early-stopping bias is real and large** (+15pp@n20 → +2.6pp@n1000). Use the analyzer's early-stopping guard;
   only the full pooled sample is a valid significance claim.
4. **"Neutral" = positive point estimate, underpowered / not established, do-not-promote** — NOT "zero effect".
5. **Naive implementations** are the recurring failure: the heuristic IDEA is usually right but impls are too
   naive/restrictive. Match impl to intent; validate every change; review scenarios yourself.
6. **Don't trust code-comment/agent citations** — verify against the real code (e.g. `SH._cs` didn't exist; the
   card-stats key is `n`/`type` not `ty`; golden fixtures are a non-Starmie deck). Run it, don't assert it.
7. **Windows concurrent `>>` to one file is not atomic** (the h2h sweep clobbered itself). One output file per
   worker/agent.
8. **Preserve gitignored data + UNCOMMITTED Model A labs before any worktree/branch surgery** — biggest refactor
   risk.

---

## 8. Recommended next steps + a refactor target structure
**Immediate next mainline step (if continuing the competition):** wait for / run Model A's C8 F1 lab; if it
returns a READY candidate, implement it as a default-off toggle and judge it through `local_meta_harness_v1` +
`local_meta_analyze_v1` (Stage A 60/cell → B 200 → C 500, primary deployed+mirror, sentinels lucario/koraidon/
abomasnow/denpa92, Holm-corrected). Promote only on a significant primary result above MDE.

**For the giant refactor itself, a suggested target layout (one repo):**
```
pokemon-tcg/
  agent/            # LIVE only: main.py (entry), heuristics, search, deck_policy, tactical_state, data jsons
  agent/vendor/     # frozen Model-A exports (default-off), clearly labeled experimental
  research/         # Model A offline labs (MERGE the .codex/0557 work here — commit it first!)
  eval/             # local_meta_harness + analyzer + parity validators (the crown jewels from tools/)
  tools/            # one-off diagnostics (archive the stale ones)
  submissions/      # built tarballs
  artifacts/        # data/generated (gitignored heavy data; keep verdict JSONs)
  docs/             # dropoff/inbox journey + roadmap + THIS file
  tests/            # run_all + modules
```
Refactor guardrails: (a) commit/copy the UNCOMMITTED Model A labs FIRST; (b) keep the live `agent/` import graph
free of selector/transplant code (preserve the default-off, off-submission-path posture verified in section 3);
(c) keep `run_all` green at each step; (d) don't delete `dropoff/inbox` — it's the journey.

---

## 9. Pointers
- **Journey log:** `dropoff/inbox/` (52 docs). Start with `2026-06-21-MASTER-HANDOFF-read-first.md`,
  `2026-06-25-roadmap.md`, `2026-06-27-unified-project-map.md`, then the dated 2026-06-27 verdicts
  (selector-v3-powered-ab, local-meta-v1, f1-energy-runtime-prep, heuristic-rule-gap-v1).
- **Run the suite:** `PYTHONIOENCODING=utf-8 .venv/Scripts/python tests/run_all.py` (expect ALL SUITES PASS).
- **Run a tactic A/B:** see section 6 + the local_meta command template in
  `dropoff/inbox/2026-06-27-f1-energy-runtime-prep-v0.md`.
- **Use the project `.venv`** for anything kaggle/engine. Branch `exp/starmie-tactical-leaf-v1`; nothing merged to
  `main` yet; no Co-Authored-By trailer in this repo.
- **Verdicts at a glance:** selector PARKED (powered-neutral) · transplant PARKED (theory) · eval harness READY ·
  F1 prep READY · heuristic floor = the product.
