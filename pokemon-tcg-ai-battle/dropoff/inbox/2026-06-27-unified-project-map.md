# Unified Project Map — Pokemon TCG AI Research Program

Date: 2026-06-27
Author: orchestrator pass (4 parallel readers over the research docs, the kaggle-fun origin folder, the dedicated repo + worktree, and git topology), synthesized.
Purpose: one reconciled map of the messy multi-home project, written before commissioning a deep-research audit of the Context-Delta Replay Transplant idea.

---

## A. North-star

The end goal is not a Pokemon agent. It is a generalizable, fail-closed methodology for estimating the value of a candidate intervention from a large corpus of logged, confounded, partial-information decisions: observational causal inference / case-based value estimation under leakage and distribution shift, with explicit support-gating and abstention. Pokemon TCG AI Battle is the high-variance testbed and the contractual deliverable; the transferable substance (residualize against a local baseline, weight analogous past situations, control same-game/same-pilot leakage, abstain under low support/OOD) is the toolkit aimed at a quant-research pivot (finance link is in user memory, not in the source docs).

Three coupled objectives govern the program:
- S = stable, legal, reproducible Simulation-category agent on a heuristic/search floor.
- R = scientific decision support (the transplant / counterfactual research line).
- W = a 2000-word Strategy-category writeup, scored Model 70% / Deck 20% / Report 10%.

Standing honest bottom line (quoted): "Learned live gameplay improvement is not yet established. The active blocker is not legality or model loading; it is useful arbitration under distribution shift."

---

## B. Relevant-repo map (the dirty mapping, reconciled)

| Path | What it is | Git status / branch | Role | State |
|---|---|---|---|---|
| `…\GithubRepos\kaggle-fun` | Monorepo for 4 competitions (pokemon-tcg, umud-muscle, ai-agent-security, predicting-stellar-class) | branch `exp/starmie-tactical-leaf-v1`, 45 ahead / 0 behind main; main HEAD `6c38e58` (2026-06-25) | integration trunk; only pokemon-tcg has active selector work | Active |
| `…\kaggle-fun\pokemon-tcg-ai-battle` | The ORIGIN folder: full working tree (agent, tools, registry, dropoff/inbox, docs) | tracked subfolder, no nested `.git`; lives on the branch above | HOME #1 — the live/source tree; the V1→V5 selector+transplant arc runs here | Active |
| `…\GithubRepos\pokemon-ai-agent` (main) | Dedicated clean-rebuild repo | single commit `8879c4d "Initial commit"`; `main.py` hello-world stub, README 2-line stub | HOME #2 — intended clean home, effectively empty at the git layer | Archive/stub |
| `…\.codex\worktrees\0557\pokemon-ai-agent` | Worktree of the dedicated repo | on `exp/c8-t0-quality-lab`, no commits of its own (points at `8879c4d`); ALL real work untracked/uncommitted | HOME #3 — where Model A's heavy offline labs actually run (transition-ranker spine, C4–C8 labs, transplant V0/V4/V5) | Active but uncommitted |
| `…\Downloads\research stuff` | 7 governing design/spec docs (v3.1 Master Reference, Replay-Transplant Note v1.1, the audit brief "1. Inputs", v0.2 HTML ledger, v2.x roadmaps, search addendum) | not a git repo; loose files dated to 2026-06-27 | source of truth for objectives S/R/W, C-stage map, claim grammar, M0–M8 / A–F transplant menu, the audit commission | Active reference |

Homes #1/#2/#3 are the SAME logical Pokemon project in three locations. #1 is the live source tree; #2 is an aspirational clean rebuild whose source snapshot is named in `pokemon-ai-agent\docs\REFACTOR_CONTROL.md` as `kaggle-fun\pokemon-tcg-ai-battle @ main 40d8e11`; #3 is a worktree of #2 where the offline V4/V5 labs live. The replay corpus (4,501 JSON files) is NOT duplicated into #2/#3 — tools resolve it from the Desktop repo's `data/external/replays`.

Other repos under GithubRepos (finance-research, mechanistic-model-inference, stable-grn-inference, structured-transform-discovery, etc.) were NOT connected to this project by any reader. The quant/finance bridge exists only in user memory.

---

## C. History narrative (chronological)

1. Heuristic/search era (origin folder). Engine is the organizer's `cabt` via `cg/api.py` (registry H001: clonable determinized forward model). Built a layered always-legal crash-safe agent: heuristic floor → 1-ply forward-model search (`search.py` → `search_v3.py`) → learned value leaf. LB outcome (`docs\SUBMISSIONS.md`): heuristic 770.9 (BEST, DENPA92 Alakazam) >> search 640–688 >> combine 422 (DEAD). Every search/learned addition scored below the plain heuristic. Anti-rehash `registry/` (24 hypotheses, 22 live, 2 graveyard) dates from here.

2. Starmie specialist + imitation pivot. Focus moved to the Cinderace/Mega-Starmie ex deck (`agent\starmie_heuristics.py`, ~52KB). Public-info-only extractor `starmie_tactical_state.py` (RACE/SWEEP/WALL/VALUE/COMMITMENT). main lineage build = `sub_starmie2` (post-audit, 6 confirmed bug fixes).

3. Selector arc V1→V5 — all on `exp/starmie-tactical-leaf-v1` (45 commits, `0ffbc1d`..`d1b07a7`), a Model-A-builds / Model-B-validates loop. Vendored bundles in `agent\vendor\portable_selector_v{1,2,3}\`, gated by `STARMIE_SELECTOR_MODE` (default off, fail-closed, never overrides into terminal ATTACK/END/RETREAT):
   - Pre-selector: tactical-leaf ATTACKER_CONTINUITY → `C_BASELINE_LEAF_PREFERRED` (n=240 refuted an n=120 directional read; early-stopping pattern #1).
   - V1 (`S5_LIGHTWEIGHT_SELECTOR`): parity 220/220 bit-exact; live 300-game smoke `top3_selector` REGRESSIVE −35pp vs mirror → `DO_NOT_PROMOTE`, `B_FAILURE_MODE_DIRECTIONAL` (the "V1 mirror crash").
   - V2 / C3 (`c3_family_limited`, blocks ATTACK/END/RETREAT): safe, key cells flat → `B_C3_SELECTOR_SMOKE_NEUTRAL` / `NEEDS_N500`. Selected policy `T1_C3_PLUS_TRANSPLANT_SCORE`; rejected `T4_AXIS_MASK_SELECTOR`.
   - V3 transplant (`portable_selector_v3` + `transplant_support_table.json`, state-blind T(a) keyed `FAMILY||COMPACT_SEMANTIC_KEY`): first smoke `D_V3_SELECTOR_UNSAFE_OR_INVALID` (INERT — runtime queried a STRUCTURED key vs COMPACT table → 0 overrides); re-vendored canonical `FAMILY||COMPACT` key (`a1d4f02`) → non-inert; repaired smoke `B_REPAIRED_V3_SMOKE_NEUTRAL` / `NEEDS_N500` (`d6bbdc8`); powered N500 A/B `B_V3_POWERED_NEUTRAL` / `PARK_V3` (`c0de28e`) — deployed+mirror +2.6pp, p=0.263, 95% CI [−1.8,+7.0], below ~6.3pp MDE; clean early-stopping demo (effect shrank +15 → +11.7 → +7.5 (p=0.040 false positive) → +2.6 n.s.); 11,599 overrides 100% table-backed, 0 terminal, over 1,700 games.
   - V4 state-conditioned T(s,a) (offline, in the 0557 worktree, not a commit on any kaggle-fun branch): `STATE_CONDITIONED_TRANSPLANT_DIRECTIONAL_ONLY`, best `B6_SELECTOR_FEATURE_ONLY`, export earned false (eq-class top-1 0.7362 but exact/MRR slightly worse).
   - V5 context-delta T(s,a,delta): HEAD `d1b07a7` (2026-06-27 08:19) `VERDICT=A_V5_RUNTIME_FEATURES_FEASIBLE` (Model B proved the live runtime can compute (s,a,delta) from public obs without a full rollout: 129 golden fixtures, ~3ms each, within the 0.6s budget). DISAGREEMENT/OVERLOAD: the worktree also holds `data\generated\starmie_transplant_v5` with verdict `T_ACTION_BASELINE_REMAINS_BEST` / `NO_V5_EXPORT_REASON.md` — an offline V5 retrieval trial that UNDERPERFORMED repaired V3. "V5" therefore names two things: Model B runtime feasibility (positive) and Model A offline retrieval (negative-so-far). Not contradictory, but easily conflated.

4. Drift into the dedicated repo / worktree. A clean rebuild (`pokemon-ai-agent`) was started but never populated at the git layer (main = hello-world stub). Real heavy offline work accumulated uncommitted in the 0557 worktree under branch `exp/c8-t0-quality-lab` (which has itself moved past C8 into the transplant arc). Cross-model handoff there runs through `audit_mailbox/inbox/`; in the live tree it runs through `dropoff/inbox/` (34+ notes).

---

## D. The idea under audit — Context-Delta Replay Transplant, T(s,a,delta) not T(a)

- Core distinction. T(s,a) is the INTENDED object ("state-conditioned replay evidence for candidate action a in current state s"). T(a) is the SHIPPED approximation: an action-type support table keyed `FAMILY||COMPACT_SEMANTIC_KEY`, so two different boards with the same key get the same signal. T(a) loses state applicability; the powered-neutral V3 result is the empirical proof a state-blind table is safe-but-neutral.
- Query object. q = (s, a, delta_a) where delta_a = phi(s_after_a) − phi(s). Memory rows m_i = (s_i, a_i, delta_i, z_i, c_i).
- Full transplant object. T(s,a) = (u_hat, c_hat, n_eff, rho_contradict, OOD, explanation); n_eff = (Σw_i)² / Σw_i².
- Kernels / similarity. K_op (operator/family), K_ctx (context), K_delta (feature delta), K_cross (applicability), K_support. Product form w_i = K_op·K_ctx·K_delta·K_support (any near-zero compatibility term collapses the evidence); w_i = exp(−d_phi/tau); u_hat = Σw_i y_i / Σw_i.
- Estimator ladder. M0 current T(a) → M1 T(a,family) → M2 T(a,coarse_bucket) → M3 context-only NN → M4 delta-only NN → M5 context+delta NN → M6 learned metric → M7 support-gated context+delta with abstain → M8 value-model advantage target. (Master Reference parallel menu A–F: A global = negative control, B family-conditioned = dominant lever, E support-gated = necessary, F = current V3.)
- Target + gating. Raw terminal y_i = z_i is too crude (winner's action ≠ good action). Preferred target = advantage A(s,a) = Q − V, approximated A(s_i,a_i) ≈ V(s_{i+1}) − V(s_i); fallback = vector of short-horizon consequences. Abstain when n_eff low, rho_contradict high, or OOD.
- Falsification target. FALSE today: "axis-delta weighting is clearly superior" — real-data micro-audit `transplant_delta_real_micro_v0` (`D_REAL_AXIS_DELTA_UNDERPOWERED`) gave AUC 0.5887 vs V0 0.5938 (−0.0043, n=500 decisions / 1,892 candidates). NOT established: that live V3 ever tested the full idea (it tested T(a), not T(s,a)), or that transplant improves win rate. NOT supported: "transplant should be abandoned" (negatives scoped to the tested formulation via FORM-NEG).
- Leakage core. Same-game continuation is often the nearest neighbor; retrieval must run inclusive / leave-one-decision / -episode / -pilot / -deck-out, reporting same-episode/pilot/deck neighbor fractions. "A transplant result that only works under inclusive retrieval is not evidence of generalization."

---

## E. Current work / active frontier (2026-06-27)

- Live state: V3 is PARKED as a safe-but-neutral baseline; selector stays default off; nothing from the selector/transplant arc has ever been submitted (best LB remains the 770.9 heuristic). V5 just established runtime feasibility.
- Docs' stated next correct step (quoted): not to tune V3 or run more V3 games, but a leakage-controlled context-delta transplant prototype comparing T(a) vs T(a,bucket) vs T(s,a) vs T(s,a,delta) offline before any live integration.
- Model A (build lane) runs in the 0557 worktree. Target namespace `data\generated\replay_delta_outcome_transplant_v0\` does NOT yet exist (net-new; nearest neighbor is `transplant_delta_real_micro_v0\`). Would build the V5 context-delta transplant, compare T(a)/T(a,bucket)/T(s,a)/T(s,a,delta) with leakage-controlled retrieval, and not export live unless offline evidence clearly beats V3/C3. Touches only `src/pokemon_ai_agent/{transplant_ranker,signals,features}` + new data/generated output; does not modify Selector V3, the portable runtime, live agent, Model B code, or weights.
- Model B (validate/stabilize lane) runs in the live tree (`exp/starmie-tactical-leaf-v1`). Standing instruction: do not run more V3 by default; support Model A only if V5 requires runtime feasibility / trace extraction — which HEAD `d1b07a7` (`A_V5_RUNTIME_FEATURES_FEASIBLE`) just fulfilled. Stabilization/merge-readiness is the real blocker for the dedicated repo: populate `pokemon-ai-agent\main.py` (hello-world stub, flagged P1 by the 0557 auditor) into a submit-ready entry point and fix `safe_agent` passing illegal successful selections through.
- Runtime gotchas Model A must respect (V5 feasibility pack): live `entity['damage']` is absent (use max(0, maxHp−hp)); unify compact-vs-structured key (the exact bug that made V3 inert); `board_hp_delta` is opp-active-only; standardize energy on UNITS (Ignition=3).

---

## F. Messiness / risks

- Three homes for one project; the worktree's real work is untracked/uncommitted (one `git clean` / worktree-remove from loss — the "preserve gitignored before worktree removal" rule applies directly).
- Empty dedicated main + label-only branches: `pokemon-ai-agent` main and the worktree's `exp/c8-t0-quality-lab` both point at the same initial commit; the branch name no longer matches the work.
- Multiple kaggle-fun exp branches, most stale: `exp/starmie-heuristic-lab-v3` (1 ahead/1 behind), `exp/starmie-heuristic-audit-v2` (became sub_starmie2), `exp/robust-learner-v2` (parked), `exp/planner-teacher-v2`, `claude/hopeful-spence-af362c` (latest work is actually UMUD, cross-competition). Only `exp/starmie-tactical-leaf-v1` is live.
- "V5" overloaded: runtime-feasibility POSITIVE vs offline retrieval NEGATIVE-so-far.
- Artifacts that must not be promoted: selector V1–V5, combine/learned leaf (422, DEAD), `STARMIE_ATTACK_FLOOR` (reverted), `STARMIE_LEAF_ATTACKER_CONTINUITY` (documented dead-end). All default-off scaffolding.
- Verdict tracking split: newer Starmie/selector/transplant arc lives in `dropoff/inbox/` notes and `data\generated\...\VERDICT.json`, NOT in `registry/` JSONL, so the registry undercounts current work.
- Minimal cleanup: (1) decide ONE canonical home; (2) preserve gitignored data/generated before any worktree removal; (3) commit/stage the worktree so "current code" is knowable; (4) one-line README in `pokemon-ai-agent` main ("scaffold only, real work in worktree/kaggle-fun"); (5) prune/annotate stale exp branches; (6) top-level note disambiguating the two "V5" verdicts.

---

## G. Open questions to confirm before deep research

1. Which home is canonical going forward (keep iterating in kaggle-fun, or finish migrating to pokemon-ai-agent)?
2. Scope of the deep research: bounded OFFLINE transplant audit (M0–M8 / A–F with leakage controls, no live integration), or also the stabilization/merge-readiness path to a real submission?
3. Framing the auditor adopts: single recommended framing, or compare local-treatment-effect vs case-based value vs offline RL vs metric learning?
4. The quant/finance bridge: map the transplant machinery to observational-causal/quant methods explicitly, or stay strictly inside the Pokemon framing?
5. N500 standard of proof for any new offline winner before a live A/B, or is offline-only evidence acceptable for the writeup (W)?
6. V5 disambiguation: is deep research meant to FIND a formulation that beats V3, or to ESTABLISH the boundary that state-conditioning does not help here?
