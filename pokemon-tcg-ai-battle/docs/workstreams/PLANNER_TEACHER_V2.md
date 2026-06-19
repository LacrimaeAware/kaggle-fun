# Branch A — Planner / Teacher V2

Worktree `planner-teacher-v2`, branch `exp/planner-teacher-v2`, forked from the SPLIT_BASE_V2 commit
`2f29e93` (codex Branch B's exact base). `data/` is junctioned to the main checkout so the engine and
the gitignored replays resolve; the frozen snapshot is consumed by manifest hash, not the live corpus.

Mission (unchanged): a stronger OFFLINE counterfactual teacher than the live N=8 search, and secondarily
a stronger live agent under the real budget. This file is the Branch A record. No merge to `main` during
development.

## A1 — reproduce + instrument Teacher V1 (done)

Teacher V1 (`agent/teacher_api_v1.py`) reproduces the deployed `agent_search` (forced-move floor then
1-ply hand-eval search, `opp_prior=None`, `opp_k=0`) and already emits per option: mean value, value
variance, completed determinizations, normalized advantage; per decision: top-two margin, soft policy,
acceptable-action set, forced flag, config hash. Engine + frozen baseline confirmed runnable in the
worktree. No change required.

## A2 — teacher stability audit (done; primary source)

Tool: `tools/audit_teacher_stability.py`. Samples non-forced single-pick decisions from the FROZEN
SPLIT_BASE_V2 snapshot `replays_20260618` (each replay's sha256 verified against the manifest), deck-
stratified. Each decision is queried two ways:

- **cross-seed** (16 different determinization seeds): varies the hidden-world draw + engine rollout RNG;
- **same-seed** (16 repeats of one seed): isolates engine rollout RNG only (the determinization draw is
  fixed; coins/shuffles inside `cg.dll` are not Python-seedable).

The gap between them attributes the instability to its source.

### Result (n=1094 decisions, 35,200 queries, ~47 min; pilot n=197 agreed on the key finding)

| metric | value |
|---|---|
| stable (top action ≥90% of seeds) | **50.5%** |
| near-tie (60–90%) | 21.2% |
| unstable (≤60%) | 28.2% |
| mean cross-seed top-action stability | 0.783 |
| mean engine-only (same-seed) stability | 0.772 |
| **determinization extra instability** | **−0.011 (≈ 0)** |
| mean top-two margin | 2722 (hand-eval scale) |
| mean acceptable-set size | 3.10 options |

Stability by action type: ability **0.678** (lowest), attach 0.737, select-card 0.756, play 0.780,
retreat 0.801, end 0.807, evolve 0.846, attack **0.849** (highest).

### The two findings (robust across the pilot and the full run)

1. **About half of non-forced strategic decisions are not cleanly decidable.** 50% stable, ~50%
   unstable or near-tie, with ~3 statistically-indistinguishable options on average. A single hard top-1
   teacher label is roughly half noise on the non-forced set.

2. **The instability is engine-rollout-RNG-dominated, not determinization-dominated.** Fixing the hidden-
   world draw (same seed) leaves stability essentially unchanged (0.772 vs 0.783; extra instability
   −0.011, and −0.019 in the pilot). The wobble is the engine's internal coin flips / shuffle effects in
   the rollout, not which world we sampled.

### What this earns for A3 (from data, not assumption)

- World-sampling levers will not fix teacher stability: more determinizations purely for world coverage,
  **shared worlds (A3.2)**, and **belief-prior sampling (A3.5)** target a noise source that is already
  negligible at N=8. This also cleanly explains why opponent-belief was parked.
- The levers that bite are **averaging more noisy rollouts** (higher N reduces engine-noise variance) and
  **selective computation (A3.3)** — spend the extra budget on the high-variance / near-tie / low-margin
  decisions, not uniformly. Candidate A3 first move: variance-triggered adaptive N, measured against V1.
- This updates the earlier opponent-sensitivity hunch: the world/belief-sampling half looks weak for
  stability; the selective-computation half is supported.

### Honest caveats

- This measures **stability/noise**, not strength. Whether opponent *information* would change the best
  move (belief-as-strength) is a separate question still gated on an opponent-sensitive leaf; the audit
  does not speak to it. Do not read "belief sampling won't reduce noise" as "opponent modelling is dead."
- The stable fraction firmed from 35% (pilot, 9 decks) to 50% (full, more decks); treat exact fractions
  as the full-run estimate, the engine-dominated decomposition as the robust qualitative result.
- Sources covered: top-player replay states across decks. Not yet covered (next A2 increment, needs state
  generation): production-search self-play states and old-ranker arena-failure states.

## For Branch B

`data/manifests/teacher_v1_stability_full.jsonl` — one record per audited decision with a stability class
(stable / near-tie / unstable), a **confidence weight** (= cross-seed top-action stability), and
**averaged soft-policy + advantage targets**. This is the down-weighting substrate for B1.2: prefer the
soft targets, weight by confidence, and exclude/curtail the ~half of decisions that are unstable or
near-ties. (B can also reproduce directly by querying Teacher V1; the records are a convenience + the
stability partition.) Share path: cherry-pick from this branch or via dropoff.

## Status

A2 primary source complete and recorded. **Auditor / review checkpoint here before any A3 work** — no
Teacher V2 candidate built yet, no search sweep rerun. Next, on sign-off: add the self-play and
ranker-failure A2 sources, then A3.3 variance-triggered selective computation as the first V2 candidate,
each gated against V1.
