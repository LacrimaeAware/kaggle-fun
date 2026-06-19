# Round-2 risk-label enrichment for Model B (search-selected-high-regret + false-positive calibration)

Model B's risk-only model failed two ways: it **missed a catastrophic search-selected high-regret state**
and it **false-positive-blocked a safe search pick**. B requested targeted enrichment
(`teacher_v2_risk_label_request_for_A.json`): the 2 seed failure states plus 40-80 analogs across three
criteria, prioritizing **c1 = states where `agent_search` itself selects a `high_regret` option**.

This **supersedes the first pass** (`teacher_v2_risk_labels_for_B_request.jsonl`, 51 labels, only 1 c1, 3
failures): the round-2 batch mines for the rare c1 class, has 0 failures, adds group/provenance fields, and
was independently verified. Both filenames now hold the round-2 content.

Branch `exp/planner-teacher-v2`. `agent_search` stays the live baseline; no live agent consumed anything; no
arena screen; `main` untouched.

## Delivered

- **Requested:** 40-80. **Labeled: 60.** **Failed / unrecoverable: 0.**
- Both seeds present and **obs_hash-verified** against their source replay step.
- All 19 of B's `labeling_requirements` present on every record (independently audited), **including `timing`**.
- Artifact (B's requested name): `data/manifests/teacher_v2_residual_risk_labels_round2.jsonl`
  (also copied to the coordinator's name `teacher_v2_risk_labels_for_B_request.jsonl`).
- Summary: `data/manifests/teacher_v2_residual_risk_labels_round2_summary.json`.
- Tools: `agent/teacher_api_v2.residual_risk_label`, `tools/label_risk_round2.py` (seed recovery + criterion
  screen + c1 ingest), `tools/mine_c1.py` (shardable c1 miner), `tools/finalize_round2.py` (post-process fields).

| class | count | what it teaches B |
|---|---|---|
| c1 search-selected-high-regret (this label) | **9** (across 5 games) | the missed-catastrophic head |
| &nbsp;&nbsp;of which seed1 (`80251230.json:12`) | 1 | B's exact canonical miss |
| c1 *candidates* (mined+seed) | 16 | provenance pool (see reproducibility below) |
| c2 safe-search false-positive (unacceptable siblings) | **49** (across 11 games) | the false-positive head |
| &nbsp;&nbsp;incl. seed2 (`80252701.json:56`) | 1 | B's exact false positive |
| c3 near-miss boundary | 21 | hard-boundary regularization |
| high_regret **options** | **127 / 588** | risk-head positive supervision |
| unacceptable **options** | 404 / 588 | sibling-danger supervision |

Densification vs round-1 (`teacher_v2_residual_risk_labels.jsonl`, 50/451): high_regret options **16 -> 127**;
search-selected-high-regret states **1 -> 9**; c2 false-positive calibration rich (49) across 11 games.

## The two findings that should change how B uses this

**1. c1 is intrinsically rare and lives only in the top criticality tier.** A 12-shard parallel mine screened
**3,224** criticality-sorted high-criticality states. It found **16** c1 candidates -- **9 in the top ~1,100 by
criticality, and ZERO below rank ~1,350**. So "search itself picks a catastrophic option" is a top-tier-only,
~0.5% event. The seed is a genuine rare outlier, not the tip of a large iceberg.

**2. The c1 label is only ~53% reproducible -- the instability is in which option search selects, not the
threshold.** Of 15 mined c1 candidates, **only 8 reproduced** the selected-high-regret property on a fresh
label (+seed1 = 9/16, ~56%). The seed's own selected option moved **0 -> 1 -> 3 across three independent
labels** (always high_regret). The engine rollout RNG is not seedable; on a knife-edge state the live N=8
search's pick flips run-to-run. Consequently the **regret MAGNITUDE is noise** (seed1 swings ~31k vs ~94k;
selected-option `value_se` ~88k is ~3x the regret it implies), while the **high_regret FLAG is the stable-ish
signal** (all 127 high_regret options have regret > 30k, none in the 5k-30k band, so the 5000 threshold is
uncritical -- though that cleanness is hand-eval lethal-scale quantization, not low noise).

## Fields added for B (post-process, no relabeling)

Per the independent adversarial review, each record also carries:

- `group_id` (= source replay file = **game**) -- **use this for a group-held-out split.** c1 positives
  cluster: the 9 reproduced c1 span 5 games but 2 games (`80270516`, `80279946`) carry 6 of 9, and within a
  game consecutive steps are near-duplicates (identical csv/sv/se). A non-grouped split would leak.
- `c1_candidate` / `c1_reproduced_this_label` -- a `c1_candidate & !c1_reproduced_this_label` row is one of the
  **7 "dead"/flipped** mined positives (looked c1 at mine time, relabeled to regret=0). **Flag or down-weight
  these**; do not feed them as clean negatives for the missed-catastrophic head.
- `selected_option_high_regret_flag` / `selected_option_unacceptable_flag` -- convenience (seed1's exact
  pattern is high_regret **and not** unacceptable; 4 states match that strict pattern, 9 match high_regret).
- `eval_only` -- True on the 2 B-failure seeds. **Hold them out as the priority eval slice; do not train on them.**

## How B should consume these (from the verification)

- **Risk head = classification on `high_regret_flag` / `unacceptable_flag`. Never regress on raw `regret`** (it
  is RNG-dominated; 38% of high_regret options have regret < 2*value_se).
- **Treat c1 as a SOFT/upweighted risk signal, not a hard label.** Use it to trigger extra search / abstention,
  not as confident "search blundered here" truth. Prefer **recall** over precision and pair the model with an
  **abstain/extra-search fallback** so a false flag triggers more search rather than hard-blocking -- exactly
  the c2 failure mode.
- **Residual head = regress `delta_to_search_norm` with a clip + robust loss (Huber/quantile).** ~74% of
  options have |delta| < 100 (predict ~0, trust search); the tail is real lethal-scale but per-option
  noise-dominated and its **sign is unstable**, so clip to ~p95/p99, **down-weight by `value_se`**, and
  integrate as `final = search_score + small_gate * predicted_residual` (small gate).
- **`value_se` as sample weight**, but the **164 zero-SE options are deterministic-across-worlds, not
  "certain"** -- do not let a zero SE inflate their weight.
- **Outcome auxiliary (`outcome_winrate`/`outcome_se`) = SE-weighted, weak.** Only k=16 playouts here
  (outcome_se median ~0.06), 47% saturate at 0/1, hand-vs-outcome argmax disagree 0.65. Soft consistency prior,
  never an argmax label. (6 of 9 reproduced c1 picks have outcome_winrate 0.0 -- a less-noisy c1 corroborator.)
- **Class-weight:** c2 dwarfs c1 ~5:1; a naive model minimizes loss by under-flagging. Do not read overall
  accuracy as evidence the catastrophic-miss head works.

## Honest limitation (what B still cannot prove)

Even with `group_id`, the 9 reproduced c1 reduce to **~3-4 independent catastrophic situations across 5 games**,
2 of which carry the mass. **B can train and demonstrate the false-positive fix (c2 is strong: 49 valid states
over 11 games), but cannot honestly demonstrate cross-game generalization of the missed-catastrophic head** --
any game-held-out split either starves train or eval of independent c1. If proving the catastrophic-miss head
is the goal, the c1 class needs more **independent games**, which requires mining a larger/disjoint corpus
(`tools/mine_c1.py` shards and scales; this run covered one snapshot's top ~3.2k states). Recommend treating
the catastrophic-miss head as **risk-recall-with-abstain calibrated on the 2 seeds**, and prioritizing the
well-supported false-positive fix, until more independent c1 games are mined.

## Verification

Independent 3-lens adversarial check (`verify-risk-round2` workflow): requirements audit **pass_with_notes**
(all 19 fields, seeds hash-consistent, JSON valid, 0 dups); label-trust **usable_with_caveats** (flag yes,
magnitude no); utility **handoff_with_notes** (addresses both failures; game-clustering is the real
constraint). The round-1 doc `teacher_v2_residual_risk_summary.md` carries round-1 numbers (50/451) -- trust
this round-2 doc and the JSONL.

Status: implemented, offline, training-ready. `agent_search` baseline unchanged; no live agent; `main` untouched.
