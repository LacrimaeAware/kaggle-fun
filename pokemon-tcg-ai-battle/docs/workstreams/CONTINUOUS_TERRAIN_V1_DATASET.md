# Continuous Terrain V1 -- expanded terrain dataset (Model A deliverable)

Joint experiment: does a learned continuous state-action representation expose risk/value structure around
safe -> borderline -> unstable -> high-regret -> catastrophic decisions, beyond root-state features,
search/criticality metadata, and the existing engineered representation? **Model A** built this dataset;
**Model B** trains the representation on it (`CONTINUOUS_TERRAIN_REPRESENTATION_V1.md`).

Branch `exp/planner-teacher-v2`. `agent_search` remains the live baseline; this is an OFFLINE dataset; no live
agent, no screen, `main` untouched. Artifact is self-contained: Model B featurizes it WITHOUT replay reparsing.

Artifacts: `data/manifests/continuous_terrain_v1.jsonl` (702 decisions, 5,743 options, 14.3 MB),
`data/manifests/continuous_terrain_v1_summary.json`, `data/manifests/continuous_terrain_v1_semantic_validation.json`.
Tools: `tools/mine_terrain_v1.py`, `tools/build_terrain_v1.py`, `tools/label_terrain_v1.py`,
`agent/action_semantics_v1.py`, `tools/refeaturize_terrain_v1.py`, `tools/finalize_terrain_v1.py`,
`tools/validate_action_semantics_v1.py`.

## Headline

| target (spec A3) | achieved |
|---|---|
| 400-800 root decisions | **702** |
| >= 3,000 option rows | **5,743** |
| >= 30 independent games | **160** |
| >= 25 reproduced c1 across >= 12 games, no game > 20% | **24** reproduced (c1-tagged) across **16 games**, top-game share **12.5%** (32 by the broader selected-high-regret measure) |
| >= 75 c2 | **416** provisional / **102** authoritative dangerous-sibling |
| >= 75 c3 / boundary | **32** provisional primary tag, **111** authoritative boundary (the primary tag undercounts; see below) |
| matched safe / background controls | **164** ring members (Ring1-4) + **594** safe-selected |
| keep 2 canonical seeds as eval_only | yes (`80251230.json:12`, `80252701.json:56`) |
| group_id = source game | yes |

Disjoint from the round-2 13-game set (those games excluded from mining). Mining scanned **~10.7k distinct
high-criticality decisions** (12 shards) + a **supplemental ~6k** over a larger pool to reach more games' top
criticality. 0 failed/unrecoverable records. All 702/702 decisions have all siblings fully determinized.

## Why two class-balance numbers (provisional vs authoritative)

The cheap single-pass mining tag is **provisional** and uses a strict priority c1 > c2 > c3, so a decision
that is BOTH a safe-pick-with-dangerous-siblings (c2) and a mixed-regret boundary (c3) is tagged c2, which
undercounts the c3 primary label (32). The **authoritative** terrain comes from the REPEATED labels (8 live +
4 stronger) and is attached per record as `terrain_authoritative`. By that measure the boundary band is well
populated:

| authoritative flag (from repeated labels) | count |
|---|---|
| `repro_c1` (search-selected option high_regret_prob >= 0.5) | 32 |
| `boundary` (mixed high_regret among siblings, or selected prob in (0.25,0.75), or unstable+dangerous) | **111** |
| `unstable` (modal_action_stability < 0.75 -- live search flips its pick across repeats) | 240 |
| `safe_selected` (selected never high-regret, no dangerous sibling) | 594 |
| `has_dangerous_sibling` (some sibling high_regret/unacceptable prob >= 0.5) | 102 |

Model B should prefer `terrain_authoritative` and the continuous per-option probabilities over the provisional
`terrain_class` tag.

## Repeated labels (A4) -- distributions, not hard labels

Every root decision carries **8 live (N_DETERM=8) + 4 stronger (N_DETERM=32)** repeated measurements; within
each run siblings share paired hidden worlds. The engine rollout RNG is not seedable (the SPLIT_BASE_V2
finding), so these repeats capture the real Monte-Carlo instability rather than hiding it behind one draw.

Per option: `mean_live_value`, `live_value_variance`, `mean_stronger_value`, `stronger_value_variance`,
`delta_to_search`, `delta_to_search_norm`, `hand_norm_advantage`, `regret`, `value_se`, `acceptable_prob`,
`high_regret_prob`, `unacceptable_prob` (probabilities across the stronger repeats), `completed_determinizations`,
`semantic_action_key`, `eq_class`, plus the A5 `semantic_vector`.

Per decision: `live_selected_distribution` (argmax option across the 8 live repeats), `live_action_entropy`,
`modal_action_stability`, `stronger_soft_policy`, `criticality` (+ components), `value_spread`,
`search_selected_option`, `stronger_argmax_option`, `forced_action_flag`, `observation`, `legal_options`,
`obs_hash`, `decision_id`, `group_id`, `source{file,step,player}`, `terrain_class`, `terrain_authoritative`,
`ring`, `anchor_id`, `eval_only`, `coverage.all_siblings_completed`, `timing`.

**Stability read:** mean `modal_action_stability` 0.793, mean `live_action_entropy` 0.449 -- i.e. on ~21% of
these hardest decisions the live N=8 search flips its top pick run-to-run, consistent with the round-2
~53% c1-reproducibility and the Teacher V1 ~0.78 stability. This instability is the SIGNAL, not noise to
average away: the `*_prob` fields and `live_selected_distribution` give Model B the soft targets the spec asks
for (do not collapse them to a single hard label).

## Matched terrain rings (A2)

For each c1 / high-regret anchor, Ring 1-4 matches were drawn from DIFFERENT games (a non-anchor game), matched
on action family + criticality band + turn proxy + option count + prize lead + board development:
Ring 0 = anchor (c1), Ring 1 = same family / similar criticality but safe selected, Ring 2 = boundary (c3),
Ring 3 = same tactical family / phase but lower-regret / stable, Ring 4 = matched background control. Counts:
Ring0 46, Ring1 44, Ring2 32, Ring3 44, Ring4 44 (`ring` + `anchor_id` per record). Matches never use the
final target label beyond the requested terrain class, and never pair within the same game.

## Action semantics (A5)

`agent/action_semantics_v1.semantic_vector` emits a 59-field self-contained vector per option:
- **identity:** opt_type, acting_card_id, attack_id, ability_flag, target_side, target_zone/index.
- **card meta:** card_type, card_stage, card_hp/prize, is_ex/mega/tera/ace_spec, card_retreat.
- **effect magnitude:** the decoded `card_effects` fields (draw, search, search_to_bench, energy_accel, heal,
  recover_discard, status, disrupt, discard_cost, shuffle_hand, has_ability) + a runtime own_switch/opp_gust
  split by target side + deterministic overrides; attack `atk_damage`/`atk_cost`.
- **forward-model option deltas (d_*):** prizes_taken, opp_ko, dmg_dealt, cards_drawn, energy_attached,
  board_dev, deck_used, discard_gain, ends_turn, wins/loses_now (one `search_step`, no rollout).
- **context / opportunity-cost (ctx_*):** immediate KO, lethal, KO-back, prize lead, backup attacker, bench
  slots, hand size, deckout risk, energy shortfall, supporter/attach availability, ability, board dev.
- **`semantic_coverage`** tier: `decoded` | `override` | `energy` | `pokemon_meta` | `tool` | `unknown`.

**Validation on the 50 most frequent acting cards:** trustworthy (any principled tier) = **1.00** unweighted
and freq-weighted; dataset-wide trust **0.877** (decoded 2255, energy 1652, pokemon_meta 563, override 353,
tool 216; only **704/5743 = 12.3% genuinely-undecoded** items/supporters/abilities). The "unknown" before
category-aware coverage was a labeling artifact: basic/special Energy (~1.6k options) is "attach energy of type
X" and Pokemon plays are establishment captured by card meta + deltas -- both understood, just not text-decoded.
Honest gaps left at 0/unknown rather than fabricated: coin-flip EV/variance, energy-type restriction, and the
specific text effect of the 12% undecoded items/abilities. Overrides fix the conflated/under-decoded frequent
cards (Buddy-Buddy Poffin, Ultra Ball, Nest Ball, Dusk Ball, Earthen Vessel, Boss's Orders gust vs Switch).

## Compute

Mining ~10,331 CPU-s (~861s wall, 12+6 shards); repeated labeling 3,061 CPU-s (~255s wall, 12 shards);
~8.7s per decision (8xN8 + 4xN32, hand leaf). Terminal-outcome playouts were NOT run in v1 (the spec marks
hand/outcome disagreement optional "if outcome is available") to keep the repeated-label budget tractable; the
B targets (residual, risk-prob, instability) are hand-based. `outcome_winrate` can be added later if needed.

## Honest limitations (for Model B)

1. **c1 reproduced = 24 (c1-tagged), 1 short of the 25 target; 32 by the broader selected-high-regret measure.**
   Across 16 games, top-game share 12.5% (the no-game-> 20% constraint holds). c1 is intrinsically ~0.5% and
   game-clustered; reaching 25 strictly-tagged would need another mining round. The reproduced-c1 head still
   cannot prove cross-game generalization from a handful of independent catastrophes -- treat it as
   risk-recall-with-abstain, hold out the 2 eval_only seeds, and lean on the well-populated boundary/unstable/
   dangerous-sibling bands and the continuous probabilities.
2. **No within-record outcome target** (hand-based labels only in v1).
3. **`value_se` from 4 stronger repeats is a coarse SE**; the per-run high_regret/unacceptable flags use the
   5000 hand-eval threshold (uncritical: high-regret options cluster well above 30k). Weight by it but do not
   over-trust small differences (engine-RNG-dominated, per round-2).
4. **12.3% of options have undecoded text effects** (`semantic_coverage == unknown`); the card embedding (B2)
   should carry residual meaning there.

Status: implemented, offline, self-contained, copied to Model B's worktree. agent_search unchanged; main
untouched.
