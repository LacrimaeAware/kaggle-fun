# Teacher V2 targeted-labeling protocol (Model A <- Model B request)

**Status:** no request from Model B yet (`data/manifests/teacher_v2_label_request_for_A.json` absent in both
worktrees and on both branches as of this check). Per the plan, Model A did NOT run a new generic batch.
Instead the request-consumer utility is built and validated, ready to label B's exact failure/test states the
moment its request lands.

Branch `exp/planner-teacher-v2`; `agent_search` remains the live baseline; no live agent consumed anything.

## How to run (once B writes its request)

```
python tools/label_requested_states.py --request data/manifests/teacher_v2_label_request_for_A.json \
       --out teacher_v2_labels_for_B_failures
```

Requested states are labeled **regardless of criticality** (B chose them). Validated: synthetic 2-state
request -> 2 self-contained labels, deck recovered from source.

## Request schema (what B should write to `teacher_v2_label_request_for_A.json`)

A JSON list, or `{"requests": [...]}`, of entries. Per entry, all fields optional EXCEPT a way to get the
root observation (provide `observation` directly, or `source` + `obs_hash` to recover it):

| field | meaning |
|---|---|
| `decision_id` | join id (e.g. `"<file>:<step>:<player>"`); echoed back |
| `obs_hash` | 12-hex hash for verification / disambiguating the player at a step |
| `observation` | the root obs dict (PREFERRED -- makes the request self-contained, no replay re-parse) |
| `deck` | 60-card determinization deck (else recovered as the deciding player's deck from the source replay; else falls back to the production deck, flagged `deck_source`) |
| `source` | `{"file": "<x>.json", "step": <int>, "player": <0|1>}` to recover obs/deck if `observation`/`deck` absent |
| `reason` | why this state is requested (echoed back) |

Best case: B echoes the `observation` from `teacher_v2_labels_scaled.jsonl` for the states it wants
re-labeled, or supplies its held-out test states' observations.

## Fields returned (per labeled state, same self-contained format)

`decision_id`, `obs_hash`, `observation`, `legal_options`, `source`, `deck_source`, `request_reason`,
`criticality` (+components), `soft_policy_target`, `acceptable_action_set`, `top_two_margin`,
`action_spread`, `hand_argmax_eq_class`, `outcome_argmax_option`, `hand_outcome_agree`, `coverage`
(`all_siblings_completed`), `timing`, `seed`, `paired_world`; and per option: `index`,
`semantic_action_key`, `eq_class`, `hand_mean_value`, `hand_value_variance`, `hand_norm_advantage`,
`completed_determinizations`, `outcome_winrate`, `outcome_playouts`, `outcome_se`.

Output: `data/manifests/teacher_v2_labels_for_B_failures.jsonl` (+ a short summary written here as
`teacher_v2_failure_label_summary.md`).

Interpretation unchanged: primary target = `hand_norm_advantage` (weight by criticality, inverse
`hand_value_variance`, coverage); auxiliary = `outcome_winrate` (confidence-weight by `outcome_se`, not its
argmax).
