# EXP55 - Class Route Search

Date: 2026-06-12

Purpose: deliberately overfit the 35-image expert benchmark to estimate class-aware routing
headroom. This is not a production rule. It tells us whether class-specific model choice has enough
signal to justify more careful routing work.

Harness: `experiments/exp55_class_route_search.py`

Disallowed route models: `DLTrack`, `SMA`, and `our_pipeline_true_scale`. These are reference or
baseline rows in the expert viewer, not production-owned candidate models for routing.

Ignored output bundle: `results/exp55_class_route_search/`

## Results

| route | overall | PA | FL | MT | read |
|---|---:|---:|---:|---:|---|
| base `median_weight_blend_best` | 0.143 | 0.149 | 0.211 | 0.070 | current local research best |
| model-level greedy class route | **0.135** | 0.147 | 0.188 | 0.072 | meaningful class-aware upper bound |
| term-level greedy route | 0.131 | 0.137 | 0.192 | 0.065 | ours-only term route; still overfit-prone |

## Model-Level Route Steps

1. `strong_lower_curve_any_direction` -> `story_stack`
2. `not_low_support_any` -> `weighted_story_fl`
3. `lower_side_angle_changes_strongly` -> `story_weight_grid_best`
4. `upper_middle_deep_sag` -> `pa_conflict_gate`
5. `lower_side_angle_changes_strongly` -> `story_weight_grid_best`

## Term-Level Route Steps

"Term-level" means PA, FL, and MT are allowed to route separately. For example, a row can keep PA
from one candidate, FL from another, and MT from a third. This is why it scores lower than the
model-level route, and also why it is more overfit-prone.

Exact allowed-only route:

1. `low_support_any` (33 rows): PA from `story_stack`
2. `not_low_support_any` (2 rows): FL from `weighted_story_fl`
3. `strong_lower_curve_any_direction` (10 rows): FL from `story_stack`
4. `not_strong_upper_curve_any_direction` (23 rows): MT from `robust_triangle_anchor`
5. `lower_side_angle_changes_strongly` (11 rows): FL from `story_weight_grid_best`
6. `sparse_fragments` (3 rows): MT from `story_stack`
7. `sparse_fragments` (3 rows): PA from `weighted_story_fl`
8. `high_PA_sensitivity` (8 rows): PA from `median_weight_blend_best`
9. `multi_band_risk` (9 rows): MT from `story_stack`
10. `high_PA_sensitivity` (8 rows): MT from `robust_triangle_anchor`
11. `upper_middle_deep_sag` (4 rows): PA from `story_stack`

## Interpretation

There is real class-routing headroom. The benchmark can be improved beyond the best single model by
choosing different model stories for curved/lower-side/deep-sag classes. The model-level route is the
most useful signal: it remains `0.135` after excluding `DLTrack`, `SMA`, and
`our_pipeline_true_scale`. The term-level route is still an overfit upper-bound, but it is now
ours-only under the same disallowed-model filter.

## Next

Validate the model-level route visually and against rough target labels. If it still looks coherent,
translate only the production-owned pieces into the 309-image pipeline and public-test as an isolated
class-routing feature.
