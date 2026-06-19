# Branch A practical: tactical-floor screens v1

Goal: a stronger submission via one hybrid tactical-search floor. Cheap directional A/B vs production
`agent_search`, DENPA92 both sides, seat-swapped, Wilson 95% CI. Tools: `agent/search_live_v2.py`
(floors), `tools/screen_tactics.py` (screen). Exploration, not promotion tests.

## Results

| candidate | what it forces | result vs search | n | verdict |
|---|---|---|---|---|
| draw   | fire available draw-engine abilities | 0.333 (10-20) | 30 | **bad, discarded** |
| evolve | evolve into our line (Dudunsparce/Kadabra/Alakazam) | 0.467 (14-16) | 30 | wash |
| tactical (gust+evolve) | gust best target + evolve line | 0.471 (33-37) | 70 | wash |

A first n=30 run of `tactical` read 0.633 (19-11); the n=40 confirmation flipped to 0.350 (14-26).
Pooled = 0.471. The 0.633 was small-sample noise. No floor beats production search.

## Why draw failed (card mechanics, not just the screen)

- Dudunsparce "Run Away Draw": draw 3, then **shuffle this Pokémon into your deck** -- using it sacrifices
  the 140-HP body. Forcing it greedily throws away board.
- Kadabra/Alakazam "Psychic Draw": only triggers **when played from hand to evolve** -- not a standalone
  repeatable draw.

So these are not free draws, and production search's low use (15% of availabilities) was partly correct
(the leaf sees the HP cost). Lesson: read the card text before building a card heuristic.

## The insight (why floors don't win here)

The search's problem on these tactical classes is **noise, not systematic wrongness** (consistent with the
A2 audit: the instability is engine-rollout variance, ~half of decisions unstable). A deterministic floor
replaces the search's choice with a fixed rule; unless that rule is clearly better than the search's
average, it nets a wash (evolve, gust) -- or worse when the rule is actually wrong about a hidden cost
(draw). Floors only win on a true systematic blind spot.

The one candidate blind spot -- the leaf eval has no card-advantage term (`W_PRIZE/W_HP/W_ENERGY` only) --
is confounded: the leaf DOES see the body-loss cost (HP), it just can't see the card benefit. So the
principled fix is a small **leaf card-advantage feature** that lets the search WEIGH draw benefit against
the cost it already sees, not a blunt floor that ignores the cost.

## Recommendation (next prototype, not yet built)

1. Small leaf card-advantage term (e.g. a low-weight hand-size / cards-vs-opponent term in `eval.evaluate`),
   single fixed weight, one cheap screen. This is in-scope ("leaf-evaluation feature inside search") and
   affects many decisions, so it is screenable. Risk: it perturbs the validated prize-dominant leaf; keep
   the weight small and gate on the screen.
2. If that also washes, the real lever is likely reducing the search's NOISE (selective computation, the
   A2/Teacher-V2 direction) rather than hand-floors -- but that is A3 and gated.

## Status

No submission candidate produced from floors. Nothing promoted, production agent/deck/`main` untouched.
The gust-target floor is mechanically clean and harmless and can be retained inside any future candidate,
but at ~1% of decisions it cannot be screened on its own cheaply.

---

# Tactic Miner V1 (state-conditioned soft prior)

Per the coordinator's redirect (floors too blunt -> mine state-conditioned tactics, implement as soft
priors). Deliverables: `agent/tactics_ontology.py` (16-tactic ontology + per-option classifier + entity
properties), `tools/tactic_miner.py` (miner), `data/manifests/mined_tactics_v1.json` (ranked artifact),
`agent/search_live_v2.agent_search_prior` (prototype).

Mined **46,889 winner sibling-decisions** from the frozen `replays_20260618` snapshot. Top clean-mechanics
context patterns (ranked by leverage): gust when `can_ko_now` (0.51 vs 0.34 base, lift 1.52); attack when
no setup left (`fixing_available=0` lift 2.87, `hand_low=1` lift 1.94); supporter-aware attach/gust shifts.

Prototype: a soft prior that acts ONLY on the search's near-ties (within 15% of the value spread, ~65% of
decisions per A2) and breaks them toward the mined high-confidence (>=0.85) pattern. Verified active
(redirects ~26% of decisions, all within ties; 0 errors). Search keeps authority on clear decisions.

**Screen: agent_search_prior 0.433 (13-17), n=30 -- wash, not a win.**

## Conclusion across the whole practical-search effort

Floors (draw bad, evolve/gust wash) AND the mined soft prior (wash) all fail to beat production
`agent_search`. The consistent reason: the search is decent-on-average; its weakness is **noise** (A2:
engine-rollout variance, ~half of decisions near-ties), **not systematic tactical wrongness**. So:
- On near-ties the choice barely changes the outcome, so a prior that only breaks ties has a small effect.
- Winner-replay "strong-player tactic" patterns are partly survivorship correlation (winners attack more
  because they are closing out games), not necessarily causal optimal play, so nudging toward them is at
  best neutral.

The Tactic Miner ontology/miner/artifact are a reusable substrate (and a clean label source Branch B can
consume), but a mined-pattern prior is not the submission lever on this deck.

## Recommendation

The evidence points away from more hand/mined tactical priors and toward the two gated levers:
1. reduce the search's noise on its non-tie decisions (selective computation / Teacher V2, A3 -- gated);
2. a contextual learned sibling-action model guiding search (Branch B's current task).
Interim, production `agent_search` remains the best available submission baseline. Awaiting review before
any A3 work; no further blunt screens planned.
