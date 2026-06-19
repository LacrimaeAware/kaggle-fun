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

No submission candidate produced. Nothing promoted, production agent/deck/`main` untouched. The gust-target
floor is mechanically clean and harmless and can be retained inside any future candidate, but at ~1% of
decisions it cannot be screened on its own cheaply.
