# Strategy Category writeup: current-state update and handoff (2026-06-24)

Purpose: a single read-first document for another model (or a future me) that is going to help write the
Kaggle Strategy Category report. It does two things: (1) points at the existing writeup-planning material
so nothing is rebuilt from scratch, and (2) updates the narrative to match what we actually know now, which
is different from the 2026-06-18 scaffold. The competition is the Pokemon Company PTCG AI Battle Challenge
Strategy Category ($240k main track, 8 finalists at $30k). Final submission 2026-09-13. Writeup limit 2000
words. Scoring: Model 70%, Deck 20%, Report 10%.

This repo's contribution is mostly HEURISTIC work plus a forward-search line. Nothing exotic. The value for
the Strategy Category is the disciplined research arc and the honest negative results, which the scoring
explicitly rewards even at middle/lower leaderboard tiers.

---

## 1. Where the existing writeup-planning material already lives

Do not start from a blank page. These exist:

- `research/competition/writeup/draft-skeleton.md` — the planned 2000-word outline, section by section, with
  word budgets, a hypothesis table, and figure candidates. Best starting structure.
- `research/competition/writeup/README.md` — the writeup workspace thesis and the per-experiment record rule.
- `research/competition/strategy/strategy-category-brief.md` — scoring weights, deadlines, requirements,
  citation block.
- `research/competition/strategy/research-ledger.md` — one-line experiment ledger, hypotheses W1-W7, planned
  table of contents, figure/table backlog.
- `research/competition/strategy/resources.md` — source map plus external papers to cite (ISMCTS/ensemble
  determinization, contextual preference ranking for Magic, ReBeL/Student of Games as blueprint-not-claim).
- `research/competition/strategy/raw-strategy-category-overview-2026-06-18.txt` — raw copied competition page.
- `writeup.md` (repo root) — an older one-page Question/Method/Result/Caveat/Lesson summary.

## 2. The big narrative change since the scaffold (read this before reusing the old outline)

The 2026-06-18 scaffold was written before any leaderboard scores existed. Its thesis was a
search-and-learning framework pivoting to "rank sibling actions, not states." That framing is now partly
outdated. What changed:

- We now have ladder scores. They are decisive and they invert the local self-play story.
- LADDER TRUTH (from `docs/SUBMISSIONS.md`, scores are facts from the Kaggle page):
  - 770.9 = pure heuristic, no search (the best agent we have).
  - 640.7 / 645.2 / 687.8 = 1-ply forward-model search variants.
  - 617.2 = day-one heuristic on the old deck.
  - 592.0 = a "hoard for Powerful Hand" eval-term variant (below plain search, so the term hurt).
  - 422.1 = search with a learned gradient-boosted value at the leaves (worst; dead, do not resubmit).
- Three independent confirmations agree: heuristic >> search >> learned-value combine. Every search or
  learned addition has scored BELOW the plain heuristic on the ladder.
- Local self-play does NOT predict the ladder. Locally, search beats heuristic; on the ladder the opposite.
  Treat local A/B as a rough filter, never as the arbiter.

So the honest 2026 writeup story is heuristic-first, with search demoted from "the agent" to "a bounded
calculator the heuristic can call." The sibling-action-ranking and learned-value lines are now NEGATIVE
RESULTS that explain why we went back to heuristics, not the headline method. That is still a strong Strategy
Category story (the scoring rewards originality, soundness, stability, and clear reasoning, not just rank).

## 3. The story to tell (updated honest narrative)

Thesis (replacement for the scaffold's thesis):

> We built both a forward-search agent and a learned-value agent for the Pokemon TCG Alakazam draw-engine
> deck, measured them against a disciplined hand-written heuristic on the real ladder, and found the
> heuristic won decisively. The contribution is the diagnosis of WHY (global value prediction does not order
> the near-identical sibling states a 1-ply search compares; local self-play does not predict the ladder; the
> deck's real win/loss conditions are Powerful Hand knockouts and self-deck-out, which a few targeted rules
> capture better than search), and a design where search is a scoped calculator the heuristic invokes, never
> the policy.

Three or four load-bearing findings:

1. The single biggest lever is PH-aware knockout detection. Alakazam's Powerful Hand deals 20 damage per card
   in hand as DAMAGE COUNTERS (weakness/resistance do not apply, and the card's static damage reads 0). A
   naive heuristic reads it as 0 and never takes the deck's main attack. Adding only PH-aware KO scoring took
   the local heuristic from 0.483 vs the baseline (a tie, cannot pilot the deck) to 0.760-0.793.
2. Deck-out is a primary loss condition, not a tail, because this is a heavy draw engine. Deck-out denial (a
   draw-safety buffer) prevents most self-deck-out losses (5/500 deck-out losses for the tuned agent vs 91/500
   for the naive baseline). This is solved on the heuristic side and is the term most worth porting anywhere
   it is missing.
3. Sequencing is the recurring misplay class: draw and search FIRST, irreversible actions (attach, evolve,
   attack) LAST, take a guaranteed knockout last (a KO cannot be lost by waiting, only by your own endangering
   actions). 1-ply search cannot represent this because it branches one decision then finishes the turn with a
   fixed rollout, which is exactly why leaf-eval tweaks keep coming back neutral.
4. Learned value lost. A gradient-boosted P(win) head had good global accuracy (AUC ~0.735) but ranked the
   sibling leaves of one decision poorly, and as a search leaf it lost to the plain hand eval (0.427) and
   tanked on the ladder (422). Good global calibration is not the same objective as ordering near-identical
   post-move states.

## 4. Deck concept (for the 20% Deck Score)

Deck: the DENPA92 Alakazam draw engine.

- Lines: Abra (741) -> Kadabra (742) -> Alakazam (743), all Psychic. Dunsparce (65/305) -> Dudunsparce (66),
  Colorless.
- Engine: it is a heavy draw/tutor deck. Kadabra/Alakazam "Psychic Draw" trigger on the EVOLVE step (draw 2 /
  draw 3), not from being active. Dudunsparce "Run Away Draw" is a bench ability (draw 3, then shuffle itself
  back). Telepath Energy fetches 2 basic Psychic (Abra is the only basic Psychic). Buddy-Buddy Poffin fetches
  any 2 basics.
- Win condition: Alakazam's Powerful Hand, which scales with hand size (20 damage counters per card in hand).
  So the deck WANTS a big hand, which interacts with hand-thinning and with deck-out risk.
- Why it co-designs with the agent: the deck creates repeated, learnable decision patterns (tutor targeting,
  draw sequencing, when to spend vs hold hand for Powerful Hand, deck-out buffer management). A small set of
  rules captures the high-value patterns; search adds value only on bounded sub-decisions where a bot can
  compute card advantage a human cannot.
- Honest caveat for the writeup: name the exact submitted deck at deadline; it may still change.

Figure candidate: deck game-plan diagram (setup tutors -> evolve Alakazam line -> bank hand -> Powerful Hand
KO -> stabilize while watching the deck-out clock).

## 5. Agent architecture (current reality)

Layered, all crash-safe and always-legal:

```
legal options (from the official cabt simulator)
  -> safety / fallback rules
  -> PH-aware KO floor (take a listed lethal/KO, correctly scoring Powerful Hand; go first; energy to active)
  -> [optional] bounded forward-model search for development decisions the floor has no opinion on
  -> final action
```

- The deployed local-best SEARCH agent is `phaware_search`: PH-aware KO floor + 1-ply determinization-averaged
  forward-model search (N_DETERM=8) for development, scored by a 4-term hand leaf eval
  (W_PRIZE=1000, W_HP=1, W_BODY=30, W_ENERGY=8).
- The ladder-best agent is a PURE heuristic with NO search (the pokemon-ai-agent registry: lethal_ko +
  alakazam_energy + deck-out denial + go-first). 770.9.
- Design rule learned the hard way: search is a calculator the heuristic calls, scoped to the heuristic's own
  candidate set and objective, with a default if it fails. It never enumerates all legal moves and never
  drives policy. Enumerate-all search lost on the ladder every time.

Figure candidate: architecture diagram (observations -> legal options -> KO floor -> scoped search -> action).

## 6. Experiment ledger (compressed, ladder facts in bold)

| Hypothesis | Method | Result | Lesson |
|---|---|---|---|
| Hand rules beat naive play | heuristic vs random/first | positive floor | legality and obvious tactics matter |
| PH-aware KO is the key fix | add PH KO scoring | 0.760-0.793 vs baseline (local) | seeing Powerful Hand KOs is the single biggest lever |
| Forward search beats heuristic | search vs heuristic | local yes, **ladder NO (640-688 < 770.9)** | local self-play does not predict the ladder |
| Heuristic + scoped search | phaware_search vs phaware | +0.25 local (0.745) | search helps only where the floor has no opinion |
| Global learned value as leaf | GBM P(win) at leaves | 0.427 local, **422 ladder** | global calibration != local sibling ordering |
| Hand-hoard eval term | W_POWERFUL_HAND | **592 ladder (< search)** | the term hurt |
| Card-advantage leaf objective | +25 per hand card | 0.525 NEUTRAL | board terms already proxy it; 1-ply cannot plan the cascade |
| Deck-out leaf term | hinge penalty near empty | fires on 10.6% of leaves, ~neutral combined | deck-out belongs in the floor as a draw-aware refusal |
| Deck-out denial (heuristic) | draw-safety buffer | 5/500 vs 91/500 deck-out losses | primary loss condition, solved on heuristic side |
| Replay-imitation ranker | sparse cross-feature model | not laddered; top-1 metric is imitation not quality | judge by deviation-decision acc, not top-1 vs option-0 |

Method discipline to state in the report: whole-game A/B noise floor is about +/-0.05 at n=300; always name
the baseline; always run a self-mirror control that must land near 0.500 (a clean n=600 self-mirror = 0.502
settled an earlier phantom side-A tilt); the ladder is the only real arbiter.

## 7. Key lessons (the methodological spine of the writeup)

1. Local self-play did not predict the ladder. This is the central cautionary result and is itself a finding.
2. Global value prediction does not order sibling states. Good AUC, bad local ranking, lost as a search leaf.
3. The deck's real edges are concrete (Powerful Hand KOs, deck-out denial, draw-first sequencing), and a few
   targeted rules captured them better than search or learned value did.
4. Search is a calculator, not a policy. Scope it to heuristic-approved candidates and a heuristic objective.
5. Negative results are the contribution. The Strategy Category rewards explaining WHY, and we can.

## 8. Open questions and future work (for the report's limitations section and for the next model)

1. Turn-level sequencing is the real bottleneck. A turn-planner where the heuristic proposes 2-6 whole-turn
   plans (KO-last, setup-first, develop-attacker, deck-safe variants) and search scores each end-of-turn state
   is the highest-leverage unbuilt item. Design is written in `dropoff/inbox/2026-06-24-turn-planner-design.md`.
   The hard part is the intent-to-option matcher mapping a plan onto the engine's per-step prompts.
2. Deck-out shaping: confirm the lift, sweep the buffer weight/shape, decide leaf-term vs decision-time floor
   refusal (the draw-amount-aware "draw 3 with 4 left -> deny" needs the floor, since the leaf only sees the
   resulting deck).
3. Contextual card-advantage objective (draw/tutor availability, hand differential, deck risk, productive hand
   spending) instead of the raw hand-count term that was neutral.
4. Eval-term ablations: the four ON weights (prize/hp/body/energy) have never been measured individually.
5. Move-ordering prior from the replay ranker to guide which options search expands first (targets sequencing,
   not the leaf number).
6. Ladder validation of `phaware_search`: it is NOT the pure search that lost (it keeps the PH KO floor), so it
   is worth one submission to confirm.

## 9. Figure / media gallery candidates (Report Score 10%)

- Architecture diagram (observations -> KO floor -> scoped search -> action).
- Deck game-plan diagram.
- Ladder bar chart: heuristic 770.9 vs search 640-688 vs combine 422 (the headline result).
- Local-vs-ladder mismatch table or scatter (the central cautionary finding).
- Powerful Hand mechanic illustration (20 counters/card, ignores weakness/resistance).
- Deck-out denial chart (5/500 vs 91/500).
- Experiment ledger table (the one in section 6, trimmed to 6-8 rows).
- Note: license. Do not include Pokemon card art unless it is within the competition's granted license, or the
  writeup is disqualified. Prefer schematic diagrams and our own charts.

## 10. Source file index (everything another model needs)

Writeup planning:
- research/competition/writeup/draft-skeleton.md, research/competition/writeup/README.md
- research/competition/strategy/{strategy-category-brief,research-ledger,resources}.md
- research/competition/strategy/raw-strategy-category-overview-2026-06-18.txt
- writeup.md

Recent work (the heuristic + search arc this points to):
- dropoff/inbox/2026-06-24-codebase-and-strategy-summary.md (read-first codebase map)
- dropoff/inbox/2026-06-24-heuristic-eval-landscape.md (every heuristic/eval term + measured deltas)
- dropoff/inbox/2026-06-24-turn-planner-design.md (the next build)
- dropoff/inbox/2026-06-24-search-ca-meta-change-audit.md (correctness audit of the current tree)
- dropoff/inbox/2026-06-23-heuristics-and-search-plan.md (search-as-a-tool plan)
- dropoff/inbox/2026-06-23-heuristics-from-replay-81555875.md (replay-derived heuristic priorities)
- dropoff/inbox/2026-06-23-phaware-plus-search-result.md (the PH KO + search local result)
- docs/SUBMISSIONS.md (canonical ladder log; the score facts)

Code that backs the claims:
- agent/main.py (agents + deck + heuristic floor), agent/search_v3.py (1-ply search),
  agent/eval.py (leaf eval terms), agent/deck_policy_v3.py (PH-aware attack math), agent/features.py
- tools/run_heuristic_ab_v1.py (agent variants registry), tools/par_ab_v1.py (parallel durable A/B),
  tools/eval_feature_importance_v1.py, tools/deck_depletion_diag_v1.py

Read-only sibling repo (the ladder-best heuristic):
- pokemon-ai-agent/src/pokemon_ai_agent/policy/heuristics/* and .../registry.py
- pokemon-ai-agent/auditor_sandbox/experiments/*.json (the n=500 heuristic ablations)
