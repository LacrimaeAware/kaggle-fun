# Pokemon TCG AI Battle: Strategy Memo

> CONSOLIDATED 2026-06-19: see [docs/OVERVIEW.md](OVERVIEW.md) for the single-document project summary. Retained for unique detail (the 15 seed hypotheses, novel angles); slated for slimming.

> SUPERSEDED IN PART, 2026-06-17. This is the original long memo from before the official Data
> tab was opened. The current front doors are `docs/LEARNING_PLAN.md` (the staged build + current
> results) and `docs/RESEARCH.md` (deep-research findings + priority plan); `docs/PLAN.md` and
> `LANDSCAPE.md` for orientation. Where this memo says in-match search is blocked or the forward
> model is unconfirmed (incl. section 3's "forward model gate"), it is WRONG: the search API
> exists (cg/api.py, registry H001 supported) and search is our strongest agent. Background only.

Numbers first. Confirmed facts come from the engine source (cabt.py, cabt.json) and the matsuoinstitute.github.io/cabt docs. Everything sourced to an unrendered Kaggle page, a press release, or analogy is marked UNCONFIRMED and must be checked live before it drives a decision. Treat this memo as a plan, not a fact ledger.

---

## 1. The two tracks and how they relate

This is one contest by The Pokemon Company (with Matsuo Institute and HEROZ, hosted on Kaggle), split into two linked Kaggle competitions that run on the same agent and the same engine.

- **Track A, Simulation** (slug `pokemon-tcg-ai-battle`). A bot-vs-bot ladder. You submit a code agent; it plays automated PTCG matches and earns a Kaggle skill rating. Reward is "Knowledge", no cash. It is the objective performance leaderboard.
- **Track B, Strategy** (slug `...-challenge-strategy`). You submit a written report explaining the same agent's strategic and deck-design logic. This is the prize-bearing track.

**Why cash sits only under Strategy.** Simulation is the measurement engine; it produces a number (your rating). Strategy is the judged track, so prize money is attached there. The reported judging blend is (a) agent stability, (b) deck-design concept, (c) Simulation-track performance, which makes your ladder rating a direct input to Strategy scoring.

**UNCONFIRMED in this section, flag before acting:**
- That the judging blend is exactly those three components, and their weights. Described as qualitative only; the strength of the "Simulation feeds Strategy" coupling is inferred, not quoted.
- The prize structure. Outlets contradict each other: Dexerto headlines a $50,000 pool, PokeBeach says $300,000+, one auto-read of the JP FAQ over-counted to ~$424,000 (double-count, do not use). The reconciliation in the brief (~$320k cash + credits, built from "Round 1 Strategy top 8 at $30k each = $240k" plus "Round 2 champion $50k / runner-up $30k") is **the researcher's own arithmetic across conflicting sources, not an organizer's verbatim figure.** Do not state any total as fact. Read the live Rules/prize page.
- Nvidia and Google Cloud as sponsors is single-outlet (Shacknews) and absent from the engine copyright line. Treat as unconfirmed.

**Confirmed:** two linked tracks, same agent/engine; Simulation is the no-cash ladder; Strategy is the judged report; team size max 5; 5 submissions/team/day; Simulation runs Jun 16 - Aug 17 2026, Strategy runs Jun 16 - Sep 14 2026.

---

## 2. The game as a formal AI problem

**Imperfect information.** Per player: a 60-card deck (multiset, hidden order), an Active and up to five Bench (each an evolution stack with HP, damage, attached Energy, Tools, status conditions), a hand, a public discard, a Lost Zone, six hidden prizes, and a shared Stadium. Your hand is visible to you; the opponent's hand is only a `handCount`; both deck orders are hidden; prizes are hidden even from their owner. Public: board, discards, Lost Zone, Stadium, prize counts, full event log. This is a genuine information-set game. A perfect-information solver would cheat by seeing hidden cards, so the principled tools are information-set MCTS, CFR, or determinization with belief sampling.

**Stochastic.** Coin flips, prize reveals, top-of-deck draws, shuffle/search effects. Chance nodes must be handled by expectation, not by chasing one lucky resolution.

**Action space (confirmed from the API).** The harness hands you a `select` of legal options plus a `maxCount`; the action is the chosen indices (single or multi-select). You never check legality yourself; only legal options are offered. Action types: bench a Basic, evolve, attach Energy, play Item/Supporter/Stadium/Tool, use Ability, retreat, choose an attack with target and coin-flip sub-choices, plus forced sub-choices (promote after a KO, pick a prize, search/discard/reorder). Per-decision branching is small, roughly 2 to 15. The difficulty is that one turn chains many micro-decisions, so a combo turn reaches thousands of end states, and games run 10 to 30+ turns. **Search depth, not per-node branching, is the binding constraint, and it is bounded by the match time budget.**

**Reward (confirmed, cabt.json):** Lost -1 / Won +1 / Draw 0. 1v1. `episodeSteps` 10000, `actTimeout` 0, `runTimeout` 3000, `remainingOverageTime` 600.

**UNCONFIRMED:** that `remainingOverageTime=600` means a 10-minute per-player wall-clock budget whose exhaustion forfeits, and whether the budget is per-move or per-match. The number 600 is real; the enforcement semantics and granularity are inference. If the budget is per-match, expensive early-turn search starves late-turn decisions, which changes all time-budget math below. Confirm on the live Rules tab.

---

## 3. Tooling: reuse vs build for an offline simulator

**Decision: reuse cabt. Do not build a rules engine.** Only the organizer ruleset matters for transfer to the leaderboard. Any non-cabt sim gives an offline win rate that does not predict the ladder, which is the global-probe / wrong-surface trap. The public engines and their honest completeness:

| Engine | Lang / License | Completeness | Use |
|---|---|---|---|
| **cabt (organizer)** | Python, license unstated | Most complete for this task. Standard rules over ~2,000 cards with documented small deviations. Imperfect-info observation built in (own hand visible, opponent `handCount`, `deckCount`, prizes None). | This is the scoring engine. Use it directly. |
| tcgone-engine-contrib | Groovy/Java, Apache 2.0 | Mature full PTCG, thousands of hand-coded cards, but partly open, online-server shaped, not a fast RL backend. | Rules reference only. Will not match cabt. |
| deckgym-core | Rust, AGPL-3.0 | ~82% of cards but TCG **Pocket** (simpler than Standard). Fast. | Architecture model only. Wrong ruleset, AGPL risk. |
| ryuu-play | TypeScript, MIT | From-scratch, moderate coverage. | Clean licensed rules reference, not cabt. |
| ptcg-sim | JS, MIT | Manual playmat, no rules enforcement. | Not usable programmatically. |
| sethkarten/tcg | C/Python, license unknown | Advertised Gym PTCG Pocket + PPO, but repo 404s and is Pocket. | Inaccessible, wrong ruleset. |
| poke-env | Python, MIT | RL envs for the **video game** on Showdown, not the TCG. | Design reference only. |

Cloning ~2,000 cards is a multi-month effort (tcgone has thousands of Groovy cards; deckgym hits only 82% on the simpler Pocket set), and the docs warn of deviations, so even a correct clone would mismatch the scoring engine. cabt already provides a kaggle_environments env, a legal-options interface, the imperfect-info observation, an event log, `all_card_data()`, and self-play hooks. It is a ready-made offline simulator.

**The one custom build worth it:** a fast rollout or leaf-value model to deepen search inside the time cap. Not a second engine.

**The gate that decides whether any of this works (UNCONFIRMED, highest priority to resolve):**
1. Does the harness expose a **forward model** you can clone and step yourself (`clone()` / `step()` / `legal_actions()` from the current node)? This is binary. If the harness is opponent-stepped (you receive one observation, return one action, no forward model), then **every search method is blocked, not merely slowed**, and even depth-1 expectimax requires you to reimplement the next-state function yourself.
2. If a forward model exists, how many sims/sec? That is a scalar that sets search depth.

The brief discusses only (2). Resolve (1) first by direct measurement: `from kaggle_environments import make` and check whether you get a steppable, clonable state object outside the hosted match server. Also confirm cabt is installable/runnable off-Kaggle at all (no PyPI `cabt` package was found; distribution appears to be only via the bundled kaggle_environments env). "An organizer-side engine exists" does **not** establish a competitor-accessible, clonable, in-process forward model. If only the hosted server is available, fall back to imitation learning from cabt-played games plus a heuristic/search agent.

---

## 4. The method plan

The build order is the load-bearing recommendation. Each step reuses the previous and never lowers your floor. The feasibility of steps 3-4 is conditional on the simulator gate in Section 3 and should not be treated as assured.

**STEP 0, ship first: a robust, always-legal heuristic agent.**
The single most important property is that it **never emits an illegal move and never times out.** A legal mediocre bot beats a clever bot that forfeits, and under a wall-clock harness an occasional illegal move or timeout loses that game outright regardless of average play quality. Enumerate legal actions from the harness; if anything is ever uncertain, fall back to a safe default (attach Energy, pass, attack for best immediate trade). Encode a static evaluation: prizes remaining (yours vs opponent), Active/Bench HP, Energy-attached vs Energy-needed for your attackers, evolution lines online, hand size and deck-out clock, and a lethal / being-KO'd-next-turn term. Greedy-play the highest-eval legal turn. Submit this. It establishes the rating baseline and is the fallback every later version reverts to. Bank the deck choice here too; a fixed card-ordering heuristic was enough to dominate the draft phase in the closest analog (LoCM) — **note that LoCM/DouZero/Hanabi citations are external analogs, currently unsourced in this repo, so treat them as motivating claims, not evidence.**

**STEP 1, first serious agent: depth-limited expectimax on the same eval.**
Enumerate your within-turn action sequences with pruning (Energy and Supporter are once-per-turn; drop dominated orderings), take expectations over coin flips and prize/draw randomness at chance nodes, assume a plausible fixed opponent response for 1-2 plies, pick the line with best expected eval. This captures almost all within-turn tactical value (lethal lines, best trades, correct setup ordering), the single highest-EV upgrade. **Caveat the brief omits:** depth-1 expectimax also needs a forward model. If the harness exposes no `step()`, you must author a rules-faithful next-state function even for one ply, which is a much larger task than "just layer expectimax on the same eval." Its low compute cost holds only given that model.

**STEP 2, the honest gate (do before escalating).**
Resolve Section 3's two questions empirically. If there is no clonable forward model, **stop escalating method complexity.** Spend the time hardening the eval (most remaining score lives there), tuning weights against a diverse opponent set, and adding opponent-aware play. Tree search and RL are worthless without a steppable, fast model.

**STEP 3, conditional, the real ceiling: hybrid ISMCTS.**
Only if a fast, faithful, clonable sim is confirmed. Graduate the expectimax into ISMCTS over information sets: use the Step-0 policy as the action prior (ordering, pruning, rollout policy) and the Step-0/1 eval as the leaf evaluation so you never pay for full rollouts. Sample determinizations of the opponent hand/deck from a belief. Keep a robust-averaging root choice and always wire the pure-heuristic timeout fallback. ISMCTS attacks the strategy-fusion error that hurts plain determinized MCTS (solving each determinization as perfect information makes the agent assume it can act differently in states it cannot distinguish). This is the architecture strong imperfect-info card-game agents converge to. **Honest rating:** it is the highest-ceiling path but also the most assumption-laden (needs the clonable sim + belief/archetype modeling + info-set bookkeeping). Its ceiling is conditional, not unconditional.

**STEP 4, late, conditional: self-play-learned eval/policy net.**
Only with a confirmed fast sim and weeks of runway. Train a DouZero/AlphaZero-style net and swap it in for the hand-built eval/prior without changing the search scaffold. Inference is cheap; the cost is offline. **The real blocker the brief understates:** before any training, you must author a faithful vectorized clone of the exact organizer ruleset. That, not GPU time, is the multi-week, possibly-infeasible task. Closer to "not viable this cycle" than "moderate-but-late" unless the clone problem is already solved.

**Do not start with:** CFR/Deep-CFR (multi-week offline cost, hard fast-sim dependency, and it optimizes toward an unexploitable equilibrium when a bot ladder rewards exploiting weak deterministic opponents), full end-to-end RL (same fast-sim gate), or an in-match LLM (empirically subpar on this exact game per PTCG-Bench/PokeAgent — unsourced here — plus latency, arithmetic fragility, and likely infeasible in a CPU-bound harness). Use an LLM only offline, as a coding aid to author the eval, encode card text, or build the deck table.

**Validation-surface discipline (overrides any tempting shortcut).** Local self-play vs a frozen pool, win-rate vs a fixed bot, and the public ladder are different exams; do not compare numbers across them. An aggregate win-rate gain is a 1-D global probe: it says the net outcome moved, it does **not** decompose into per-matchup mechanism. Do not infer "the agent now beats aggressive openings" from an aggregate. State the surface's noise band before calling any delta real; PTCG per-game variance (prizes, coin flips, draw order) is high, so most apparent gains from a few hundred games will not survive a properly sized null.

---

## 5. Novel-but-effective angles, each with how it buys win-rate

1. **Opponent modeling for exploitation, not equilibrium.** This is a skill-rating ladder, not a worst-case match. Maintain a light online belief over opponent tendencies (does it pass with lethal, over-extend the Bench, retreat predictably) from observed moves and bias the response model toward it. *Buys rating because beating the actual weak/deterministic pool scores more than being unexploitable; this is exactly why CFR's objective is the wrong target.*

2. **Belief-conditioned determinization sampling.** ISMCTS strength is bottlenecked by how you sample the hidden hand/deck. Uniform-over-consistent is a weak belief. Infer the opponent archetype from revealed cards, then sample hands/decks consistent with that archetype's known list. *Buys win-rate by making every search iteration realistic; it is the highest-leverage upgrade that makes ISMCTS beat heuristics rather than tie them.*

3. **Deck-archetype matchup matrix.** If you control deck selection, precompute (offline self-play between candidate decks) an empirical win-rate matrix across archetypes. *Buys win-rate two ways: pick a deck with a good worst-case column, and at inference read the opponent's archetype early and switch to the line the matrix says wins that pairing.*

4. **Metagame deck selection as a small explicit game.** Treat "which deck to bring" as a separate, tiny payoff matrix and solve it for a maximin / mixed strategy over your pool against the observed field. *Buys win-rate by avoiding being hard-countered, solvable in milliseconds, decoupled from expensive in-match search. This is the one place a Nash mindset earns its keep, at near-zero cost.*

5. **Heuristic-as-prior to make search budget-feasible.** Reframe the heuristic not as an alternative to search but as the prior that collapses search's iteration requirement. A good action prior plus leaf eval reaches strong play with an order of magnitude fewer playouts. *Buys win-rate by turning the "do we have a fast sim?" constraint from fatal into merely limiting, which is what makes any tree search viable under the time cap.*

6. **Risk-aware, board-state-conditioned play.** Add a term that values information-gaining or low-variance lines depending on board state: when behind, value forcing the opponent to commit attackers/Energy (information); when ahead on prizes, value low-variance lines. *Buys win-rate by shaping risk to the situation instead of playing risk-neutral throughout, converting marginal positions more reliably.*

**Cross-repo honesty note that constrains 3, 4, 6.** Across five prior repos, the durable output of every "clever" structured method was a boundary map and a controlled explanation, not a method that beat a strong discriminative baseline on accuracy. The matchup-matrix and metagame angles are legitimate for *diagnosis and deck selection*, but do not expect a low-rank "denoiser" of the matchup matrix to recover a hidden deck edge that sample-size-discounted empirical win rates don't already contain — three convergent negatives there. Before crediting any latent-archetype model for predicting match outcomes, run the dumb baseline (logistic regression / gradient boosting on hand-crafted deck features) as the yardstick. And the deck-strength vs pilot-skill confound is structural: from ladder win/loss alone you cannot separate "the deck is good" from "good players pick it." Break it with fixed-policy self-play (same agent across decks), or report the joint deck-pilot quantity and flag the non-identifiable split.

---

## 6. Open questions that need the live Kaggle pages

These cannot be settled by reconciliation or analogy. Read them on the live competition before any of them drives a decision.

1. **Forward-model access (highest priority).** Does the harness let you clone the state and call `step()` / `legal_actions()` yourself, or is it opponent-stepped with no rollout? This single fact decides whether expectimax/det-MCTS/ISMCTS are possible at all.
2. **Fast local simulator of the real Standard ruleset** provided to competitors, vs only the hosted match server. The gate for det-MCTS/ISMCTS/CFR/RL.
3. **Time budget: value, granularity, hardware.** Confirm ~10 min, confirm per-move vs per-match, confirm CPU-only vs GPU-in-harness, and confirm whether exhausting `remainingOverageTime` forfeits.
4. **Prize structure, verbatim.** Resolves the $50k vs $300k+ vs $424k vs $320k conflict. Organizer's exact figure only.
5. **Internet access and external pretrained models/weights** permitted inside the submitted agent at evaluation time. The "no GPU, large models impractical" claim is community analysis, not a quoted rule.
6. **Strategy judging criteria and weights**, and confirmation that stability + deck design + Simulation performance are exactly the official components.
7. **Submission mechanics, verbatim.** The `.tar.gz` / `main.py`-at-top / `deck.csv` packaging and the "self-play validation then matchmaking" flow currently come from an unrendered search snippet, not a read page. Also: leaderboard rating model for *this* competition (the Gaussian N(mu, sigma^2) claim is inferred from ConnectX/Lux), and the literal 5/day limit on the page.
8. **Data/Code tab:** whether organizers ship a downloadable card-pool CSV and/or an official starter notebook (only in-engine `all_card_data()` is confirmed).
9. **Eligibility/region restrictions** and the official team-size cap (max 5 from organizer material only).

---

## 7. Seed hypotheses for the registry

Each is a falsifiable claim with a test and an explicit refutation condition. Status `open` until a result on the named surface decides it. The surface is named because a verdict is only valid on the surface that produced it. "Refutes" rows are written to satisfy the tombstone contract (falsifying number + surface + re-open gate) when they fire.

1. **H1 (forward-model present).** The cabt harness exposes a clonable, steppable forward model usable for search. *Test:* instantiate the env locally, attempt to clone the state and step it forward from an arbitrary node. *Refuted if:* no clone/step is available and the agent only receives one observation per call. Re-open gate: a new engine release or doc that exposes a forward API.

2. **H2 (always-legal floor).** A heuristic agent that returns only `select`-offered indices and falls back to a safe default never emits an illegal move and never times out across N self-play games. *Test:* run N (target 2,000) self-play games, count illegal/timeout forfeits. *Refuted if:* any forfeit occurs from an illegal action or time-out.

3. **H3 (heuristic beats random decisively).** The Step-0 heuristic beats `random_agent` at a win rate whose lower CI bound exceeds 0.5 by a wide margin. *Test:* heuristic vs random_agent, n games, Wilson interval. *Refuted if:* win-rate CI includes or sits near 0.5.

4. **H4 (expectimax > heuristic, on the ladder surface).** Depth-limited expectimax on the same eval raises the **public ladder** rating over the Step-0 heuristic by more than the surface's noise band. *Test:* submit both, compare ratings after sigma shrinks; state the band first. *Refuted if:* the rating delta is inside the noise band. Do not promote a within-band move.

5. **H5 (timing budget granularity).** The match time budget is allocated per-move, not per-match. *Test:* read the live Rules tab; if ambiguous, instrument an agent that spends heavily on turn 1 and observe whether late turns are starved. *Refuted if:* the budget is documented or observed to be a single per-match pool.

6. **H6 (within-turn enumeration tractable).** Within-turn legal action sequences, after once-per-turn and dominated-line pruning, stay small enough for depth-1 expectimax to complete inside the per-decision budget on real decks. *Test:* instrument max and median sequence count per turn over n self-play games. *Refuted if:* the budget is exceeded on a non-negligible fraction of turns even after pruning.

7. **H7 (exploitation > equilibrium on this pool).** An opponent-adaptive response model out-rates a fixed (non-adaptive) policy of equal search depth on the live ladder. *Test:* submit adaptive and non-adaptive variants differing only in the opponent model; compare ratings. *Refuted if:* the adaptive variant does not exceed the fixed one beyond the noise band.

8. **H8 (archetype-conditioned determinization > uniform).** Sampling ISMCTS determinizations from an inferred-archetype belief beats uniform-over-consistent sampling, holding everything else fixed. *Test:* A/B self-play where only the belief sampler differs, plus a ladder check. *Refuted if:* the archetype sampler does not beat uniform beyond the noise band on the ladder. (Conditional on H1 being supported.)

9. **H9 (matchup residual below the noise floor at realistic n).** After deflating the global power-level mode from the deck matchup matrix, the residual rock-paper-scissors structure is below the specific-SNR floor at the per-matchup game counts we can actually log. *Test:* split-half stability of residual matchup edges (recompute on two random halves, correlate). *Refuted if:* residual edges are split-half stable above chance at the available n. *If supported, the decision is to play the global-power read, not the residual.*

10. **H10 (plain classifier ties or beats latent-archetype model).** A logistic-regression / gradient-boosting baseline on hand-crafted deck features predicts match outcome at accuracy at least as high as any latent-archetype embedding model. *Test:* same train/test split, both models, compare held-out accuracy. *Refuted if:* the embedding model beats the baseline beyond CI.

11. **H11 (deck-vs-pilot confound is real).** Ladder win rate by deck does not equal fixed-policy self-play win rate by deck; raw ladder tier lists are partly pilot selection. *Test:* run the same fixed agent across the candidate decks in self-play and compare the resulting deck ranking to the ladder ranking. *Refuted if:* the two rankings agree within sampling error.

12. **H12 (intervention reveals counter-structure that co-occurrence cannot).** Forcing a deck into the self-play field and measuring the directed win-rate response of the rest recovers an "A suppresses B" relation that symmetric usage/co-occurrence data does not show. *Test:* compare the forced-flow directed response to the co-occurrence matrix on the same decks. *Refuted if:* the directed response adds no information beyond the symmetric statistic. *Caveat: decidable in sim is not verified real-meta truth; adaptation mutes the signal.*

13. **H13 (decision is load-bearing, counterfactual test).** A specific agent decision (example: tech-card play timing) actually moves win rate. *Test:* re-simulate matches with only that decision randomized, everything else fixed; measure the win-rate flip. *Refuted if:* win rate is unchanged within the noise band, i.e. the decision was cosmetic.

14. **H14 (metagame mixed-strategy deck selection > best single deck).** Solving the small deck-selection payoff matrix for a maximin/mixed strategy out-rates always bringing the single highest-average deck against a non-stationary field. *Test:* A/B on the ladder, single-deck vs matrix-mixed selection. *Refuted if:* the mixed strategy does not exceed the single deck beyond the noise band.

15. **H15 (heuristic-as-prior cuts playouts an order of magnitude).** ISMCTS with the Step-0 policy as prior + leaf eval reaches the same playing strength as vanilla ISMCTS with roughly 10x fewer playouts. *Test:* fix a strength target (self-play win rate vs a frozen reference); measure playouts to reach it, prior vs vanilla. *Refuted if:* the playout reduction is far below an order of magnitude. (Conditional on H1.)

---

**Through-line.** The floor (Step 0) is the highest-EV engineering, not boilerplate. The ceiling (Steps 3-4) is real but gated on a forward model that is currently unconfirmed; resolve that gate before paying for search or RL. The cross-repo evidence says the structured matchup/metagame methods buy diagnosis, deck selection, and sample-size honesty, not raw win-rate over a well-tuned baseline. Bank conclusions to the registry, name the surface, and state the noise band before calling any delta real.
