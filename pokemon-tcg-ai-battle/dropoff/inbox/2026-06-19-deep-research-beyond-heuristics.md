# Deep research: beyond-heuristics / "true AI" methods for imperfect-info card games (2026-06-19)

Provenance: deep-research workflow run `wf_9f783553-b0e` (106 agents, 24 sources fetched, 118 claims
extracted, top 25 adversarially verified 3-vote, 22 confirmed / 3 killed, merged to 2 high-confidence
findings). This file is the saved record so the findings are usable later. A SECOND broad-methods run
(`wf_6d233761-717`: ISMCTS/determinization limits, expert-iteration, DAgger/covariate-shift, policy-as-
search-prior, representations) was still running when this was saved; append its results here when done.

## Question
Do imperfect-info / stochastic card-game agents go beyond hand heuristics via self-play? is RL required,
or do neural-guided ISMCTS / CFR / expert-iteration / supervised-from-replays suffice? how are heuristics
integrated without dominating? Grounded in our setup: forward model available, ~0.6s/move budget, single
fixed deck, thousands of replays, and our learned attempts only MATCH the hand-eval determinized search.

## Two high-confidence findings (post-verification)
1. **Self-play deep RL converges from scratch in stochastic card games**, but the closest analog (ByteRL,
   COG2022 Legends of Code and Magic winner) DROPPED neural-guided MCTS under a fixed budget in favor of a
   learned policy. Evidence: DouZero ranked 1st of 344 bots; ByteRL won LoCM; MCTS used fewer samples vs a
   learned policy. Sources: DouZero https://arxiv.org/pdf/2106.06135 , ByteRL https://arxiv.org/abs/2303.04096
2. **Learned-guided LOOK-AHEAD (CFR-family) beats a pure policy — not IS-MCTS — and heuristics integrate
   best as a PRUNING PRIOR, not as the value function.** Evidence: LAMIR (test-time depth-limited look-ahead)
   beats RNaD by up to 80; ReBeL beat poker pros; ODMC filters actions into Deep-Monte-Carlo for faster
   training. Sources: LAMIR https://arxiv.org/html/2510.05048v1 , ReBeL https://arxiv.org/abs/2007.13544 ,
   ODMC https://www.sciencedirect.com/science/article/abs/pii/S1568494624003193

## Confirmed claims (verifier previews; 3-0 unless noted)
- Self-play deep RL converged to state-of-the-art in card games.
- DeepNash mastered Stratego (imperfect-info) via model-free RL (RNaD), explicitly orthogonal to tree search.
  (Caveat: Stratego is deterministic.) Source: https://www.science.org/doi/10.1126/science.add4679
- DouZero's core algorithm is Deep Monte-Carlo (DMC); compute/sample cost is modest and fits a single GPU;
  the learned self-play policy beat eight state-of-the-art bots. (DMC = self-play + MC returns + value net,
  NO MCTS, NO CFR.)
- AlphaHoldem is an end-to-end deep-RL agent; a learned policy meets a tight per-move budget (fast inference).
- A PPO + ensemble-self-play deep-RL agent / a single end-to-end neural policy trained by self-play (LoCM).
- ByteRL (self-play learned agent) decisively beat opponents (2-1).
- ODMC injects a heuristic search as a PRUNING PRIOR over actions (not the value); speeds DMC training.
- ReBeL combines deep RL with search, trains a value AND policy net, achieved superhuman poker; "first
  algorithm enabling learning+search in imperfect-info games." LAMIR combines self-play policy-gradient with
  test-time depth-limited look-ahead that beats the pure policy (2-1). Student of Games (SoG) is a single
  unified RL+search algorithm (2-1).

## KILLED claims — do NOT cite these (failed adversarial verification)
- "AlphaHoldem self-play beat CFR agents (Slumbot/OpenStack) to SOTA HUNL" — refuted 0-3.
- "AlphaHoldem self-play training was cheap (3 days, 1 server) vs CFR" — refuted 1-2.
- "Self-play RL failed to beat search: 37.3% vs tree-search AI" — refuted 0-3. (So do NOT cite that specific
  stat; the *qualitative* caveat below stands, the number does not.)

## Caveats / open questions
- Caveat that survived: a card agent that beat a rule-based AI LOST to a tree-search AI — mirrors our own
  result (learning beats heuristics but not search). poker is simpler than a TCG; Stratego is deterministic.
- OPEN (literature gave no clean answer): does learned-guided search beat a hand-eval determinized search at
  EQUAL compute in a card game? Is offline RL better than self-play for a single fixed deck?

## Implications for THIS project
- The branch plan's direction is the evidence-backed one: keep SEARCH authoritative; use learning as a
  pruning prior / candidate proposer (heuristic-search-v2 fail-closed propose-and-verify is exactly the
  ODMC-style "heuristic = pruning prior" pattern) plus a fast policy/value to guide it; and prefer
  LOOK-AHEAD over a pure policy (LAMIR/ReBeL).
- The highest-ceiling "true AI" swing with real domain proof is **self-play Deep Monte-Carlo (DouZero-style)**
  — simpler than AlphaZero (no tree search at train or inference; instant policy fits the 0.6s budget). The
  blocker is sample/sim volume, i.e. our simulator wall-clock. Only viable if sim throughput is raised
  (parallelism / faster or learned simulator).
- CFR-family (ReBeL/SoG) is the theoretically correct imperfect-info tool and what wins poker, but needs
  belief-state subgame solving; feasibility under 0.6s/move is the uncertain part. High effort.
- RL is NOT strictly required to make progress now; neural-guided search + heuristics-as-prior is lower-risk
  and evidence-backed. The big swing (self-play DMC) is gated on fixing the sim bottleneck.

## Full source list (angle)
self-play RL cost: arxiv 2106.06135 (DouZero), science add4679 (DeepNash), aaai 20394 (AlphaHoldem),
arxiv 2002.06290, arxiv 2303.04096 (ByteRL), sciencedirect S1568494624003193 (ODMC).
CFR variants: arxiv 2007.13544 (ReBeL), arxiv 2510.05048v1 (LAMIR), sciadv adg3256 (SoG), arxiv 1701.01724 (DeepStack).
neural-guided ISMCTS vs heuristic: frontiersin frai.2023.1014561, arxiv 2106.06135, ar5iv 1709.09451.
heuristic integration: dl.acm 645528.657613, arxiv 2410.01458, 2204.02558, 1808.10120, 1808.04794.
offline RL / BC / replays: arxiv 2404.16689, 2403.00841, mlr v139/zha21a (RLCard/DouZero), DouZero-opponent-modeling, openreview MT2l4ziaxeE.

## Per-paper detail (the specific findings)
Verification key: [V] = claim verified in our run; [K] = claim KILLED in verification (do NOT cite); [bg] =
established background (not specifically re-verified here). Be conservative with exact numbers; several were killed.

- **DouZero** — "Mastering DouDizhu with Self-Play Deep RL" (arXiv 2106.06135).
  Method: **Deep Monte-Carlo (DMC)** [V] = average full self-play episode returns into a deep net + action
  encoding + massive parallel self-play. NO tree search, NO CFR [V]. Result: beat 8 prior SOTA bots [V];
  ranked 1st of 344 on Botzone [V]; compute modest / single-GPU-class server [V]. For us: the strongest
  precedent for a large-action imperfect-info card game won by SELF-PLAY without search; cost = self-play
  VOLUME (our sim-throughput bottleneck). DouZero+ added opponent modeling + coach-guided learning [bg].

- **DeepNash** — "Mastering Stratego" (Science 2022, add4679).
  Method: model-free RL via Regularized Nash Dynamics (R-NaD), converging toward Nash; NO search [V]. Result:
  expert / top-human level on Gravon [bg]. CAVEAT [V]: Stratego is imperfect-info but DETERMINISTIC (no
  chance) — less transferable to a stochastic TCG with draws/flips.

- **ReBeL** — "Combining Deep RL and Search for Imperfect-Information Games" (arXiv 2007.13544).
  Method: operates on PUBLIC BELIEF STATES; RL + CFR-style subgame search; trains a value net AND a policy
  net [V]. Result: superhuman heads-up no-limit poker [V]; "first algorithm enabling learning+search in
  imperfect-info games" [V]. For us: the principled RL+search recipe; belief-state subgame solving is heavy,
  feasibility under 0.6s/move is the open question.

- **Student of Games (SoG)** — (Science Advances 2023, adg3256).
  Method: one unified algorithm (growing-tree CFR + self-play RL + search) across perfect AND imperfect-info
  games [V, 2-1]. Strong on chess/Go/poker/Scotland Yard. Generalization of ReBeL.

- **LAMIR** — (arXiv 2510.05048v1).
  Finding [V]: test-time depth-limited LOOK-AHEAD on top of a learned policy beats the pure policy (RNaD) by
  a large margin (up to ~80 in the run). Evidence that look-ahead > pure policy in imperfect info -> keep
  search on top of any learned policy.

- **AlphaHoldem** — (AAAI 2022).
  Method: end-to-end deep RL (self-play, pseudo-Siamese), no explicit search/CFR, fast inference [V].
  IMPORTANT [K]: the claims that it BEAT CFR agents (Slumbot/OpenStack) and trained CHEAPLY were KILLED in
  verification. Treat it as "an end-to-end self-play RL poker agent with fast inference," NOT as proof that
  self-play beats CFR.

- **ODMC** — (sciencedirect S1568494624003193).
  Finding [V]: injects a heuristic search as a PRUNING PRIOR over actions feeding Deep-Monte-Carlo, speeding
  training. The "heuristic = prune the action set, not set the value" evidence -> validates fail-closed
  propose/prune (heuristic-search-v2).

- **ByteRL / LoCM** — (arXiv 2303.04096).
  Finding [V, 2-1]: self-play deep-RL agent won COG2022 Legends of Code and Magic; it DROPPED neural-guided
  MCTS under a fixed budget for a learned policy (MCTS needed fewer samples vs the learned policy). Under a
  tight budget, a fast learned policy was preferred over MCTS.

- **DeepStack** — (Science 2017, arXiv 1701.01724) [bg].
  Depth-limited continual re-solving + a learned counterfactual value net at leaves; first to beat pros at
  HUNL. Evidence for learned VALUE at search leaves in imperfect info.

## Monte Carlo in OUR project (clarification)
We ALREADY use Monte Carlo, of the SEARCH kind: `agent_search` is determinized Monte-Carlo / Perfect-
Information Monte Carlo (PIMC) -- sample N_DETERM hidden worlds, short forward rollout in each, score leaf,
average. That is DIFFERENT from DouZero's "Deep Monte Carlo (DMC)", which is an RL method (average self-play
returns into a net, no search). CFR/ReBeL is a third family (regret minimization over belief states). Same
word, three different machines. So "did we use Monte Carlo" = yes, the determinized-search kind.

# Run B (2026-06-19, wf_a8c22da9): history + what RELIABLY improves game AI
Provenance: deep-research run wf_a8c22da9-475, 103 agents, 21 sources, 102 claims, 25 verified
(23 confirmed / 2 killed). Answers: how poker/chess were actually built, does simulating more help,
and what is closest to "guaranteed if done right." [V]=verified, [K]=killed (do not cite).

## Headline ranking of reliable levers (the answer to "what is guaranteed")
1. **Real-time SEARCH on an approximate value/equilibrium** — strongest causal evidence. BUT it must be a
   CONSISTENT / equilibrium-aware search (CFR/OOS-style), NOT naive determinization.
2. **A good LEARNED leaf evaluation** — necessary (search alone has a low ceiling), but only SOUND in
   imperfect-info games when defined over BELIEF STATES (ReBeL), not action/observation history.
3. **Raw compute** — lawful (power-law in params/compute) but only LOGARITHMIC in search count, saturates,
   and mostly demonstrated for perfect-information self-play.

## Q1 Poker — NOT pure math, NOT heuristics, NOT pure deep learning [V]
- Limit hold'em was essentially solved by PURE CFR+ regret math (Cepheus, 2015; no neural net, no online
  search, a precomputed table).
- No-limit required SEARCH. The Libratus ablation is the cleanest causal evidence in the whole field: the
  raw self-play blueprint LOST to the prior best bot (-8 mbb/g); adding full nested subgame solving WON
  (+63 mbb/g) vs the SAME opponent. Search was the decisive lever. Libratus used no neural nets / no human data.
- Modern (ReBeL, 2020): self-play deep RL training a value+policy net over PUBLIC BELIEF STATES + CFR search
  at train AND test time; provably converges to Nash; beat pros under a <2s/decision budget.
- Sources: Cepheus http://johanson.ca/publications/poker/2015-science-hulhe/2015-science-hulhe.html ,
  Libratus https://noambrown.com/papers/17-Science-Superhuman.pdf ,
  ReBeL https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf

## Q2 Chess — almost entirely search + handcrafted eval until ~2017; now a HYBRID [V]
Classical alpha-beta + handcrafted eval (Deep Blue, classical Stockfish) until AlphaZero (2017, RL+MCTS),
Leela (2018), and Stockfish NNUE (2020, learned eval, ~80-100 Elo). Top chess today = classical search +
a learned (NNUE) evaluation TOGETHER, not one replacing the other. Source: https://arxiv.org/pdf/2209.01506

## Q3 Are we under-simulating? Split the blocker in TWO [V — the key insight]
- Determinized / Perfect-Information Monte Carlo (PIMC) -- which is exactly what agent_search is -- has TWO
  STRUCTURAL failure modes that more simulation provably CANNOT fix:
  - **Strategy fusion**: each determinization is solved as if fully observable, so the solver wrongly assumes
    it can pick a different strategy in each world; it OVERESTIMATES option values and can pick inferior moves.
  - **Non-locality**: search wastes effort on worlds the opponent can make vanishingly likely; breaks the
    minimax assumption even in tiny games (Biased RPS, Kuhn poker).
  These are BIAS, not sampling variance. ISMCTS exploitability stays FLAT with more search time (0.95/0.96/
  0.91); a consistent OOS/CFR method's exploitability FALLS with time.
- Determinization scaling SATURATES: strength rises with determinizations only to ~20 (in Dou Di Zhu), then
  flat; logarithmic in iterations per world. PIMC is WORST when leaf correlation is LOW. Whether PIMC works
  is predictable a priori from 3 measurable game-tree properties: leaf correlation, bias, disambiguation.
- [K] "doubling MCTS rollouts = +200 Elo" (refuted 0-3); [K] exact "saturate at 20 determ / 300 iter"
  thresholds (refuted 1-2) -- the qualitative saturation holds, the exact numbers do not.
- Sources: PIMC bias https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf ; OOS vs ISMCTS
  https://mlanctot.info/files/papers/aaai14-w4-iioos.pdf ; determinization scaling
  http://orangehelicopter.com/academic/papers/aisb11.pdf

## Q4/Q5 What reliably FAILS, and THE DIAGNOSIS FOR US [V — most important]
- RELIABLY BACKFIRES: naive AlphaZero-style RL+search in imperfect-info games is PROVABLY UNSOUND -- an
  action's value can depend on the probability it is played, so a state defined by action/observation
  history has no unique value (ReBeL paper). Also: "just simulate more" on determinized search; and trusting
  determinized values where leaf correlation is low.
- **Our pattern -- every learned addition only matches the search at equal budget -- is the EXPECTED
  signature of training on biased/high-variance DETERMINIZED labels. It is a methodology signature, not a
  modeling-capacity failure.** Two fixes depending on which problem dominates:
  - If it is sampling VARIANCE: **common random numbers** (evaluate compared options on the SAME sampled
    rollouts so the noise cancels = paired/antithetic evaluation), more determinizations up to saturation,
    and a lower-variance leaf value. (heuristic-search-v2's compare_selections paired evaluation IS this.)
  - If it is structural BIAS (strategy fusion / non-locality): no rollout count or learned ranker trained on
    those labels will help; the fix is a CONSISTENT / belief-state search (CFR / ReBeL-style).
- **DECISIVE DIAGNOSTIC (do this before more learning):** measure leaf correlation / bias / disambiguation
  on the Pokemon TCG game tree, and test whether a consistent-search's error FALLS with budget while the
  determinized one PLATEAUS. That tells us whether we are under-simulating (variance, cheaply fixable) or
  hitting PIMC bias (needs a different algorithm). Also: is the deployed agent past the determinization
  saturation point, or genuinely under-simulating?
- Sources: ReBeL (unsoundness) as above; Stratego PC-PIMC
  https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full ;
  variance reduction in MCTS https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf ;
  AlphaZero scaling laws https://arxiv.org/pdf/2210.00849 .
