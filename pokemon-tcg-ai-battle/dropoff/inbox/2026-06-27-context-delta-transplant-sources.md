# Context-Delta Transplant Audit — Source List (URLs)

Date: 2026-06-27
These are the actual sources the deep-research run fetched and verified. 24 sources fetched, 103 claims extracted, 25 adversarially verified.
Identification confidence is marked: [confirmed] = I am sure of the paper from the URL; [inferred] = identified from the URL/arXiv id, title not independently re-verified by me, click through to confirm.

## Prior-art landscape (primary)
- [confirmed] Ormoneit & Sen (2002), Kernel-Based Reinforcement Learning — https://www.researchgate.net/publication/2353479_Kernel-Based_Reinforcement_Learning
- [confirmed] Blundell et al. (2016), Model-Free Episodic Control (MFEC) — https://www.gatsby.ucl.ac.uk/~ucgtcbl/papers/BluUriPriLiRudLeiRaeWieHas2016a.pdf
- [confirmed] Pritzel et al. (2017), Neural Episodic Control (NEC) — https://arxiv.org/pdf/1703.01988 ; ICML page: https://proceedings.mlr.press/v70/pritzel17a.html
- [confirmed] Goyal et al. (2022), Retrieval-Augmented Reinforcement Learning (R2A) — https://arxiv.org/pdf/2202.08417
- [confirmed] Shah & Xie (2018), Q-learning with Nearest Neighbors (sample complexity + minimax lower bound) — https://arxiv.org/pdf/1802.03900

## Offline RL / OOD-action / conservative methods (primary unless noted)
- [confirmed] Levine, Kumar, Tucker, Fu (2020), Offline RL: Tutorial, Review, and Perspectives — https://arxiv.org/abs/2005.01643
- [confirmed] Fujimoto et al. (2019), Off-Policy Deep RL without Exploration (BCQ) — https://www.researchgate.net/publication/329525481_Off-Policy_Deep_Reinforcement_Learning_without_Exploration
- [confirmed] Fujimoto & Gu (2021), A Minimalist Approach to Offline RL (TD3+BC) — https://openreview.net/pdf?id=Q32U7dzWXpc
- [inferred] Implicit Q-Learning (IQL, Kostrikov et al. 2021) overview (secondary, explainer site) — https://www.emergentmind.com/topics/implicit-q-learning-iql

## Causal / treatment-effect / post-treatment bias (primary unless noted)
- [confirmed] Cinelli, Forney & Pearl, A Crash Course in Good and Bad Controls — https://ftp.cs.ucla.edu/pub/stat_ser/r493.pdf
- [confirmed] Montgomery, Nyhan & Torres (2018), How Conditioning on Post-Treatment Variables Can Ruin Your Experiment (AJPS) — https://onlinelibrary.wiley.com/doi/abs/10.1111/ajps.12357
- [inferred] Künzel, Sekhon, Bickel, Yu (2017/2019), Meta-learners for estimating heterogeneous treatment effects (CATE / T-/X-learner) — https://arxiv.org/pdf/1706.03461
- [inferred] Causal/post-treatment-conditioning epidemiology paper — https://pubmed.ncbi.nlm.nih.gov/30111904/
- [inferred] Causal mediation / overcontrol paper (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC5784842/
- [confirmed] Facure, Causal Inference for the Brave and True — Ch.21 Meta-Learners (secondary, textbook site) — https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html

## Off-policy evaluation (better-choice benchmark) (primary)
- [confirmed] Thomas & Brunskill (2016), Data-Efficient Off-Policy Policy Evaluation (doubly robust) — https://arxiv.org/abs/1511.03722
- [confirmed] Voloshin et al. (2019/2021), Empirical Study of Off-Policy Policy Evaluation for RL — https://arxiv.org/abs/1911.06854
- [inferred] Off-policy evaluation paper (OpenReview) — https://openreview.net/pdf?id=kWSeGEeHvF8

## State/action abstraction & metric learning for retrieval (primary)
- [confirmed] Zhang et al. (2020), Learning Invariant Representations for RL without Reconstruction (Deep Bisimulation for Control, DBC) — https://arxiv.org/abs/2006.10742
- [inferred] Castro et al. (2021), MICo: Improved representations via sampling-based state similarity (bisimulation-style metric) — https://arxiv.org/abs/2110.14096
- [confirmed] Barreto et al. (2016), Successor Features for Transfer in Reinforcement Learning — https://arxiv.org/abs/1606.05312
- [inferred] State/action abstraction or metric-learning paper (2023) — https://arxiv.org/abs/2301.11490
- [inferred] State/action abstraction or metric-learning paper (2024) — https://arxiv.org/pdf/2406.04056

## Claims that were REFUTED in verification — do NOT cite these as support
1. "R2A conditions on the ENTIRE dataset = the same novel thesis as the transplant" — refuted 0-3 (src: R2A, arXiv 2202.08417)
2. "NN regression gives a provably ε-accurate Q estimate from a single logged path under arbitrary policy" — refuted 0-3 (src: Shah & Xie, arXiv 1802.03900)
3. "The estimator is statistically consistent without any causal/ignorability assumption" — refuted 0-3 (src: KBRL). Consistency of this estimator form cannot be assumed.
4. "DQN/DDPG cannot learn at all from an off-distribution batch" — refuted 1-2 (src: BCQ)
5. "All support-constraint offline-RL methods differ only in how 'closeness' is implemented" — refuted 1-2 (src: TD3+BC openreview). Support-gating is ONE instance, not the whole family.

## Verification stats
5 search angles; 24 sources fetched; 103 claims extracted; 25 verified by 3-vote adversarial refute; 20 confirmed, 5 killed; 9 findings after synthesis; 106 agent calls total.
Full machine-readable output (claims, votes, per-claim sources) was produced by the run; this file is the durable human-readable index. The companion analysis is in 2026-06-27-context-delta-transplant-research-audit.md in this same folder.
