# Strategy write-up resources

This file maps the evidence and source documents that can support the eventual Kaggle Strategy Category write-up.

## Competition and rules

| Resource | Location | Use |
|---|---|---|
| Raw Strategy Category overview | `research/competition/strategy/raw-strategy-category-overview-2026-06-18.txt` | Scoring weights, deadlines, submission requirements, citation. |
| Competition research notes | `research/competition/findings.md` | Engine/API facts, track distinction, rule/source caveats. |
| Official Strategy URL | `https://kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy` | Primary citation and final rule checks. |

## Project methodology sources

| Resource | Location | Use |
|---|---|---|
| Deep research report | `research/deep-research-report-2026-06-18.md` | Main methodology: sibling-action ranking, belief-conditioned search, representation upgrades. |
| Deep research report copy | `dropoff/inbox/2026-06-18-deep-research-report.md` | Inbox-accessible copy for other agents. |
| Methodology compliance review | `dropoff/inbox/2026-06-18-methodology-compliance-review.md` | Where implementation followed or failed to follow the research plan. |
| Card effects/action-prior handoff | `dropoff/inbox/2026-06-18-card-effects-action-prior-handoff.md` | Guardrails around card effects, embeddings, and action ranking. |
| Current research doc | `docs/RESEARCH.md` | Living project research direction. |
| Learning plan | `docs/LEARNING_PLAN.md` | Experiment discipline, learning branch assumptions. |
| Model communication | `docs/MODEL_COMMUNICATION.md` | Known issues, model-to-model handoff cautions. |

## Experiment and result sources

| Resource | Location | Use |
|---|---|---|
| Experiments registry | `registry/experiments.jsonl` | Hypotheses and experiment definitions. |
| Results registry | `registry/results.jsonl` | Canonical completed metrics where present. |
| A/B runner | `tools/run_ab.py` | Head-to-head evaluation procedure. |
| Arena wrapper | `agent/cabt_arena.py` | Seat swapping, deck wrappers, evaluation mechanics. |
| Search agent | `agent/search.py` | Forward-model search and option evaluation. |
| Main agent entrypoint | `agent/main.py` | Current live agent families and deck selection. |
| Card effects decoder | `tools/build_card_effects.py` | Decoded card text/effect features. |
| Card effects artifact | `agent/card_effects.json` | Effect features, only meaningful when consumed by live policy. |

## External method references to cite later

These are listed in the deep-research report and should be checked before final citation formatting.

| Theme | Candidate sources | Write-up use |
|---|---|---|
| Imperfect-information search | Cowling et al. on ISMCTS / ensemble determinization | Justifies determinization and hidden-state modeling. |
| Search and rollout caution | Gelly and Silver / MCTS rollout policy cautions | Explains why stronger rollout heuristics can bias search. |
| Card-game ranking | Bertram et al. contextual preference ranking for Magic | Supports sibling/action ranking framing. |
| Contrastive preference learning | Bertram et al. contextual InfoNCE | Supports decision-supervised embeddings. |
| Generalized card representations | Bertram et al. generalized card reps for Magic | Supports magnitude-aware card features / embeddings. |
| Search + learning in imperfect information | ReBeL / Student of Games | Long-term blueprint, not near-term claim. |
| Strategy card-game RL | LOCM / ByteRL / Hearthstone opponent modeling | Background for possible future work. |

## Source discipline

- Use primary/local project files for claims about this repo.
- Use Kaggle page/rules for competition requirements.
- Use external papers only to justify method families, not to claim our implementation already achieved their results.
- Before final submission, verify all URLs and paper citations.

