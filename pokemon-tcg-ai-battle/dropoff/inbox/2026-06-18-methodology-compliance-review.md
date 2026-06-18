# Methodology compliance review

Date: 2026-06-18

## Files reviewed

- `research/deep-research-report-2026-06-18.md`
- User-provided current-state review, saved as `dropoff/inbox/2026-06-18-external-current-state-methodology-review.txt`
- Prior inbox handoffs in `dropoff/inbox/`

## Image / asset note

The deep-research report exists locally as:

```text
research/deep-research-report-2026-06-18.md
```

There is also a copy at:

```text
dropoff/inbox/2026-06-18-deep-research-report.md
```

The markdown begins with links to:

```text
sandbox:/mnt/data/pokemon_tcg_research_report.md
sandbox:/mnt/data/pokemon_tcg_research_report.pdf
```

Those are stale links from the original generation environment and should not be expected to work locally.

In the local markdown report itself, I see Mermaid diagram blocks, not normal embedded image files.

Relevant diagram blocks:

```text
```mermaid
gantt
...
```
```

and:

```text
```mermaid
flowchart TD
...
```
```

So if the user remembered "images," they were likely rendered Mermaid diagrams in the original UI/PDF, not downloaded PNG/JPG assets inside this repo. The report can still render diagrams in a Mermaid-capable viewer, but there do not appear to be local report image assets to preserve.

Some research notebooks do contain remote or attachment image references, but those are separate notebook assets, not the deep-research methodology markdown.

## How seriously should the project stick to the deep-research methodology?

The project should treat the methodology as a strong steering document, not as a rigid script.

Stick closely to these core methodological claims:

- The central bottleneck is state-value-to-action conversion.
- The main target should be sibling legal-action ranking within the same decision.
- State-only value prediction and global AUC/Pearson are not sufficient success metrics.
- The learner needs root context, option/action identity, root-to-leaf deltas, and preferably effect/action descriptors.
- Labels based only on the current hand evaluator are circular and unlikely to surpass the teacher.
- Stronger labels should include at least one source of information beyond the current hand evaluator.
- Criticality matters: high-impact decisions should be analyzed separately from low-impact decisions.
- Learned representations and embeddings should be judged by sibling-action discrimination and gameplay, not by pretty embedding spaces or global prediction.
- Belief-conditioned determinization and replay-derived opponent information remain important, but should not be mixed up with the action-ranking question.
- Full A/B win-rate should be the final gate, not the only diagnostic.

Be flexible on these implementation details:

- Exact sample sizes like 800-1600 games may be too slow for daily iteration; use them for final gates, not every check.
- The roadmap order can be adjusted if the current work already has a partial neural/action pipeline.
- "Start with transparent baselines before neural encoders" is good scientific hygiene, but if the user explicitly wants the integrated neural/effect system, build the integrated spine and use ablations to keep it interpretable.
- Mermaid/PDF rendering is cosmetic; do not block research work on recovering rendered images.

## Did the implementation actually follow the methodology?

Short answer:

```text
Partially, but not enough.
```

The project listened to the diagnosis rhetorically, but repeatedly implemented easier surrogates.

The deep report said:

```text
Rank legal sibling actions from the same root decision.
```

The project often implemented:

```text
Fit absolute resulting-state values or hand-search leaf scores.
```

Those are related, but they are not equivalent.

## Compliance scorecard

### Sibling-action ranking

Status:

```text
Partially followed, not fully executed.
```

Evidence:

- The project created action-data machinery and grouped decisions.
- It discussed sibling ranking repeatedly.
- But the visible/current experiments often used pointwise regression/classification over leaf or state features.
- E013/H024-like work appears to have been treated as "done" despite not satisfying the full grouped pairwise/listwise methodology.

Gap:

```text
The learner still did not consistently learn Q(root, action) or advantage(root, action).
```

Needed:

- Root features.
- Action/option descriptors.
- Leaf features.
- Delta features.
- Group-aware pairwise/listwise objective.
- Within-decision metrics.

### Card embeddings / learned representations

Status:

```text
Mostly not followed.
```

The methodology specifically discussed magnitude-aware card representations and learned encoders/embeddings.

What happened instead:

- Engineered tags and board-summary features remained the main live representation.
- `card_effects.json` was built, which is useful.
- But `card_effects.json` was not consumed by `search_v`.
- Learned card embeddings were not clearly implemented as a live decision feature.
- Recent effect-aware work reportedly wired effects into a hand heuristic, not into a learned model.

This is one of the user's strongest complaints, and I think it is valid.

The project did not really test:

```text
card id embedding + decoded effects + state context + action ranking
```

It tested nearby things.

### Card effects

Status:

```text
Decoder built, decision integration weak/late.
```

The decoder is a real foundation. But until it is used inside the live action scorer or learned model, it is not evidence for or against the user's idea.

Recent hand-weighted effect-policy failure should not be interpreted as the methodology failing. It mostly shows that naive hand weights can bulldoze the baseline by choosing setup over attacks.

Correct next interpretation:

```text
Card effects need to become state-conditioned action features or learned residuals, not standalone hand bonuses.
```

### Root/action/delta modeling

Status:

```text
Recognized, partially scaffolded, not consistently central.
```

The deep report emphasized:

```text
root features + action descriptor + leaf features + leaf-minus-root delta
```

The project has some `option_deltas` machinery, but prior learned branches mostly remained state/leaf value models.

This is a major methodology miss.

### Multi-head / auxiliary future-option objectives

Status:

```text
Mostly not implemented.
```

The methodology discussed future option counts and auxiliary heads as a medium-priority direction.

The current project appears to have talked about these more than it used them as first-class training targets.

This is less severe than missing action ranking, but it is still a gap.

### Belief-conditioned determinization

Status:

```text
Not yet meaningfully followed.
```

The report strongly recommended replay-derived hidden-state/opponent-deck modeling.

The project found replay access and noted both decks are available in replay JSON, which is good.

But a learned or replay-grounded belief sampler does not appear to be driving `search_begin` yet.

### Search-budget sweep

Status:

```text
Partially discussed, not cleanly completed as methodology.
```

The report recommended disciplined sweeps over determinizations, rollout policy, and horizon.

The project has done search experiments, but the user is right to be wary if these became ad hoc long A/B runs rather than a controlled matrix with clear logs and acceptance criteria.

### Evaluation discipline

Status:

```text
Improving, but still leaky.
```

Good:

- The project uses registries and A/B tests.
- It has found real bugs.
- It has Wilson intervals in some places.

Bad:

- Some statuses appear to conflate implemented/trained/evaluated/accepted.
- E013-like work was marked done without clearly satisfying H024's intended methodology.
- Results were sometimes interpreted as testing ideas they did not actually test.
- `search_v` was interpreted in the neighborhood of card-effect learning even though it ignored `card_effects.json`.

This is the control-plane bug that keeps making the project feel maddening.

## How much did it "listen"?

My honest assessment:

```text
It listened to the words and diagnosis, but often implemented weaker substitutes.
```

It did listen on:

- The importance of search.
- The importance of action ranking as a concept.
- The existence of replay data.
- The need for registries and confidence intervals.
- The idea that global value metrics can fail to improve play.

It did not sufficiently listen on:

- Learned card embeddings / representation learning as a live decision path.
- Explicit decoded effects as input to the learned action model.
- True sibling-action ranking rather than leaf/state scoring.
- Root-action-delta inputs.
- Non-circular or stronger-than-hand-eval labels.
- Multi-head/future-option objectives.
- Separating "implemented" from "validated."

This explains why the user feels like the same idea keeps being "tested" without ever actually being built.

## Current-state review assessment

The user-provided external current-state review is largely aligned with this read.

Its most important claim is:

```text
The project keeps implementing nearby, easier surrogates for the real research question.
```

I agree.

Its strongest specific criticism is:

```text
The action-ranking experiment is not yet a true action-ranking experiment.
```

I also agree, based on the described mismatch between H024/E013 and the implemented target/objective.

Its warning about teacher circularity is important:

```text
A learner trained only on hand-search leaf scores should not be expected to systematically beat the hand-search teacher.
```

That does not mean teacher distillation is useless. It means it is mainly useful for compression/stabilization unless the teacher target includes stronger information.

## What should be considered binding from here?

The next model should be held to these requirements:

1. Do not call a run "card-effect learning" unless `card_effects.json` is consumed by the live action model.

2. Do not call a run "action ranking" unless options are grouped by root decision and trained/evaluated with within-decision ranking metrics.

3. Do not call a run "embedding learning" unless card identity/effect vectors are trainable or explicitly represented inside the model that scores actions.

4. Do not call a result "validated" unless it has the relevant ablation:

```text
full model vs feature-zeroed / component-removed model
```

5. Do not use global AUC/Pearson as a success claim for action quality.

6. Do not treat hand-eval-only labels as a plausible route to beating hand search unless there is another source of information in the target.

7. Do not let "done" mean merely "script ran."

Use statuses like:

```text
specified
implemented
data-generated
trained
offline-evaluated
arena-evaluated
accepted
refuted
inconclusive
```

## Recommended next plan

Build the integrated action-conditioned model, but make it auditable.

The minimum real spine:

```text
root state
+ option/action descriptor
+ card id embedding
+ decoded card effects
+ state x effect interactions
+ option_deltas / root-to-leaf delta
-> shared model
-> action-ranking logits over siblings
```

This is not a toy proof step. It is the actual methodology's central object.

Required offline reports:

- top-1 within-decision accuracy
- pairwise preference accuracy
- selected-action regret
- high-criticality subset performance
- option-0 baseline
- current heuristic/search teacher baseline
- ablations for no effects, no card id, no deltas

Only after that should it become:

```text
policy prior
search tie-breaker
search leaf/ranker hybrid
full arena candidate
```

## Bottom line

The deep-research methodology was directionally good and should be followed on the core question.

The project did not actually execute the most important parts cleanly.

The user's intuition that "it specifically told us to do embeddings/effects/action ranking and then we somehow did adjacent stuff instead" is basically correct.

The next work should not be another broad reset. It should be one real H024-v2/action-conditioned ranker path that includes the representation pieces the report called out, with ablations that make it impossible to confuse "artifact exists" with "agent used it."

