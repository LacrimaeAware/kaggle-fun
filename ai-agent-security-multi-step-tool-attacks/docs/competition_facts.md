# Competition facts

Verified meta-facts about the competition, with confidence levels. Two source classes
are used. The bundled SDK on disk (`data/competition/aicomp_sdk`, version 3.1.0) and the
Kaggle evaluation harness (`data/competition/kaggle_evaluation/jed_attack_134815`) are
primary sources and are treated as ground truth for mechanics. The live Kaggle pages are
JavaScript-rendered and could not be fetched in full, so meta-facts taken from them are
marked accordingly and should be re-checked on the site before any decision depends on
the exact number.

## Identity

| Field | Value | Confidence |
| --- | --- | --- |
| Name | AI Agent Security: Multi-Step Tool Attacks | High |
| URL | https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks | High |
| Hosts | OpenAI, Google, IEEE | High (page and reference doc) |
| Category | Featured code competition | High |
| Prize pool | 50,000 USD | High |
| Working Note awards | Two awards of 2,500 USD each for the best written method and findings reports | Medium (single page snippet, consistent across two searches) |
| Internal code-name | "JED Red-Team Attack Competition", id 134815. SDK title: "JED: Replay-Based Security Benchmark for Tool-Using AI Agents" | High (harness package name) |
| Benchmark source | github.com/mbhatt1/competitionscratch, docs at mbhatt1.github.io/competitionscratch | Medium |

Note on "JED": it is the internal code-name in the evaluation harness. The acronym is
never expanded in any primary source found. The web claim that it means "Jailbreak,
Exploit, Defend" traces only to AI-generated search summaries and should be treated as
unverified. There is no confirmed link to a specific IEEE workshop (the four IEEE SaTML
2026 competitions do not include this one).

## Timeline

All from the local reference doc, consistent with web mentions. Confidence high for the
start and final dates, medium for the rest.

| Event | Date (UTC) |
| --- | --- |
| Start | 2026-06-11 |
| Entry deadline | 2026-08-25, 23:59 |
| Team-merger deadline | 2026-08-25, 23:59 |
| Final submission | 2026-09-01, 23:59 |
| Optional Working Note | 2026-09-08, 23:59 |

The window is about three months from start to final submission, with this work
beginning around 2026-06-13, so roughly eleven weeks remain.

## Format and rules

| Rule | Value | Confidence |
| --- | --- | --- |
| Submission | Notebook that writes `attack.py` (class `AttackAlgorithm`) to `/kaggle/working/` | High |
| Track | Red-team (attack) only for the public submission file | High |
| Internet | Disabled in the scored notebook | High |
| External and pretrained models | Allowed (freely and publicly available) | High |
| GPU runtime | At most 9 hours per notebook | High |
| Max team size | Not separately verified; standard Kaggle code-competition rules apply, solo allowed | Low |
| Daily submission cap | Not verified; code competitions are commonly about 5 per day | Low |
| Kaggle medals | Not separately confirmed for this competition | Low |

## Targets

Two open-weight models, attacked sequentially, each in its own server. Both run as
4-bit GGUF quantizations through llama.cpp on a T4-class GPU. Confirmed from the model
servers in the harness.

| Selector | Model | Repo and file | Tool-call format |
| --- | --- | --- | --- |
| `gpt_oss` | openai/gpt-oss-20b (about 21B total, 3.6B active, MoE) | unsloth/gpt-oss-20b-GGUF, gpt-oss-20b-Q4_K_M.gguf | OpenAI "harmony" channel tokens |
| `gemma` | google/gemma-4-26B-A4B-it (about 26B total, 3.8B active, MoE) | unsloth/gemma-4-26B-A4B-it-GGUF, gemma-4-26B-A4B-it-UD-Q4_K_M.gguf | Gemma 4 tool-call tags |

The "Gemma 4 26B-A4B-it" name in the prior reference doc is correct. The SDK also ships a
separate legacy `gemma` agent for google/gemma-3-4b-it, but it is not wired to any model
server and is not the target. Model details and attacker implications are in
[methods.md](methods.md) and [strategy.md](strategy.md).

## Scoring and leaderboard

- Public leaderboard: candidates are replayed against the source-available
  `OptimalGuardrail` (permissive). This is what you see while competing.
- Private leaderboard: candidates are also replayed against a stricter hidden guardrail.
  Final standing uses the private result.
- The submission produces one score row per model per guardrail:
  `gpt_oss_public`, `gpt_oss_private`, `gemma_public`, `gemma_private`. Each is a
  normalized score from 0 to 1000.
- Score formula and exact mechanics: see [sdk_reference.md](sdk_reference.md).

Top public score around 2026-06-13 was reported as 48.600, on the 0 to 1000 scale. This
was not independently verified (the leaderboard is JavaScript-rendered). If accurate, it
means the field is early and no one has run away with it. Participation was reported by a
single page snippet as roughly 1,988 entrants, 222 teams, 632 submissions, which moves
daily and is medium confidence at best.

## Time budgets (resolving the documented inconsistency)

The Kaggle pages and prior notes gave conflicting budget numbers (1,800 vs 9,000 vs
18,000 seconds). The harness source resolves it. There are two different layers.

- Real Kaggle gateway: 9,000 seconds per model for the attack-generation phase
  (`DEFAULT_BUDGET_S = 9000.0` in the gateway). With two models that is about 18,000
  seconds of generation total, plus replay time. Tool hops are hard-capped at 8 by the
  gateway (attacker requests are clamped down, and `reset` arguments that would raise the
  cap are dropped).
- Local CLI default: 1,800 seconds per track when `--budget-s` is omitted, and the
  local evaluator uses 4 tool hops by default in the `sandbox` env. To approximate the
  real gateway locally, use `aicomp evaluate redteam attack.py --env gym` and pass a
  budget explicitly.

So 1,800 is the local default, 9,000 is the real per-model budget, and 18,000 is the
aggregate across both models. They are not contradictory, they are different layers.

## What to re-verify on the live site

Before committing engineering decisions to a specific number, confirm on the Kaggle
Overview, Rules, Data, Evaluation, and Leaderboard tabs: exact max team size, daily
submission limit, whether Kaggle medals are awarded, the precise prize split beyond the
two Working Note awards, and the current leaderboard top and team count.

## Sources

- Kaggle competition pages (JS-rendered): the Overview, Data, Evaluation, Leaderboard,
  and Rules tabs of the URL above.
- Local primary sources: `data/competition/aicomp_sdk` (v3.1.0) and
  `data/competition/kaggle_evaluation/jed_attack_134815`, plus this folder's
  `competition_reference.md`.
- Benchmark repo and docs: github.com/mbhatt1/competitionscratch,
  mbhatt1.github.io/competitionscratch.
