# Methods and prior art

A reference library of automated red-teaming methods and comparable competitions, mapped
to this competition. Use it to choose seed templates and a search engine. The
recommended stack is in [strategy.md](strategy.md).

The task: write code that automatically searches a large space of multi-step prompt
chains to make a tool-using agent emit unsafe tool calls, scored by severity-weighted
predicate hits plus a coverage bonus. So the relevant methods are search and mutation
methods, and the relevant prior art is agent red-teaming contests.

## Comparable competitions and what won

These are public contests on essentially the same problem. Their winning tactics are your
seed library.

AgentDojo (ETH Zurich, NeurIPS 2024). The closest academic analogue: a benchmark of
tool-using agents over untrusted data, 97 tasks and 629 security cases across email,
banking, and travel. The single most reliable attack is the "Important Instructions"
pattern: inject a forcefully phrased fake authoritative instruction into a tool output.
Read this first for concrete templates. (arxiv.org/abs/2406.13352)

Gray Swan Arena agent red-teaming (with UK AISI). The largest live event: 1.8M attempts
against 22 tool-using agents, 62k breaches. Its four vulnerability classes mirror this
competition's predicates (confidentiality breach, information-hierarchy violation,
action-hierarchy violation, conflicting objectives). It scored unique breaks, the same
diversity idea as cells, and the leaders won on breadth. Winning style was creative
prompt engineering: fake approval messages, spoofed system or tool messages, and XML or
structured-format confusion. (grayswan.ai news; arxiv.org/abs/2507.20526)

HackAPrompt (EMNLP 2023). About 600k adversarial prompts, with a useful taxonomy ranked
by generality: compound instruction attacks (chaining two or more instructions, present
in nearly all successful injections), context ignoring ("ignore your instructions and"),
context overflow (padding), special case ("special instruction:"), refusal suppression
(forbidding refusal language), and the two-token attack. (arxiv.org/abs/2311.16119)

LLMail-Inject (Microsoft, IEEE SaTML 2025). The best methodological match: inject into an
email so an email agent performs an unauthorized tool action, a multi-step tool-call
exfiltration scenario. 292 teams, 208k unique attacks. Winning tactics: special tokens in
the body, using the subject line both to win retrieval and to carry the injection while
the body stays benign, and tailoring to each model's attention behavior. All automatable.
(microsoft.github.io/llmail-inject)

SaTML 2024 LLM CTF (ETH). Secret-extraction attack and defense, 137k multi-turn chats,
every defense bypassed at least once. Confirms that multi-turn extraction reliably beats
prompt-level defenses. (spylab.ai/blog/results-competition)

Predecessor note. The earlier gpt-oss-20b red-teaming Kaggle event (500k pool, 600-plus
teams) was a human-judged writeup hackathon, where headcount and report volume mattered.
This competition is the opposite: automated, leaderboard-scored code. That difference is
why solo is more viable here (see [solo_outlook.md](solo_outlook.md)).

## Automated attack-search methods

Each entry: the idea, then how it maps to searching multi-step tool-call chains here.

Go-Explore and quality-diversity (MAP-Elites, Rainbow Teaming). Keep an archive of
reached cells, return to a promising cell, explore one step further, and keep any new
cell. This is the structural match: the competition literally calls its novelty unit a
cell, and the score rewards quality-diversity. Define a coarse predicate-aware cell, keep
the best partial trace per cell, replay to a prefix, then mutate the next step. A 2026
case study adapting Go-Explore to AI red-teaming reports four lessons that should shape
your engine: seed variance dominates outcomes (use multiple seeds), reward shaping
consistently hurts (keep reward sparse and binary), simple state signatures beat complex
ones, and ensembles of templates buy attack-type diversity while a single agent maximizes
coverage within one type. (arxiv.org/abs/2601.00042; Rainbow Teaming
arxiv.org/abs/2402.16822)

PAIR and TAP (attacker-LLM refinement). An attacker model proposes an attack, reads the
target's response, and refines, looping. TAP adds branching and pruning and reports high
success with far fewer queries. Map: at each step, an attacker model proposes the next
injected content or message; branch on the predicate signal and prune to stay in budget.
Best used as the per-cell mutation operator inside a Go-Explore archive so refinement
targets new cells. (arxiv.org/abs/2312.02119)

GCG and AutoDAN (suffix optimization). GCG uses gradients to optimize an adversarial
suffix; AutoDAN uses a genetic algorithm for fluent suffixes. Applicability caveat: the
targets here are GGUF models served through llama.cpp, which is effectively black-box with
no convenient gradient API, so vanilla GCG does not fit. Options are to run GCG offline
against a full-precision copy of the same model and transfer the suffix, or to prefer
AutoDAN's genetic variant, which only needs forward scores. Gibberish suffixes also risk
tripping a guardrail, so fluent is safer. Treat this family as an optional booster, not
the backbone. (arxiv.org/abs/2407.01599)

Fuzzing and mutation search (GPTFuzzer and successors). Seed with human templates,
mutate by paraphrase, expansion, crossover, and shortening, and select seeds with a
tree search. Reports high transferable success, and AgentFuzzer extends it to black-box
indirect injection against agents. This is the cheapest high-coverage engine for a solo
competitor: seed with the template library above, mutate, and route survivors into the
archive. (arxiv.org/abs/2309.10253; AgentFuzzer arxiv.org/abs/2505.05849)

Crescendo and many-shot (multi-turn). Crescendo starts benign and escalates over turns,
referencing the model's own prior replies, which evades per-turn input filters. This fits
the multi-step framing: a benign tool call establishes context, then a follow-up turns it
into a destructive write or exfiltration, which is harder for a guardrail to catch than a
single overt step. (arxiv.org/abs/2404.01833)

Attacker LLM as the generator. A capable model can serve as the mutation and
next-move operator inside any of the above. Useful if budget allows, but the offline
template-plus-archive engine should work without it.

## Why quality-diversity fits, and the one correction

The score is quality-diversity by construction: coverage of distinct cells plus depth of
severity within them. A single perfect jailbreak (the one-suffix objective of GCG or PAIR)
maxes one cell and forfeits coverage. So a Go-Explore archive with a template-and-fuzzing
mutation operator is the best fit for a solo competitor.

The one correction to that received wisdom for this specific SDK: because severity is
summed over findings with no de-duplication and findings are capped at 2,000 (see
[strategy.md](strategy.md), Lever 1), raw severity volume can dominate the coverage bonus
by a large factor. So the engine should first secure a few reliable high-severity chains
and exploit the finding budget, and use the quality-diversity machinery to broaden
coverage as the secondary term. Verify the duplication mechanic with a controlled
submission before committing to it.

## Target-model notes for crafting tool calls

The unsafe artifact must be a parseable tool call, so the attack has to produce the
model's tool-call format. From [competition_facts.md](competition_facts.md):

- GPT-OSS 20B uses the harmony format. A tool call is a `commentary` channel message with
  a `to=functions.<name>` recipient and a JSON body. Its deliberative-alignment safety is
  brittle in this exact deployment (4-bit GGUF on a Turing T4): reported effects include
  guardrail erosion under low-bit quantization, inconsistent refusal by reasoning depth,
  and long-reasoning bypasses. Levers: reasoning-level manipulation and structural or
  format-token injection that gets a `commentary` tool call out directly.
- Gemma 4 26B-A4B-it uses simple tool-call tags and has historically leaned on prompt
  engineering rather than hard gating, with a shallow refusal posture (community work
  drives residual refusals below one percent). It is the softer target. Levers: direct
  task framing, persona and roleplay, and benign-looking multi-step decomposition. There
  is a documented chat-template bug around messages that carry both content and a tool
  call.
- General principle for both, being small and quantized: multi-turn decomposition,
  authority and roleplay framing, fictional or historical wrappers, and injecting the
  model's own format tokens transfer well. Quantization makes both more compliant than
  their full-precision reputations.

## Pointers

- Lethal trifecta: simonwillison.net/2025/Jun/16/the-lethal-trifecta
- OWASP LLM Top 10 (2025), prompt injection LLM01 and excessive agency LLM06:
  genai.owasp.org
- Indirect prompt injection origin: Greshake et al., arxiv.org/abs/2302.12173
- AgentDojo, Gray Swan, HackAPrompt, LLMail-Inject as cited above.
