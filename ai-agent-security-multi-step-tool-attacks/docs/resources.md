# Resources: people, tools, and practice

A curated, maintained link library for learning AI security and practicing hands-on. This
is the most volatile document in the folder: links rot, tools change hands, and challenges
open and close. Re-verify before relying on any single item.

Last reviewed: 2026-06-14. Concepts behind these links are in [field_guide.md](field_guide.md).

## People to follow

Most beginner-friendly first.

- Simon Willison. Coined "prompt injection"; the best plain-English explainer in the field
  and the single best subscription for a beginner. Blog tag: simonwillison.net/tags/prompt-injection.
  Newsletter: simonw.substack.com.
- Johann Rehberger (Embrace The Red, "wunderwuzzi"). Prolific practical exploit researcher
  (ChatGPT memory poisoning, Copilot RCE, many MCP data-exfiltration bugs). Concrete and
  reproducible. embracethered.com/blog.
- Rich Harang (NVIDIA AI Red Team; a PhD statistician, relevant to your background).
  Rigorous defender-oriented guidance and the "AI Kill Chain." rharang.github.io.
- Nicholas Carlini (Anthropic). The leading academic on breaking ML defenses (adversarial
  examples, training-data extraction, the Carlini-Wagner attack). Deeper; read once you have
  footing. nicholas.carlini.com/writing.
- Pliny the Liberator (@elder_plinius). Runs the L1B3RT4S jailbreak-prompt repo. Useful as a
  live technique feed, heavy on hype; treat as a primary-source feed, not analysis.

## Labs and company research blogs

- Trail of Bits (blog.trailofbits.com/categories/machine-learning): rigorous technical AI
  audits; good for seeing how real audits are done.
- Lakera (lakera.ai/blog): runs Gandalf; agent and visual injection research. Beginner-friendly.
- Gray Swan AI (grayswan.ai/blog, app.grayswan.ai/arena/blog): runs large public red-teaming
  arenas; technique write-ups. Directly relevant to a red-teaming competition.
- Dreadnode (dreadnode.io/research): offensive-AI startup; runs Crucible CTFs and publishes
  AIRTBench (can LLMs red-team other LLMs).
- HiddenLayer (hiddenlayer.com): annual AI Threat Landscape Report; big-picture threat view.
- Promptfoo (promptfoo.dev/docs/red-team): open-source eval and red-team tool; docs double as
  a free curriculum. Beginner-friendly.
- NVIDIA AI Red Team (developer.nvidia.com/blog, filter AI Red Team): practical defender
  guidance.
- Microsoft MSRC and AI Red Team (microsoft.com/security/blog): maintains PyRIT.
- Anthropic (alignment.anthropic.com, red.anthropic.com): safeguards and frontier red-team
  research.
- Google Project Zero and DeepMind (projectzero.google): "Big Sleep" agent finding real
  0-days; the frontier of AI for offense.

## Academic groups

- ETH Zurich SPY Lab (Florian Tramer): authors of AgentDojo, the standard agent-security
  benchmark. If you study one academic artifact for this competition, make it this.
  spylab.ai, github.com/ethz-spylab/agentdojo.
- Invariant Labs (ETH-affiliated, now part of Snyk): AgentDojo tooling, MCP security,
  MCP-Scan. invariantlabs.ai.
- UK AI Security Institute (AISI): maintains Inspect, a widely used eval framework.
  aisi.gov.uk, inspect.aisi.org.uk.
- Berkeley RDI: runs an Agentic AI MOOC (beginner-friendly) and benchmark competitions.
  rdi.berkeley.edu.
- Andy Zou (CMU and Gray Swan): GCG attack, HarmBench; bridges academia and the arenas.

## Newsletters, podcasts, communities, conferences

- Newsletters: Simon Willison's newsletter (best single subscription); CAIS AI Safety
  Newsletter (safe.ai/newsletter); Adversarial AI Digest (github.com/TalEliyahu/AI-Security-Newsletter).
- Podcasts: MLSecOps Podcast (mlsecops.com/podcast); The AI Security Podcast.
- Communities: OWASP GenAI Security Project and its Slack (genai.owasp.org; most
  beginner-friendly community); DEF CON AI Village (aivillage.org); Gray Swan Arena Discord;
  Dreadnode Crucible; r/LocalLLaMA and r/netsec.
- Conferences: IEEE SaTML (satml.org; the dedicated academic venue, has a competition track);
  USENIX Security and IEEE S&P; NeurIPS and ICML safety workshops; Black Hat and DEF CON.

## Tools (hands-on)

"Local" runs on your machine; "needs key" means point it at a model endpoint (a paid API or
a local model via Ollama or Hugging Face). The tools are free either way.

Scanners (point and shoot):

- NVIDIA garak (github.com/NVIDIA/garak): LLM vulnerability scanner, 50-plus probe modules.
  `pip install garak`, point at any model including local Ollama. Beginner-friendly. Best
  first automated sweep.
- promptfoo (promptfoo.dev/docs/red-team): YAML-driven eval and red-team CLI, attacks across
  50-plus vulnerability types, report mapped to OWASP. Beginner-friendly. Note: acquired by
  OpenAI (announced March 2026), so watch its open-source direction.
- Giskard (github.com/Giskard-AI/giskard-oss): Python library that auto-generates an
  OWASP-aligned adversarial test suite from a description.

Frameworks (you script the attacks):

- Microsoft PyRIT (github.com/Azure/PyRIT): orchestration framework, 50-plus datasets,
  70-plus obfuscation converters (Base64, ROT13, leetspeak), scorers. "Metasploit for LLMs."
- UK AISI Inspect and inspect_evals (github.com/UKGovernmentBEIS/inspect_ai): reproducible
  eval framework with Docker sandboxing and 200-plus benchmarks. Rigorous, citable; a good
  fit for a leaderboard mindset.

Agent-specific (most relevant here):

- AgentDojo (github.com/ethz-spylab/agentdojo): live attacks and defenses against tool-using
  agents in four simulated environments, scoring utility and security jointly. The closest
  open analog to this competition.
- MCP-Scan (invariantlabs-ai.github.io/docs/mcp-scan): scans MCP servers for tool poisoning,
  rug pulls, and shadowing. `uvx mcp-scan@latest`, zero config. Beginner-friendly.
- Dreadnode AIRTBench (github.com/dreadnode/AIRTBench-Code): 70 black-box red-team CTF
  challenges as a structured problem set.

Classic adversarial ML (the pre-LLM surface):

- IBM Adversarial Robustness Toolbox (github.com/Trusted-AI/adversarial-robustness-toolbox):
  the mature library for evasion, poisoning, extraction, and inference attacks.
- Microsoft Counterfit (github.com/Azure/counterfit): CLI over ART and TextAttack. Effectively
  dormant (last release 2022); instructive, not current.

Guardrails (the blue-team side, know them so you can break them):

- Meta LlamaFirewall and Llama Guard (ai.meta.com/research): agent-focused guardrail
  framework plus a content-moderation classifier.
- NVIDIA NeMo Guardrails (github.com/NVIDIA-NeMo/Guardrails): programmable rule-based
  guardrails, with a garak integration.

## Practice platforms and challenges

Live or recurring as of mid-2026:

- Lakera Gandalf (gandalf.lakera.ai): trick an LLM into revealing a password across levels.
  Free, browser, no signup. The single most beginner-friendly on-ramp.
- Lakera Gandalf Agent Breaker (gandalf.lakera.ai/agent-breaker): newer, agent-specific
  (tool abuse, data exfiltration, indirect injection). Start here after classic Gandalf if
  your goal is agents.
- HackAPrompt 2.0 and the Learn Prompting playground (hackaprompt.com,
  learnprompting.org/hackaprompt-playground): large red-teaming competition plus a persistent
  practice playground. The site blocks automated fetches; check the live track in a browser.
- Gray Swan Arena (app.grayswan.ai/arena): the serious competitive venue, frequent
  high-stakes events with real leaderboards and cash pools. Excellent once you have cleared
  Gandalf; the agent and indirect-injection challenges mirror this competition.
- Dreadnode Crucible (app.dreadnode.io): 80-plus AI red-team scenarios, many requiring you to
  write Python to interact with the target, so it builds real exploit-coding skill.
- DEF CON AI Village Generative Red Team (aivillage.org): the flagship in-person event,
  recurring annually.

Bug bounties (real targets, real payouts):

- huntr (huntr.com): the first AI/ML-specific bug-bounty platform (model-file and open-source
  vulnerabilities), now under Palo Alto Networks. More software security than prompt craft;
  attempt once you can audit Python ML code.

Archived but instructive:

- Microsoft LLMail-Inject (github.com/microsoft/llmail-inject-challenge): craft indirect
  injections against a simulated LLM email client. The full submission dataset is open
  (Hugging Face: microsoft/llmail-inject-challenge), a goldmine of winning injections.
- HackAPrompt 1.0 dataset: the original 2023 corpus of working jailbreaks.

## Start here this weekend

A concrete order, lightest setup first, building toward agent red-teaming:

1. Saturday morning: Lakera Gandalf (classic), then Agent Breaker. Zero install. Internalize
   how system-prompt plus input-guard plus output-guard layers stack, then practice the
   agent patterns.
2. Saturday afternoon: NVIDIA garak against a local Ollama model (no API key, no cost). Run a
   couple of probes and read the HTML report. This turns "I can jailbreak by hand" into "I
   understand systematic automated probing," the core competition skill.
3. Sunday: clone and run AgentDojo to see indirect injection against tool-calling agents in a
   measured utility-versus-security frame, then sign up for Gray Swan Arena and attempt the
   indirect-injection or agent challenge for live scored practice.

Bonus, not the weekend: download the LLMail-Inject dataset and read winning indirect-injection
submissions.

## Items flagged as unverified at last review

- OpenAI's acquisition of promptfoo (announced March 2026) is from secondary sources; no
  primary OpenAI announcement was confirmed.
- The Pliny "Fable 5" jailbreak claim (June 2026) is secondary-sourced and not independently
  confirmed.
- Several challenge schedules and prize pools change frequently; the Gray Swan, HackAPrompt,
  and DEF CON specifics should be checked live before you plan around them.
