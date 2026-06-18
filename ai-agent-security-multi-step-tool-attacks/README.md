# AI Agent Security: Multi-Step Tool Attacks

Kaggle competition: https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks

Hosted by OpenAI, Google, and IEEE. A red-team code competition: submit a Kaggle notebook
that writes `attack.py`, defining `AttackAlgorithm`, which searches a sandboxed tool-using
agent (GPT-OSS-20B and Gemma-4-26B-A4B) for unsafe multi-step behaviors and returns
replayable user-message chains. The grader replays each chain at a fixed seed and scores
it. Full meta-facts are in [docs/competition_facts.md](docs/competition_facts.md).

## Start here

Read the docs in this order.

1. [docs/explainer.md](docs/explainer.md): plain-language on-ramp, no security background
   needed. What the competition is, the skill tested, and a glossary.
2. [docs/competition_facts.md](docs/competition_facts.md): verified rules, timeline,
   prizes, targets, and budgets, with confidence levels and what to re-check on the site.
3. [docs/sdk_reference.md](docs/sdk_reference.md): the authoritative mechanics from the
   bundled SDK source: the contract, scoring formula, predicates, cells, attack surface,
   guardrails, budgets, the CLI, and the local-vs-real-gateway divergences.
4. [docs/guardrail_reachability.md](docs/guardrail_reachability.md): the central finding.
   Under the public guardrail only EXFILTRATION and CONFUSED_DEPUTY can fire. Verified by
   `verify_reachability.py`.
5. [docs/strategy.md](docs/strategy.md): the scoring levers and a phased roadmap with the
   80/20 path.
6. [docs/methods.md](docs/methods.md): automated red-teaming methods and comparable
   competitions, with what won.
7. [docs/solo_outlook.md](docs/solo_outlook.md): the honest solo-vs-teams assessment and
   the realistic outcome ladder.
8. [docs/audit.md](docs/audit.md): review of the prior scaffold and what changed.

To learn the broader field (not competition-specific):

9. [docs/field_guide.md](docs/field_guide.md): the AI and LLM security landscape: taxonomy,
   frameworks, the canon, defenses, the agent frontier, and a beginner learning path.
10. [docs/resources.md](docs/resources.md): a curated, dated link library of people, tools,
    practice sites, and communities, with a "start here this weekend" path.

[docs/INDEX.md](docs/INDEX.md) is the maintenance map: it lists every document, tags each as
stable, semi-stable, or volatile, and gives a short routine for keeping the folder fresh.

## Status

Analysis and a revised starter, no scored submission yet. Local checks pass; attack
effectiveness against the real models is unmeasured (the local agent is a stub). See
[writeup.md](writeup.md) for the repository-standard five-section summary.

Key result so far: under the public `OptimalGuardrail`, only EXFILTRATION (severity 5) and
CONFUSED_DEPUTY (severity 3) can fire. UNTRUSTED_TO_ACTION (the namesake attack) and
DESTRUCTIVE_WRITE are blocked. This is verified against the SDK code by
`verify_reachability.py` (10 of 10 checks pass).

## Files

- `attack.py`: the prior scaffold (a trace-guided enumerator). Kept for reference. Audited
  in [docs/audit.md](docs/audit.md); it has a backwards secret-avoidance rule and does not
  reflect the reachability finding.
- `attack_v2.py`: the revised starter, aimed at the two public-reachable predicates, with
  generation-time confirmation, an optional duplication mode, and control candidates. This
  is the one to develop. It is contract-valid and runs; effectiveness is unmeasured.
- `verify_reachability.py`: reproducible checks of the reachability analysis against the
  real SDK predicate scorer and guardrail.
- `run_local.py`: the prior custom smoke runner. The bundled `aicomp` CLI is preferred.
- `competition_reference.md`: the prior reference doc. Accurate; superseded as canonical by
  [docs/competition_facts.md](docs/competition_facts.md).
- `data/`: gitignored Kaggle download and the extracted SDK (`data/competition`).
- `results/`: gitignored local evaluation artifacts.

## How to run locally

The bundled SDK lives in `data/competition`. From the repository root, with the project
venv:

```powershell
$env:PYTHONPATH = ".\ai-agent-security-multi-step-tool-attacks\data\competition"
$env:PYTHONUTF8 = "1"
$venv = ".\.venv\Scripts\python.exe"

# Static contract check (fast)
& $venv -X utf8 -m aicomp_sdk.cli.main validate redteam .\ai-agent-security-multi-step-tool-attacks\attack_v2.py

# Verify the reachability analysis against the real SDK code
& $venv -X utf8 .\ai-agent-security-multi-step-tool-attacks\verify_reachability.py

# Deterministic smoke run (plumbing only; expect score 0.0)
& $venv -X utf8 .\ai-agent-security-multi-step-tool-attacks\run_local.py --attack .\ai-agent-security-multi-step-tool-attacks\attack_v2.py --budget-s 20
```

Note `PYTHONUTF8=1`: the CLI's pretty-printer uses Unicode glyphs that fail on a cp1252
Windows console without it. The deterministic agent is a stub, so a 0.0 local score is
structural, not a measure of attack quality. Real effectiveness needs the GPU targets,
either on a Kaggle GPU notebook or a local GPU with the GGUF weights and `llama-cpp-python`
(see [docs/sdk_reference.md](docs/sdk_reference.md)).

## Kaggle submission

This is a notebooks-only code competition. The notebook must write `attack.py` (the
class `AttackAlgorithm`) into `/kaggle/working/`, with Internet disabled and a GPU runtime.
Develop in `attack_v2.py` here, then copy its contents into the submitted `attack.py`.
