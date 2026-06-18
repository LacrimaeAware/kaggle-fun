# Documentation index and maintenance

The map of every document in this folder, and the system for keeping them fresh over the
roughly eleven weeks of the competition. Read the folder [README](../README.md) first for
the reading order; this file is the maintenance backbone.

## How the system works

Every doc is tagged with a volatility level. The level tells you how often it needs a look
and what triggers a rewrite. The point is that you never have to re-check everything: you
refresh the volatile docs on a cadence and leave the stable ones alone unless their source
changes.

- Stable: derived from the bundled SDK source. Changes only if the competition ships a new
  SDK version. Refresh trigger: a new `aicomp_sdk` version in `data/competition`.
- Semi-stable: concepts, strategy, and synthesis. Changes when our understanding or approach
  changes, or after a scored run teaches us something. Refresh trigger: a new result or a
  decision to change approach.
- Volatile: time-sensitive external facts (the leaderboard, entrant counts, live links,
  tool ownership, competition schedules). Refresh trigger: a calendar cadence (suggest every
  2 to 3 weeks) and always re-verify before a claim depends on it.

When a doc is updated, change its "Last reviewed" date at the top. When a scored run
happens, update [../writeup.md](../writeup.md) first (the five-section summary), then any
strategy or audit doc it affects, then the Status line in the folder README and the row in
the top-level repository README.

## Documents

Tier 1, the competition.

| Doc | Purpose | Volatility |
| --- | --- | --- |
| [explainer.md](explainer.md) | Plain-language on-ramp: what the competition is, the skill, scoring, glossary | Semi-stable |
| [competition_facts.md](competition_facts.md) | Verified rules, timeline, prizes, targets, budgets, with confidence levels | Volatile |
| [sdk_reference.md](sdk_reference.md) | Authoritative mechanics from the SDK source: contract, scoring, predicates, cells, tools, guardrails, budgets, CLI | Stable |
| [guardrail_reachability.md](guardrail_reachability.md) | The central finding: only EXFILTRATION and CONFUSED_DEPUTY can fire on public, with the proof | Stable |
| [strategy.md](strategy.md) | Scoring levers and the phased roadmap with the 80/20 path | Semi-stable |
| [methods.md](methods.md) | Automated red-teaming methods and comparable competitions, with what won | Semi-stable |
| [solo_outlook.md](solo_outlook.md) | Honest solo-vs-teams assessment and the outcome ladder | Volatile |
| [audit.md](audit.md) | Review of the prior scaffold and what changed in attack_v2 | Semi-stable |

Tier 2, the field (learning material, broader than the competition).

| Doc | Purpose | Volatility |
| --- | --- | --- |
| [field_guide.md](field_guide.md) | The AI security landscape: taxonomy, frameworks, canon, defenses, agent frontier, learning path | Semi-stable |
| [resources.md](resources.md) | Living link library: people, labs, tools, practice sites, communities | Volatile |

Root of the folder.

| File | Purpose | Volatility |
| --- | --- | --- |
| [../README.md](../README.md) | Folder index, status, and how to run | Semi-stable |
| [../writeup.md](../writeup.md) | Repository-standard five-section summary | Semi-stable |
| [../attack_v2.py](../attack_v2.py) | The revised starter to develop | Semi-stable |
| [../verify_reachability.py](../verify_reachability.py) | Reproducible proof of the reachability finding | Stable |
| [../attack.py](../attack.py) | The prior scaffold, kept for reference | Frozen |
| [../run_local.py](../run_local.py) | The prior smoke runner; the `aicomp` CLI is preferred | Frozen |
| [../competition_reference.md](../competition_reference.md) | The prior reference doc; superseded by competition_facts.md | Frozen |

## Maintenance checklist

A short routine, run roughly every 2 to 3 weeks and before any submission.

1. Re-verify the volatile docs against the live site: the leaderboard top and team count,
   the schedule, and any rule changes (competition_facts.md, solo_outlook.md).
2. Spot-check the links and tool status in resources.md; remove dead links, note ownership
   or schedule changes.
3. If a scored run happened, update writeup.md, then the affected strategy or audit doc,
   then the two README status lines.
4. If the SDK version changed, re-run verify_reachability.py and review the stable docs.
5. Update the "Last reviewed" date on any doc you touched.

## Provenance note

The competition mechanics (the stable docs) are derived from reading the bundled SDK source
directly, and the reachability finding is checked by verify_reachability.py. The field
material (Tier 2) and the volatile competition facts come from web research with sources
cited inline; treat any single web claim as needing re-verification, especially the ones
flagged in each doc's own caveats.
