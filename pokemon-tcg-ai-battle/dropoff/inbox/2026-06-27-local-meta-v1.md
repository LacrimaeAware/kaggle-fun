# Local meta / evaluation harness V1 (2026-06-27)

Model B eval-infrastructure lane. **Verdict: LOCAL_META_V1_READY. Default behavior: NOT_PIPELINE_DIRTY.** No
gameplay change, no large A/B, no merge, no submission, selector default off, 13/13 test suites pass.

Purpose: make local evaluation trustworthy enough that future tactical candidates (Model A's C8 atlas labs)
stop looking like "falls flat" vs "maybe works" depending on noise. This bakes in every lesson from the
selector arc.

## What shipped (durable, reusable)
- **`tools/local_meta_harness_v1.py`** — reusable runner with an EXTENDED opponent roster. Beyond the 6 builtins
  (deployed, mirror, alakazam, denpa92, first, random) it wires the real archetype decks from
  `registry/decks.json` as field pilots: **lucario (D003 Mega Lucario), koraidon (D002), abomasnow (D001
  wall/control)** — so we finally have genuine non-Starmie SENTINEL cells, not just the weak field. Supports
  opponent subset, per-opponent counts, staged runs, full per-game + per-decision logging. Reuses the smoke
  engine verbatim (no gameplay change).
- **`tools/local_meta_analyze_v1.py`** — the trustworthy ANALYSIS TEMPLATE in code, generic over (baseline,
  treatment) so it serves every future candidate. Reports: PRIMARY deployed+mirror combined (+ each cell
  separate), sentinels with **Holm** multiple-comparison correction, negative controls + weak field reported
  SEPARATELY (never decisive), Wilson CIs, Fisher exact, two-proportion delta CI, **MDE**, an **early-stopping
  trajectory + warning**, trigger coverage, override-intensity-by-result (with the game-length-confound caveat),
  family matrix, examples.
- **`tests/test_local_meta_analyze_v1.py`** (in run_all): proves the trust anchors — correct POOLING (sum W/L,
  not average %), the **early-stopping warning fires** when an interim look is significant but the full sample
  isn't (the +15pp->+2.6pp trap), and the **primary cell stays isolated from a field landslide** (the C3 trap).

## Verified
- Tiny harness smoke: 24 games incl lucario/koraidon/abomasnow sentinels, **0 errors** — the registry archetypes
  run fine as opponents.
- Analyzer smoke: produced the full standard report; at n=4/arm it correctly reports MDE=99pp (flagging the
  sample as useless) rather than over-reading noise.
- Default behavior: selector mode default off; `main.py` does not call `choose_action` or import the selector;
  the eval harness imports only eval modules; analyzer is pure stats. NOT_PIPELINE_DIRTY.

## How to use it for the next tactic (the workflow this enables)
1. Wire the candidate behind an env toggle (default off), as we did for the selector.
2. `local_meta_harness_v1.py --modes off,<candidate> --opponents deployed:60,mirror:60,denpa92:40,lucario:40 --stage A`
3. `local_meta_analyze_v1.py --dir <run> --baseline off --treatment <candidate> --primary deployed,mirror --sentinels denpa92,lucario,koraidon,abomasnow,alakazam --neg first,random`
4. Read the PRIMARY cell + Holm-corrected sentinels. Escalate Stage B/C only if Stage A is non-negative and
   non-regressive. Promote only on a significant PRIMARY result above MDE — and remember local self-play does
   not predict the ladder.

## Benchmark mixtures (in benchmark_mixtures.json)
M0 PRIMARY = deployed, mirror · M1 SENTINELS = denpa92, alakazam, lucario, koraidon, abomasnow · M2 NEG = first,
random · M3 FIELD = all non-primary. Rule: promotion never on M2/M3 alone; field always secondary.

## Two documented (non-blocking) partials
- Staging is operator-gated across invocations (deliberate — prevents auto-running a powered Stage C on a flat
  Stage B), not a single auto-stop runner.
- Runtime is captured at the decision/selector level, not full per-game wall-clock.

## Artifacts
`data/generated/local_meta_v1/`: opponent_inventory.json, benchmark_mixtures.json, harness_capability_report.json,
analysis_template.json, report_template.md, default_behavior_report.json, closeout.json. Tools:
local_meta_harness_v1.py, local_meta_analyze_v1.py, local_meta_reports_v1.py. Test: test_local_meta_analyze_v1.py.

## Lane status
Model B eval harness is ready. When Model A's C8 atlas names the top 1-3 tactical labs, each can be tested
through this harness + analyzer with primary/sentinel separation and early-stopping protection built in.
