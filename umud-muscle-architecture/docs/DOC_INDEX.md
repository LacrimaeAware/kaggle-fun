# UMUD documentation index

The repo was consolidated on 2026-06-14. There are now six living docs; everything else is history in
`../archive/` (preserved via git, summarized in the registry).

## Living docs (read these)

1. `HANDOFF.md` (this folder) - full self-contained resume document: current state, evidence,
   methodology reset, and the decision tree for what to do next. Read this first if coming in cold.
2. `CURRENT_STATE.md` (this folder) - terse canonical single source of truth: best score, per-term
   state, open levers, validation rules, next plan. Wins over every other short doc.
3. `../VERIFIED_FACTS.md` - only code-line- or LB-backed facts.
4. `../FINDINGS_REGISTRY.md` - every idea/feature/experiment organized by concept, each tagged
   LIVE / FACT / UNTESTED / REJECTED / FALSE / PAST. This is the merged replacement for the old
   front-door docs and the dated EXP journals.
5. `../EXPERIMENT_LOG.md` - chronological public-submission changelog (append-only).
6. `../competition_reference.md` - host rules, data facts, and forum clarifications.

Plus `../FEATURE_DATABASE.md`/`.csv` (feature ledger) and `../METHODOLOGY_AUDIT_2026-06-13.md` (the
prior decision audit, kept as a long-form read; its conclusions are folded into the registry).

## Archive

`../archive/` holds the merged narrative docs (synthesis, MASTER_REVIEW, handoff_brief, the SUBMISSION_*
notes, the dated STATE/audit files, etc.) and `../archive/exp/` holds the EXP38–EXP79 journals. They are
kept for history only; if one disagrees with the living docs, the living docs win.

## Rule

When a public score lands or a finding changes status, update `CURRENT_STATE.md`, `HANDOFF.md` if the
resume narrative changed, `EXPERIMENT_LOG.md`, and the relevant `FINDINGS_REGISTRY.md` entry. Do not
create a new front-door doc; resolve contradictions in place. A claim may drive a decision only if it
carries a code `file:line` or an LB number.
