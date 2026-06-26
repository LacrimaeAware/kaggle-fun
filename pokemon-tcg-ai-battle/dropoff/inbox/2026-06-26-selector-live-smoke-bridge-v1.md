# Starmie Selector Live-Smoke Bridge V1 — closeout (2026-06-26)

Model B. Wires Model A's exported learned proposer+selector into the live Starmie agent behind a
default-OFF, fail-closed gate, and runs a tiny live smoke. **Outcome: keep it OFF. Not promoted, not submitted.**

## What was built
- `agent/vendor/portable_selector_v1/` — vendored Model A bundle (runtime + **official Feature-V2 packer**, read-only).
- `agent/learned_selector_bridge.py` — CABT observation → public Feature-V2 payload adapter (Model-B side).
- `agent/starmie_heuristics.py` — `STARMIE_SELECTOR_MODE` (off | top1_gate | top3_selector), default off, fail-closed.
- Tests: `tests/test_selector_wiring_v1.py`. Harness fix: `tests/run_all.py` now subprocess-runs each module
  (a `sys.exit` in one module was silently skipping 6 of 8 suites). All 8 suites pass.
- Tools: `validate_selector_bridge_v1.py`, `selector_live_smoke_v1.py`, `selector_changed_decisions_v1.py`.

## Pipeline and gates (all passed)
`raw CABT obs → learned_selector_bridge → official Model A packer → portable selector runtime`
- **Raw-observation parity** (official packer): 220/220 selected match, 100% packed fields (type/attack/source/
  target card_id, semantic key, raw-option grouping). The earlier SELECT_CARD/context gap is solved by the official packer.
- **Runtime self-reproduction**: portable runtime reproduces Model A's recorded selections 100% on Model A's own packed options.
- **Adapter feature parity** (my CABT→payload vs Model A ground truth, 220 decisions): tactical 100% (all 20 keys),
  board_entities 100%, state 100% except `opp_attack_ready_count` 91.4%, option exact-IDs 95–100%.
- **Adapter exact-selection vs Model A**: 91.4%. Residual is near-indifferent selector calls + minor key-format grouping.
- **Off-mode**: action-identical to the heuristic baseline (strict identity test).
- **Live smoke**: 0 errors / 0 illegal selections across 300 games.

## Key reverse-engineering findings (for future selector work)
- Model A `board_entities`: `hp` = CABT remaining hp (already net of damage); `damage` = explicit CABT damage
  counter (0 default); `attached_energy_count` = energy **units** (Ignition = 3, from CABT `energies`); active
  slot always emitted (placeholder when empty).
- `action_family` is deterministic from `type_id`; `select_context_id` = `select.context`; ATTACK source = our active.
- **The selector is dominated by tactical features**: 149 of 185 selector weights are `tactical:* && family:*`
  crosses (e.g. `board.energy_on_main_attackers&&family:END = -4.039`). Our public `starmie_tactical_state`
  extractor already produces all 20 keys (board./commitment./value.). Omitting them dropped selection parity to ~80%.

## Smoke results (20 games/matchup, budget 0.2s, 0 errors)
| Opponent | S0 off | S1 top1_gate | S2 top3_selector |
|---|---|---|---|
| **deployed (mirror)** | 55% | 45% | **20%** |
| alakazam | 85% | 90% | 95% |
| denpa92 | 95% | 100% | 95% |
| first (cabt) | 70% | 75% | 80% |
| random (cabt) | 90% | 95% | 100% |
| **aggregate** | 79% | 81% | 78% |

Changed-decision trace: the selector overrides the heuristic on 38.7% (top1_gate) / 48.0% (top3_selector) of
single-select decisions, shifting **develop (ATTACH/PLAY) → ATTACK/END**.

## Verdict
- **top1_gate (S1): B_NEUTRAL** — field neutral-to-slightly-positive; mirror 45% vs 55% is within n=20 noise.
- **top3_selector (S2): C_REGRESSIVE vs the deployed mirror** — 20% vs 55% off (−35pp, well outside n=20 noise),
  while neutral vs field decks.
- This **confirms the heuristic's develop-before-attack policy**: the learned selector's bias toward attacking
  earlier loses against the strongest opponent (the mirror), exactly where tempo/board discipline matters most.

## PROMOTION_STATUS: DO_NOT_PROMOTE
`STARMIE_SELECTOR_MODE` stays **off** by default. Nothing submitted. The infrastructure (faithful adapter,
official packer, fail-closed wiring, 100% tactical parity) is proven and reusable; the **current** selector
model does not improve local win rate and its aggressive mode regresses in the mirror.

**Caveat (recurring):** local self-play does NOT predict the ladder (sub_starmie went 33–7 local, 0.480 ladder).
This is a safety/direction smoke, not a ranking. The mirror regression is a real local signal that the aggressive
mode is risky; the field-neutral result plus ladder-uncertainty means "keep off" rather than "this is worse on the ladder".

## If revisited
- The lever is the selector MODEL, not the wiring. A selector retrained to respect develop-before-attack (or a
  top1_gate restricted to high-margin overrides only) is the path. The pipeline is ready to A/B any new selector export.
