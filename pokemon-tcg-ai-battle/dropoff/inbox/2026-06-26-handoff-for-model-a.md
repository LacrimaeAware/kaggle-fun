# Handoff — Model B session 2026-06-26 (facts + experiments only)

Audience: Model A (features lane) + any follow-up. States what was run and the numbers. No interpretation.

## Repo state
- All work on branch `exp/starmie-tactical-leaf-v1` (created from `main` @ 6c38e58). `main` not modified.
- Commits on the branch: 0ffbc1d, 01b6e20, cf2c1b3, 227543e, dc2cfca.
- `agent/starmie_heuristics.py` is identical to 6c38e58 (the `STARMIE_ATTACK_FLOOR` edit below was added then reverted).
- `agent/eval.py` on the branch has an added `ATTACKER_CONTINUITY_V1` term, default OFF (env `STARMIE_LEAF_ATTACKER_CONTINUITY=1`).
- Submission bundle: none built this session. `sub_starmie2` is the current ladder submission.

## Definitions used in the A/Bs
- `heavy` = `starmie_heuristics.agent` (heuristics + search_v3 + veto).
- `deployed` = `main.agent_starmie` = KO-floor + `search_v3.best_option(leaf_mode="deckout", rollout_mode="develop")`, no heuristics, no veto.
- field pilots `alakazam`/`denpa92` = KO-floor + `search_v3.best_option(leaf_mode="hand")` on their decks. `leaf_mode="hand"` does NOT use `evaluate_deck_v3`, so they are unaffected by the continuity term.
- search config: `DEFAULT_BUDGET` 0.6 (A/Bs overrode to 0.3/0.4), `N_DETERM` 8.

## Experiment set 1 — mirror (heavy vs deployed), tool `tools/starmie_ab_v1.py`, budget 0.3
- baseline heavy vs deployed: n=40 17-23 (42%); n=60 27-33 (45%); n=120 57-63 (48%).
- `STARMIE_DISABLE=R2` (dig + bench-dev off): n=60 28-32 (47%).
- `STARMIE_DISABLE=R13` (whole veto off): n=100 48-52 (48%).
- `STARMIE_DISABLE=R0..R14` (all rules off): n=100 47-53 (47%).

## Experiment set 2 — behavioral comparison, tool `tools/mirror_behavior_v1.py` (records both agents), n=80, budget 0.3
- heavy LOST (n=46): heavy/deployed -> attacks 3.07/5.02; first_attack@ decision 16.0/13.24; prizes_taken 1.26/2.43; end_board 1.43/2.07; digs 5.07/6.28.
- heavy WON (n=34): attacks 5.0/3.71; first_attack@ 14.32/14.68; prizes_taken 2.65/1.91; end_board 1.91/1.32; digs 6.09/6.0.
- decision scan over captured losses (`data/starmie_losses.json`): of decisions where heavy's active is a Mega Starmie with >=1 energy unit AND an ATTACK option is on the menu, the share where heavy did NOT pick an attack: deployed-mirror losses 69/107 (64%); alakazam 20/41 (49%); denpa92 18/33 (55%).

## Experiment set 3 — STARMIE_ATTACK_FLOOR candidate (added then reverted)
- Change: in `_main_action`, after the KO floor (R9) and retreat-pivot (R10), if any attacker can attack, take `_best_attack_index` (develop steps run first). Env `STARMIE_ATTACK_FLOOR=1`.
- A/B `tools/starmie_ab_v1.py`, n=120, budget 0.3:
  - floor OFF: mirror 57-63 (48%); alakazam 97-23 (81%); denpa92 105-15 (88%).
  - floor ON: mirror 37-83 (31%); alakazam 81-39 (68%); denpa92 96-24 (80%).
- behavioral with floor ON (`mirror_behavior_v1.py`, n=60, heavy LOST n=35): heavy attacks 3.03; digs 4.11; end_board 1.23; end_prizes_left 5.49; decisions 28.54. (baseline-off losses for comparison: attacks 3.07; digs 5.07; end_board 1.43; end_prizes_left 4.74; decisions 34.5.)
- Reverted: `git checkout -- agent/starmie_heuristics.py`.

## Tactical-leaf task (S0–S9). Artifacts in `data/generated/starmie_tactical_leaf_v1/`.
- S0 `baseline_manifest.json` (`tools/starmie_baseline_freeze_v1.py`): HEAD 6c38e58; 9 correctness checks all OK (opp-meta prior, Boss/Wally KO gating, Ignition=3 units on Mega, tutor zone, no-suicide, develop-before-attack, deck==main, head includes 359f9ae).
- S1 extractor `agent/starmie_tactical_state.py`: public-info-only `extract(obs)` -> entity_features, board_features, tactical_coordinates (RACE/SWEEP/WALL/VALUE/COMMITMENT). Roles: Mega Starmie 1031=main_attacker, Cinderace 666=energy_engine, Staryu 1030=setup_basic. Energy units Ignition-aware (3 on Mega). Verified attacks: Jetting Blow 1487 (>=1 unit, 120 +50 snipe), Nebula Beam 1488 (>=3 units, 210 flat), Turbo Flare 965 (>=1 unit, 50).
- S2 `starmie_tactical_state_v1.jsonl` (`tools/starmie_tactical_export_v1.py`): 22,083 rows (our-seat, non-trivial). cohorts C0 18794, C2 3289; splits train 14595 / val 3813 / test 3675; families PLAY 6472, SELECT_CARD 8425, ATTACK 2915, ATTACH 2454, EVOLVE 813, RETREAT 260, OTHER 744. 0 unresolved obs, 0 errors. Row = `{decision_id, runtime{observable_state_hash, action_family, n_options, option_types, legal_grouped_actions, baseline_action, baseline_source, entity_features, board_features, tactical_coordinates}, eval_meta{replay_id, step, seat, split, cohort, deck_distance, is_opponent_seat, pilot_action, pilot, won, same_turn_sequence, in_disagreement_class}}`. Leak check: pilot/won/replay_id absent from `runtime`.
- S3 `current_leaf_failure_audit.json` (`tools/starmie_leaf_audit_v1.py`):
  - structural: leaf `evaluate_deck_v3(leaf_mode="deckout")`. With identical 3 Basic Water, Cinderace-active and Mega-active both score 424.0 (W_ENERGY term equal); `_active_energy` counts energy cards.
  - prevalence over 22,083 (count, of which in pilot-disagreement): cinderace_active_while_mega_ready_or_short 2339 (768); no_main_attacker_continuity 2753 (939); mega_one_attachment_short 6523 (2392); energy_on_engine_overinvestment 131 (41); redundant_concentration_no_threshold 2504 (576).
- S4 `agent/eval.py` `ATTACKER_CONTINUITY_V1` (default off). Vector components: ready_main_active, ready_main_bench, one_short_main, viable_backups, no_main_online, engine_overinvest, redundant_energy, exposed_concentration. Frozen weights `ACW` (hand-set): +20/+15/+8/+6 reward, -25/-6/-4/-5 penalize. Checks: toggle OFF -> leaf unchanged (424=424); toggle ON -> mega-active 444 > cinderace-active 395; max term magnitude observed 29 (W_PRIZE=1000).
- S5 `tests/test_attacker_continuity_v1.py`: 10/10 pass (added to `tests/run_all.py`); `tests/test_heuristics_fixed_state.py` 11/11 and `tests/test_starmie_audit_fixes_v1.py` still pass.
- S6 `offline_selection_audit.json` (`tools/starmie_leaf_offline_audit_v1.py`): 232 triggered+search-decided roots, budget 0.15. picks_changed 65 (28.0%); changed_toward_continuity 3 (4.6% of changed); pilot_agreement_triggered baseline 58.2% / continuity 57.8%. by-stratum (n, changed, cont_dir): value_even (100,31,2), value_ahead (105,29,1), race (17,2,0), value_behind (27,5,0).
- S7 `tools/starmie_leaf_ab_v1.py` (heavy_continuity vs heavy_baseline head-to-head + each vs field), budget 0.4:
  - n=120: mirror 65-55 (54.2% for continuity); alakazam continuity 95-25 (79.2%) vs baseline 94-26 (78.3%); denpa92 continuity 107-13 (89.2%) vs baseline 103-17 (85.8%).
  - n=240 (`leaf_ab_n240.txt`): mirror 113-127 (47.1%); alakazam continuity 191-49 (79.6%) vs baseline 200-40 (83.3%); denpa92 continuity 208-32 (86.7%) vs baseline 210-30 (87.5%).
  - pooled mirror (120+240): 178-182 (49.4%).
- S8 `tactical_feature_schema.json` + `tactical_coordinate_summary.json` (`tools/starmie_handoff_v1.py`). Feature classes labeled deck_independent / starmie_semantic_role_dependent / exact_card_id_dependent / uncertain_for_opponents; runtime=public, eval_meta=eval-only. COMMITMENT_STATE true-rates over 22,083: game_winning_attack 0.03, guaranteed_ko 0.208, nonterminal_attack 0.218, safe_development 0.527, attachment_unused 0.333, supporter_or_play 0.459, information_action 0.376, retreat 0.241, end 0.604.
- S9 verdicts recorded in `closeout.json`: ATTACKER_CONTINUITY = C_BASELINE_LEAF_PREFERRED; TACTICAL_STATE_EXPORT = A_TACTICAL_STATE_DATA_READY_FOR_MODEL_A.

## For Model A specifically
- Consume `starmie_tactical_state_v1.jsonl` `runtime.*` as a public situation representation; do NOT feed any `eval_meta.*` (pilot/outcome/replay/split) as a model input. `tactical_feature_schema.json` lists which features are deck-independent vs Starmie-role vs exact-card-id.
- The leaf deck-blindness finding (S3) is the input motivation; the hand-weighted `ATTACKER_CONTINUITY_V1` term did not change win rate (S7 numbers above). The term is left in `eval.py` default OFF as a reference implementation of the feature vector.
- STOP conditions honored: no learned policy trained, no C8 weights used, no Model A files modified, one leaf term only, no merge/push to main, no Kaggle bundle built.
