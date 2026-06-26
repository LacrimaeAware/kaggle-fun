# Master checklist — Model B (Starmie practical lane) — 2026-06-25

Consolidated from every prompt + heuristic discussion so nothing gets dropped. Model B = heuristics + corpus
(this repo). Model A = C8/features learned model (separate Codex 0557 pokemon-ai-agent worktree). The ladder
is the ONLY real test; local A/B + imitation mislead (sub_starmie2 went 7-6 despite "strong" local numbers).

## DONE (committed to main, with verification)
- [x] Imitation-gap pipeline (feed top-pilot replay states to our agent, compare picks). tools/imitation_gap_v1.py
- [x] Develop-before-attack: implemented (free dev before the turn-ending attack) AND VALIDATED on the replay
      sequence (96.5% of "develop-instead-of-attack" cases attack later same turn; holds wins+losses).
- [x] Forensic audit fixes (6 confirmed bugs): opp meta prior (was assuming a mirror); Boss/Wally gated on
      no-active-KO; Ignition counted as 3 energy units; tutor _sel_card respects option.area; no_suicide only
      self-removing abilities.
- [x] Tutor/snipe/gust target selection; Crushing Hammer; don't-field-a-3rd-Mega guard.
- [x] Empty-bench fix: bench development (Poffin/Staryu when in-play<=2).
- [x] Ladder-loss fixes (from real submission 54063181 games): premature-END veto; dig-for-the-Mega-line
      instead of looping Cinderace Turbo Flare; never waste Ignition on a non-attacker.
- [x] 15 rule toggles (STARMIE_DISABLE) + Stage-A ablation (every rule earns its place except Cape, in noise).
- [x] STARMIE_SPECIALIST_CORPUS_V1: 52,335 rows for Model A (data/starmie_corpus + manifest). THE features bridge.
- [x] Loss-fetch tooling: pokemon-ai-agent/tools/analyze_submission_losses_v1.py (any submission's ladder games).
- [x] Learned-adviser interface stub (agent/learned_adviser.py) -- hard-disabled, waiting on Model A coverage.

## IN PROGRESS / ON A BRANCH (not promoted)
- [~] Heuristic Lab v3 candidates (branch exp/starmie-heuristic-lab-v3, default-off toggles):
      retreat_guard=DIRECTIONAL, hammer_threshold=BASELINE_PREFERRED, cape_conditional=DIRECTIONAL. Need a
      field A/B or ladder to verify (low-frequency; imitation can't rank them).

## TODO — heuristics (ranked by likely ladder impact)
- [~] **MIRROR MISPLAY = develop-XOR-attack (DIAGNOSED 2026-06-26).** Heavy under-attacks: in mirror losses it
      makes 3.07 attacks vs deployed's 5.02, first attack ~3 decisions later; 64% of armed-Mega-with-attack
      decisions don't attack (it digs/passes). Root: the chain's only turn-ending attack is the KO floor (R9),
      no chip floor; and the develop-rollout digs forever and never attacks. FIX candidate (default off):
      `STARMIE_ATTACK_FLOOR=1` extends R9 to chip with an armed Mega Starmie (develop THEN attack). A/B running.
      Deeper half (TODO): fix the develop-rollout so it attacks after building, not digs forever. See
      dropoff/inbox/2026-06-26-mirror-misplay-develop-xor-attack.md.
- [ ] **Search-verified "material impact" guard (user's idea).** Use deck_policy_v3.compare_selections (paired
      hidden-world sim) to check a heuristic/search action MATERIALLY improves the state in 1-2 plies vs the
      default/END; veto no-op plays (redundant attach, pointless search) that change nothing. HIGH value:
      addresses the recurring "did anything happen?" complaint directly. Isolated candidate + tests + A/B.
- [ ] **eval.py Starmie ATTACK-READINESS leaf term (the diagnosed ROOT).** The search leaf is deck-agnostic --
      W_ENERGY rewards energy on ANY active equally, so search can't prefer a Mega Starmie attacker (120/210)
      over Cinderace (Turbo Flare 50). This is the search-level cause of the Turbo-loop / 0-prize losses. Add a
      term valuing a promotable/ready Mega Starmie attacker; then three-prize-liability; then backup-attacker.
      (eval_feature_audit.json). Isolated leaf change + fixed tests + A/B.
- [ ] **Promote/transition: Cinderace -> Mega Starmie.** Build + energize + promote the Mega line reliably; we
      still collapse to a stuck Cinderace. Tie to the readiness leaf term above.
- [ ] **Backup attacker.** Keep a 2nd Staryu/Mega developing so we don't lose when the active Mega dies.
- [ ] **Wally trigger = real incoming-KO check** (currently an HP<=50% proxy).
- [ ] **Tutor _need_value richer** (hand/board counts, line redundancy, energy shortfall) -- audit H.
- [ ] **Retreat guard verify** (candidate A) -- field A/B before promoting.
- [ ] Counter-specific play vs Lightning (Mega Starmie weak x2) / big-Mega tanks.

## TODO — submission / evaluation
- [ ] Compare current main (all fixes) vs previous baselines BEFORE submitting (this turn).
- [ ] Build/update submission from main and recommend upload (manual decision).
- [ ] Re-run analyze_submission_losses on EVERY new ladder submission (the real diagnosis loop).

## Model A lane (NOT Model B -- separate Codex worktree). My only job here is the corpus (DONE).
- [ ] Model A: train STARMIE specialist behavior model on the corpus; export starmie_behavior_proposer_v0.
- [ ] Model A: Starmie T0 quality plan (ATTACH/tutor/retreat/attack/Wally/Hammer/Boss).
- [ ] Then: integrate behavior top-k + heuristic + search behind safety/abstention gates (offline first).
- [ ] (If Model A is NOT being run, Model B can take this on -- pending user confirmation.)

## Standing rules / lessons
- Ladder is the only real test; never cite local A/B/imitation as "it's good." Deck is one of the BEST lists --
  do NOT chase decklists. Piloting is the lever. Don't bundle changes; isolate + measure each. Don't conclude
  from tiny samples. Don't auto-merge candidates or auto-submit.
