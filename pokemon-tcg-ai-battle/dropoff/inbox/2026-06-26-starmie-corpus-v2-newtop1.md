# Starmie Specialist Corpus V2 — new #1 pilot cohort (Model B) — 2026-06-26

Executed the Model B prompt only (corpus V2 + zero-shot eval prep). No training, no heuristic/submission changes,
V1 not overwritten. The Model A semantic-feature/student-repair prompt is a SEPARATE task in the .codex worktree.

## S1 — identify pilot + deck
- New top pilot = **Yushin Ito**, leaderboard **#1** (frozen Elo-like 1397.2, source: kaggle competition_leaderboard_view),
  submission **54038721**. (Confirmed: appears in all 65 fetched episodes.) Opponents in those games include
  keidroid (#2, 1334.7), Mogja J (#3), tomatomato (#4).
- Deck: **EXACT match** to our Starmie deck — all 65 games at deck_distance 0, zero card diffs. Cohort
  `C_NEW_TOP1_EXACT`.
- New games: 65, W/L **54-11** (83%). New decisions: **2,182** (PLAY 560, SELECT_CARD 697, ATTACH 299,
  ATTACK 331, EVOLVE 101, RETREAT 34, YES_NO 84, OTHER 76).
- Replays fetched into the LARGE corpus only: `pokemon-ai-agent/data/external/replays` (5289 -> 5354).
  kaggle-fun replays unchanged (gitignored, never committed).

## S2-S3 — corpus V2 (V1 preserved) + splits
- `data/starmie_corpus/starmie_specialist_corpus_v2.jsonl` = all 52,335 V1 rows VERBATIM + 2,182 new Yushin Ito
  rows (cohort `C_NEW_TOP1_EXACT`, `source=new_top1_fetch_v2`, `new_top1_zero_shot=true`). Total 54,517.
- V1 untouched: `starmie_specialist_corpus_v1.jsonl` (52,335) and `starmie_corpus_manifest_v1.json` unchanged.
- Manifest: `data/starmie_audit/starmie_corpus_manifest_v2.json` (references V1; new-cohort stats; split views).
- Note: Yushin Ito had **0 exact primary-seat rows in V1** (their 398 V1 rows were all as an OPPONENT seat), so
  these 2,182 are their first primary-seat cohort. The 65 episodes are absent from V1 entirely.
- **Both top pilots now in V2's exact-deck data**: #2 keidroid 11,791 rows (V1, preserved) + #1 Yushin Ito 2,182
  (new). Replay-grouped splits via sha1(episode)%100; new episodes disjoint from V1, so no split overlap.
- Split views (manifest filters, no row duplication): A NEW_TOP1_EXACT_SINGLE_PILOT (Yushin exact, train 1515 /
  val 508 / test 159); B OLD_STARMIE_EXACT_ALL_PILOTS (V1 C0); C OLD_PLUS_NEW_EXACT (C0 + C_NEW_TOP1_EXACT);
  D NEW_TOP1_ZERO_SHOT_EVAL_ONLY (the 2,182 new rows).

## S4 — zero-shot eval prep (NOT run)
- `data/starmie_corpus/starmie_v2_zero_shot_newtop1.jsonl` = 2,182 held-out decisions (all n_legal>=2), each with
  legal sibling group + pilot_action label + root_obs_pointer for feature resolution.
- `data/starmie_audit/starmie_v2_zero_shot_eval.json` = family counts + the exact command for Model A.
- Model B did NOT run the teacher: the frozen Starmie teacher + its feature pipeline live in Model A's `.codex`
  worktree (off-limits) and are not in the main pokemon-ai-agent checkout. Command emitted for Model A:
  `python tools/eval_starmie_teacher.py --teacher <ckpt> --eval-jsonl <KF>/.../starmie_v2_zero_shot_newtop1.jsonl
  --replays <PAA>/data/external/replays --group-by legal_actions --label pilot_action_label --by-family
  --report data/generated/starmie_v2_newtop1_zeroshot_metrics.json`.

## S5 — VERDICT
**A. NEW_TOP1_CORPUS_READY.** Deck exact-match, V1 preserved, V2 + manifest + splits + held-out zero-shot set all
built; the new #1 pilot (exact deck) is a clean held-out cohort. Pending: Model A runs the emitted zero-shot
command on its frozen teacher to get NEW_TOP1 metrics (the GOOD-vs-WEAK split is Model A's to report).

## Tools added (kaggle-fun)
`tools/build_starmie_corpus_v2.py`, `tools/build_starmie_v2_zeroshot.py`. Large jsonls gitignored; manifests
committed.
