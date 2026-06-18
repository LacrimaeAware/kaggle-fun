# Submission log (canonical)

The real signal is the Kaggle public score. Compare OUR OWN submissions to each other (relative deltas
are informative); do not over-read the absolute number or the global ranking (see the LB-aggregation
caveat in memory). Claude builds the tarball under `submissions/` (gitignored, token rule); the human
uploads it. **Every build that gets uploaded must get a row here, with the commit it was built from.**

## History (06-17 / 06-18)
Times are relative to when this was logged (2026-06-18, ~12:1x local). Scores + status + filenames are
FACTS from the Kaggle submissions page. "Config" is reconstructed from git commit timestamps where it was
not logged at build time -> treat reconstructed configs as best-effort, not certain.

| When (rel) | ~Clock | File | Public score | Status | Config (reconstructed unless noted) | Commit |
|---|---|---|---|---|---|---|
| 27s ago | 06-18 ~12:1x | sub_search.tar.gz | (pending) | Pending | agent_search, DENPA92 deck, **N_DETERM=8**, 1-ply hand-eval, aggro continuation | dc1fba8 (this is the FIRST N=8 submission) |
| 9h ago | 06-18 ~03:00 | sub_search.tar.gz | **697.7** | Complete | agent_search, DENPA92 deck, N_DETERM=4 (pre-N=8 commit 07:13) | ~between 5e71444 and 747d006 |
| 14h ago | 06-17 ~22:00 | sub_search.tar.gz | **640.7** | Complete | agent_search, DENPA92 deck, N_DETERM=4 (deck swap was 21:21) | ~5e71444 |
| 14h ago | 06-17 ~22:00 | sub_combine.tar.gz | **422.1** | Complete | agent_combine (heuristic floor + blended leaf eval) | ~8d82b46 |
| 14h ago | 06-17 ~22:00 | sub_combine.tar.gz | (none) | Error | combine, failed to run | |
| 14h ago | 06-17 ~22:00 | sub_search.tar.gz | (none) | Error | search, failed to run | |
| 1d ago | 06-17 ~12:00 | submission.tar.gz | **617.2** | Complete | earliest upload, pre-DENPA92-deck (old deck) | (pre 5e71444) |

## Reads (relative, our own subs only)
- **combine is much worse than search on the real LB**: 422.1 (combine) vs 640.7 (search) at the same
  time. This matches local self-play (combine loses) -> combine is dead, do not resubmit it.
- **search has been climbing**: 617.2 (old deck) -> 640.7 (DENPA92 deck) -> 697.7. The DENPA92 deck swap
  and later fixes moved it up. Do NOT assert which single change caused each delta (not isolated on the LB).
- **The pending submission (dc1fba8) is the first with N_DETERM=8** (local: N=8 beat N=4 ~0.675 head-to-head).
  Its score vs the 697.7 (N=4) is the real test of whether N=8 helps on the LB. **Record it when it completes.**
- agent_rank (the learned net), belief, and continuation were tested locally this session and did NOT beat
  agent_search, so none were submitted.

## Going-forward protocol
1. On every upload, add a row: time, file, the commit hash it was built from (`git rev-parse --short HEAD`),
   the agent + key config (deck, N_DETERM, leaf eval, continuation), and later the public score + status.
2. Build with `PYTHON=<repo venv python> bash tools/build_submission.sh <variant>` so verify runs.
3. When a score lands, fill it in and add a one-line read vs the previous best.
