"""Powered / sequential A/B runner for the repaired Selector V3 artifact (NEEDS_N500 follow-up).

Differs from the tiny smoke (selector_v2_smoke_v1.py) in exactly three ways, all required by the powered task:
  1. opponent SUBSETTING + PER-OPPONENT game counts (primary deployed/mirror heavy, denpa92 sentinel lighter),
  2. no 50-game cap (this is the authorised N500 run; the smoke-era cap does not apply),
  3. stage-labelled output files so Stage A/B/C accumulate side by side.

Everything else -- the agent under test, the engine, the per-decision transplant logging -- is REUSED verbatim
from selector_v2_smoke_v1 so gameplay and logging are byte-identical to the smoke. We only change task generation
and aggregation. Selector default stays off; this harness sets STARMIE_SELECTOR_MODE per chunk inside workers.

Local self-play does NOT predict the ladder; this measures whole-game effect direction + safety, not rank.

  PYTHONIOENCODING=utf-8 python tools/selector_v3_powered_ab_v1.py \
      --modes off,selector_v3_transplant --opponents deployed:60,mirror:60,denpa92:40 \
      --budget 0.2 --stage A --out data/generated/starmie_selector_v3_powered_ab
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import selector_v2_smoke_v1 as H  # reuse engine setup, agent-under-test, run_chunk, logging

ROOT = H.ROOT
DEFAULT_OUT = ROOT / "data" / "generated" / "starmie_selector_v3_powered_ab"
# value->key so --modes can be given as mode values (off, selector_v3_transplant) or keys (S0,S3)
_VAL_TO_KEY = {v: k for k, v in H.MODES.items()}


def _parse_opponents(spec: str) -> dict:
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, n = item.split(":", 1)
            out[name.strip()] = int(n)
        else:
            out[item.strip()] = None  # filled with --games default later
    return out


def _resolve_modes(spec: str) -> dict:
    req = [m.strip() for m in spec.split(",") if m.strip()]
    selected = {}
    for token in req:
        if token in H.MODES:           # already a key (S0/S3)
            selected[token] = H.MODES[token]
        elif token in _VAL_TO_KEY:     # a value (off / selector_v3_transplant)
            selected[_VAL_TO_KEY[token]] = token
        else:
            raise SystemExit(f"unknown mode {token!r}; valid: {list(H.MODES.items())}")
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="off,selector_v3_transplant")
    ap.add_argument("--opponents", default="deployed:60,mirror:60,denpa92:40",
                    help="comma list name[:games]; per-opponent game count overrides --games")
    ap.add_argument("--games", type=int, default=60, help="default per-opponent games when not given as name:n")
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stage", default="A")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-games", type=int, default=520, help="hard safety ceiling per opponent (N500 + headroom)")
    args = ap.parse_args()

    selected = _resolve_modes(args.modes)
    opp_games = _parse_opponents(args.opponents)
    for name in list(opp_games):
        if opp_games[name] is None:
            opp_games[name] = args.games
        if opp_games[name] > args.max_games:
            raise SystemExit(f"{name}={opp_games[name]} exceeds --max-games {args.max_games}")
        if name not in H.OPPONENTS:
            raise SystemExit(f"unknown opponent {name!r}; valid: {H.OPPONENTS}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    trace_file = outdir / f"stage_{stage}_changed_decisions.jsonl"
    summary_file = outdir / f"stage_{stage}_summary.json"
    games_file = outdir / f"stage_{stage}_game_summary.jsonl"

    # build tasks: per (mode, opponent) split into chunks, globally-unique tidx -> unique game ids
    tasks = []
    t = 0
    for mk in selected:
        for opp, n in opp_games.items():
            rem, seat = n, 0
            while rem > 0:
                k = min(args.chunk, rem)
                tasks.append((mk, opp, k, seat, t))
                t += 1
                seat = (seat + k) % 2
                rem -= k

    print(f"POWERED A/B stage {stage}: modes {list(selected.items())} | opponents {opp_games} | "
          f"budget {args.budget}s | workers {args.workers} | {len(tasks)} chunks", flush=True)
    print("CAVEAT: local self-play does NOT predict the ladder; whole-game direction + safety only.\n", flush=True)

    agg = Counter()
    metrics_by = {}
    all_records = []
    done, total = 0, len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=H._winit, initargs=(args.budget,)) as ex:
        for f in as_completed([ex.submit(H.run_chunk, tk) for tk in tasks]):
            done += 1
            try:
                mk, opp, wt, wo, dr, err, illegal, metrics, records = f.result()
            except Exception as exc:  # pragma: no cover
                print(f"  [{done}/{total}] ERROR {exc!r}", flush=True)
                continue
            agg[(mk, opp, "wt")] += wt
            agg[(mk, opp, "wo")] += wo
            agg[(mk, opp, "dr")] += dr
            agg[(mk, opp, "err")] += err
            mb = metrics_by.setdefault((mk, opp), Counter())
            for k, v in metrics.items():
                mb[k] += v
            all_records.extend(records)
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] chunks done", flush=True)

    # per-game rollup from the trace (override count + result per game_id)
    per_game = {}
    for r in all_records:
        g = per_game.setdefault(r["game_id"], {"game_id": r["game_id"], "mode": r["mode"], "matchup": r["matchup"],
                                               "result": r.get("game_result"), "overrides": 0, "blocked_terminal": 0,
                                               "table_hits": 0, "terminal_overrides": 0, "first_changed_family": None})
        if r.get("blocked_terminal"):
            g["blocked_terminal"] += 1
        elif r.get("selector_raw") != r.get("baseline_raw"):
            g["overrides"] += 1
            if r.get("source") == "selector":
                g["table_hits"] += 1
            if r.get("selector_family") in {"ATTACK", "END", "RETREAT"}:
                g["terminal_overrides"] += 1
            if r.get("first_changed") and g["first_changed_family"] is None:
                g["first_changed_family"] = r.get("selector_family")
    with open(games_file, "w", encoding="utf-8") as fh:
        for g in per_game.values():
            fh.write(json.dumps(g, default=str) + "\n")
    with open(trace_file, "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, default=str) + "\n")

    summary = {"stage": stage, "budget": args.budget, "modes": selected, "opponents": opp_games,
               "games_per_opponent": opp_games, "results": {}, "metrics": {}}
    print("\n=== win rate by mode x opponent ===", flush=True)
    for mk, mname in selected.items():
        row = {}
        for opp in opp_games:
            wt, wo, dr, err = (agg[(mk, opp, k)] for k in ("wt", "wo", "dr", "err"))
            tot = wt + wo
            wr = round(100 * wt / tot, 1) if tot else None
            row[opp] = {"win": wt, "loss": wo, "draw": dr, "err": err, "win_pct": wr}
            mb = dict(metrics_by.get((mk, opp), {}))
            summary["metrics"][f"{mk}:{opp}"] = mb
            print(f"  {mk}({mname:20s}) vs {opp:9s}: {wt}-{wo} ({wr}% win, {dr}d, {err}e) "
                  f"ov={mb.get('overrides', 0)} blkT={mb.get('blocked_terminal', 0)} veto={mb.get('veto', 0)}",
                  flush=True)
        summary["results"][mk] = row
    summary_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {summary_file}\n      {trace_file} ({len(all_records)} rows)\n      {games_file} "
          f"({len(per_game)} games)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
