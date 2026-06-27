"""Local meta / evaluation harness V1 (Model B eval-infrastructure).

A reusable, trustworthy local runner for testing tactical candidates. Generalizes the V3 powered-A/B runner with
an EXTENSIBLE opponent roster: the 6 builtins (deployed, mirror, alakazam, denpa92, first, random) PLUS the
real archetype decks in registry/decks.json (Mega Lucario, Koraidon, Mega Abomasnow) wired as field pilots --
so wall/control/Lucario SENTINEL cells are runnable, not just the weak field.

Reuses selector_v2_smoke_v1's engine setup, agent-under-test, run_chunk and per-decision logging VERBATIM (no
gameplay change); it only extends opponent resolution and the staged aggregation. Selector stays default off.

Local self-play does NOT predict the ladder; this measures whole-game direction + safety + trigger behavior.

  PYTHONIOENCODING=utf-8 python tools/local_meta_harness_v1.py \
      --modes off,selector_v3_transplant --opponents deployed:20,mirror:20,lucario:20,denpa92:20 \
      --budget 0.2 --stage A --out data/generated/local_meta_v1/run_demo
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import selector_v2_smoke_v1 as H

ROOT = H.ROOT
DEFAULT_OUT = ROOT / "data" / "generated" / "local_meta_v1"
_REG = {x["id"]: x for x in json.load(open(ROOT / "registry" / "decks.json", encoding="utf-8"))}
# friendly roster name -> registry deck cards (archetype field/sentinel pilots)
ROSTER = {
    "lucario": _REG["D003"]["cards"],     # Mega Lucario ex (validated-matchup) -- Mega attacker sentinel
    "koraidon": _REG["D002"]["cards"],    # Koraidon ex (crustle notebook) -- aggro/alt archetype sentinel
    "abomasnow": _REG["D001"]["cards"],   # Mega Abomasnow ex (official sample) -- wall/control sentinel
}
_BUILTINS = list(H.OPPONENTS)             # deployed, mirror, alakazam, denpa92, first, random
ALL_OPPONENTS = _BUILTINS + list(ROSTER) + list(_REG)
_OPP_ORIG = H._opp                         # original 6-opponent resolver (captured at import)


def _opp_ext(name):
    """Extended opponent resolver: builtins via the original map; roster/registry ids via a field pilot."""
    if name in ROSTER:
        return H._field(list(ROSTER[name]))
    if name in _REG:
        return H._field(list(_REG[name]["cards"]))
    return _OPP_ORIG(name)


def _winit_meta(budget):
    """Worker init: set up the engine (H._winit) then rebind H._opp so run_chunk resolves roster opponents."""
    H._winit(budget)
    H._opp = _opp_ext


_VAL_TO_KEY = {v: k for k, v in H.MODES.items()}


def _resolve_modes(spec):
    selected = {}
    for tok in [m.strip() for m in spec.split(",") if m.strip()]:
        if tok in H.MODES:
            selected[tok] = H.MODES[tok]
        elif tok in _VAL_TO_KEY:
            selected[_VAL_TO_KEY[tok]] = tok
        else:
            raise SystemExit(f"unknown mode {tok!r}; valid {list(H.MODES.items())}")
    return selected


def _parse_opponents(spec, default_games):
    out = {}
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        name, _, n = item.partition(":")
        name = name.strip()
        if name not in ALL_OPPONENTS:
            raise SystemExit(f"unknown opponent {name!r}; valid {ALL_OPPONENTS}")
        out[name] = int(n) if n else default_games
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="off")
    ap.add_argument("--opponents", default="deployed:20,mirror:20,denpa92:20")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stage", default="A")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-games", type=int, default=520)
    args = ap.parse_args()

    selected = _resolve_modes(args.modes)
    opp_games = _parse_opponents(args.opponents, args.games)
    for name, n in opp_games.items():
        if n > args.max_games:
            raise SystemExit(f"{name}={n} exceeds --max-games {args.max_games}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    trace_file = outdir / f"stage_{stage}_changed_decisions.jsonl"
    summary_file = outdir / f"stage_{stage}_summary.json"
    games_file = outdir / f"stage_{stage}_game_summary.jsonl"

    tasks, t = [], 0
    for mk in selected:
        for opp, n in opp_games.items():
            rem, seat = n, 0
            while rem > 0:
                k = min(args.chunk, rem)
                tasks.append((mk, opp, k, seat, t))
                t += 1
                seat = (seat + k) % 2
                rem -= k

    print(f"LOCAL META stage {stage}: modes {list(selected.items())} | opponents {opp_games} | "
          f"budget {args.budget}s | {len(tasks)} chunks", flush=True)
    print("CAVEAT: local self-play does NOT predict the ladder.\n", flush=True)

    agg, metrics_by, all_records = Counter(), {}, []
    done, total = 0, len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit_meta, initargs=(args.budget,)) as ex:
        for f in as_completed([ex.submit(H.run_chunk, tk) for tk in tasks]):
            done += 1
            try:
                mk, opp, wt, wo, dr, err, illegal, metrics, records = f.result()
            except Exception as exc:  # pragma: no cover
                print(f"  [{done}/{total}] ERROR {exc!r}", flush=True)
                continue
            for k, v in (("wt", wt), ("wo", wo), ("dr", dr), ("err", err)):
                agg[(mk, opp, k)] += v
            mb = metrics_by.setdefault((mk, opp), Counter())
            for k, v in metrics.items():
                mb[k] += v
            all_records.extend(records)
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] chunks done", flush=True)

    per_game = {}
    for r in all_records:
        g = per_game.setdefault(r["game_id"], {"game_id": r["game_id"], "mode": r["mode"], "matchup": r["matchup"],
                                               "result": r.get("game_result"), "overrides": 0, "blocked_terminal": 0,
                                               "terminal_overrides": 0})
        if r.get("blocked_terminal"):
            g["blocked_terminal"] += 1
        elif r.get("selector_raw") != r.get("baseline_raw"):
            g["overrides"] += 1
            if r.get("selector_family") in {"ATTACK", "END", "RETREAT"}:
                g["terminal_overrides"] += 1
    with open(games_file, "w", encoding="utf-8") as fh:
        for g in per_game.values():
            fh.write(json.dumps(g, default=str) + "\n")
    with open(trace_file, "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, default=str) + "\n")

    summary = {"stage": stage, "budget": args.budget, "modes": selected, "opponents": opp_games,
               "roster_used": {o: ("builtin" if o in _BUILTINS else "registry") for o in opp_games},
               "results": {}, "metrics": {}}
    print("\n=== win rate by mode x opponent ===", flush=True)
    for mk, mname in selected.items():
        row = {}
        for opp in opp_games:
            wt, wo, dr, err = (agg[(mk, opp, k)] for k in ("wt", "wo", "dr", "err"))
            tot = wt + wo
            wr = round(100 * wt / tot, 1) if tot else None
            row[opp] = {"win": wt, "loss": wo, "draw": dr, "err": err, "win_pct": wr}
            summary["metrics"][f"{mk}:{opp}"] = dict(metrics_by.get((mk, opp), {}))
            print(f"  {mk}({mname:20s}) vs {opp:10s}: {wt}-{wo} ({wr}% win, {dr}d, {err}e)", flush=True)
        summary["results"][mk] = row
    summary_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {summary_file}\n      {trace_file} ({len(all_records)} rows)\n      {games_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
