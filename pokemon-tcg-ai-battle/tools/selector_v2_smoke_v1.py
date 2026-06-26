"""Second tiny live smoke for the conservative C3 selector (c3_family_limited), with per-game changed-decision
logging (the key fix vs the first smoke). Modes: S0 off, S1 top1_gate (V1 reference), S2 c3_family_limited.

Each test decision is traced (baseline vs selector pick + family + confidence + terminal-block + tactical state),
tagged with game id / matchup / mode / result / first-changed flag. Opponents never use the Starmie heuristic
agent except the pinned-OFF mirror (it forces STARMIE_SELECTOR_MODE=off for its own turns).

Local self-play does NOT predict the ladder; this is a safety/direction smoke, not a ranking.

  PYTHONIOENCODING=utf-8 python tools/selector_v2_smoke_v1.py --games 20 --budget 0.2 --workers 6
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_selector_v2_smoke"
ALAKAZAM = ([5]*3 + [13] + [19]*4 + [66]*3 + [305]*4 + [741]*4 + [742]*4 + [743]*4 + [1079]*4 + [1081]*4
            + [1086]*4 + [1097]*3 + [1129] + [1152]*4 + [1182]*3 + [1184] + [1225]*4 + [1231]*4 + [1264])
DENPA92 = ([5]*3 + [19]*4 + [65]*4 + [66]*4 + [741]*4 + [742]*4 + [743]*3 + [1079]*3 + [1081]*3 + [1086]*4
           + [1097] + [1129] + [1146] + [1152]*4 + [1159] + [1182]*3 + [1184] + [1225]*4 + [1231]*4 + [1264]*4)
MODES = {"S0": "off", "S1": "top1_gate", "S2": "c3_family_limited"}
OPPONENTS = ["deployed", "mirror", "alakazam", "denpa92", "first", "random"]
_G: dict = {}
_LOG: list = []
_CTX: dict = {}


@contextlib.contextmanager
def _quiet():
    old = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(dn)
        os.close(old)


def _winit(budget):
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet(), contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make  # noqa
        import deck_policy_v3 as DP
        import search_v3 as S
        import starmie_heuristics as SH
        import learned_proposer_adapter as AD
        import learned_selector_bridge as BR
        import main as M
    S.USE_DYNAMIC_ATTACKS = True
    with contextlib.suppress(Exception):
        S.DEFAULT_BUDGET = budget
    _G.update(make=make, DP=DP, S=S, SH=SH, AD=AD, BR=BR, M=M)


def _fam(obs, raw):
    try:
        return (_G["AD"].option_index_to_key(obs).get(raw) or "?").split(":")[0]
    except Exception:
        return "?"


def test_agent(obs):
    """The agent under test: production choose_action, but logs each single-select decision it changes."""
    SH = _G["SH"]
    if obs.get("select") is None:
        return list(SH.STARMIE_DECK)
    base = SH._baseline_pick(obs)
    mode = os.environ.get("STARMIE_SELECTOR_MODE", "off")
    _CTX["calls"] = _CTX.get("calls", 0) + 1
    if mode == "off" or not (isinstance(base, list) and len(base) == 1):
        return SH._selector_override(obs, list(base)) if isinstance(base, list) else base
    t0 = time.perf_counter()
    rec = SH.selector_trace(obs, base[0])
    _CTX["sel_ms"] = _CTX.get("sel_ms", 0.0) + (time.perf_counter() - t0) * 1000
    _CTX["sel_calls"] = _CTX.get("sel_calls", 0) + 1
    final = [rec["selector_raw"]] if rec else SH._selector_override(obs, list(base))
    if rec and rec.get("changed"):
        _CTX["overrides"] = _CTX.get("overrides", 0) + 1
        _LOG.append({
            "game_id": _CTX.get("game_id"), "matchup": _CTX.get("matchup"), "mode": mode,
            "step": _CTX.get("calls"), "baseline_raw": base[0], "selector_raw": rec["selector_raw"],
            "baseline_family": _fam(obs, base[0]), "selector_family": _fam(obs, rec["selector_raw"]),
            "source": rec.get("source"), "confidence": rec.get("confidence"), "entropy": rec.get("entropy"),
            "top1_margin": rec.get("top1_margin"), "terminal_override_blocked": rec.get("terminal_override_blocked"),
            "hard_veto": rec.get("hard_veto"), "first_changed": _CTX.get("first_changed_done") is None,
            "tactical": _tac(obs),
        })
        _CTX["first_changed_done"] = True
    if rec and rec.get("terminal_override_blocked"):
        _CTX["blocked_terminal"] = _CTX.get("blocked_terminal", 0) + 1
        # log blocked terminal decisions too (they keep baseline -> not a "change", but we need outcome linkage)
        _LOG.append({
            "game_id": _CTX.get("game_id"), "matchup": _CTX.get("matchup"), "mode": mode,
            "step": _CTX.get("calls"), "baseline_raw": base[0], "selector_raw": base[0],
            "baseline_family": _fam(obs, base[0]), "selector_family": _fam(obs, base[0]),
            "blocked_terminal": True, "blocked_override_reason": rec.get("blocked_override_reason"),
            "source": rec.get("source"), "confidence": rec.get("confidence"), "first_changed": False,
            "tactical": _tac(obs), "terminal_override_blocked": True, "hard_veto": rec.get("hard_veto"),
        })
    return final


def _tac(obs):
    try:
        t = _G["BR"].tactical_state_features(obs)
        return {k: t.get(k) for k in ("commitment.guaranteed_ko_available", "commitment.game_winning_attack_available",
                                      "commitment.safe_development_available", "board.prize_diff")}
    except Exception:
        return {}


def _mirror_off(obs):
    """Current Starmie baseline mirror, pinned to off regardless of the global test mode."""
    SH = _G["SH"]
    saved = os.environ.get("STARMIE_SELECTOR_MODE")
    os.environ["STARMIE_SELECTOR_MODE"] = "off"
    try:
        return SH.agent(obs)
    finally:
        if saved is None:
            os.environ.pop("STARMIE_SELECTOR_MODE", None)
        else:
            os.environ["STARMIE_SELECTOR_MODE"] = saved


def _field(deck):
    DP, S = _G["DP"], _G["S"]

    def pilot(obs):
        if obs.get("select") is None:
            return list(deck)
        try:
            ko = DP.best_ko_attack(obs)
            if ko is not None:
                return [ko[0]]
            mv = S.best_option(obs, deck, leaf_mode="hand")
            if mv:
                return list(mv)
        except Exception:
            pass
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        k = sel.get("minCount") or 1
        return list(range(min(max(k, 1), len(opts)))) if opts else []
    return pilot


def _opp(name):
    if name == "deployed":
        return _G["M"].agent_starmie
    if name == "mirror":
        return _mirror_off
    if name == "alakazam":
        return _field(ALAKAZAM)
    if name == "denpa92":
        return _field(DENPA92)
    return name  # "first" / "random"


def _winner(env):
    last = env.steps[-1]
    r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def run_chunk(task):
    mode_key, opp_name, n, seat0, tidx = task
    os.environ["STARMIE_SELECTOR_MODE"] = MODES[mode_key]
    make, SH = _G["make"], _G["SH"]
    opp = _opp(opp_name)
    wt = wo = dr = err = illegal = 0
    metrics = Counter()
    records = []
    for i in range(n):
        t_seat = (seat0 + i) % 2
        pair = [test_agent, opp] if t_seat == 0 else [opp, test_agent]
        _LOG.clear()
        _CTX.clear()
        gid = f"{mode_key}:{opp_name}:t{tidx}:{i}"  # tidx is globally unique per task -> game_id unique
        _CTX.update(game_id=gid, matchup=opp_name)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt")
                env.run(pair)
            w = _winner(env)
        except Exception:
            err += 1
            continue
        res = "draw" if w is None else ("win" if w == t_seat else "loss")
        if w is None:
            dr += 1
        elif w == t_seat:
            wt += 1
        else:
            wo += 1
        metrics["selector_calls"] += _CTX.get("sel_calls", 0)
        metrics["overrides"] += _CTX.get("overrides", 0)
        metrics["blocked_terminal"] += _CTX.get("blocked_terminal", 0)
        metrics["sel_ms"] += _CTX.get("sel_ms", 0.0)
        for rec in _LOG:
            rec["game_result"] = res
            if rec.get("hard_veto"):
                metrics["veto"] += 1
            if rec.get("source") == "fallback":
                metrics["fallback"] += 1
            records.append(rec)
    return (mode_key, opp_name, wt, wo, dr, err, illegal, dict(metrics), records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=2)
    args = ap.parse_args()
    args.games = min(args.games, 50)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    _t = 0
    for mk in MODES:
        for opp in OPPONENTS:
            rem, seat = args.games, 0
            while rem > 0:
                k = min(args.chunk, rem)
                tasks.append((mk, opp, k, seat, _t))
                _t += 1
                seat = (seat + k) % 2
                rem -= k
    print(f"V2 second smoke: modes {list(MODES.items())} vs {OPPONENTS}, n={args.games}/matchup, budget={args.budget}s",
          flush=True)
    print("CAVEAT: local self-play does NOT predict the ladder; safety/direction smoke only.\n", flush=True)

    agg = Counter()
    metrics_by = {}
    all_records = []
    done, total = 0, len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit, initargs=(args.budget,)) as ex:
        for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
            done += 1
            try:
                mk, opp, wt, wo, dr, err, illegal, metrics, records = f.result()
            except Exception as exc:
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
            if done % 12 == 0 or done == total:
                print(f"  [{done}/{total}] chunks done", flush=True)

    with open(OUT / "changed_decisions.jsonl", "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, default=str) + "\n")

    summary = {"games_per_matchup": args.games, "budget": args.budget, "modes": MODES, "opponents": OPPONENTS,
               "results": {}, "metrics": {}}
    print("\n=== win rate by mode x opponent ===", flush=True)
    for mk, mname in MODES.items():
        row = {}
        for opp in OPPONENTS:
            wt, wo, dr, err = (agg[(mk, opp, k)] for k in ("wt", "wo", "dr", "err"))
            tot = wt + wo
            wr = round(100 * wt / tot, 1) if tot else None
            row[opp] = {"win": wt, "loss": wo, "draw": dr, "err": err, "win_pct": wr}
            mb = dict(metrics_by.get((mk, opp), {}))
            summary["metrics"][f"{mk}:{opp}"] = mb
            print(f"  {mk}({mname:16s}) vs {opp:9s}: {wt}-{wo} ({wr}% win, {dr}d, {err}e) "
                  f"ov={mb.get('overrides',0)} blkT={mb.get('blocked_terminal',0)} veto={mb.get('veto',0)}", flush=True)
        summary["results"][mk] = row
    (OUT / "live_smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT/'live_smoke_summary.json'} + changed_decisions.jsonl ({len(all_records)} changed decisions)",
          flush=True)


if __name__ == "__main__":
    main()
