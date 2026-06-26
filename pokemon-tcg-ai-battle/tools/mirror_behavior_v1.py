"""Behavioral comparison of HEAVY (full heuristics+veto) vs DEPLOYED (KO-floor + search, no heuristics) in the
same-deck MIRROR, to characterize WHY heavy loses the mirror (~44%). Records BOTH agents in every game and
aggregates per-agent behavior split by outcome (heavy won / heavy lost):

  - attacks       : ATTACK (type 13) actions chosen
  - searches      : PLAY of a card with a search/draw/recover effect (the "dig" plays)
  - decisions     : total decisions made
  - end_board     : active(0/1) + bench size at the agent's LAST decision
  - prizes_taken  : 6 - remaining prizes at the agent's last decision
  - first_attack  : decision-index of the first ATTACK (None if never attacked)

Hypothesis: in heavy's LOSSES, heavy attacks fewer times / digs more / ends thinner than deployed -> heavy
over-develops and under-races the mirror. Output: data/mirror_behavior.json

  python tools/mirror_behavior_v1.py --games 80 --budget 0.3
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@contextlib.contextmanager
def _quiet_import():
    old = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2); yield
    finally:
        os.dup2(old, 2); os.close(dn); os.close(old)


_G = {}


def _winit(budget):
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet_import(), contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make
        import deck_policy_v3 as DP, search_v3 as S, starmie_heuristics as SH, main as M
    S.USE_DYNAMIC_ATTACKS = True
    try: S.DEFAULT_BUDGET = budget
    except Exception: pass
    CEFF = json.load(open(ROOT/"agent"/"card_effects.json", encoding="utf-8"))
    _G.update(make=make, DP=DP, S=S, SH=SH, M=M, CEFF=CEFF)


def _is_dig(o, obs):
    """A PLAY of a card whose effect searches/draws/recovers -- the 'dig' plays."""
    DP = _G["DP"]; CEFF = _G["CEFF"]
    if o.get("type") != 7:  # PLAY
        return False
    try: cid = DP.option_card_id(o, obs)
    except Exception: cid = None
    eff = CEFF.get(str(cid), {}) or {}
    return bool(eff.get("search") or eff.get("search_to_bench") or eff.get("draw")
                or eff.get("recover_discard"))


def _stats_recorder(agent_fn, st):
    """Wrap an agent; tally its behavior into dict st across one game."""
    DP = _G["DP"]
    def agent(obs):
        sel = obs.get("select")
        pick = agent_fn(obs)
        if sel is None:
            return pick
        opts = sel.get("option") or []
        st["dec"] += 1
        chose_attack = any(0 <= i < len(opts) and opts[i].get("type") == 13 for i in pick)
        if chose_attack:
            st["atk"] += 1
            if st["first_atk"] is None:
                st["first_atk"] = st["dec"]
        if any(0 <= i < len(opts) and _is_dig(opts[i], obs) for i in pick):
            st["dig"] += 1
        # board + prizes from our own view
        cur = obs.get("current") or {}; pls = cur.get("players") or []; yi = cur.get("yourIndex", 0)
        me = pls[yi] if yi < len(pls) else {}
        board = (1 if (me.get("active") or [None])[0] else 0) + len([b for b in (me.get("bench") or []) if b])
        st["end_board"] = board
        st["end_pz"] = len(me.get("prize") or [])
        return pick
    return agent


def _winner(env):
    last = env.steps[-1]; r0, r1 = last[0].get("reward"), last[1].get("reward")
    if r0 is None or r1 is None or r0 == r1: return None
    return 0 if r0 > r1 else 1


def _new_st():
    return {"dec": 0, "atk": 0, "dig": 0, "first_atk": None, "end_board": 0, "end_pz": 6}


def run_chunk(task):
    n, seat0, budget = task
    make = _G["make"]; M = _G["M"]; SH = _G["SH"]
    games = []
    for i in range(n):
        hseat = (seat0 + i) % 2
        hst, dst = _new_st(), _new_st()
        heavy = _stats_recorder(SH.agent, hst)
        deploy = _stats_recorder(M.agent_starmie, dst)
        pair = [heavy, deploy] if hseat == 0 else [deploy, heavy]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make("cabt"); env.run(pair)
            w = _winner(env)
        except Exception:
            continue
        if w is None:
            continue
        heavy_won = (w == hseat)
        games.append({"heavy_won": heavy_won, "heavy": hst, "deployed": dst})
    return games


def _agg(rows, key):
    """Average a numeric field over rows (skip None for first_atk)."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--budget", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=4)
    a = ap.parse_args()
    tasks = []; rem, seat = a.games, 0
    while rem > 0:
        k = min(a.chunk, rem); tasks.append((k, seat, a.budget)); seat = (seat + k) % 2; rem -= k
    print(f"mirror behavior: heavy vs deployed, n={a.games}, budget={a.budget}s", flush=True)
    games = []; done = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_winit, initargs=(a.budget,)) as ex:
        for f in as_completed([ex.submit(run_chunk, t) for t in tasks]):
            done += 1
            try: gs = f.result()
            except Exception as e: print(f"  chunk err {e!r}", flush=True); continue
            games.extend(gs)
            print(f"  [{done}/{len(tasks)}] +{len(gs)} games (total {len(games)})", flush=True)

    wins = [g for g in games if g["heavy_won"]]
    losses = [g for g in games if not g["heavy_won"]]
    def block(label, gs):
        h = [g["heavy"] for g in gs]; d = [g["deployed"] for g in gs]
        print(f"\n=== {label} (n={len(gs)}) ===")
        print(f"  {'metric':12} {'HEAVY':>8} {'DEPLOYED':>10}")
        for k, nm in [("atk","attacks"),("dig","digs"),("dec","decisions"),
                      ("end_board","end_board"),("end_pz","end_prizes_left"),("first_atk","first_attack@")]:
            print(f"  {nm:12} {str(_agg(h,k)):>8} {str(_agg(d,k)):>10}")
    print(f"\nheavy win rate: {len(wins)}/{len(games)} = {round(100*len(wins)/max(1,len(games)),1)}%")
    block("HEAVY WON", wins)
    block("HEAVY LOST", losses)
    out = ROOT/"data"/"mirror_behavior.json"
    out.write_text(json.dumps({
        "n_games": len(games), "heavy_wins": len(wins),
        "win": {"heavy": [g["heavy"] for g in wins], "deployed": [g["deployed"] for g in wins]},
        "loss": {"heavy": [g["heavy"] for g in losses], "deployed": [g["deployed"] for g in losses]},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
