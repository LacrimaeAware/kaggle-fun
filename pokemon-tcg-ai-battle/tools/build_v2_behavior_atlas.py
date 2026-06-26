"""STARMIE V2 BEHAVIORAL ATLAS -- foundation table (Section 2).

For every new-#1-pilot (Yushin Ito) decision in the V2 zero-shot cohort: resolve the recorded observation, run
the CURRENT Starmie heuristic/search agent, extract public tactical-state features (agent/starmie_tactical_state),
and characterize the pilot's vs the agent's chosen option. Read-only; no training, no heuristic changes.

Output: data/starmie_audit/v2_behavior_atlas/newtop1_decisions.jsonl  (one flat row per decision)

  python tools/build_v2_behavior_atlas.py --budget 0.2 --workers 6
Also supports --pilot keidroid --no-agent for the cross-pilot feature table (no agent run).
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATLAS = ROOT / "data" / "starmie_audit" / "v2_behavior_atlas"
V2 = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v2.jsonl"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
_G = {}


@contextlib.contextmanager
def _quiet():
    old = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2); yield
    finally:
        os.dup2(old, 2); os.close(dn); os.close(old)


def _winit(budget, run_agent):
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet(), contextlib.redirect_stdout(io.StringIO()):
        import deck_policy_v3 as DP, starmie_heuristics as SH, starmie_tactical_state as TS
        if run_agent:
            import search_v3 as S
            S.USE_DYNAMIC_ATTACKS = True
            try: S.DEFAULT_BUDGET = budget
            except Exception: pass
    _G.update(DP=DP, SH=SH, TS=TS, run_agent=run_agent)


_EPC = {}


def _obs(ptr):
    e = ptr["episode"]
    if e not in _EPC:
        if len(_EPC) > 16:
            _EPC.clear()
        try:
            _EPC[e] = json.load(open(REPLAYS / f"{e}.json", encoding="utf-8"))
        except Exception:
            _EPC[e] = None
    ep = _EPC[e]
    if not ep:
        return None
    try:
        return ep["steps"][ptr["step"]][ptr["seat"]].get("observation")
    except Exception:
        return None


def _opt_label(o):
    """Interpretable label for a chosen option (from the corpus legal_actions entry)."""
    t = o.get("type"); tn = o.get("tname")
    if t == 13:
        a = (o.get("attack") or "").lower()
        for k in ("jetting", "nebula", "turbo", "water gun"):
            if k in a:
                return f"ATTACK:{k.split()[0].title()}"
        return f"ATTACK:{o.get('attack') or '?'}"
    if t in (7, 8, 9):  # PLAY/ATTACH/EVOLVE
        return f"{tn}:{o.get('card') or '?'}"
    if t == 3:
        return f"SELECT_CARD:{o.get('card') or '?'}"
    return tn or str(t)


ROLE = {1031: "Mega", 666: "Cinderace", 1030: "Staryu"}


def _target_role(DP, TS, o, obs):
    """Role of the entity an option targets (attach/tool/heal target), if any."""
    try:
        ent = DP.option_target_entity(o, obs)
        if ent:
            cid = DP._cid(ent)
            return ROLE.get(cid, f"#{cid}"), TS.energy_units(ent)
    except Exception:
        pass
    return None, None


def run_chunk(task):
    rows, budget = task
    DP, SH, TS = _G["DP"], _G["SH"], _G["TS"]
    out = []
    for r in rows:
        obs = _obs({"episode": r["episode"], "step": r["step"], "seat": r["seat"]})
        if not obs:
            continue
        legal = r["legal_actions"]
        pilot = r["pilot_action_label"]
        rec = {"decision_id": r["decision_id"], "episode": r["episode"], "step": r["step"], "seat": r["seat"],
               "family": r["family"], "n_legal": r["n_legal"], "split": r.get("split")}
        # pilot choice
        pi = pilot[0] if pilot and 0 <= pilot[0] < len(legal) else None
        po = legal[pi] if pi is not None else {}
        rec["pilot_choice"] = _opt_label(po) if po else "?"
        tr, tu = _target_role(DP, TS, po, obs) if po else (None, None)
        rec["pilot_target_role"], rec["pilot_target_units"] = tr, tu
        # agent choice (run current agent)
        if _G["run_agent"]:
            try:
                ag = SH.agent(obs)
            except Exception:
                ag = None
            try:
                src = "heuristic" if SH.choose(obs) is not None else "search_default"
            except Exception:
                src = "?"
            rec["agent_action"] = ag
            rec["agent_source"] = src
            ai = ag[0] if ag and 0 <= ag[0] < len(legal) else None
            rec["agent_choice"] = _opt_label(legal[ai]) if ai is not None else "?"
            rec["agree"] = (sorted(ag) == sorted(pilot)) if (ag and pilot) else False
        # tactical features
        try:
            feats = TS.extract(obs)
            bf = feats.get("board_features", {}); co = feats.get("tactical_coordinates", {})
            cm = co.get("COMMITMENT_STATE", {}); rs = co.get("RACE_STATE", {})
            rec["feat"] = {
                "prize_diff": bf.get("prize_diff"), "my_ready_main": bf.get("my_ready_main_attackers"),
                "my_backup_ready": bf.get("my_backup_ready"), "my_main_one_short": bf.get("my_main_one_short"),
                "my_units": bf.get("my_units"), "opp_units": bf.get("opp_units"),
                "my_immediate_ko": bf.get("my_immediate_ko"), "opp_immediate_ko": bf.get("opp_immediate_ko"),
                "engine_overinvest": bf.get("engine_overinvestment_units"),
                "max_conc": round(bf.get("max_energy_concentration") or 0, 2),
                "my_deck": bf.get("my_deck_count"), "my_hand": bf.get("my_hand_count"),
                "bench_cap": bf.get("my_bench_capacity"),
                "gw_attack": cm.get("game_winning_attack_available"), "ko_avail": cm.get("guaranteed_ko_available"),
                "nonterm_attack": cm.get("nonterminal_attack_available"), "attach_avail": cm.get("attachment_unused"),
                "info_action": cm.get("information_action_available"), "retreat_avail": cm.get("retreat_available"),
                "both_mains_online": rs.get("both_mains_online"),
            }
        except Exception as e:
            rec["feat"] = {"error": repr(e)}
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="Yushin Ito")
    ap.add_argument("--cohort-zero-shot", action="store_true", default=True)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--no-agent", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default="newtop1_decisions.jsonl")
    a = ap.parse_args()
    ATLAS.mkdir(parents=True, exist_ok=True)
    run_agent = not a.no_agent

    rows = []
    for line in open(V2, encoding="utf-8"):
        r = json.loads(line)
        m = r.get("meta") or {}
        if a.pilot == "Yushin Ito":
            if not r.get("new_top1_zero_shot"):
                continue
        else:
            if m.get("pilot") != a.pilot or r.get("deck_distance") != 0 or r.get("is_opponent"):
                continue
        rows.append({"decision_id": r["id"], "episode": r["episode"], "step": r["step"], "seat": r["seat"],
                     "family": r["family"], "n_legal": r["n_legal"], "split": r.get("split"),
                     "legal_actions": r["legal_actions"], "pilot_action_label": r["pilot_action"]})
        if a.max and len(rows) >= a.max:
            break
    print(f"pilot={a.pilot} decisions={len(rows)} run_agent={run_agent} budget={a.budget}", flush=True)
    chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
    out_rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_winit, initargs=(a.budget, run_agent)) as ex:
        for f in as_completed([ex.submit(run_chunk, (c, a.budget)) for c in chunks]):
            done += 1
            out_rows.extend(f.result())
            if done % 10 == 0:
                print(f"  {done}/{len(chunks)} chunks ({len(out_rows)} rows)", flush=True)
    outp = ATLAS / a.out
    with open(outp, "w", encoding="utf-8") as o:
        for r in out_rows:
            o.write(json.dumps(r, default=str) + "\n")
    if run_agent:
        agree = sum(1 for r in out_rows if r.get("agree"))
        print(f"overall agreement: {agree}/{len(out_rows)} = {round(100*agree/max(1,len(out_rows)),1)}%")
    print(f"wrote {outp} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
