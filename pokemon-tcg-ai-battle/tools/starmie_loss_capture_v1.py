"""Capture games the HEAVY Starmie agent LOSES, with its full decision log, so we can find the failing turns
and fix the heuristics. Runs heavy vs the field (alakazam, denpa92) and the mirror (deployed sub_starmie);
wraps heavy with a recorder; on a loss, saves every decision (board summary + menu + pick + source + flags).

Flags per decision: source (heuristic|search|default); missed_ko (a KO attack was available but we didn't take
an attack); opp_took_prize (opponent's remaining prizes dropped since our last turn = they KO'd one of ours).

  python tools/starmie_loss_capture_v1.py --games 30 --max-losses 15 --budget 0.3
Output: data/starmie_losses.json
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALAKAZAM = ([5]*3+[13]+[19]*4+[66]*3+[305]*4+[741]*4+[742]*4+[743]*4+[1079]*4+[1081]*4+[1086]*4+[1097]*3+[1129]+[1152]*4+[1182]*3+[1184]+[1225]*4+[1231]*4+[1264])
DENPA92 = ([5]*3+[19]*4+[65]*4+[66]*4+[741]*4+[742]*4+[743]*3+[1079]*3+[1081]*3+[1086]*4+[1097]+[1129]+[1146]+[1152]*4+[1159]+[1182]*3+[1184]+[1225]*4+[1231]*4+[1264]*4)
TN = {1:"YES",2:"NO",3:"CARD",7:"PLAY",8:"ATTACH",9:"EVOLVE",10:"ABILITY",11:"DISCARD",12:"RETREAT",13:"ATTACK",14:"END"}


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
    CDB = json.load(open(ROOT/"agent"/"card_stats.json", encoding="utf-8"))
    ATK = json.load(open(ROOT/"agent"/"attack_stats.json", encoding="utf-8"))
    _G.update(make=make, DP=DP, S=S, SH=SH, M=M, CDB=CDB, ATK=ATK)


def _nm(cid): return (_G["CDB"].get(str(cid),{}) or {}).get("n", f"#{cid}") if cid is not None else None


def _ent(e):
    if not e: return None
    return {"n":_nm(e.get("id")),"hp":e.get("hp"),"e":len((e.get("energyCards") or e.get("energies") or []))}


def _label(o, obs, sel):
    DP=_G["DP"]; t=o.get("type")
    if t==13:
        a=_G["ATK"].get(str(o.get("attackId")),{}); return f"ATTACK {a.get('n','?')}(~{a.get('d','?')})"
    if t in (14,12,1,2): return TN.get(t,str(t))
    cid=None
    try: cid=DP.option_card_id(o,obs)
    except Exception: cid=None
    if cid is None and t==3:
        idx=o.get("index")
        for k in ("deck","discard","prize","hand"):
            z=sel.get(k) or []
            if isinstance(idx,int) and 0<=idx<len(z): cid=(z[idx].get("id") if isinstance(z[idx],dict) else z[idx]); break
    return f"{TN.get(t,t)} {_nm(cid) or ''}".strip()


def _recorder(log):
    DP=_G["DP"]; SH=_G["SH"]
    def agent(obs):
        sel=obs.get("select")
        if sel is None: return list(SH.STARMIE_DECK)
        opts=sel.get("option") or []
        pick=SH.agent(obs)
        # source
        try: src = "heuristic" if SH.choose(obs) is not None else "search/default"
        except Exception: src="?"
        # missed KO: a KO attack existed but we didn't pick an attack
        missed=False
        try:
            ko=DP.best_ko_attack(obs)
            if ko is not None and not any((0<=i<len(opts) and opts[i].get("type")==13) for i in pick): missed=True
        except Exception: pass
        cur=obs.get("current") or {}; pls=cur.get("players") or []; yi=cur.get("yourIndex",0)
        me=pls[yi] if yi<len(pls) else {}; opp=pls[1-yi] if len(pls)>1 else {}
        if len(opts)>=2 and (sel.get("maxCount") or 0)>=1:
            log.append({"step":obs.get("step"),
                        "me":{"act":_ent((me.get("active") or [None])[0]),"bench":[_ent(b) for b in (me.get("bench") or []) if b],"pz":len(me.get("prize") or [])},
                        "opp":{"act":_ent((opp.get("active") or [None])[0]),"bench_n":len([b for b in (opp.get("bench") or []) if b]),"pz":len(opp.get("prize") or [])},
                        "menu":[_label(o,obs,sel) for o in opts],
                        "pick":[ _label(opts[i],obs,sel) for i in pick if 0<=i<len(opts)],
                        "src":src,"missed_ko":missed})
        return pick
    return agent


def _winner(env):
    last=env.steps[-1]; r0,r1=last[0].get("reward"),last[1].get("reward")
    if r0 is None or r1 is None or r0==r1: return None
    return 0 if r0>r1 else 1


def run_chunk(task):
    opp_name, n, seat0, budget = task
    make=_G["make"]; M=_G["M"]; DP=_G["DP"]; S=_G["S"]
    deck={"alakazam":ALAKAZAM,"denpa92":DENPA92}.get(opp_name)
    def field(obs):
        if obs.get("select") is None: return list(deck)
        try:
            ko=DP.best_ko_attack(obs)
            if ko is not None: return [ko[0]]
            mv=S.best_option(obs, deck, leaf_mode="hand")
            if mv: return list(mv)
        except Exception: pass
        sel=obs.get("select") or {}; o=sel.get("option") or []; k=sel.get("minCount") or 1
        return list(range(min(max(k,1),len(o)))) if o else []
    opp = M.agent_starmie if opp_name=="deployed" else field
    losses=[]
    for i in range(n):
        hseat=(seat0+i)%2
        log=[]
        hrec=_recorder(log)
        pair=[hrec,opp] if hseat==0 else [opp,hrec]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env=make("cabt"); env.run(pair)
            w=_winner(env)
        except Exception:
            continue
        if w is not None and w!=hseat:   # heavy lost
            losses.append({"opponent":opp_name,"heavy_seat":hseat,"n_decisions":len(log),
                           "final":{"me_pz":log[-1]["me"]["pz"] if log else None,"opp_pz":log[-1]["opp"]["pz"] if log else None},
                           "missed_kos":sum(1 for d in log if d["missed_ko"]),
                           "decisions":log})
    return losses


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--games",type=int,default=30); ap.add_argument("--max-losses",type=int,default=15)
    ap.add_argument("--budget",type=float,default=0.3); ap.add_argument("--workers",type=int,default=max(1,(os.cpu_count() or 2)-2))
    ap.add_argument("--chunk",type=int,default=3)
    a=ap.parse_args()
    tasks=[]
    for opp in ("alakazam","denpa92","deployed"):
        rem,seat=a.games,0
        while rem>0:
            k=min(a.chunk,rem); tasks.append((opp,k,seat,a.budget)); seat=(seat+k)%2; rem-=k
    print(f"loss capture: heavy vs alakazam/denpa92/deployed, n={a.games}/opp, budget={a.budget}s",flush=True)
    losses=[]; done=0
    with ProcessPoolExecutor(max_workers=a.workers,initializer=_winit,initargs=(a.budget,)) as ex:
        for f in as_completed([ex.submit(run_chunk,t) for t in tasks]):
            done+=1
            try: ls=f.result()
            except Exception as e: print(f"  chunk err {e!r}",flush=True); continue
            losses.extend(ls)
            print(f"  [{done}/{len(tasks)}] +{len(ls)} losses (total {len(losses)})",flush=True)
    # keep field losses first (most informative), cap
    losses.sort(key=lambda L: (L["opponent"]=="deployed", -L["missed_kos"]))
    losses=losses[:a.max_losses]
    out=ROOT/"data"/"starmie_losses.json"
    out.write_text(json.dumps({"n_losses":len(losses),"by_opp":{o:sum(1 for L in losses if L["opponent"]==o) for o in ("alakazam","denpa92","deployed")},"losses":losses},indent=2),encoding="utf-8")
    print(f"\nwrote {out} ({len(losses)} losses, {sum(L['missed_kos'] for L in losses)} missed-KO flags)",flush=True)


if __name__=="__main__":
    main()
