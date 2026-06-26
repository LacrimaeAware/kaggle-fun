"""Dump the real structure of Starmie SUB-prompts (fetch/search, energy-accel target, bench snipe, gust target)
so the heuristics for them are grounded, not guessed. Scans top-pilot Starmie games for the seat's ACTIVE
decisions that are NOT the main action menu, and prints one full example per distinct prompt signature:
select context/contextCard/effect/remainDamageCounter/remainEnergyCost + the deck/discard arrays (ids) + the
first options as full dicts + what the pilot picked (card id/name)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
CDB = json.load(open(ROOT / "agent" / "card_stats.json", encoding="utf-8"))
TN = {0:"NUMBER",1:"YES",2:"NO",3:"CARD",4:"TOOL",5:"ENERGY_CARD",6:"ENERGY",7:"PLAY",8:"ATTACH",9:"EVOLVE",10:"ABILITY",11:"DISCARD",12:"RETREAT",13:"ATTACK",14:"END"}
ACTION_TYPES = {7,8,9,10,12,13,14}

def nm(cid): return (CDB.get(str(cid),{}) or {}).get("n", f"#{cid}") if cid is not None else None

def main():
    pilots = json.loads((ROOT/"data"/"starmie_top_pilots.json").read_text(encoding="utf-8"))
    games = []
    for p in pilots["pilots"][:6]:
        for ep,seat in (p.get("winning_episodes") or [])[:4]:
            games.append((ep,seat))
    seen = set()
    for ep, seat in games:
        try:
            d = json.load(open(os.path.join(REPLAYS,f"{ep}.json"),encoding="utf-8"))
        except Exception:
            continue
        steps = d["steps"]
        for t in range(len(steps)-1):
            e = steps[t][seat]
            if e.get("status")!="ACTIVE": continue
            obs = e.get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if len(opts)<2: continue
            types = {o.get("type") for o in opts}
            if types & ACTION_TYPES: continue   # skip main action menus
            ctxcard = sel.get("contextCard")
            ctxname = nm(ctxcard.get("id")) if isinstance(ctxcard,dict) else nm(ctxcard)
            sig = (sel.get("context"), ctxname, tuple(sorted(types)),
                   sel.get("remainDamageCounter") is not None, sel.get("remainEnergyCost") is not None,
                   bool(sel.get("deck")))
            if sig in seen: continue
            seen.add(sig)
            pilot = steps[t+1][seat].get("action")
            rec = {"episode":ep,"step":t,"context":sel.get("context"),"contextCard":ctxname,
                   "effect":sel.get("effect"),"min":sel.get("minCount"),"max":sel.get("maxCount"),
                   "remainDamageCounter":sel.get("remainDamageCounter"),"remainEnergyCost":sel.get("remainEnergyCost"),
                   "deck_ids":[ (c.get("id") if isinstance(c,dict) else c) for c in (sel.get("deck") or [])][:14],
                   "n_opt":len(opts),"opt_types":{TN.get(k,k):sum(1 for o in opts if o.get('type')==k) for k in types},
                   "first_opts":[{k:v for k,v in o.items()} for o in opts[:4]],
                   "pilot_pick":pilot}
            print(json.dumps(rec, indent=2)); print("-"*70)

if __name__ == "__main__":
    main()
