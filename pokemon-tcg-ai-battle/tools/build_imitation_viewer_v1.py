"""Build a simple replay-state visualizer for the top imitation-gap moments.

Reads data/imitation_gap.json (top disagreements between our agent and top Starmie pilots). For each
moment it reopens the replay, pulls the exact observation, resolves card ids -> images, and emits a
self-contained HTML page: opponent board, our board, our hand, and the option menu -- with the PILOT's
pick (green) and OUR pick (blue) highlighted, and hover-to-enlarge on every card. The user reviews these
to tell us which heuristics to add.

Writes to pokemon-ai-agent/imitation_review.html (next to data/external/official/card_images/), so the
image paths resolve. Run after tools/imitation_gap_v1.py.

  python tools/build_imitation_viewer_v1.py --top 36
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
REPLAYS = r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"
GAP = ROOT / "data" / "imitation_gap.json"
OUT = Path(r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/imitation_review.html")
IMG_REL = "data/external/official/card_images"

import deck_policy_v3 as DP  # noqa: E402  (id/target resolution; loads card stats, no cg)

CDB = json.load(open(ROOT / "agent" / "card_stats.json", encoding="utf-8"))
ATK = json.load(open(ROOT / "agent" / "attack_stats.json", encoding="utf-8"))

TYPE_NAMES = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL", 5: "ENERGY_CARD", 6: "ENERGY",
              7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT",
              13: "ATTACK", 14: "END"}


def _name(cid):
    if cid is None:
        return None
    return (CDB.get(str(cid), {}) or {}).get("n", f"#{cid}")


def _ent(e):
    if not e:
        return None
    cid = e.get("id")
    return {"id": cid, "name": _name(cid), "hp": e.get("hp"),
            "energy": len((e.get("energyCards") or e.get("energies") or []))}


def _side(p):
    act = (p.get("active") or [None])[0]
    return {"active": _ent(act),
            "bench": [_ent(b) for b in (p.get("bench") or []) if b],
            "hand": [{"id": (c.get("id") if isinstance(c, dict) else c),
                      "name": _name(c.get("id") if isinstance(c, dict) else c)} for c in (p.get("hand") or [])],
            "handCount": p.get("handCount"), "prizeLeft": len(p.get("prize") or []),
            "deckCount": p.get("deckCount")}


def _opt_view(o, obs, sel):
    t = o.get("type")
    out = {"type": t, "tname": TYPE_NAMES.get(t, str(t))}
    if t == 13:  # attack
        arow = ATK.get(str(o.get("attackId")), {})
        out["label"] = f"{arow.get('n','?')}"
        out["sub"] = f"~{arow.get('d','?')} dmg (static; conditional dmg not shown)"
        return out
    if t == 14:
        out["label"] = "End turn"; return out
    if t == 12:
        out["label"] = "Retreat"; return out
    if t in (1, 2):
        out["label"] = TYPE_NAMES[t]; return out
    cid = None
    try:
        cid = DP.option_card_id(o, obs)
    except Exception:
        cid = None
    if cid is None and t == 3:
        idx = o.get("index")
        for key in ("deck", "discard", "prize", "hand"):
            zone = sel.get(key) or []
            if isinstance(idx, int) and 0 <= idx < len(zone):
                cid = (zone[idx].get("id") if isinstance(zone[idx], dict) else zone[idx])
                break
    out["cid"] = cid
    out["name"] = _name(cid)
    verb = {8: "Attach", 9: "Evolve→", 7: "Play", 3: "Choose", 11: "Discard", 10: "Ability"}.get(t, out["tname"])
    if t == 8:
        try:
            tgt = DP.option_target_entity(o, obs)
            out["target"] = _name(DP._cid(tgt)) if tgt else None
        except Exception:
            out["target"] = None
    out["label"] = f"{verb} {out.get('name') or ''}".strip()
    return out


def build_moment(rec):
    epid = rec["episode"]; seat = rec["seat"]; step = rec["step"]
    fn = os.path.join(REPLAYS, f"{epid}.json")
    try:
        d = json.load(open(fn, encoding="utf-8"))
    except Exception:
        return None
    steps = d.get("steps") or []
    if step >= len(steps):
        return None
    obs = (steps[step][seat] or {}).get("observation") or {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    yi = cur.get("yourIndex", 0)
    me = players[yi] if yi < len(players) else {}
    opp = players[1 - yi] if len(players) > 1 else {}
    return {
        "episode": epid, "seat": seat, "step": step,
        "category": rec["category"], "significance": rec["significance"], "our_source": rec["our_source"],
        "pilot_pick": rec["pilot_pick"], "our_pick": rec["our_pick"],
        "options": [_opt_view(o, obs, sel) for o in opts],
        "me": _side(me), "opp": _side(opp),
        "context": sel.get("context"), "maxCount": sel.get("maxCount"),
    }


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Imitation gap review</title>
<style>
:root{--bg:#0f1115;--panel:#1a1d24;--ink:#e6e8ec;--mut:#8b93a3;--green:#2ecc71;--blue:#4aa3ff;--line:#2a2e38}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,Segoe UI,Arial}
header{position:sticky;top:0;background:#0b0d11;border-bottom:1px solid var(--line);padding:10px 16px;z-index:20}
h1{font-size:16px;margin:0 0 6px}
.stat{color:var(--mut);font-size:13px}
.filters{margin-top:8px}
.filters button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:14px;padding:4px 10px;margin:2px;cursor:pointer;font-size:12px}
.filters button.on{background:var(--blue);border-color:var(--blue);color:#04121f}
.wrap{max-width:1100px;margin:0 auto;padding:14px}
.moment{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}
.mhead{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#222633;color:var(--mut);border:1px solid var(--line)}
.cat{color:var(--blue)}
.picks{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 12px}
.pickbox{flex:1;min-width:240px;border:2px solid var(--line);border-radius:8px;padding:8px}
.pickbox.pilot{border-color:var(--green)}
.pickbox.ours{border-color:var(--blue)}
.pickbox .ttl{font-size:12px;font-weight:600;margin-bottom:6px}
.pilot .ttl{color:var(--green)} .ours .ttl{color:var(--blue)}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end}
.lbl{color:var(--mut);font-size:11px;width:64px;flex:none}
.zone{display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end;flex:1}
.card{position:relative;width:52px}
.card img{width:52px;border-radius:4px;display:block;border:1px solid #000}
.card .nm{font-size:9px;color:var(--mut);text-align:center;max-width:52px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.card .hp{position:absolute;top:0;left:0;background:#000a;color:#fff;font-size:9px;padding:0 3px;border-radius:0 0 4px 0}
.card .en{position:absolute;top:0;right:0;background:#3a7;color:#021;font-size:9px;padding:0 3px;border-radius:0 0 0 4px}
.card.big img{width:52px}
.card:hover .pop{display:block}
.pop{display:none;position:absolute;bottom:60px;left:50%;transform:translateX(-50%);z-index:50}
.pop img{width:240px;border:1px solid #000;border-radius:8px;box-shadow:0 8px 30px #000a}
.opt{display:flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:6px;padding:4px 8px;margin:3px 0}
.opt.pilot{border-color:var(--green);background:#0e2417}
.opt.ours{border-color:var(--blue);background:#0c1d31}
.opt.both{border-color:#caa400;background:#241f08}
.opt .oi{width:26px;color:var(--mut);font-size:11px;flex:none}
.opt .ti{flex:none}
.opt .ti img{width:34px;border-radius:3px;display:block}
.opt .tx{flex:1}.opt .tx .s{color:var(--mut);font-size:11px}
.opt .tag{font-size:10px;padding:1px 6px;border-radius:8px}
.opt .tag.p{background:var(--green);color:#04140a}.opt .tag.o{background:var(--blue);color:#04121f}
.board{display:flex;flex-direction:column;gap:8px;margin:6px 0}
.bside{border:1px solid var(--line);border-radius:8px;padding:8px}
.bside .who{font-size:11px;color:var(--mut);margin-bottom:4px}
.opp .who{color:#ff8a8a}
.note{color:var(--mut);font-size:12px;margin-top:4px;font-style:italic}
.counts{color:var(--mut);font-size:11px}
a{color:var(--blue)}
</style></head><body>
<header>
<h1>Imitation gap: our Starmie agent vs top pilots</h1>
<div class="stat" id="stat"></div>
<div class="filters" id="filters"></div>
</header>
<div class="wrap" id="wrap"></div>
<script>
const IMG="__IMG__";
const DATA=__DATA__;
const META=__META__;
function img(cid,cls){if(cid==null)return"";return `<div class="card ${cls||''}"><img src="${IMG}/${cid}.jpg" onerror="this.style.display='none'"><div class="pop"><img src="${IMG}/${cid}.jpg" onerror="this.parentNode.style.display='none'"></div></div>`}
function ent(e){if(!e)return '<div class="card"><div class="nm">(empty)</div></div>';
 let h=`<div class="card"><img src="${IMG}/${e.id}.jpg" onerror="this.style.display='none'">`;
 if(e.hp!=null)h+=`<div class="hp">${e.hp}</div>`; if(e.energy)h+=`<div class="en">${'\\u26a1'.repeat(Math.min(e.energy,3))}</div>`;
 h+=`<div class="pop"><img src="${IMG}/${e.id}.jpg" onerror="this.parentNode.style.display='none'"></div><div class="nm">${e.name||''}</div></div>`;return h}
function side(s,who,cls){
 let bench=(s.bench||[]).map(ent).join('')||'<span class="counts">none</span>';
 let h=`<div class="bside ${cls}"><div class="who">${who} &middot; prizes left ${s.prizeLeft} &middot; deck ${s.deckCount} &middot; hand ${s.handCount}</div>`;
 h+=`<div class="row"><div class="lbl">active</div><div class="zone">${ent(s.active)}</div></div>`;
 h+=`<div class="row"><div class="lbl">bench</div><div class="zone">${bench}</div></div></div>`;return h}
function pickrow(m,idxs){return idxs.map(i=>{const o=m.options[i]||{};return img(o.cid)+`<span style="margin:0 8px 0 2px">${o.label||''}</span>`}).join('')}
function note(m){
 const cats={attack:'Attack timing/selection differs.',play:'Pilot played a different card (or chose to develop vs attack).',
  energy_attach:'Energy routing differs.',evolve:'Evolution timing/target differs.',select_card:'Different search/fetch target.'};
 let n=cats[m.category]||'';
 const our=new Set(m.our_pick), pil=new Set(m.pilot_pick);
 const ourT=m.our_pick.map(i=>(m.options[i]||{}).type), pilT=m.pilot_pick.map(i=>(m.options[i]||{}).type);
 if(ourT.includes(13)&&!pilT.includes(13))n+=' WE attack; pilot develops first.';
 if(pilT.includes(13)&&!ourT.includes(13))n+=' PILOT attacks; we develop instead.';
 return n}
function render(){
 const wrap=document.getElementById('wrap');wrap.innerHTML='';
 const active=window._cat||'all';
 let shown=DATA.filter(m=>active=='all'||m.category==active);
 document.getElementById('stat').innerHTML=`agreement ${META.agreements}/${META.total_decisions} = ${(META.agreement_rate*100).toFixed(1)}% over ${META.n_games} games &middot; ${META.disagreements} disagreements &middot; showing ${shown.length} top moments`;
 shown.forEach((m,k)=>{
  const both=(i)=>m.pilot_pick.includes(i)&&m.our_pick.includes(i);
  let opts=m.options.map((o,i)=>{
   let cls=both(i)?'both':m.pilot_pick.includes(i)?'pilot':m.our_pick.includes(i)?'ours':'';
   let tags=(m.pilot_pick.includes(i)?'<span class="tag p">PILOT</span>':'')+(m.our_pick.includes(i)?'<span class="tag o">OURS</span>':'');
   return `<div class="opt ${cls}"><div class="oi">${i}</div><div class="ti">${o.cid!=null?`<img src="${IMG}/${o.cid}.jpg" onerror="this.style.display='none'">`:''}</div><div class="tx">${o.label||''}${o.sub?`<div class="s">${o.sub}</div>`:''}</div>${tags}</div>`}).join('');
  wrap.innerHTML+=`<div class="moment">
   <div class="mhead"><div><b>#${k+1}</b> <span class="badge cat">${m.category}</span> <span class="badge">sig ${m.significance}</span> <span class="badge">our pick via ${m.our_source}</span></div>
   <div class="badge">ep ${m.episode} &middot; step ${m.step} &middot; <a href="https://pobservable.web.app/?id=${m.episode}" target="_blank">replay</a></div></div>
   <div class="note">${note(m)}</div>
   <div class="picks">
     <div class="pickbox pilot"><div class="ttl">PILOT played</div><div class="row">${pickrow(m,m.pilot_pick)||'(none)'}</div></div>
     <div class="pickbox ours"><div class="ttl">OUR agent played</div><div class="row">${pickrow(m,m.our_pick)||'(none)'}</div></div>
   </div>
   <div class="board">${side(m.opp,'OPPONENT','opp')}${side(m.me,'US','me')}
     <div class="bside"><div class="who">our hand</div><div class="zone">${(m.me.hand||[]).map(c=>img(c.id)).join('')||'<span class="counts">empty</span>'}</div></div>
   </div>
   <div><div class="who" style="color:var(--mut);font-size:11px;margin:4px 0">all options (${m.options.length})</div>${opts}</div>
  </div>`});
}
function filters(){
 const cats=['all',...Array.from(new Set(DATA.map(m=>m.category)))];
 const f=document.getElementById('filters');
 cats.forEach(c=>{const b=document.createElement('button');b.textContent=c;b.onclick=()=>{window._cat=c;[...f.children].forEach(x=>x.classList.remove('on'));b.classList.add('on');render()};f.appendChild(b)});
 f.firstChild.classList.add('on');
}
filters();render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=36)
    ap.add_argument("--in", dest="inp", default=str(GAP), help="imitation_gap json to render")
    ap.add_argument("--out", default=str(OUT), help="output html path")
    args = ap.parse_args()
    out_path = Path(args.out)
    gap = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    recs = gap.get("top_disagreements") or []
    moments = []
    for rec in recs[: args.top]:
        m = build_moment(rec)
        if m:
            moments.append(m)
    meta = {"agreements": gap.get("agreements"), "total_decisions": gap.get("total_decisions"),
            "agreement_rate": gap.get("agreement_rate"), "n_games": gap.get("n_games"),
            "disagreements": gap.get("disagreements")}
    html = (HTML.replace("__IMG__", IMG_REL)
            .replace("__DATA__", json.dumps(moments))
            .replace("__META__", json.dumps(meta)))
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({len(moments)} moments)", flush=True)


if __name__ == "__main__":
    main()
