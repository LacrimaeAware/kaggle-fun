"""Draw ALONG my masks. Renders, per image: the cyan apo mask, the red fascicle mask, and my computed
apo edges + fascicle waves as toggleable layers. The user draws the fascicle waves and apo ridges they
MEAN on top, to show the bend and the merging a human intends (not to hand-label every fascicle).
DOWNLOAD saves draw_<image>.json (real image-pixel coords) to Downloads; Claude reads it back.

Run per_gap_viewer with UMUD_PERGAP_PROTO=1 first (it writes the masks + geom.json this reads).

    python experiments/draw_tool.py
"""
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "results" / "per_gap_viewer"
OUT = ROOT / "results" / "draw_tool"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gp = SRC / "geom.json"
    if not gp.exists():
        print("no geom.json - run: UMUD_PERGAP_PROTO=1 python experiments/per_gap_viewer.py  first")
        return
    geom = json.loads(gp.read_text())
    stems = sorted(geom.keys())
    dims, slim = {}, {}
    for s in stems:
        ok = True
        for suf in ("_base.jpg", "_apo.png", "_fasc.png"):
            src = SRC / f"{s}{suf}"
            if src.exists():
                shutil.copy(src, OUT / f"{s}{suf}")
            else:
                ok = False
        if not ok:
            continue
        im = cv2.imread(str(OUT / f"{s}_base.jpg"))
        dims[s] = [int(im.shape[1]), int(im.shape[0])]
        slim[s] = {"gaps": [{"sup_pts": g.get("sup_pts", []), "deep_pts": g.get("deep_pts", []),
                             "waves": g.get("waves", [])} for g in geom[s]["gaps"]]}
    stems = [s for s in stems if s in dims]
    html = (HTML.replace("__IMGS__", json.dumps(stems)).replace("__DIMS__", json.dumps(dims))
            .replace("__GEOM__", json.dumps(slim)))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"draw tool for {len(stems)} images -> results/draw_tool/index.html")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>draw along masks</title><style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 #bar{padding:8px;background:#1c1c1c;position:sticky;top:0;display:flex;gap:7px;align-items:center;flex-wrap:wrap;z-index:20}
 button,select{background:#333;color:#eee;border:1px solid #555;border-radius:5px;padding:6px 10px;cursor:pointer;font-size:13px}
 button.on{background:#2a7d4f;border-color:#3fa}
 label{font-size:13px;cursor:pointer}
 #stage{position:relative;margin:6px auto;width:fit-content}
 #base{display:block;max-width:1180px;height:auto}
 .ov{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
 #cv{position:absolute;left:0;top:0;cursor:crosshair}
 #info{padding:5px 10px;font-size:13px;color:#aab}
 textarea{width:98%;height:60px;margin:6px 1%;background:#0a0a0a;color:#6f6;font-family:monospace;font-size:11px}
</style></head><body>
<div id="bar">
 <select id="imgsel"></select>
 <label><input type="checkbox" id="t_apo" checked> cyan apo mask</label>
 <label><input type="checkbox" id="t_fasc" checked> red fasc mask</label>
 <label><input type="checkbox" id="t_mine" checked> my apo edges</label>
 <label><input type="checkbox" id="t_wave" checked> my waves</label>
 <span style="width:10px"></span>
 <button id="bt_fasc" class="on" onclick="setType('fascicle')">DRAW fascicle</button>
 <button id="bt_apo" onclick="setType('apo')">DRAW apo</button>
 <button onclick="finishLine()">finish [Enter]</button>
 <button onclick="undoPt()">undo [Bksp]</button>
 <button onclick="delLine()">del line</button>
 <button onclick="clearAll()">clear</button>
 <button onclick="dl()" style="background:#36c">DOWNLOAD</button>
 <span id="cnt"></span>
</div>
<div id="info">Draw the fascicle WAVES (yellow) and apo ridges (magenta) you mean, ON TOP of my masks. You don't need every fascicle - just show the bend and the merging you intend. Enter ends a line.</div>
<div id="stage">
 <img id="base">
 <img id="apoL" class="ov"><img id="fascL" class="ov">
 <svg id="mine" class="ov" preserveAspectRatio="none"></svg>
 <canvas id="cv"></canvas>
</div>
<textarea id="out" readonly></textarea>
<script>
const IMGS=__IMGS__, DIMS=__DIMS__, GEOM=__GEOM__, GC=['#0dd','#f0c','#0f0','#fb0'];
let cur=IMGS[0], type='fascicle', lines=[], pts=[];
const base=document.getElementById('base'), cv=document.getElementById('cv'), ctx=cv.getContext('2d'),
      sel=document.getElementById('imgsel'), mine=document.getElementById('mine');
IMGS.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;sel.appendChild(o)});
sel.onchange=()=>{cur=sel.value;lines=[];pts=[];load()};
function load(){
 base.src=cur+'_base.jpg';
 document.getElementById('apoL').src=cur+'_apo.png';
 document.getElementById('fascL').src=cur+'_fasc.png';
 base.onload=()=>{cv.width=base.clientWidth;cv.height=base.clientHeight;
   mine.setAttribute('viewBox','0 0 '+DIMS[cur][0]+' '+DIMS[cur][1]);applyToggles();redraw();out()};
}
function PLp(p,c,w,dash){return '<path d="'+p.map((q,i)=>(i?'L':'M')+q[0]+' '+q[1]).join(' ')+'" fill="none" stroke="'+c+'" stroke-width="'+w+'"'+(dash?' stroke-dasharray="6 5"':'')+'/>'}
function applyToggles(){
 document.getElementById('apoL').style.display=document.getElementById('t_apo').checked?'block':'none';
 document.getElementById('fascL').style.display=document.getElementById('t_fasc').checked?'block':'none';
 let s='';const g=GEOM[cur];
 if(g)g.gaps.forEach((gp,gi)=>{
   if(document.getElementById('t_mine').checked){if(gp.sup_pts&&gp.sup_pts.length)s+=PLp(gp.sup_pts,'#fff',2);if(gp.deep_pts&&gp.deep_pts.length)s+=PLp(gp.deep_pts,'#fff',2);}
   if(document.getElementById('t_wave').checked)(gp.waves||[]).forEach(w=>s+=PLp(w,GC[gi%4],1.5));
 });
 mine.innerHTML=s;
}
function nat(e){const r=base.getBoundingClientRect();return [Math.round((e.clientX-r.left)*DIMS[cur][0]/base.clientWidth), Math.round((e.clientY-r.top)*DIMS[cur][1]/base.clientHeight)]}
function disp(p){return [p[0]*base.clientWidth/DIMS[cur][0], p[1]*base.clientHeight/DIMS[cur][1]]}
cv.onclick=e=>{pts.push(nat(e));redraw()};
function setType(t){type=t;document.getElementById('bt_fasc').className=(t=='fascicle')?'on':'';document.getElementById('bt_apo').className=(t=='apo')?'on':''}
function finishLine(){if(pts.length>=2)lines.push({type,pts:pts.slice()});pts=[];redraw();out()}
function undoPt(){pts.pop();redraw()}
function delLine(){lines.pop();redraw();out()}
function clearAll(){lines=[];pts=[];redraw();out()}
function poly(p,c){if(!p.length)return;ctx.strokeStyle=c;ctx.fillStyle=c;ctx.lineWidth=3;ctx.beginPath();p.forEach((q,i)=>{const d=disp(q);i?ctx.lineTo(d[0],d[1]):ctx.moveTo(d[0],d[1])});ctx.stroke();p.forEach(q=>{const d=disp(q);ctx.beginPath();ctx.arc(d[0],d[1],3.5,0,7);ctx.fill()})}
function redraw(){ctx.clearRect(0,0,cv.width,cv.height);lines.forEach(l=>poly(l.pts,l.type=='apo'?'#f3f':'#ff2'));poly(pts,type=='apo'?'#f3f':'#ff2');document.getElementById('cnt').textContent=lines.length+' lines drawn'}
function out(){document.getElementById('out').value=JSON.stringify({image:cur,W:DIMS[cur][0],H:DIMS[cur][1],lines:lines})}
function dl(){finishLine();const b=new Blob([JSON.stringify({image:cur,W:DIMS[cur][0],H:DIMS[cur][1],lines:lines})],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='draw_'+cur+'.json';a.click()}
['t_apo','t_fasc','t_mine','t_wave'].forEach(id=>document.getElementById(id).onchange=applyToggles);
document.onkeydown=e=>{if(e.key=='Enter'){e.preventDefault();finishLine()}if(e.key=='Backspace'){e.preventDefault();undoPt()}};
load();
</script></body></html>"""


if __name__ == "__main__":
    main()
