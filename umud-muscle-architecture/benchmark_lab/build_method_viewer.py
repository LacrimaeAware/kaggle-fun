"""Build a static, interactive method-comparison viewer.

For each image (GM calf-raise frames + the 35-image architecture benchmark) it bakes the layers we keep
arguing about, as toggleable overlays:
  - aponeurosis lines (sup/deep)
  - fascicle mask (the U-Net output the pipeline measures from)
  - pipeline fragments, kept vs rejected
  - raw brightness walk lines
  - gated walk lines (walk on mask-gated brightness)
  - truth fascicle angle (the rater PA, drawn at the band centre)
plus the PA numbers (truth / pipeline / raw / gated).

Everything is written as STATIC files under results/method_viewer/ (images, mask PNGs, manifest.json,
viewer.html), so it is served by a plain http.server with no backend logic.

    python umud-muscle-architecture/benchmark_lab/build_method_viewer.py
    python -m http.server 8791 --directory umud-muscle-architecture/results/method_viewer
    open http://127.0.0.1:8791/viewer.html
"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark_lab"))
import segment_then_measure as M
import benchmark_validate as BV
import field_fascicle as FF

OUT = ROOT / "results" / "method_viewer"
GM_STEP = 6   # subsample the (correlated) calf-raise frames


def load(t):
    m = M.build_model(encoder_weights=None)
    m.load_state_dict(M.checkpoint_state(torch.load(M.weights_path(t), map_location="cpu")))
    return m.eval().to(M.DEVICE)


def gated(gray, fm):
    soft = cv2.GaussianBlur((fm > 0).astype(np.float32), (0, 0), 10.0); soft = soft / (soft.max() + 1e-9)
    gn = gray.astype(np.float32) / (gray.max() + 1e-9)
    return (gn * (0.15 + 0.85 * soft) * 255.0).astype(np.float32)


def band_tex_angle(gray, sup, deep):
    """Dominant orientation (deg) of the muscle texture inside the band, via a global structure tensor.
    This is the objective fascicle direction, used only to pick which SIDE the truth line tilts."""
    sm, sb = FF.line_from_pts(sup); dm, db = FF.line_from_pts(deep)
    H, W = gray.shape; xs = np.arange(W); sy = sm * xs + sb; dy = dm * xs + db
    lo = np.minimum(sy, dy); hi = np.maximum(sy, dy); pad = (hi - lo) * 0.2
    yy = np.arange(H)[:, None]; band = (yy >= (lo + pad)[None, :]) & (yy <= (hi - pad)[None, :])
    if band.sum() < 20:
        return None
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    jxx = (gx * gx)[band].mean(); jyy = (gy * gy)[band].mean(); jxy = (gx * gy)[band].mean()
    return float(np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2))


def truth_line(sup, deep, tex_ang_deg, truth_pa):
    """Truth PA line through the band centre. Magnitude is the rater PA; the SIDE (deep_ang + PA vs
    deep_ang - PA) is whichever matches the measured band texture orientation."""
    sm, sb = FF.line_from_pts(sup); dm, db = FF.line_from_pts(deep)
    ddir = np.array([deep[1][0] - deep[0][0], deep[1][1] - deep[0][1]], float)
    if ddir[0] < 0:
        ddir = -ddir
    deep_ang = np.degrees(np.arctan2(ddir[1], ddir[0]))
    cands = [deep_ang + truth_pa, deep_ang - truth_pa]
    if tex_ang_deg is None:
        a = cands[0]
    else:
        a = min(cands, key=lambda c: abs((c - tex_ang_deg + 90) % 180 - 90))  # closer to the texture line
    ar = np.radians(a)
    cx = (sup[0][0] + sup[1][0]) / 2.0
    cy = ((sm * cx + sb) + (dm * cx + db)) / 2.0
    dv = np.array([np.cos(ar), np.sin(ar)])
    u = FF._line_line(np.array([cx, cy]), dv, sm, sb); l = FF._line_line(np.array([cx, cy]), dv, dm, db)
    return [list(map(float, u)), list(map(float, l))] if (u is not None and l is not None) else None


def process(img, geo_src, apo, fasc):
    gray = img[..., 0] if img.ndim == 3 else img
    am = M.predict_mask(apo, img, "apo"); fm = M.predict_mask(fasc, img, "fasc")
    g = M.measure(am, fm, return_geometry=True)
    if not g or not g.get("geometry"):
        return None
    geo = g["geometry"]["apo"]; cols = np.where((am > 0).sum(0) >= 3)[0]
    xr = (int(cols.min()), int(cols.max())) if len(cols) else None
    rw = FF.measure_walk(gray, geo["superficial"], geo["deep"], x_range=xr)
    gw = FF.measure_walk(gated(gray, fm), geo["superficial"], geo["deep"], x_range=xr)
    rec = {
        "w": int(gray.shape[1]), "h": int(gray.shape[0]),
        "sup": [[round(p[0], 1), round(p[1], 1)] for p in geo["superficial"]],
        "deep": [[round(p[0], 1), round(p[1], 1)] for p in geo["deep"]],
        "pipeline": [{"up": [round(f["upper"][0], 1), round(f["upper"][1], 1)],
                      "lo": [round(f["lower"][0], 1), round(f["lower"][1], 1)], "kept": bool(f["kept"])}
                     for f in g["geometry"]["fragments"] if f.get("upper") and f.get("lower")],
        "raw": [[[round(u[0], 1), round(u[1], 1)], [round(d[0], 1), round(d[1], 1)]] for u, d, _ in rw["lines"]],
        "gated": [[[round(u[0], 1), round(u[1], 1)], [round(d[0], 1), round(d[1], 1)]] for u, d, _ in gw["lines"]],
        "pa": {"pipe": round(g["pa_deg"], 1) if g["pa_deg"] else None,
               "raw": round(rw["pa_deg"], 1) if rw["pa_deg"] is not None else None,
               "gated": round(gw["pa_deg"], 1) if gw["pa_deg"] is not None else None},
        "fm": fm,
        "tex_ang": band_tex_angle(gray, geo["superficial"], geo["deep"]),
    }
    return rec, geo["superficial"], geo["deep"]


def write_image_and_mask(stem, img, fm):
    bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(OUT / "img" / f"{stem}.png"), bgr)
    h, w = fm.shape
    rgba = np.zeros((h, w, 4), np.uint8); m = fm > 0
    rgba[m] = (0, 220, 0, 255)
    cv2.imwrite(str(OUT / "mask" / f"{stem}.png"), rgba)


def main():
    import shutil; shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "img").mkdir(parents=True); (OUT / "mask").mkdir(parents=True)
    apo, fasc = load("apo"), load("fasc")
    records = []

    # GM calf-raise (held-out)
    gmroot = ROOT / "data/gm_dynamic/benchmark_dataset_architecture_GM_dynamic_v0.1.0"
    if gmroot.exists():
        pa = pd.read_excel(list(gmroot.glob("*.xlsx"))[0], sheet_name="PA")
        pa = pa[np.isfinite(pa["frame"])].copy(); pa["frame"] = pa["frame"].astype(int); pa = pa.set_index("frame")
        frames = sorted(int(p.stem.split("_")[1]) for p in (gmroot / "frames").glob("frame_*.png"))
        for f in frames[::GM_STEP]:
            if f not in pa.index:
                continue
            img = M.read_rgb(gmroot / "frames" / f"frame_{f:05d}.png")
            out = process(img, None, apo, fasc)
            if out is None:
                continue
            rec, sup, deep = out; fm = rec.pop("fm")
            stem = f"GM_{f:05d}"
            write_image_and_mask(stem, img, fm)
            rec.update({"id": stem, "dataset": "GM calf-raise (held-out)",
                        "img": f"img/{stem}.png", "mask": f"mask/{stem}.png"})
            rec["pa"]["truth"] = round(float(pa.loc[f, "Mean"]), 1)
            rec["truth_line"] = truth_line(sup, deep, rec["tex_ang"], rec["pa"]["truth"])
            records.append(rec)
            print(f"  GM {f}: truth {rec['pa']['truth']} pipe {rec['pa']['pipe']} raw {rec['pa']['raw']} gated {rec['pa']['gated']}", flush=True)

    # 35-image architecture benchmark (note: 21/35 leaked into training)
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench is not None:
        for _, r in truth.iterrows():
            img = M.read_rgb(bench / f"{r.ImageID}.tif")
            out = process(img, None, apo, fasc)
            if out is None:
                continue
            rec, sup, deep = out; fm = rec.pop("fm")
            stem = str(r.ImageID)
            write_image_and_mask(stem, img, fm)
            rec.update({"id": stem, "dataset": "35-benchmark (21/35 leaked)",
                        "img": f"img/{stem}.png", "mask": f"mask/{stem}.png"})
            rec["pa"]["truth"] = round(float(r.pa_deg_true), 1)
            rec["truth_line"] = truth_line(sup, deep, rec["tex_ang"], rec["pa"]["truth"])
            records.append(rec)
            print(f"  {r.ImageID}: truth {rec['pa']['truth']} pipe {rec['pa']['pipe']} raw {rec['pa']['raw']} gated {rec['pa']['gated']}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(records), encoding="utf-8")
    (OUT / "viewer.html").write_text(VIEWER_HTML, encoding="utf-8")
    print(f"\nwrote {len(records)} records -> {OUT}", flush=True)


VIEWER_HTML = r"""<!doctype html><html><head><meta charset=utf-8><title>method viewer</title>
<style>
 *{box-sizing:border-box} body{margin:0;background:#14171c;color:#d7dce3;font:13px/1.4 system-ui,sans-serif}
 #bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:8px 12px;border-bottom:1px solid #2a2f37;position:sticky;top:0;background:#14171c;z-index:5}
 #layers{display:flex;gap:12px;flex-wrap:wrap;padding:8px 12px;border-bottom:1px solid #2a2f37}
 label.lay{display:inline-flex;gap:5px;align-items:center;cursor:pointer}
 .sw{width:12px;height:12px;border-radius:2px;display:inline-block}
 button{background:#232934;color:#d7dce3;border:1px solid #3a414c;border-radius:5px;padding:4px 10px;cursor:pointer}
 button:hover{background:#2c3340}
 #stagewrap{overflow:hidden;height:calc(100vh - 96px);background:#000;position:relative}
 #stage{position:absolute;transform-origin:0 0;left:0;top:0}
 #stage img,#stage canvas{position:absolute;left:0;top:0}
 .pa b{margin-right:4px;color:#9aa3af} .pa span{margin-right:12px;font-weight:600}
 select{background:#232934;color:#d7dce3;border:1px solid #3a414c;border-radius:5px;padding:4px}
</style></head><body>
<div id=bar>
 <select id=ds></select>
 <button id=prev>&larr; prev</button><button id=next>next &rarr;</button>
 <span id=idx></span><span id=name style="font-weight:600"></span>
 <span class=pa><b>PA</b><span id=pt style="color:#ffffff">truth -</span><span id=pp style="color:#ff9d3c">pipe -</span><span id=pr style="color:#df46e6">raw -</span><span id=pg style="color:#4dd0e1">gated -</span></span>
 <span style="margin-left:auto">mask <input id=op type=range min=0 max=100 value=45 style="vertical-align:middle"></span>
</div>
<div id=layers></div>
<div id=stagewrap><div id=stage><img id=base><img id=maskimg><canvas id=cv></canvas></div></div>
<script>
const LAYERS=[
 ['apo','aponeuroses','#ffd24a'],['mask','fascicle mask','#00dc00'],
 ['pkept','pipeline kept','#ff9d3c'],['prej','pipeline rejected','#ff4d4d'],
 ['raw','raw walk','#df46e6'],['gated','gated walk','#4dd0e1'],['truth','truth angle','#ffffff']];
const on={apo:1,mask:0,pkept:1,prej:0,raw:1,gated:0,truth:1};
let DATA=[],view=[],i=0,zoom=1,ox=0,oy=0;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),base=document.getElementById('base'),mk=document.getElementById('maskimg'),stage=document.getElementById('stage');
const layEl=document.getElementById('layers');
LAYERS.forEach(([k,lbl,c])=>{const l=document.createElement('label');l.className='lay';l.innerHTML=`<input type=checkbox ${on[k]?'checked':''}><span class=sw style="background:${c}"></span>${lbl}`;l.querySelector('input').onchange=e=>{on[k]=e.target.checked;draw()};layEl.appendChild(l)});
document.getElementById('op').oninput=e=>{mk.style.opacity=e.target.value/100;};
function line(a,b,c,w){ctx.strokeStyle=c;ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
function draw(){const r=view[i];if(!r)return;cv.width=r.w;cv.height=r.h;ctx.clearRect(0,0,r.w,r.h);
 mk.style.display=on.mask?'block':'none';
 if(on.pkept)r.pipeline.filter(f=>f.kept).forEach(f=>line(f.up,f.lo,'#ff9d3c',1.5));
 if(on.prej)r.pipeline.filter(f=>!f.kept).forEach(f=>line(f.up,f.lo,'#ff4d4d',1));
 if(on.raw)r.raw.forEach(s=>line(s[0],s[1],'#df46e6',1));
 if(on.gated)r.gated.forEach(s=>line(s[0],s[1],'#4dd0e1',1));
 if(on.apo){line(r.sup[0],r.sup[1],'#4ab3ff',2.5);line(r.deep[0],r.deep[1],'#ffd24a',2.5);}
 if(on.truth&&r.truth_line)line(r.truth_line[0],r.truth_line[1],'#ffffff',2.5);
}
function show(){const r=view[i];if(!r)return;base.src=r.img;mk.src=r.mask;mk.style.opacity=document.getElementById('op').value/100;
 base.onload=()=>{draw();fit();};
 document.getElementById('idx').textContent=`${i+1}/${view.length}`;document.getElementById('name').textContent=r.id;
 const p=r.pa;document.getElementById('pt').textContent='truth '+(p.truth??'-');document.getElementById('pp').textContent='pipe '+(p.pipe??'-');document.getElementById('pr').textContent='raw '+(p.raw??'-');document.getElementById('pg').textContent='gated '+(p.gated??'-');
 draw();
}
function fit(){const r=view[i];const wrap=document.getElementById('stagewrap');zoom=Math.min(wrap.clientWidth/r.w,wrap.clientHeight/r.h);ox=0;oy=0;apply();}
function apply(){stage.style.transform=`translate(${ox}px,${oy}px) scale(${zoom})`;}
document.getElementById('prev').onclick=()=>{i=(i-1+view.length)%view.length;show()};
document.getElementById('next').onclick=()=>{i=(i+1)%view.length;show()};
document.onkeydown=e=>{if(e.key=='ArrowLeft')document.getElementById('prev').click();if(e.key=='ArrowRight')document.getElementById('next').click();};
const wrap=document.getElementById('stagewrap');
wrap.onwheel=e=>{e.preventDefault();const f=e.deltaY<0?1.1:0.9;const rx=e.offsetX,ry=e.offsetY;ox=rx-(rx-ox)*f;oy=ry-(ry-oy)*f;zoom*=f;apply();};
let drag=null;wrap.onmousedown=e=>drag=[e.clientX-ox,e.clientY-oy];window.onmouseup=()=>drag=null;window.onmousemove=e=>{if(drag){ox=e.clientX-drag[0];oy=e.clientY-drag[1];apply();}};
function setDS(){const d=document.getElementById('ds').value;view=DATA.filter(r=>d=='all'||r.dataset==d);i=0;show();}
document.getElementById('ds').onchange=setDS;
fetch('manifest.json').then(r=>r.json()).then(d=>{DATA=d;const dss=[...new Set(d.map(r=>r.dataset))];const sel=document.getElementById('ds');sel.innerHTML='<option value=all>all</option>'+dss.map(x=>`<option>${x}</option>`).join('');view=DATA;show();});
</script></body></html>"""


if __name__ == "__main__":
    main()
