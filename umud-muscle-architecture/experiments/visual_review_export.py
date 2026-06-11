"""Build a self-contained interactive review page for the benchmark images.

Exports, per image: the base frame, the fascicle mask (red), the aponeurosis mask (cyan), and the
exact geometry the pipeline measures (aponeurosis fit lines, the fascicle fragments we USE, the
fragments we EXCLUDE as too-small/too-flat, and every fragment's extrapolated full length). Writes
results/visual_review/index.html with all layers toggleable and prev/next navigation, sorted worst-FL
first. Open the html in any browser - no server needed.

    python umud-muscle-architecture/experiments/visual_review_export.py
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402

OUT = ROOT / "results" / "visual_review"


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def rgba(mask, rgb, a=170):
    h, w = mask.shape
    out = np.zeros((h, w, 4), np.uint8)
    m = mask > 0
    out[m] = (rgb[2], rgb[1], rgb[0], a)  # cv2 writes BGRA
    return out


def isect_par(cs, b, par, xref):
    """Intersect line y=cs*x+b with parabola y=par(x); return the (x,y) root nearest xref."""
    A, B, C = float(par[0]), float(par[1]), float(par[2])
    a2, b2, c2 = A, B - cs, C - b
    if abs(a2) < 1e-9:
        if abs(b2) < 1e-12:
            return None
        x = -c2 / b2
    else:
        disc = b2 * b2 - 4 * a2 * c2
        if disc < 0:
            return None
        sq = disc ** 0.5
        r1, r2 = (-b2 + sq) / (2 * a2), (-b2 - sq) / (2 * a2)
        x = r1 if abs(r1 - xref) <= abs(r2 - xref) else r2
    return (x, cs * x + b)


def draw_geometry(am, fm):
    """Replicate measure()'s apo + fragment selection, returning drawable coords (image space)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(am, 8)
    bands = sorted([(stats[i, 4], i) for i in range(1, n)], reverse=True)[:2]
    if len(bands) < 2:
        return None
    binfo = []
    for _, i in bands:
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            return None
        binfo.append((float(np.mean(ys)), xs, ys))
    binfo.sort()
    fit, edges = [], []
    for role, (_, xs, ys) in zip(("sup", "deep"), binfo):
        ux, inv = np.unique(xs, return_inverse=True)
        if role == "sup":
            ey = np.full(len(ux), -1.0); np.maximum.at(ey, inv, ys.astype(float))
        else:
            ey = np.full(len(ux), 1e18); np.minimum.at(ey, inv, ys.astype(float))
        fit.append(M.fit_line(ey, ux.astype(float)))
        edges.append((ux.astype(float), ey.astype(float)))
    sup, deep = fit[0], fit[1]
    deep_s = deep[0]
    W = am.shape[1]
    sup_par = np.polyfit(*edges[0], 2) if len(edges[0][0]) >= 6 else None   # parabolic apo (your bend idea)
    deep_par = np.polyfit(*edges[1], 2) if len(edges[1][0]) >= 6 else None
    used, excluded = [], []
    nf, labf, statsf, _ = cv2.connectedComponentsWithStats(fm, 8)
    for i in range(1, nf):
        area = int(statsf[i, 4])
        ys, xs = np.where(labf == i)
        if len(xs) < 8:
            continue
        fs, fb = M.pca_line(ys, xs)
        ang = abs(np.degrees(np.arctan(fs) - np.arctan(deep_s)))
        if ang > 90:
            ang = 180 - ang
        x0, x1 = int(xs.min()), int(xs.max())
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        seg = {"x0": x0, "y0": round(fs * x0 + fb, 1), "x1": x1, "y1": round(fs * x1 + fb, 1),
               "area": area, "ang": round(float(ang), 1), "fs": float(fs), "cx": cx, "cy": cy}
        up = M.line_intersection((fs, fb), sup)
        lo = M.line_intersection((fs, fb), deep)
        if up is not None and lo is not None:
            fl = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
            seg["extra"] = {"x0": round(lo[0], 1), "y0": round(lo[1], 1),
                            "x1": round(up[0], 1), "y1": round(up[1], 1), "fl": round(fl, 0)}
        if area >= M.FASC_MIN_AREA and M.FASC_MIN_ANG <= ang <= 75:
            used.append(seg)
        else:
            excluded.append(seg)
    # CONSENSUS field: one shared angle (robust median of fragment angles, length-weighted), then
    # cast every used fragment from its own centroid at that angle -> parallel, non-crossing fascicles.
    new_fl_px = new_fl_on_px = new_fl_mx_px = new_fl_vw_px = new_fl_par_px = new_fl_pf_px = None
    new_fl_pfon_px = new_fl_pfmx_px = None
    # facing constraint: keep an apo's curve ONLY if it bows toward the muscle (gap can't diverge at
    # the edges). superficial is the top band, so it should curve down in the middle (A>=0); deep is the
    # bottom band, so it should curve up in the middle (A<=0). Otherwise fall back to the straight line.
    sup_f = sup_par if (sup_par is not None and float(sup_par[0]) >= 0) else None
    deep_f = deep_par if (deep_par is not None and float(deep_par[0]) <= 0) else None
    if used:
        angs = np.array([np.arctan(u["fs"]) for u in used])
        wts = np.array([u["area"] for u in used], float)
        cs = float(np.tan(float(M.weighted_median(angs, wts))))
        all_fl, on_fl, mx_fl, vw_fl, vw_w, par_fl, pf_fl, pf_on_fl, pf_mx_fl = [[] for _ in range(9)]
        for u in used:
            b = u["cy"] - cs * u["cx"]
            up = M.line_intersection((cs, b), sup)
            lo = M.line_intersection((cs, b), deep)
            if up is None or lo is None:
                continue
            fl = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
            onscr = (-2 <= up[0] <= W + 2) and (-2 <= lo[0] <= W + 2)
            vlen = float(np.hypot(u["x1"] - u["x0"], u["y1"] - u["y0"]))
            vfrac = min(1.0, vlen / (fl + 1e-6))                          # how much of the line is real, not guessed
            u["cons"] = {"x0": round(lo[0], 1), "y0": round(lo[1], 1),
                         "x1": round(up[0], 1), "y1": round(up[1], 1), "on": bool(onscr)}
            all_fl.append(fl)
            if onscr: on_fl.append(fl)
            if vfrac >= 0.25: mx_fl.append(fl)
            vw_fl.append(fl); vw_w.append(vfrac * vfrac)                  # graded-by-visibility (your idea)
            if sup_par is not None and deep_par is not None:              # parabolic-apo length (your idea)
                pu = isect_par(cs, b, sup_par, u["cx"]); pl = isect_par(cs, b, deep_par, u["cx"])
                if pu and pl:
                    par_fl.append(float(np.hypot(pu[0] - pl[0], pu[1] - pl[1])))
                    u["par"] = {"x0": round(pl[0], 1), "y0": round(pl[1], 1),
                                "x1": round(pu[0], 1), "y1": round(pu[1], 1)}
            fu = isect_par(cs, b, sup_f, u["cx"]) if sup_f is not None else up   # facing: curve only where it bows in
            fd = isect_par(cs, b, deep_f, u["cx"]) if deep_f is not None else lo
            if fu and fd:
                pf = float(np.hypot(fu[0] - fd[0], fu[1] - fd[1]))
                pf_fl.append(pf)
                if (-2 <= fu[0] <= W + 2) and (-2 <= fd[0] <= W + 2):   # facing + on-screen (stacked)
                    pf_on_fl.append(pf)
                if vlen / (pf + 1e-6) >= 0.25:                          # facing + minimize-extrapolation (stacked)
                    pf_mx_fl.append(pf)
                u["pf"] = {"x0": round(fd[0], 1), "y0": round(fd[1], 1),
                           "x1": round(fu[0], 1), "y1": round(fu[1], 1)}
        med = lambda L: float(np.median(L)) if L else None
        new_fl_px, new_fl_on_px = med(all_fl), med(on_fl)
        new_fl_mx_px, new_fl_par_px, new_fl_pf_px = med(mx_fl), med(par_fl), med(pf_fl)
        new_fl_pfon_px, new_fl_pfmx_px = med(pf_on_fl), med(pf_mx_fl)
        if vw_fl:
            new_fl_vw_px = float(M.weighted_median(np.array(vw_fl), np.array(vw_w)))
    pp = lambda P: [[round(float(x), 1), round(float(np.polyval(P, x)), 1)]
                    for x in np.linspace(0, W, 9)] if P is not None else None
    line = lambda L: {"x0": 0, "y0": round(M.line_y(L, 0), 1), "x1": W, "y1": round(M.line_y(L, W), 1)}
    return {"sup": line(sup), "deep": line(deep), "sup_par": pp(sup_par), "deep_par": pp(deep_par),
            "used": used, "excluded": excluded,
            "new_fl_px": new_fl_px, "new_fl_on_px": new_fl_on_px, "new_fl_mx_px": new_fl_mx_px,
            "new_fl_vw_px": new_fl_vw_px, "new_fl_par_px": new_fl_par_px, "new_fl_pf_px": new_fl_pf_px,
            "new_fl_pfon_px": new_fl_pfon_px, "new_fl_pfmx_px": new_fl_pfmx_px}


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for _, r in truth.iterrows():
        iid = r.ImageID
        img = M.read_rgb(bench / f"{iid}.tif")
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        g = M.measure(am, fm)
        if g is None or not g.get("fl_px"):
            continue
        geo = draw_geometry(am, fm)
        if geo is None:
            continue
        cv2.imwrite(str(OUT / f"{iid}_base.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(OUT / f"{iid}_fasc.png"), rgba(fm, (255, 40, 40)))
        cv2.imwrite(str(OUT / f"{iid}_apo.png"), rgba(am, (0, 220, 255)))
        ppm = float(r.scale_px_per_cm) / 10.0
        pa = g["pa_deg"]
        cv = lambda k: round(geo[k] / ppm, 1) if geo.get(k) else 0
        recs.append({
            "id": iid, "w": int(am.shape[1]), "h": int(am.shape[0]),
            "geo": geo,
            "ours_fl": round(g["fl_px"] / ppm, 1),
            "new_fl": cv("new_fl_px"), "new_fl_on": cv("new_fl_on_px"), "new_fl_mx": cv("new_fl_mx_px"),
            "new_fl_vw": cv("new_fl_vw_px"), "new_fl_par": cv("new_fl_par_px"), "new_fl_pf": cv("new_fl_pf_px"),
            "new_fl_pfon": cv("new_fl_pfon_px"), "new_fl_pfmx": cv("new_fl_pfmx_px"),
            "id_fl": round((g["mt_px"] / np.sin(np.radians(max(pa, 1)))) / ppm, 1) if pa else 0,
            "true_fl": round(float(r.fl_mm_true), 1),
            "ours_pa": round(pa, 1) if pa else 0, "true_pa": round(float(r.pa_deg_true), 1),
            "ours_mt": round(g["mt_px"] / ppm, 1), "true_mt": round(float(r.mt_mm_true), 1),
            "n_used": len(geo["used"]), "n_excl": len(geo["excluded"]),
        })
    recs.sort(key=lambda d: -abs((d["new_fl"] or d["ours_fl"]) - d["true_fl"]))  # worst CONSENSUS first
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(recs)), encoding="utf-8")
    tru = np.array([d["true_fl"] for d in recs])
    def report(key, label):
        v = np.array([d[key] for d in recs], float)
        ok = v > 0
        err = np.abs(v[ok] - tru[ok]); bias = (v[ok] - tru[ok])
        print(f"  {label:26s} mean|err| {err.mean():.1f}mm  signed bias {bias.mean():+.1f}mm")
    print(f"wrote {len(recs)} images + results/visual_review/index.html")
    print("RAW FL error vs expert (no recentering):")
    report("ours_fl", "current (per-fragment)")
    report("new_fl", "consensus (parallel)")
    report("new_fl_on", "consensus + on-screen only")
    report("new_fl_mx", "consensus + min-extrapolation")
    report("new_fl_par", "consensus + parabolic apo")
    report("new_fl_pf", "consensus + FACING parabola")
    report("new_fl_pfon", "FACING + on-screen (stacked)")
    report("new_fl_pfmx", "FACING + min-extrap (stacked)")
    report("id_fl", "identity MT/sin(PA)")
    for iid in ("im_27_arch", "im_29_arch", "im_12_arch"):
        d = next((x for x in recs if x["id"] == iid), None)
        if d:
            print(f"  {iid}: true {d['true_fl']}  current {d['ours_fl']}  parabola {d['new_fl_par']}  "
                  f"facing {d['new_fl_pf']}")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>UMUD visual review</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 #bar{padding:8px 12px;background:#1c1c1c;position:sticky;top:0;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #stage{position:relative;margin:10px auto;max-width:1100px}
 #stage img,#stage svg{position:absolute;top:0;left:0;width:100%;height:auto}
 #base{position:relative !important}
 label{font-size:13px;cursor:pointer;user-select:none}
 .num{font-variant-numeric:tabular-nums}
 button{background:#333;color:#eee;border:1px solid #555;border-radius:5px;padding:4px 10px;cursor:pointer}
 .bad{color:#ff7070}.ok{color:#79e07a}
 #read{padding:6px 12px;font-size:14px;line-height:1.6}
</style></head><body>
<div id="bar">
 <button onclick="go(-1)">&larr;</button><button onclick="go(1)">&rarr;</button>
 <span id="pos" class="num"></span>
 <label><input type="checkbox" id="L_fasc" checked> red mask</label>
 <label><input type="checkbox" id="L_apo" checked> cyan apo</label>
 <label><input type="checkbox" id="L_used" checked> yellow fits</label>
 <label><input type="checkbox" id="L_excl"> gray excluded</label>
 <label><input type="checkbox" id="L_apoln" checked> apo lines</label>
 <label><input type="checkbox" id="L_extra"> green=current</label>
 <label><input type="checkbox" id="L_cons" checked> magenta=consensus</label>
 <label>x&le; <input type="range" id="xs" min="0" max="100" value="100" style="vertical-align:middle;width:130px"> <span id="xlab" class="num">100%</span></label>
</div>
<div id="read"></div>
<div id="stage">
 <img id="base"><img id="fasc"><img id="apo"><svg id="ov" preserveAspectRatio="none"></svg>
</div>
<script>
const D=__DATA__; let i=0;
const $=id=>document.getElementById(id);
function seg(o,c,wd,dash){return `<line x1="${o.x0}" y1="${o.y0}" x2="${o.x1}" y2="${o.y1}" stroke="${c}" stroke-width="${wd}"${dash?' stroke-dasharray="7 4"':''}/>`}
function poly(pts,c){return `<polyline points="${pts.map(p=>p[0]+','+p[1]).join(' ')}" fill="none" stroke="${c}" stroke-width="2.5" stroke-dasharray="5 3"/>`}
function sgn(x){return (x>0?'+':'')+(+x).toFixed(0)}
function render(){
 const r=D[i], g=r.geo;
 $('base').src=r.id+'_base.jpg'; $('fasc').src=r.id+'_fasc.png'; $('apo').src=r.id+'_apo.png';
 $('ov').setAttribute('viewBox',`0 0 ${r.w} ${r.h}`);
 const xp=+$('xs').value, xt=xp/100*r.w; $('xlab').textContent=xp+'%';
 const vis=u=>(u.cx===undefined)||(u.cx<=xt);
 let s='';
 if($('L_extra').checked) for(const u of g.used) if(u.extra&&vis(u)) s+=seg(u.extra,'rgba(0,255,90,.6)',2);
 if($('L_cons').checked) for(const u of g.used) if(u.cons&&vis(u)){
   const off=!u.cons.on; s+=seg(u.cons, off?'rgba(255,90,0,.95)':'rgba(255,0,200,.75)',2,off);
 }
 if($('L_apoln').checked){ s+=seg(g.sup,'#ff9b2e',3)+seg(g.deep,'#2ec5ff',3);
   if(g.sup_par) s+=poly(g.sup_par,'#ffd089'); if(g.deep_par) s+=poly(g.deep_par,'#9fe6ff'); }
 if($('L_excl').checked) for(const e of g.excluded) if(vis(e)) s+=seg(e,'rgba(170,170,170,.85)',2);
 if($('L_used').checked) for(const u of g.used) if(vis(u)) s+=seg(u,'#ffe000',3);
 $('ov').innerHTML=s;
 for(const k of ['fasc','apo']) $(k).style.display=$('L_'+k).checked?'block':'none';
 $('read').innerHTML=`<b>${r.id}</b> &nbsp; used ${r.n_used}, excluded ${r.n_excl} &nbsp; <span class="num">[x&le;${xp}%]</span><br>`+
  `FL &nbsp; <b>TRUE ${r.true_fl}</b> &nbsp;|&nbsp; `+
  `<span style="color:#3f9">current ${r.ours_fl}</span> ${sgn(r.ours_fl-r.true_fl)} &nbsp; `+
  `<span style="color:#f4c">consensus ${r.new_fl}</span> ${sgn(r.new_fl-r.true_fl)} &nbsp; `+
  `<b style="color:#7df">parabola ${r.new_fl_par}</b> ${sgn(r.new_fl_par-r.true_fl)} &nbsp; `+
  `<b style="color:#6f6">facing ${r.new_fl_pf}</b> ${sgn(r.new_fl_pf-r.true_fl)} &nbsp; `+
  `on-scr ${r.new_fl_on} &nbsp; identity ${r.id_fl}<br>`+
  `PA ${r.ours_pa}/${r.true_pa} &nbsp; MT ${r.ours_mt}/${r.true_mt} &nbsp; <i>(dashed apo curves = parabolic fit; orange dashed fascicle = runs off-screen)</i>`;
 $('pos').textContent=`${i+1}/${D.length}`;
}
function go(d){i=(i+d+D.length)%D.length;render()}
for(const id of ['L_fasc','L_apo','L_used','L_excl','L_apoln','L_extra','L_cons']) $(id).onchange=render;
$('xs').oninput=render;
document.onkeydown=e=>{if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1)};
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
