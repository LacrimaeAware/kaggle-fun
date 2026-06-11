"""Per-gap geometry in an INTERACTIVE viewer. For each test image: fit ALL apo bands, form a gap per
consecutive pair, assign each fascicle to its gap, run the full facing method (consensus + facing
parabola + minimize-extrapolation) PER GAP, and report a fragment-count-weighted average of the gaps'
lengths. Writes results/per_gap_viewer/index.html (toggles, x-slider, prev/next, sorted by biggest
move vs the 0.61918 baseline). Open in a browser.

    python umud-muscle-architecture/experiments/per_gap_viewer.py
"""
import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import segment_then_measure as M  # noqa: E402
import visual_review_export as V  # noqa: E402  (rgba helper)

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT = ROOT / "results" / "per_gap_viewer"
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def apo_bands(am):
    """Group the apo MASK into separate aponeuroses and trace each one's edges PER-X from the mask.
    Fragments that overlap in depth (y) are the same aponeurosis (a broken-up line) and get merged;
    fragments separated by the black muscle gap (no y-overlap) are kept as DIFFERENT aponeuroses.
    No global line/parabola is fit -- the boundary is the mask edge itself."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(am, 8)
    raw = []
    for i in range(1, n):
        if stats[i, 4] < 200:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            continue
        raw.append({"ys": ys.astype(float), "xs": xs.astype(float),
                    "y0": float(ys.min()), "y1": float(ys.max())})
    raw.sort(key=lambda b: 0.5 * (b["y0"] + b["y1"]))
    merged = []
    for b in raw:
        hit = next((m for m in merged if min(b["y1"], m["y1"]) >= max(b["y0"], m["y0"])), None)  # depth overlap
        if hit is None:
            merged.append({"ys": b["ys"], "xs": b["xs"], "y0": b["y0"], "y1": b["y1"]})
        else:
            hit["ys"] = np.concatenate([hit["ys"], b["ys"]]); hit["xs"] = np.concatenate([hit["xs"], b["xs"]])
            hit["y0"] = min(hit["y0"], b["y0"]); hit["y1"] = max(hit["y1"], b["y1"])
    bands = []
    for m in merged:
        ux, inv = np.unique(m["xs"].astype(int), return_inverse=True)
        top = np.full(len(ux), 1e18); np.minimum.at(top, inv, m["ys"])
        bot = np.full(len(ux), -1.0); np.maximum.at(bot, inv, m["ys"])
        bands.append({"my": float(m["ys"].mean()), "ux": ux.astype(float), "top": top, "bot": bot})
    bands.sort(key=lambda b: b["my"])
    return bands


def _line(ux, ey, W):
    s, b = M.fit_line(ey, ux)
    return (s, b), {"x0": 0, "y0": round(b, 1), "x1": W, "y1": round(s * W + b, 1)}


def _fit_slope_field(frags):
    """The fascicle WAVE, per level. slope(x,y) = s0 + a*(y-yc) + b*(x-xc): s0 = consensus angle
    (0th order), a = BEND (slope changes with depth along a fascicle, 1st order), b = DRIFT (slope
    changes left-to-right across the muscle, 2nd order). Weighted least squares on the fragment PCA
    slopes; weights = sqrt(fragment area). Returns (s0,a,b,xc,yc) or None."""
    if not frags:
        return None
    X = np.array([f[1] for f in frags], float); Y = np.array([f[2] for f in frags], float)
    S = np.array([f[0] for f in frags], float); Wt = np.sqrt(np.array([f[3] for f in frags], float))
    xc = float(np.average(X, weights=Wt ** 2)); yc = float(np.average(Y, weights=Wt ** 2))
    if len(frags) < 4:                                        # too few to fit a bend -> consensus only
        return (float(np.median(S)), 0.0, 0.0, xc, yc)
    Amat = np.stack([np.ones_like(X), Y - yc, X - xc], 1) * Wt[:, None]
    try:
        c = np.linalg.lstsq(Amat, S * Wt, rcond=None)[0]
        return (float(c[0]), float(c[1]), float(c[2]), xc, yc)
    except Exception:
        return (float(np.median(S)), 0.0, 0.0, xc, yc)


def _trace_wave(field, sup, deep, x0):
    """Trace one fascicle from the deep apo up to the superficial apo, stepping along the BENDING slope
    field (so it follows the wave, not a straight line off-frame). Returns (points, arc_length)."""
    s0, a, b, xc, yc = field
    x, y = float(x0), float(M.line_y(deep, x0))
    pts = [(x, y)]; L = 0.0; step = 3.0
    for _ in range(2000):
        sl = s0 + a * (y - yc) + b * (x - xc)
        nrm = float(np.hypot(1.0, sl))
        dx, dy = (1.0 / nrm, sl / nrm) if sl < 0 else (-1.0 / nrm, -sl / nrm)   # head toward the sup apo (smaller y)
        x += step * dx; y += step * dy; L += step
        pts.append((x, y))
        if y <= M.line_y(sup, x) or L > 3000 or not (-300.0 < x < 4000.0) or y < 0:
            break
    return pts, L


def per_gap(am, fm, W):
    bands = apo_bands(am)
    if len(bands) < 2:
        return None
    xchk = np.linspace(0, W, 60)
    gaps = []
    for k in range(len(bands) - 1):
        bt, bb = bands[k], bands[k + 1]
        sup = M.fit_line(bt["bot"], bt["ux"])             # straight trend of the upper apo's bottom mask edge
        deep = M.fit_line(bb["top"], bb["ux"])            # straight trend of the lower apo's top mask edge
        sy = np.array([M.line_y(sup, x) for x in xchk]); dy = np.array([M.line_y(deep, x) for x in xchk])
        valid = bool(np.all(sy < dy))                     # the two apo lines must not cross (no top<->bottom swap)
        gaps.append({"sup": sup, "deep": deep, "valid": valid,
                     "sx": bt["ux"], "sy": bt["bot"], "dx": bb["ux"], "dy": bb["top"], "fr": []})
    nf, labf, statsf, _ = cv2.connectedComponentsWithStats(fm, 8)
    for i in range(1, nf):
        if statsf[i, 4] < M.FASC_MIN_AREA:
            continue
        ys, xs = np.where(labf == i)
        if len(xs) < 8:
            continue
        fs, _fb = M.pca_line(ys, xs)
        cx, cy = float(xs.mean()), float(ys.mean())
        vlen = M.fragment_visible_length(xs, ys, fs)
        for g in gaps:                                    # fragment must sit between this gap's two apos
            if g["valid"] and M.line_y(g["sup"], cx) <= cy <= M.line_y(g["deep"], cx):
                g["fr"].append((fs, cx, cy, int(statsf[i, 4]), vlen)); break
    out, fls, wts = [], [], []
    for gi, g in enumerate(gaps):
        rec = {"sup_pts": [[round(float(x), 1), round(float(y), 1)] for x, y in zip(g["sx"], g["sy"])],
               "deep_pts": [[round(float(x), 1), round(float(y), 1)] for x, y in zip(g["dx"], g["dy"])],
               "waves": [], "fl": None, "n": len(g["fr"]), "valid": g["valid"], "bend": 0.0, "drift": 0.0,
               "field": None}
        if g["valid"] and len(g["fr"]) >= 2:
            field = _fit_slope_field(g["fr"])                  # consensus + bend + drift, fit to the fragments
            if field is not None:
                rec["bend"], rec["drift"] = round(field[1], 5), round(field[2], 6)
                rec["field"] = [round(float(v), 6) for v in field]
                lens = []
                for x0 in np.linspace(0.1 * W, 0.9 * W, 9):    # a family of BENT fascicles spanning the muscle
                    pts, L = _trace_wave(field, g["sup"], g["deep"], x0)
                    if 10 <= L <= 2000 and len(pts) >= 2:
                        lens.append(L)
                        rec["waves"].append([[round(x, 1), round(y, 1)] for x, y in pts])
                if lens:
                    gfl = float(np.median(lens))               # FL = the wave's arc length, not a straight line
                    rec["fl"] = round(gfl, 0)
                    fls.append(gfl); wts.append(len(g["fr"]))
        out.append(rec)
    avg = float(np.average(fls, weights=wts)) if fls else None  # fragment-count-weighted average of gaps
    return {"gaps": out, "avg_px": avg, "W": int(W), "H": int(am.shape[0])}


def main():
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(r"C:/Users/EcceNihilum/Downloads/0P61918_submission_local.csv").set_index("image_id")
    files = sorted(p for p in TEST.iterdir() if p.suffix.lower() in EXTS)
    proto = os.environ.get("UMUD_PERGAP_PROTO", "")          # prototype on a small named set for fast review
    if proto:
        want = proto.split(",") if proto != "1" else [
            "IMG_00001", "IMG_00013", "IMG_00025", "IMG_00145",          # cleaner single-muscle examples
            "IMG_00039", "IMG_00127", "IMG_00116", "IMG_00121", "IMG_00125", "IMG_00193",
            "IMG_00091", "IMG_00211", "IMG_00037", "IMG_00073", "IMG_00285", "IMG_00277"]
        files = [p for p in files if p.stem in want]
    recs = []
    for p in files:
        img = M.read_rgb(p)
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        g = per_gap(am, fm, img.shape[1])
        if g is None:
            continue
        cv2.imwrite(str(OUT / f"{p.stem}_base.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 84])
        cv2.imwrite(str(OUT / f"{p.stem}_apo.png"), V.rgba(am, (0, 220, 255)))
        cv2.imwrite(str(OUT / f"{p.stem}_fasc.png"), V.rgba(fm, (255, 40, 40)))
        bfl = float(base.loc[p.name, "fl_mm"]) if p.name in base.index else 0.0
        g["id"] = p.stem
        g["n_gaps"] = len(g["gaps"])
        g["base_fl"] = round(bfl, 1)
        g["delta"] = round((g["avg_px"] or 0) / 10.0 - bfl, 1) if g["avg_px"] else 0  # rough, px scale unknown
        recs.append(g)
    recs.sort(key=lambda d: -abs(d["delta"]))
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(recs)), encoding="utf-8")
    (OUT / "geom.json").write_text(json.dumps({r["id"]: r for r in recs}), encoding="utf-8")  # for the draw tool
    multi = sum(1 for r in recs if r["n_gaps"] >= 2)
    print(f"wrote {len(recs)} images + results/per_gap_viewer/index.html | {multi} are multi-gap (>=2 gaps)")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>per-gap viewer</title><style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 #bar{padding:8px 12px;background:#1c1c1c;position:sticky;top:0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 #stage{position:relative;margin:8px auto;max-width:1100px}#stage img,#stage svg{position:absolute;top:0;left:0;width:100%;height:auto}#base{position:relative !important}
 label{font-size:13px;cursor:pointer}button{background:#333;color:#eee;border:1px solid #555;border-radius:5px;padding:4px 9px;cursor:pointer}
 #read{padding:6px 12px;font-size:14px;line-height:1.6}.num{font-variant-numeric:tabular-nums}
</style></head><body><div id="bar">
 <button onclick="go(-1)">&larr;</button><button onclick="go(1)">&rarr;</button><span id="pos" class="num"></span>
 <label><input type="checkbox" id="L_fasc" checked> red mask</label><label><input type="checkbox" id="L_apo" checked> cyan apo</label>
 <label><input type="checkbox" id="L_lines" checked> apo edges <span style="font-size:11px;color:#aaa">(traced straight from the mask, per-x &middot; each ends where its apo actually ends)</span></label><label><input type="checkbox" id="L_gap" checked> per-gap fascicles</label>
 <label>x&le;<input type="range" id="xs" min="0" max="100" value="100" style="width:120px;vertical-align:middle"> <span id="xl" class="num">100%</span></label>
</div><div id="read"></div><div id="stage"><img id="base"><img id="fasc"><img id="apo"><svg id="ov" preserveAspectRatio="none"></svg></div>
<script>
const D=__DATA__,GC=['rgba(0,220,255,.85)','rgba(255,0,200,.8)','rgba(0,255,0,.8)','rgba(255,180,0,.8)'];let i=0;const $=id=>document.getElementById(id);
function L(o,c,w){return `<line x1="${o.x0}" y1="${o.y0}" x2="${o.x1}" y2="${o.y1}" stroke="${c}" stroke-width="${w}"/>`}
function PL(pts,c,w,dash){return `<path d="${pts.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' ')}" fill="none" stroke="${c}" stroke-width="${w}"${dash?' stroke-dasharray="7 5"':''}/>`}
function render(){const r=D[i];$('base').src=r.id+'_base.jpg';$('fasc').src=r.id+'_fasc.png';$('apo').src=r.id+'_apo.png';
 $('ov').setAttribute('viewBox',`0 0 ${r.W} ${r.H}`);const xp=+$('xs').value,xt=xp/100*r.W;$('xl').textContent=xp+'%';let s='';
 r.gaps.forEach((g,gi)=>{if($('L_lines').checked){const c=g.valid?'#fff':'#ff5555';s+=PL(g.sup_pts,c,2,!g.valid)+PL(g.deep_pts,c,2,!g.valid);}
   if($('L_gap').checked&&g.valid)for(const wv of g.waves){const mx=wv[Math.floor(wv.length/2)][0];if(mx<=xt)s+=PL(wv,GC[gi%4],2);}});
 $('ov').innerHTML=s;for(const k of ['fasc','apo'])$(k).style.display=$('L_'+k).checked?'block':'none';
 const gi=r.gaps.map((g,j)=>`gap${j}: ${g.n} frags, FL ${g.fl||'-'}px${g.valid?'':' [REJECTED-apos cross]'} &nbsp;bend ${g.bend} drift ${g.drift}`).join(' &nbsp;|&nbsp; ');
 $('read').innerHTML=`<b>${r.id}</b> &nbsp; ${r.n_gaps} gap(s) &nbsp; [x&le;${xp}%]<br>`+gi+`<br>weighted-avg FL ${r.avg_px?r.avg_px.toFixed(0):'-'}px &nbsp; (baseline submitted ${r.base_fl}mm)`;
 $('pos').textContent=`${i+1}/${D.length}`;}
function go(d){i=(i+d+D.length)%D.length;render()}
for(const id of['L_fasc','L_apo','L_lines','L_gap'])$(id).onchange=render;$('xs').oninput=render;
document.onkeydown=e=>{if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1)};render();
</script></body></html>"""


if __name__ == "__main__":
    main()
