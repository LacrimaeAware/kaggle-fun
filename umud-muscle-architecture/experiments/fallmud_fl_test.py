"""Independent FL test on FALLMUD/NeilCronin (GT apo + GT fascicle masks on the same image, never
trained on). Two FL estimates, both in pixels (scale-free), both shown so the user can eyeball them:
  REFERENCE (yellow) = straight extrapolation of each GT fascicle to the GT apos (the host's method).
  OURS     (green)  = the per-gap wave geometry on the same GT masks.
Neither is absolute truth (both extrapolate), which is why they are both drawn. -> results/fallmud_fl/

    python experiments/fallmud_fl_test.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import segment_then_measure as M       # noqa: E402
import per_gap_viewer as PGV           # noqa: E402

B = ROOT / "data/dropoff/FALLMUD/NeilCronin"
OUT = ROOT / "results/fallmud_fl"


def rgba(mask, rgb, a=185):
    h, w = mask.shape
    o = np.zeros((h, w, 4), np.uint8); o[mask > 0] = (rgb[2], rgb[1], rgb[0], a)
    return o


def reference_fl(am, fm):
    """Host method: each GT fascicle straight-extrapolated to the GT apo lines. Returns (median FL, lines)."""
    bands = PGV.apo_bands(am)
    if len(bands) < 2:
        return None, []
    bt, bb = bands[0], bands[-1]
    sup = M.fit_line(bt["bot"], bt["ux"]); deep = M.fit_line(bb["top"], bb["ux"])
    nf, labf, st, _ = cv2.connectedComponentsWithStats(fm, 8)
    fls, lines = [], []
    for i in range(1, nf):
        if st[i, 4] < 6:
            continue
        ys, xs = np.where(labf == i)
        if len(xs) < 4:
            continue
        fs, _ = M.pca_line(ys, xs); cx, cy = float(xs.mean()), float(ys.mean()); b = cy - fs * cx
        up = M.line_intersection((fs, b), sup); lo = M.line_intersection((fs, b), deep)
        if up is None or lo is None:
            continue
        fl = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
        if 10 <= fl <= 2000:
            fls.append(fl)
            lines.append([[round(lo[0], 1), round(lo[1], 1)], [round(up[0], 1), round(up[1], 1)]])
    return (float(np.median(fls)) if fls else None), lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted((B / "images").glob("*.tif"))[::10][:26]
    recs, ratios = [], []
    for p in imgs:
        img = cv2.imread(str(p)); h, w = img.shape[:2]
        amp = next((B / "aponeurosis_masks").glob(p.stem + ".*")); fmp = next((B / "fascicle_masks").glob(p.stem + ".*"))
        am = (cv2.resize(cv2.imread(str(amp), 0), (w, h)) > 127).astype(np.uint8)
        fm = (cv2.resize(cv2.imread(str(fmp), 0), (w, h)) > 127).astype(np.uint8)
        ref_fl, ref_lines = reference_fl(am, fm)
        g = PGV.per_gap(am, fm, w)
        our_fl = g["avg_px"] if g else None
        waves = [wv for gp in (g["gaps"] if g else []) for wv in gp.get("waves", [])]
        if ref_fl is None or our_fl is None:
            print(f"{p.stem}: skipped (ref {ref_fl} ours {our_fl})", flush=True); continue
        ratios.append(our_fl / ref_fl)
        cv2.imwrite(str(OUT / f"{p.stem}_base.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(OUT / f"{p.stem}_apo.png"), rgba(am, (0, 220, 255)))
        cv2.imwrite(str(OUT / f"{p.stem}_fasc.png"), rgba(fm, (255, 40, 40)))
        recs.append({"id": p.stem, "W": w, "H": h, "ref_fl": round(ref_fl, 0), "our_fl": round(our_fl, 0),
                     "ref_lines": ref_lines, "waves": waves})
        print(f"{p.stem}: ref(straight) {ref_fl:.0f}px  ours(wave) {our_fl:.0f}px  ratio {our_fl/ref_fl:.2f}", flush=True)
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(recs)), encoding="utf-8")
    if ratios:
        r = np.array(ratios)
        print(f"\n{len(recs)} images. our-wave / ref-straight FL ratio: median {np.median(r):.2f}, "
              f"mean {r.mean():.2f} (1.0 = match; <1 our wave is shorter/stays on-screen, >1 longer).")
        print(f"viewer -> results/fallmud_fl/index.html")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>FALLMUD FL test</title><style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 #bar{padding:8px 12px;background:#1c1c1c;position:sticky;top:0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 #stage{position:relative;margin:8px auto;max-width:760px}#stage img,#stage svg{position:absolute;top:0;left:0;width:100%;height:auto}#base{position:relative!important}
 label{font-size:13px;cursor:pointer}button{background:#333;color:#eee;border:1px solid #555;border-radius:5px;padding:5px 11px;cursor:pointer}
 #read{padding:6px 12px;font-size:15px}
</style></head><body><div id="bar">
 <button onclick="go(-1)">&larr;</button><button onclick="go(1)">&rarr;</button><span id="pos"></span>
 <label><input type="checkbox" id="L_apo" checked> GT apo</label><label><input type="checkbox" id="L_fasc" checked> GT fascicle</label>
 <label><input type="checkbox" id="L_ref" checked> <span style="color:#ff2">REFERENCE straight FL</span></label>
 <label><input type="checkbox" id="L_our" checked> <span style="color:#3f6">OUR wave FL</span></label>
</div><div id="read"></div><div id="stage"><img id="base"><img id="fasc"><img id="apo"><svg id="ov" preserveAspectRatio="none"></svg></div>
<script>
const D=__DATA__;let i=0;const $=id=>document.getElementById(id);
function PL(p,c,w){return `<path d="${p.map((q,k)=>(k?'L':'M')+q[0]+' '+q[1]).join(' ')}" fill="none" stroke="${c}" stroke-width="${w}"/>`}
function render(){const r=D[i];$('base').src=r.id+'_base.jpg';$('fasc').src=r.id+'_fasc.png';$('apo').src=r.id+'_apo.png';
 $('ov').setAttribute('viewBox',`0 0 ${r.W} ${r.H}`);let s='';
 if($('L_ref').checked)r.ref_lines.forEach(l=>s+=PL(l,'#ff2',2));
 if($('L_our').checked)(r.waves||[]).forEach(w=>s+=PL(w,'#3f6',2));
 $('ov').innerHTML=s;for(const k of['fasc','apo'])$(k).style.display=$('L_'+k).checked?'block':'none';
 $('read').innerHTML=`<b>${r.id}</b> &nbsp; REFERENCE straight FL <b style="color:#ff2">${r.ref_fl}px</b> &nbsp; OUR wave FL <b style="color:#3f6">${r.our_fl}px</b> &nbsp; ratio ${(r.our_fl/r.ref_fl).toFixed(2)}`;
 $('pos').textContent=`${i+1}/${D.length}`;}
function go(d){i=(i+d+D.length)%D.length;render()}
for(const id of['L_apo','L_fasc','L_ref','L_our'])$(id).onchange=render;
document.onkeydown=e=>{if(e.key=='ArrowLeft')go(-1);if(e.key=='ArrowRight')go(1)};render();
</script></body></html>"""


if __name__ == "__main__":
    main()
