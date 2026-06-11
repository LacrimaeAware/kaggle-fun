"""A clean, presentable flip-through viewer (arrow keys / buttons) of the muscle analysis on many test
images: each frame shows the ultrasound with the detected muscle boundaries (cyan), the fibre lines
(green), and the three measurements. -> results/show_viewer/index.html

    python experiments/show_viewer.py
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
import segment_then_measure as M  # noqa: E402

TEST = ROOT / "data/test_images_v2/test_set_v2"
OUT = ROOT / "results" / "show_viewer"
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
W = 760


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / f"results/seg_{t}.pt", map_location="cpu"))
    return m.eval()


def bands(am):
    n, lab, st, _ = cv2.connectedComponentsWithStats(am, 8)
    bs = [(st[i, 4], float(np.where(lab == i)[0].mean())) for i in range(1, n) if st[i, 4] >= 200]
    bs.sort(key=lambda r: -r[0])
    ys = sorted(y for _, y in bs[:2])
    return ys if len(ys) == 2 else None


def annotate(img, am, fm, by, pa, fl, mt):
    h, w = img.shape[:2]
    vis = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
    ov = vis.copy(); ov[am > 0] = (255, 200, 25)                  # muscle boundaries = cyan
    vis = cv2.addWeighted(ov, 0.40, vis, 0.60, 0)
    ys, xs = np.where(fm > 0)
    if len(xs) > 8:
        slope, _ = M.pca_line(ys, xs)               # true fibre direction
        sup_y, deep_y = by
        L = 66.0
        nrm = float(np.hypot(1.0, slope)); dx = L / nrm; dy = slope * L / nrm
        rows = np.arange(sup_y + 10, deep_y - 6, 32) if deep_y - sup_y > 30 else [(sup_y + deep_y) / 2]
        for yy in rows:                             # short fibre strokes filling the muscle at the real angle
            for xx in np.arange(0.07 * w, 0.94 * w, 0.105 * w):
                a = (int(xx - dx / 2), int(yy - dy / 2)); b = (int(xx + dx / 2), int(yy + dy / 2))
                cv2.line(vis, a, b, (60, 255, 90), 2, cv2.LINE_AA)
    scale = W / w
    vis = cv2.resize(vis, (W, int(h * scale)))
    # measurement panel
    txt = [f"Fibre tilt:        {pa:.0f} deg"]
    if fl:
        txt.append(f"Fibre length:    {fl:.0f} mm")
    if mt:
        txt.append(f"Muscle thickness: {mt:.1f} mm")
    y0 = vis.shape[0] - 18 * len(txt) - 16
    cv2.rectangle(vis, (10, y0 - 8), (270, vis.shape[0] - 8), (25, 22, 18), -1)
    cv2.rectangle(vis, (10, y0 - 8), (270, vis.shape[0] - 8), (90, 255, 110), 1)
    for k, t in enumerate(txt):
        cv2.putText(vis, t, (18, y0 + 14 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 255, 235), 1, cv2.LINE_AA)
    return vis


def main():
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in TEST.iterdir() if p.suffix.lower() in EXTS)
    sample = files[:: max(1, len(files) // 24)][:24]
    recs = []
    for p in sample:
        img = M.read_rgb(p)
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        g = M.measure(am, fm)
        by = bands(am)
        if not g or not g.get("pa_deg") or by is None:
            continue
        cal = M.calibrate_image(p)
        ppm = cal.px_per_mm if (cal and cal.confidence >= M.CALIBRATION_MIN_CONF) else None
        pa = g["pa_deg"]
        fl = g["fl_px"] / ppm if (ppm and g.get("fl_px")) else None
        mt = g["mt_px"] / ppm if (ppm and g.get("mt_px")) else None
        cv2.imwrite(str(OUT / f"{p.stem}.jpg"), annotate(img, am, fm, by, pa, fl, mt),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        recs.append({"id": p.stem, "pa": round(pa), "fl": round(fl) if fl else None,
                     "mt": round(mt, 1) if mt else None})
        print(f"  {p.stem}: tilt {pa:.0f}deg" + (f" len {fl:.0f}mm thick {mt:.1f}mm" if fl else ""), flush=True)
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(recs)), encoding="utf-8")
    print(f"\nwrote {len(recs)} frames + results/show_viewer/index.html")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Muscle analysis</title><style>
 body{margin:0;background:#0d0f1a;color:#e8ecff;font-family:system-ui,sans-serif;text-align:center}
 h2{margin:14px 0 2px}.sub{color:#9aa3c7;font-size:14px;margin-bottom:10px}
 #bar{display:flex;gap:14px;align-items:center;justify-content:center;margin:10px}
 button{background:#1d2236;color:#e8ecff;border:1px solid #3a4264;border-radius:7px;padding:8px 16px;font-size:16px;cursor:pointer}
 img{max-width:820px;width:95%;border-radius:8px;box-shadow:0 6px 30px #0008}
 #cap{margin:10px;color:#bfc7ea;font-size:15px}
</style></head><body>
 <h2>Automatic Muscle Analysis from Ultrasound</h2>
 <div class="sub">Each scan is read by the program: cyan = muscle boundaries, green = muscle fibres. Use the arrows.</div>
 <div id="bar"><button onclick="go(-1)">&larr; Prev</button><span id="pos"></span><button onclick="go(1)">Next &rarr;</button></div>
 <div><img id="im"></div><div id="cap"></div>
<script>
const D=__DATA__;let i=0;const $=id=>document.getElementById(id);
function render(){const r=D[i];$('im').src=r.id+'.jpg';
 $('pos').textContent=`${i+1} / ${D.length}`;
 let m=`Fibre tilt ${r.pa}°`;if(r.fl)m+=` · length ${r.fl} mm · thickness ${r.mt} mm`;
 $('cap').textContent=m;}
function go(d){i=(i+d+D.length)%D.length;render()}
document.onkeydown=e=>{if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1)};render();
</script></body></html>"""


if __name__ == "__main__":
    main()
