"""Fascicle compare viewer, CORRECTLY paired: uses the fascicle image set (fasc_imgs) as background,
host fascicle mask (green) vs OUR fascicle prediction (red), aligned the host's way (resize to square).
Toggle each, flip with arrows, you judge. -> results/compare_fasc_viewer/index.html

    python umud-muscle-architecture/experiments/compare_fasc_viewer.py
    UMUD_COMPARE_N=150 python ... compare_fasc_viewer.py
"""
import os
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

FI = ROOT / "data/fasc_imgs_v1/fasc_images_new_model_v1"
FM = ROOT / "data/fasc_masks_v1/fasc_masks_new_model_v1"
OUT = ROOT / "results" / "compare_fasc_viewer"
S = 640


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def rgba(mask01, rgb, a=190):
    out = np.zeros((S, S, 4), np.uint8)
    out[mask01 > 0] = (rgb[2], rgb[1], rgb[0], a)
    return out


def main():
    fasc = load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    N = int(os.environ.get("UMUD_COMPARE_N", "60"))
    files = sorted(p for p in FI.iterdir() if p.suffix.lower() == ".tif")
    sample = files[:: max(1, len(files) // N)][:N]
    recs = []
    for j, p in enumerate(sample):
        gf = cv2.imread(str(FM / p.name), 0)
        if gf is None:
            continue
        img = M.read_rgb(p)                                   # fascicle image (the correct one)
        cv2.imwrite(str(OUT / f"{p.stem}_base.jpg"),
                    cv2.cvtColor(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (S, S)), cv2.COLOR_GRAY2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(OUT / f"{p.stem}_host.png"), rgba(cv2.resize(gf, (S, S)) > 127, (0, 255, 0)))
        cv2.imwrite(str(OUT / f"{p.stem}_ours.png"),
                    rgba(cv2.resize(np.asarray(M.predict_mask(fasc, img), np.uint8), (S, S)) > 0, (255, 40, 40)))
        recs.append({"id": p.stem})
        if (j + 1) % 15 == 0:
            print(f"  {j+1}/{len(sample)}", flush=True)
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(recs)), encoding="utf-8")
    print(f"wrote {len(recs)} fascicle images + results/compare_fasc_viewer/index.html")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>fascicle: ours vs host</title><style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 #bar{padding:8px 12px;background:#1c1c1c;position:sticky;top:0;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #stage{position:relative;margin:10px auto;max-width:900px}#stage img{position:absolute;top:0;left:0;width:100%}#base{position:relative !important}
 label{font-size:15px;cursor:pointer}button{background:#333;color:#eee;border:1px solid #555;border-radius:5px;padding:5px 11px;cursor:pointer}
</style></head><body><div id="bar">
 <button onclick="go(-1)">&larr;</button><button onclick="go(1)">&rarr;</button><span id="pos"></span>
 <label><input type="checkbox" id="host" checked> <span style="color:#3f3">host fascicle</span></label>
 <label><input type="checkbox" id="ours" checked> <span style="color:#f55">OUR fascicle</span></label>
 <span id="name" style="font-weight:bold"></span>
</div><div id="stage"><img id="base"><img id="L_host"><img id="L_ours"></div>
<script>
const D=__DATA__;let i=0;const $=id=>document.getElementById(id);
function render(){const r=D[i];$('base').src=r.id+'_base.jpg';
 $('L_host').src=r.id+'_host.png';$('L_host').style.display=$('host').checked?'block':'none';
 $('L_ours').src=r.id+'_ours.png';$('L_ours').style.display=$('ours').checked?'block':'none';
 $('pos').textContent=`${i+1}/${D.length}`;$('name').textContent=r.id;}
function go(d){i=(i+d+D.length)%D.length;render()}
$('host').onchange=render;$('ours').onchange=render;
document.onkeydown=e=>{if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1)};render();
</script></body></html>"""


if __name__ == "__main__":
    main()
