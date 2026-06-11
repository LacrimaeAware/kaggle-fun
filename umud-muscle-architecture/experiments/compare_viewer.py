"""Interactive compare viewer: OUR predicted masks vs the HOST ground-truth masks on training images,
aligned the host's way (resize both to a square). Toggle each of the 4 layers, flip through with the
arrow keys, you judge. No metrics. -> results/compare_viewer/index.html (open in a browser).

    python umud-muscle-architecture/experiments/compare_viewer.py            # default ~60 images
    UMUD_COMPARE_N=200 python ... compare_viewer.py                           # more
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

AI = ROOT / "data/apo_imgs_v1/apo_images_new_model_v1"
AM = ROOT / "data/apo_masks_v1/apo_masks_new_model_v1"
FM = ROOT / "data/fasc_masks_v1/fasc_masks_new_model_v1"
OUT = ROOT / "results" / "compare_viewer"
S = 640


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def rgba(mask01, rgb, a=180):
    out = np.zeros((S, S, 4), np.uint8)
    m = mask01 > 0
    out[m] = (rgb[2], rgb[1], rgb[0], a)
    return out


def main():
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    N = int(os.environ.get("UMUD_COMPARE_N", "60"))
    files = sorted(p for p in AI.iterdir() if p.suffix.lower() == ".tif")
    sample = files[:: max(1, len(files) // N)][:N]
    recs = []
    for j, p in enumerate(sample):
        ga = cv2.imread(str(AM / p.name), 0)
        gf = cv2.imread(str(FM / p.name), 0)
        if ga is None or gf is None:
            continue
        img = M.read_rgb(p)
        cv2.imwrite(str(OUT / f"{p.stem}_base.jpg"),
                    cv2.cvtColor(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (S, S)), cv2.COLOR_GRAY2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(OUT / f"{p.stem}_hapo.png"), rgba(cv2.resize(ga, (S, S)) > 127, (255, 140, 0)))
        cv2.imwrite(str(OUT / f"{p.stem}_hfasc.png"), rgba(cv2.resize(gf, (S, S)) > 127, (0, 255, 0)))
        cv2.imwrite(str(OUT / f"{p.stem}_oapo.png"),
                    rgba(cv2.resize(np.asarray(M.predict_mask(apo, img), np.uint8), (S, S)) > 0, (0, 200, 255)))
        cv2.imwrite(str(OUT / f"{p.stem}_ofasc.png"),
                    rgba(cv2.resize(np.asarray(M.predict_mask(fasc, img), np.uint8), (S, S)) > 0, (255, 40, 40)))
        recs.append({"id": p.stem})
        if (j + 1) % 15 == 0:
            print(f"  {j+1}/{len(sample)}", flush=True)
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(recs)), encoding="utf-8")
    print(f"wrote {len(recs)} images + results/compare_viewer/index.html")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>compare: ours vs host</title><style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 #bar{padding:8px 12px;background:#1c1c1c;position:sticky;top:0;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #stage{position:relative;margin:10px auto;max-width:900px}#stage img{position:absolute;top:0;left:0;width:100%}#base{position:relative !important}
 label{font-size:14px;cursor:pointer}button{background:#333;color:#eee;border:1px solid #555;border-radius:5px;padding:5px 11px;cursor:pointer}
 b1{color:#ff9b2e}b2{color:#2ec5ff}b3{color:#3f3}b4{color:#f55}
</style></head><body><div id="bar">
 <button onclick="go(-1)">&larr;</button><button onclick="go(1)">&rarr;</button><span id="pos"></span>
 <label><input type="checkbox" id="hapo" checked> <b1>host apo</b1></label>
 <label><input type="checkbox" id="oapo" checked> <b2>OUR apo</b2></label>
 <label><input type="checkbox" id="hfasc"> <b3>host fascicle</b3></label>
 <label><input type="checkbox" id="ofasc"> <b4>OUR fascicle</b4></label>
 <span id="name" style="font-weight:bold"></span>
</div><div id="stage"><img id="base"><img id="L_hapo"><img id="L_oapo"><img id="L_hfasc"><img id="L_ofasc"></div>
<script>
const D=__DATA__;let i=0;const $=id=>document.getElementById(id);
const M={hapo:'hapo',oapo:'oapo',hfasc:'hfasc',ofasc:'ofasc'};
function render(){const r=D[i];$('base').src=r.id+'_base.jpg';
 for(const k in M){const el=$('L_'+k);el.src=r.id+'_'+M[k]+'.png';el.style.display=$(k).checked?'block':'none';}
 $('pos').textContent=`${i+1}/${D.length}`;$('name').textContent=r.id;}
function go(d){i=(i+d+D.length)%D.length;render()}
for(const k in M)$(k).onchange=render;
document.onkeydown=e=>{if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1)};render();
</script></body></html>"""


if __name__ == "__main__":
    main()
