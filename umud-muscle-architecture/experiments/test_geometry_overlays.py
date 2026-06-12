"""Draw the facing-geometry FL on the actual 309 TEST images (no truth) - to SEE whether it stays
sane on the hard out-of-distribution images, especially the ones where it diverged most from the
0.61918 baseline. Writes results/test_geometry/index.html (interactive, same viewer) sorted by
|facing - baseline| so the biggest movers are first.

    python umud-muscle-architecture/experiments/test_geometry_overlays.py
"""
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
import visual_review_export as V  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT = ROOT / "results" / "test_geometry"
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def main():
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(Path.home() / "Downloads" / "0P61918_submission_local.csv").set_index("image_id")
    cur = pd.read_csv(ROOT / "results" / "submission_local.csv").set_index("image_id")
    files = sorted(p for p in TEST.iterdir() if p.suffix.lower() in EXTS)
    recs = []
    for p in files:
        img = M.read_rgb(p)
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        geo = V.draw_geometry(am, fm)
        if geo is None:
            continue
        cv2.imwrite(str(OUT / f"{p.stem}_base.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 84])
        cv2.imwrite(str(OUT / f"{p.stem}_fasc.png"), V.rgba(fm, (255, 40, 40)))
        cv2.imwrite(str(OUT / f"{p.stem}_apo.png"), V.rgba(am, (0, 220, 255)))
        bfl = float(base.loc[p.name, "fl_mm"]) if p.name in base.index else 0.0
        ffl = float(cur.loc[p.name, "fl_mm"]) if p.name in cur.index else 0.0
        recs.append({
            "id": p.stem, "w": int(am.shape[1]), "h": int(am.shape[0]), "geo": geo,
            "ours_fl": round(ffl, 1), "new_fl": round(ffl, 1), "new_fl_par": round(ffl, 1),
            "new_fl_pf": round(ffl, 1), "new_fl_on": 0, "id_fl": round(bfl, 1),
            "true_fl": None,  # no truth on the test set
            "ours_pa": 0, "true_pa": 0, "ours_mt": 0, "true_mt": 0,
            "n_used": len(geo["used"]), "n_excl": len(geo["excluded"]),
            "delta": round(ffl - bfl, 1),
        })
    recs.sort(key=lambda d: -abs(d["delta"]))  # biggest movers vs baseline first
    (OUT / "index.html").write_text(V.HTML.replace("__DATA__", json.dumps(recs)), encoding="utf-8")
    print(f"wrote {len(recs)} test overlays + results/test_geometry/index.html (sorted: biggest FL movers first)")
    print("readout 'identity' column = the 0.61918 BASELINE FL; 'facing' = the submitted value.")
    big = sorted(recs, key=lambda d: -abs(d["delta"]))[:8]
    print("biggest movers:", [(r["id"], r["delta"]) for r in big])


if __name__ == "__main__":
    main()
