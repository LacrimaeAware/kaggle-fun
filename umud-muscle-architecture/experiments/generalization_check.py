"""Generalization check: run the FINAL geometry (consensus + facing parabola + minimize-extrapolation)
on a diverse sample of TRAINING images (more muscles than the 35-image benchmark) and verify it stays
sane - no exploded lengths, no haywire parabolas - on muscle shapes the benchmark never saw.

We have no expert FL truth on the training set, so we check scale-independent SANITY instead:
  - geometry success rate (two apo bands + usable fascicles),
  - the facing/straight length ratio (on the benchmark the facing fix lands ~0.8-0.9; a wildly
    different distribution on training would mean the fix behaves differently on other muscles),
  - "wild" fraction (ratio <0.5 or >1.4, i.e. the geometry doing something drastic),
  - and 24 drawn overlays for the eye.

    python umud-muscle-architecture/experiments/generalization_check.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import segment_then_measure as M  # noqa: E402
import benchmark_validate as BV  # noqa: E402
import visual_review_export as V  # noqa: E402  (reuse draw_geometry)

TRAIN = ROOT / "data" / "fasc_imgs_v1" / "fasc_images_new_model_v1"
OUT = ROOT / "results" / "generalization_train"
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def run(paths, apo, fasc):
    rows = []
    for p in paths:
        img = M.read_rgb(p)
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        geo = V.draw_geometry(am, fm)
        if geo is None or not geo.get("new_fl_px"):
            rows.append(None)
            continue
        straight = geo["new_fl_px"]
        facing = geo.get("new_fl_pfmx_px") or geo.get("new_fl_pf_px") or straight
        rows.append({"straight": straight, "facing": facing, "ratio": facing / straight,
                     "n_used": len(geo["used"]), "geo": geo, "img": img, "p": p})
    return rows


def summarize(name, rows):
    ok = [r for r in rows if r]
    rat = np.array([r["ratio"] for r in ok]) if ok else np.array([0.0])
    wild = int(((rat < 0.5) | (rat > 1.4)).sum())
    print(f"{name:10s} success {len(ok)}/{len(rows)} ({100*len(ok)/max(len(rows),1):.0f}%) | "
          f"facing/straight ratio  median {np.median(rat):.2f}  p10 {np.percentile(rat,10):.2f}  "
          f"p90 {np.percentile(rat,90):.2f}  | wild {wild} ({100*wild/max(len(ok),1):.0f}%) | "
          f"median used-frags {int(np.median([r['n_used'] for r in ok])) if ok else 0}")
    return rat


def overlay(r, out):
    vis = cv2.cvtColor(r["img"], cv2.COLOR_RGB2BGR).copy()
    g = r["geo"]
    L = lambda o, c, w=2: cv2.line(vis, (int(o["x0"]), int(o["y0"])), (int(o["x1"]), int(o["y1"])), c, w)
    for u in g["used"]:
        L(u, (0, 230, 230), 2)                                  # yellow fits
        if u.get("pf"):
            L(u["pf"], (0, 220, 0), 2)                          # green facing lengths
    L(g["sup"], (255, 150, 0), 1); L(g["deep"], (0, 150, 255), 1)
    for key, col in (("sup_par", (255, 200, 120)), ("deep_par", (150, 230, 255))):
        pts = g.get(key)
        if pts:
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 2)
    cv2.putText(vis, f"{r['p'].name}  straight {r['straight']/10:.0f} -> facing {r['facing']/10:.0f}px*  n={r['n_used']}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(vis, f"{r['p'].name}  straight {r['straight']/10:.0f} -> facing {r['facing']/10:.0f}px*  n={r['n_used']}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 1)
    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    # benchmark (reference: where the method was tuned)
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    bpaths = [bench / f"{r.ImageID}.tif" for _, r in truth.iterrows()]
    # training: even stride across the sorted set -> spans the muscle/source groups
    tfiles = sorted(p for p in TRAIN.iterdir() if p.suffix.lower() in EXTS)
    step = max(1, len(tfiles) // 220)
    tpaths = tfiles[::step][:220]
    print(f"benchmark: {len(bpaths)} images | training sample: {len(tpaths)} of {len(tfiles)} (every {step}th)\n")

    brows = run(bpaths, apo, fasc)
    trows = run(tpaths, apo, fasc)
    print("=== scale-independent sanity (does the geometry behave the same off the benchmark?) ===")
    summarize("BENCHMARK", brows)
    summarize("TRAINING", trows)

    # draw 24 diverse training overlays for the eye
    ok = [r for r in trows if r]
    pick = ok[:: max(1, len(ok) // 24)][:24]
    for j, r in enumerate(pick):
        overlay(r, OUT / f"train_{j:02d}_{r['p'].stem}.jpg")
    print(f"\nwrote {len(pick)} training overlays to {OUT}")
    print("read: if TRAINING ratio distribution ~ BENCHMARK and wild% is low, the fix is general, not")
    print("a 35-image artifact. (*lengths are px/10 since training has no known scale.)")


if __name__ == "__main__":
    main()
