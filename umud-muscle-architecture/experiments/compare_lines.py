"""Compare my fitted fascicle field against the user's hand-drawn fascicles (the geometry ground truth
they made in draw_tool). For each drawn fascicle segment, my slope field predicts an angle at that
(x,y); report the mean absolute angle difference (degrees). This is "how far off are my waves", numeric.

    python experiments/compare_lines.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import segment_then_measure as M       # noqa: E402
import per_gap_viewer as PGV           # noqa: E402

DL = Path.home() / "Downloads"
TEST = ROOT / "data/test_images_v2/test_set_v2"


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / f"results/seg_{t}.pt", map_location="cpu"))
    return m.eval()


def edge_at(pts, x):
    ux = [p[0] for p in pts]; ey = [p[1] for p in pts]
    if not ux or x < ux[0] or x > ux[-1]:
        return None
    return float(np.interp(x, ux, ey))


def segs(pts):
    out = []
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        if x1 == x0:
            continue
        sl = (y1 - y0) / (x1 - x0)
        out.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, sl))
    return out


def main():
    apo, fasc = load("apo"), load("fasc")
    allf = []
    print(f"{'image':12} {'my-vs-your fascicle angle':>26} {'segs':>5}  {'your bend?':>10}")
    for jf in sorted(DL.glob("draw_IMG_*.json")):
        d = json.loads(jf.read_text()); stem = d["image"]
        p = next(TEST.glob(stem + ".*"), None)
        if p is None:
            continue
        img = M.read_rgb(p)
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        g = PGV.per_gap(am, fm, img.shape[1])
        gaps = [gp for gp in (g["gaps"] if g else []) if gp.get("field")]
        diffs, curves = [], []
        for ln in d["lines"]:
            if ln["type"] != "fascicle":
                continue
            sg = segs(ln["pts"])
            if len(sg) < 1:
                continue
            mx = float(np.mean([s[0] for s in sg])); my = float(np.mean([s[1] for s in sg]))
            curves.append(abs(np.degrees(np.arctan(sg[-1][2])) - np.degrees(np.arctan(sg[0][2]))))  # your own bend
            gg = None
            for gp in gaps:
                sy = edge_at(gp["sup_pts"], mx); dy = edge_at(gp["deep_pts"], mx)
                if sy is not None and dy is not None and sy <= my <= dy:
                    gg = gp; break
            if gg is None and gaps:
                gg = min(gaps, key=lambda gp: abs(gp["field"][4] - my))
            if gg is None:
                continue
            s0, a, b, xc, yc = gg["field"]
            for sx, sy_, sl_u in sg:
                sl_m = s0 + a * (sy_ - yc) + b * (sx - xc)
                diffs.append(abs(np.degrees(np.arctan(sl_u)) - np.degrees(np.arctan(sl_m))))
        if diffs:
            allf += diffs
            cb = f"{np.mean(curves):.1f}deg" if curves else "-"
            print(f"{stem:12} {f'mean {np.mean(diffs):.1f} / med {np.median(diffs):.1f} deg':>26} {len(diffs):>5}  {cb:>10}")
    if allf:
        a = np.array(allf)
        print(f"\nOVERALL: my fitted fascicle angle is off from your drawn lines by "
              f"mean {a.mean():.1f} deg, median {np.median(a):.1f} deg, 90th pct {np.percentile(a,90):.1f} deg "
              f"(over {len(a)} segments). Pennation tolerance is 6 deg.")


if __name__ == "__main__":
    main()
