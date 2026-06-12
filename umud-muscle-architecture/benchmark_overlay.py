"""Diagnostic overlays on the benchmark images: draw what the pipeline sees and measures next to
the expert truth, so we (and a human) can see WHERE the geometry goes wrong - especially FL.

For each image it draws: the two predicted aponeurosis lines, the fascicle fragment fits, and a
representative straight fascicle (deep->superficial at the measured angle, = the FL the identity
uses). It prints our PA/FL/MT vs the expert consensus on the image. Saves to results/benchmark_overlay/.

    python umud-muscle-architecture/benchmark_overlay.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402

OUT = ROOT / "results" / "benchmark_overlay"


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def apo_lines(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bands = sorted([(stats[i, 4], i) for i in range(1, n)], reverse=True)[:2]
    if len(bands) < 2:
        return None
    lines = []
    for _, i in bands:
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            return None
        s, b = np.polyfit(xs, ys, 1)
        lines.append((np.mean(ys), float(s), float(b)))
    lines.sort()
    return (lines[0][1], lines[0][2]), (lines[-1][1], lines[-1][2])  # superficial, deep


def fascicles(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs, slopes, wts = [], [], []
    for i in range(1, n):
        if stats[i, 4] < 20:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        s, b = M.pca_line(ys, xs)
        segs.append((s, b, int(xs.min()), int(xs.max())))
        slopes.append(s); wts.append(int(stats[i, 4]))
    return segs, slopes, wts


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        fmd = cv2.dilate(fm, np.ones((3, 3), np.uint8), iterations=1)  # thicken so the mask is visible
        ov = vis.copy(); ov[fmd > 0] = (0, 0, 255)
        vis = cv2.addWeighted(ov, 0.55, vis, 0.45, 0)  # clear red where the model predicts fascicle
        al = apo_lines(am)
        h, w = vis.shape[:2]
        ppm = float(r.scale_px_per_cm) / 10.0
        txt = [f"{r.ImageID}"]
        if al is not None:
            sup, deep = al
            for ln, col in [(sup, (255, 140, 0)), (deep, (0, 140, 255))]:
                cv2.line(vis, (0, int(ln[1])), (w - 1, int(ln[0] * (w - 1) + ln[1])), col, 2)
            segs, slopes, wts = fascicles(fm)
            for s, b, x0, x1 in segs:
                up = M.line_intersection((s, b), sup)
                lo = M.line_intersection((s, b), deep)
                if up is not None and lo is not None:
                    cv2.line(vis, (int(up[0]), int(up[1])), (int(lo[0]), int(lo[1])), (255, 255, 0), 1)
                cv2.line(vis, (x0, int(s * x0 + b)), (x1, int(s * x1 + b)), (0, 255, 255), 2)
            if slopes:
                fsl = float(np.median(slopes))            # representative fascicle direction
                xc = w / 2.0
                # walk from deep up to superficial along the fascicle slope -> the FL the identity uses
                yc_deep = deep[0] * xc + deep[1]
                pts = []
                x, y = xc, yc_deep
                for _ in range(4000):
                    if y <= sup[0] * x + sup[1] or not (0 <= x < w and 0 <= y < h):
                        break
                    y -= 1.0; x -= 1.0 / fsl if abs(fsl) > 1e-6 else 0
                    pts.append((x, y))
                if pts:
                    cv2.line(vis, (int(xc), int(yc_deep)), (int(pts[-1][0]), int(pts[-1][1])), (0, 255, 0), 2)
                geom = M.measure(am, fm)
                pa = geom["pa_deg"] or 0
                fl = (geom["mt_px"] / np.sin(np.radians(max(pa, 1)))) / ppm if pa else 0
                mt = geom["mt_px"] / ppm
                txt.append(f"OURS  PA {pa:.1f}  FL {fl:.0f}  MT {mt:.1f}")
        txt.append(f"EXPERT PA {r.pa_deg_true:.1f}  FL {r.fl_mm_true:.0f}  MT {r.mt_mm_true:.1f}")
        for k, t in enumerate(txt):
            y = 26 + 26 * k
            cv2.putText(vis, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{r.ImageID}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {len(truth)} overlays to {OUT}")
    print("apo: superficial=orange deep=blue | projected fragment spans=cyan | visible fits=yellow | representative FL=green | red=fasc mask")


if __name__ == "__main__":
    main()
