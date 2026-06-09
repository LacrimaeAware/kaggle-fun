"""Experiment 04: real curved fascicle tracking (the proper version of ranked idea #5).

Instead of fitting one straight line, build a per-pixel fascicle ORIENTATION field (structure
tensor on the image, smoothed for coherence), then TRACE streamlines from the deep aponeurosis up
to the superficial one, following the local orientation. The streamline bends as the orientation
changes with depth, so its arc length captures fascicle curvature. FL = median streamline length.

Scored against the 35 experts with TRUE scale, vs the straight identity (0.528 recentered / 0.680
raw) and DL-Track (0.312). CPU. Needs results/seg_apo.pt and results/seg_fasc.pt.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402


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
    return (lines[0][1], lines[0][2]), (lines[-1][1], lines[-1][2])  # superficial(top), deep(bottom)


def orientation_field(gray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 6)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 6)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 6)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)      # dominant gradient orientation
    return theta + np.pi / 2.0                         # fibre direction (perpendicular to gradient)


def trace(fiber, x, y, sup, step=2.0, max_steps=900):
    h, w = fiber.shape
    length = 0.0
    for _ in range(max_steps):
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            break
        ang = fiber[iy, ix]
        dx, dy = np.cos(ang), np.sin(ang)
        if dy > 0:                      # always move upward (toward the superficial aponeurosis)
            dx, dy = -dx, -dy
        x += step * dx; y += step * dy; length += step
        if y <= sup[0] * x + sup[1]:    # reached the superficial aponeurosis
            return length
    return None                          # never reached it -> unreliable, drop


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    rows = []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        al = apo_lines(am)
        if al is None:
            rows.append(dict(image_id=r.ImageID, fl_mm=M.PRIOR["fl_mm"]))
            continue
        sup, deep = al
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fiber = orientation_field(gray)
        w = img.shape[1]
        lens = []
        for x0 in np.linspace(w * 0.2, w * 0.8, 25):       # seed along the deep aponeurosis
            y0 = deep[0] * x0 + deep[1] - 3
            L = trace(fiber, x0, y0, sup)
            if L is not None and L > 5:
                lens.append(L)
        if len(lens) < 5:
            rows.append(dict(image_id=r.ImageID, fl_mm=M.PRIOR["fl_mm"]))
            continue
        fl_px = float(np.median(lens))
        rows.append(dict(image_id=r.ImageID, fl_mm=fl_px / (float(r.scale_px_per_cm) / 10.0)))
    pred = pd.DataFrame(rows)

    def term(fl):
        return float((np.clip(fl, 30, 200) - truth["fl_mm_true"].values).__abs__().mean() / 12.0)

    tracked = pred["fl_mm"].values
    rec = tracked * (truth["fl_mm_true"].mean() / np.clip(tracked, 30, 200).mean())
    print("fascicle tracking (streamlines through the orientation field):")
    print(f"  tracked FL mean {np.mean(tracked):.1f} mm; FL-term raw {term(tracked):.3f}, "
          f"recentered {term(rec):.3f}")
    print("  references: straight MT/sin(PA) recentered 0.528 | good constant 0.682 | DL-Track 0.312")


if __name__ == "__main__":
    main()
