"""Prototype: handle MULTI-LEVEL images by assigning each fascicle to the gap it sits in, then
building a separate consensus angle + extrapolation PER GAP. Compare to the current naive single-pile
consensus. Draws top-gap fascicles in cyan, bottom-gap in magenta, apo lines in white. No production
change - this is a look-first prototype on the extreme two-level test images.

    python umud-muscle-architecture/experiments/per_gap_prototype.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import segment_then_measure as M  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT = ROOT / "results" / "per_gap"
NAMES = ["IMG_00127.tif", "IMG_00121.tif", "IMG_00116.tif", "IMG_00125.tif"]


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def apo_lines_all(am):
    """Fit a straight line to EVERY apo band (not just the 2 largest), sorted top->bottom."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(am, 8)
    lines = []
    for i in range(1, n):
        if stats[i, 4] < 200:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            continue
        s, b = np.polyfit(xs, ys, 1)
        lines.append((float(np.mean(ys)), float(s), float(b)))
    lines.sort()
    return [(s, b) for _, s, b in lines]


def main():
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        img = M.read_rgb(TEST / name)
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        lines = apo_lines_all(am)
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        W = img.shape[1]
        for s, b in lines:                                  # all apo lines = white
            cv2.line(vis, (0, int(b)), (W - 1, int(s * (W - 1) + b)), (255, 255, 255), 2)
        if len(lines) < 2:
            cv2.imwrite(str(OUT / f"pergap_{Path(name).stem}.jpg"), vis); continue
        # one gap per consecutive pair of apo lines
        gaps = [(lines[k], lines[k + 1]) for k in range(len(lines) - 1)]
        gap_frags = [[] for _ in gaps]
        nf, labf, statsf, _ = cv2.connectedComponentsWithStats(fm, 8)
        for i in range(1, nf):
            if statsf[i, 4] < M.FASC_MIN_AREA:
                continue
            ys, xs = np.where(labf == i)
            if len(xs) < 8:
                continue
            fs, fb = M.pca_line(ys, xs)
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            for gi, (top, dp) in enumerate(gaps):           # which gap is this fascicle's centroid in?
                yt = top[0] * cx + top[1]; yd = dp[0] * cx + dp[1]
                if yt <= cy <= yd:
                    gap_frags[gi].append((fs, cx, cy, int(statsf[i, 4]))); break
        colors = [(255, 220, 0), (255, 0, 200), (0, 200, 255), (0, 255, 0)]  # cyan, magenta, ...
        summary = []
        for gi, (top, dp) in enumerate(gaps):
            frs = gap_frags[gi]
            if len(frs) < 2:
                summary.append(f"gap{gi}:{len(frs)}frag"); continue
            ang = np.array([np.arctan(f[0]) for f in frs]); wt = np.array([f[3] for f in frs], float)
            cs = float(np.tan(float(M.weighted_median(ang, wt))))   # consensus angle for THIS gap only
            fls = []
            for fsl, cx, cy, ar in frs:
                bb = cy - cs * cx
                up = M.line_intersection((cs, bb), top); lo = M.line_intersection((cs, bb), dp)
                if up and lo:
                    fls.append(float(np.hypot(up[0] - lo[0], up[1] - lo[1])))
                    cv2.line(vis, (int(lo[0]), int(lo[1])), (int(up[0]), int(up[1])), colors[gi % 4], 2)
            summary.append(f"gap{gi}:{len(frs)}frag FL~{int(np.median(fls)) if fls else 0}px")
        lab = f"{name}  {len(lines)} apo bands -> {len(gaps)} gaps | " + "  ".join(summary)
        cv2.putText(vis, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 1)
        cv2.imwrite(str(OUT / f"pergap_{Path(name).stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{name}: {len(lines)} apo bands, {len(gaps)} gaps | " + "  ".join(summary))
    print("wrote per-gap overlays to", OUT)


if __name__ == "__main__":
    main()
