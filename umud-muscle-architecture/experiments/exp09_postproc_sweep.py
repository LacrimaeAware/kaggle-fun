"""Experiment 09: sweep the fascicle post-processing on top of TTA - the cheapest FL lever.

Levers (no retraining, local CPU): binarization threshold, fascicle fragment min-area, and the
minimum-orientation filter that rejects near-horizontal (apo-parallel) fragments. TTA prob maps are
computed ONCE per image then the cheap post-proc is swept. Scored vs the 35 experts, recentered
identity FL. No submission.
"""

import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def _one(model, img, size):
    h, w = img.shape[:2]
    t = M.tf(False)(image=cv2.resize(img, (size, size)), mask=np.zeros((size, size), np.float32))
    with torch.no_grad():
        p = torch.sigmoid(model(t["image"].unsqueeze(0).to(M.DEVICE)))[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h))


def tta_prob(model, img):  # same ensemble as the wired UMUD_TTA
    fl = np.ascontiguousarray(img[:, ::-1])
    return (_one(model, img, M.IMG_SIZE) + _one(model, fl, M.IMG_SIZE)[:, ::-1] + _one(model, img, 448)) / 3.0


def apo_geom(am):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(am, connectivity=8)
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
    sup, deep = (lines[0][1], lines[0][2]), (lines[-1][1], lines[-1][2])
    xc = am.shape[1] / 2.0
    mt = abs((deep[0] * xc + deep[1]) - (sup[0] * xc + sup[1])) / np.sqrt(1 + deep[0] ** 2)
    return deep[0], float(mt)


def pa_from(fm, deep_s, min_area, min_ang):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fm, connectivity=8)
    vals, wts = [], []
    for i in range(1, n):
        if stats[i, 4] < min_area:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        s, _ = M.pca_line(ys, xs)
        a = abs(np.degrees(np.arctan(s) - np.arctan(deep_s)))
        a = 180 - a if a > 90 else a
        if min_ang <= a <= 75:
            vals.append(a); wts.append(int(stats[i, 4]))
    if not vals:
        return None
    order = np.argsort(vals)
    v = np.asarray(vals)[order]; c = np.cumsum(np.asarray(wts, float)[order])
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")

    cache = []  # (row_truth, deep_s, mt_mm, fasc_prob)
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = (tta_prob(apo, img) > 0.5).astype(np.uint8)
        g = apo_geom(am)
        fp = tta_prob(fasc, img)
        ppm = float(r.scale_px_per_cm) / 10.0
        if g is None:
            cache.append((r, None, M.PRIOR["mt_mm"], fp))
        else:
            cache.append((r, g[0], float(np.clip(g[1] / ppm, M.MT_MIN, M.MT_MAX)), fp))

    fl_true = truth["fl_mm_true"].mean()
    print(f"{'thr':>4} {'area':>4} {'ang':>4}   overall    pa     fl     mt")
    results = []
    for thr, area, ang in product([0.45, 0.50, 0.55], [20, 40, 60], [3.0, 6.0, 9.0]):
        recs = []
        for r, deep_s, mt_mm, fp in cache:
            pa = None
            if deep_s is not None:
                pa = pa_from((fp > thr).astype(np.uint8), deep_s, area, ang)
            pa = M.PRIOR["pa_deg"] if pa is None else float(np.clip(pa, M.PA_MIN, M.PA_MAX))
            fl = float(np.clip(mt_mm / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX))
            recs.append((r.pa_deg_true, r.fl_mm_true, r.mt_mm_true, pa, fl, mt_mm))
        a = np.array(recs, float)
        flv = a[:, 4] * (fl_true / a[:, 4].mean())  # recenter
        pa_t = np.abs(a[:, 3] - a[:, 0]).mean() / TOL["pa_deg"]
        fl_t = np.abs(flv - a[:, 1]).mean() / TOL["fl_mm"]
        mt_t = np.abs(a[:, 5] - a[:, 2]).mean() / TOL["mt_mm"]
        ov = (pa_t + fl_t + mt_t) / 3
        results.append((ov, thr, area, ang, pa_t, fl_t, mt_t))
    results.sort()
    for ov, thr, area, ang, pa_t, fl_t, mt_t in results:
        print(f"{thr:>4} {area:>4.0f} {ang:>4.0f}   {ov:.3f}  {pa_t:.3f}  {fl_t:.3f}  {mt_t:.3f}")
    print("\ncurrent wired (thr 0.5, area 20, ang 2): overall 0.370 (pa .171 fl .449 mt .490)")
    print("refs: human 0.307 | DL-Track 0.331")


if __name__ == "__main__":
    main()
