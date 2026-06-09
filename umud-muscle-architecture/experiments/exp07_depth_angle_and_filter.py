"""Experiment 07: test the user's visual observations against the experts.

(1) Bend: the user observes fascicles steepen toward the SUPERFICIAL aponeurosis and our straight
    line is too shallow. Test which depth's angle best matches the expert PA: all fragments (current),
    near-deep, near-superficial, or depth-weighted toward the top.
(2) Filter: the user notes faint near-horizontal mid-lines that are not fascicles. Test a stricter
    minimum angle to exclude them.

Scored vs the 35 experts (PA-term and the FL-term it implies). CPU.
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
    sup, deep = (lines[0][1], lines[0][2]), (lines[-1][1], lines[-1][2])
    xc = mask.shape[1] / 2.0
    mt = abs((deep[0] * xc + deep[1]) - (sup[0] * xc + sup[1])) / np.sqrt(1 + deep[0] ** 2)
    return sup, deep, float(mt)


def frags(fasc_mask, deep_s, min_ang):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fasc_mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, 4] < 20:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        s, _ = M.pca_line(ys, xs)
        a = abs(np.degrees(np.arctan(s) - np.arctan(deep_s)))
        if a > 90:
            a = 180 - a
        if min_ang <= a <= 75:
            out.append((a, float(ys.mean()), int(stats[i, 4])))
    return out


def wmed(vals, wts):
    if not vals:
        return None
    order = np.argsort(vals)
    v = np.asarray(vals, float)[order]
    c = np.cumsum(np.asarray(wts, float)[order])
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    rows = []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        al = apo_lines(am)
        row = dict(ImageID=r.ImageID, mt_mm=M.PRIOR["mt_mm"])
        if al is not None:
            sup, deep, mt_px = al
            row["mt_mm"] = mt_px / (float(r.scale_px_per_cm) / 10.0)
            xc = am.shape[1] / 2.0
            ysup, ydeep = sup[0] * xc + sup[1], deep[0] * xc + deep[1]
            fr = frags(fm, deep[0], 2.0)
            frs = frags(fm, deep[0], 6.0)  # stricter horizontal filter
            if fr:
                a, yc, wt = zip(*fr)
                a, yc, wt = np.array(a), np.array(yc), np.array(wt)
                depth = np.clip((yc - ysup) / max(ydeep - ysup, 1), 0, 1)  # 0=superficial,1=deep
                row["pa_all"] = wmed(list(a), list(wt))
                top = depth < 0.5
                row["pa_top"] = wmed(list(a[top]), list(wt[top])) if top.any() else row["pa_all"]
                row["pa_bot"] = wmed(list(a[~top]), list(wt[~top])) if (~top).any() else row["pa_all"]
                row["pa_wtop"] = wmed(list(a), list(wt * (1.0 - depth + 0.2)))  # weight toward superficial
            if frs:
                a2, _, w2 = zip(*frs)
                row["pa_strict"] = wmed(list(a2), list(w2))
        rows.append(row)
    m = truth.merge(pd.DataFrame(rows), on="ImageID")

    def pa_term(col):
        return float((np.clip(m[col].fillna(M.PRIOR["pa_deg"]), 5, 45) - m.pa_deg_true).abs().mean() / 6.0)

    def fl_term(col):
        pa = np.clip(m[col].fillna(M.PRIOR["pa_deg"]), 5, 45)
        fl = np.clip(m.mt_mm / np.sin(np.radians(pa)), 30, 200)
        fl = fl * (m.fl_mm_true.mean() / fl.mean())
        return float((fl - m.fl_mm_true).abs().mean() / 12.0)

    print("PA angle source        PA-term   FL-term   (current = pa_all)")
    for col in ["pa_all", "pa_top", "pa_bot", "pa_wtop", "pa_strict"]:
        if col in m:
            print(f"  {col:10s}            {pa_term(col):.3f}     {fl_term(col):.3f}")
    print("  refs: DL-Track PA-term 0.242 FL-term 0.312 | current best FL 0.476")


if __name__ == "__main__":
    main()
