"""Experiment 05: iterate on the two failures, intelligently.

(A) Better PA: our angle uses np.polyfit (minimizes vertical residual -> biased for steep fascicles).
    Total-least-squares (PCA) orientation is unbiased. Also try length-weighting the fragments.
    exp02 showed PA precision bounds the straight FL, so a sharper PA helps FL too.
(B) Banded curved FL: instead of streamlines through the noisy IMAGE (exp04 failed), measure the
    fascicle angle in horizontal depth-bands from the cleaner MASK and integrate the path
    band-by-band: FL = sum over bands of dy / sin(theta_band). Captures depth-varying bend without
    wandering. (Straight identity = the special case of one band.)

Scored vs the 35 experts (TRUE scale). CPU. Needs results/seg_apo.pt and results/seg_fasc.pt.
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
    mt_px = abs((deep[0] * xc + deep[1]) - (sup[0] * xc + sup[1])) / np.sqrt(1 + deep[0] ** 2)
    return sup, deep, float(mt_px)


def pca_slope(xs, ys):
    p = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(float)
    cov = p @ p.T / max(len(xs), 1)
    w, v = np.linalg.eigh(cov)
    vx, vy = v[:, 1]
    return (vy / vx) if abs(vx) > 1e-6 else 1e6


def fragment_pa(fasc_mask, deep_s, mode):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fasc_mask, connectivity=8)
    angs, wts = [], []
    for i in range(1, n):
        if stats[i, 4] < 20:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8 or xs.max() - xs.min() < 1e-6:
            continue
        s = pca_slope(xs, ys) if mode in ("pca", "wpca") else np.polyfit(xs, ys, 1)[0]
        a = abs(np.degrees(np.arctan(s) - np.arctan(deep_s)))
        if a > 90:
            a = 180 - a
        if 2 <= a <= 75:
            angs.append(a); wts.append(len(xs))
    if not angs:
        return None
    angs, wts = np.array(angs), np.array(wts)
    if mode == "wpca":  # length-weighted median
        order = np.argsort(angs)
        c = np.cumsum(wts[order])
        return float(angs[order][np.searchsorted(c, c[-1] / 2)])
    return float(np.median(angs))


def banded_fl_px(fasc_mask, sup, deep, nbands=6):
    h, w = fasc_mask.shape
    xc = w / 2.0
    y_sup, y_deep = sup[0] * xc + sup[1], deep[0] * xc + deep[1]
    if y_deep - y_sup < 20:
        return None
    edges = np.linspace(y_sup, y_deep, nbands + 1)
    ys_all, xs_all = np.where(fasc_mask > 0)
    fl, ok = 0.0, 0
    for i in range(nbands):
        m = (ys_all >= edges[i]) & (ys_all < edges[i + 1])
        if m.sum() < 25 or xs_all[m].max() - xs_all[m].min() < 5:
            fl += (edges[i + 1] - edges[i]) / np.sin(np.radians(15.0))  # fallback angle
            continue
        s = pca_slope(xs_all[m], ys_all[m])
        th = max(np.radians(3), abs(np.arctan(s)))
        fl += (edges[i + 1] - edges[i]) / np.sin(th)
        ok += 1
    return fl if ok >= nbands * 0.5 else None


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    rec = []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        al = apo_lines(am)
        ppm = float(r.scale_px_per_cm) / 10.0
        row = dict(ImageID=r.ImageID, mt_mm=M.PRIOR["mt_mm"])
        if al is not None:
            sup, deep, mt_px = al
            row["mt_mm"] = mt_px / ppm
            for mode in ("poly", "pca", "wpca"):
                row[f"pa_{mode}"] = fragment_pa(fm, deep[0], mode)
            bfl = banded_fl_px(fm, sup, deep)
            row["banded_fl_mm"] = (bfl / ppm) if bfl else None
        rec.append(row)
    g = pd.DataFrame(rec)
    m = truth.merge(g, on="ImageID")

    def fl_term(fl):
        return float((np.clip(fl, 30, 200) - m["fl_mm_true"].values).__abs__().mean() / 12.0)

    def pa_term(pa):
        return float((np.clip(pa.fillna(M.PRIOR["pa_deg"]), 5, 45) - m["pa_deg_true"].values).__abs__().mean() / 6.0)

    def recenter(x):
        x = np.clip(x, 30, 200)
        return x * (m["fl_mm_true"].mean() / np.nanmean(x))

    print("(A) PA precision (MAE deg / PA-term):")
    for mode in ("poly", "pca", "wpca"):
        pa = m[f"pa_{mode}"]
        print(f"    {mode:5s}  MAE {(pa - m.pa_deg_true).abs().mean():.2f}   PA-term {pa_term(pa):.3f}")
    print("\n(B) FL-term (recentered) by method:")
    best_pa = m["pa_pca"].fillna(M.PRIOR["pa_deg"]).clip(5, 45)
    straight = m["mt_mm"] / np.sin(np.radians(best_pa))
    print(f"    straight identity (PCA PA):  {fl_term(recenter(straight)):.3f}")
    print(f"    banded curved FL:            {fl_term(recenter(m['banded_fl_mm'].fillna(straight))):.3f}")
    print("    refs: straight (polyfit PA) recentered 0.528 | good constant 0.682 | DL-Track 0.312")


if __name__ == "__main__":
    main()
