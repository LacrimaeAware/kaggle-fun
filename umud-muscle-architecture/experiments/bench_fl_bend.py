"""Does the fascicle BEND actually improve FL, on real FL ground truth? The 35-expert benchmark has
measured FL + true scale. Fit the slope field once; trace it STRAIGHT (bend off) and CURVED (bend on)
from deep apo to sup apo; score both FL terms against the expert FL. Same everything, only the bend
differs -> isolates the bend's effect on the metric.

    python experiments/bench_fl_bend.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import benchmark_validate as BV        # noqa: E402
import segment_then_measure as M       # noqa: E402
import per_gap_viewer as PGV           # noqa: E402

TOL_FL = 12.0


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / f"results/seg_{t}.pt", map_location="cpu"))
    return m.eval()


def two_fls(am, fm, W):
    """Trace each fascicle straight (bend off) and curved (bend on); also a MINIMIZE-EXTRAPOLATION
    variant that keeps only fascicles that are mostly visible (visible/FL >= 0.25, the host's rule).
    Returns medians: straight-all, wave-all, straight-minextrap, wave-minextrap (px)."""
    bands = PGV.apo_bands(am)
    if len(bands) < 2:
        return None
    bt, bb = bands[0], bands[-1]
    sup = M.fit_line(bt["bot"], bt["ux"]); deep = M.fit_line(bb["top"], bb["ux"])
    nf, labf, st, _ = cv2.connectedComponentsWithStats(fm, 8)
    frags = []
    for i in range(1, nf):
        if st[i, 4] < M.FASC_MIN_AREA:
            continue
        ys, xs = np.where(labf == i)
        if len(xs) < 8:
            continue
        fs, _ = M.pca_line(ys, xs); cx, cy = float(xs.mean()), float(ys.mean())
        if M.line_y(sup, cx) <= cy <= M.line_y(deep, cx):
            frags.append((fs, cx, cy, int(st[i, 4]), M.fragment_visible_length(xs, ys, fs)))
    if len(frags) < 2:
        return None
    field = PGV._fit_slope_field(frags)
    straight = (field[0], 0.0, 0.0, field[3], field[4])
    s_all, w_all, s_me, w_me = [], [], [], []
    for fs, cx, cy, ar, vlen in frags:
        _, Ls = PGV._trace_wave(straight, sup, deep, cx)
        _, Lw = PGV._trace_wave(field, sup, deep, cx)
        if 10 <= Ls <= 2000:
            s_all.append(Ls)
            if vlen / Ls >= 0.25:
                s_me.append(Ls)
        if 10 <= Lw <= 2000:
            w_all.append(Lw)
            if vlen / Lw >= 0.25:
                w_me.append(Lw)
    md = lambda v: float(np.median(v)) if v else None
    return md(s_all), md(w_all), md(s_me), md(w_me)


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    rows = []
    for r in truth.itertuples():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        res = two_fls(am, fm, img.shape[1])
        ppm = float(r.scale_px_per_cm) / 10.0
        if res and all(v is not None for v in res):
            rows.append((float(r.fl_mm_true),) + tuple(v / ppm for v in res))
    a = np.array(rows, float)
    true = a[:, 0]
    cols = {"straight (all frags)": a[:, 1], "wave/bend (all frags)": a[:, 2],
            "straight + minextrap": a[:, 3], "wave/bend + minextrap": a[:, 4]}

    def term(pred, recenter):
        p = pred * (true.mean() / pred.mean()) if recenter else pred
        return float(np.abs(p - true).mean() / TOL_FL)
    print(f"n = {len(a)} images. expert FL mean {true.mean():.1f}mm. (production FL term here ~ 0.353)\n")
    print(f"{'method':24} {'FL mean':>9} {'term RAW':>10} {'term RECENTER':>14}")
    for name, pred in cols.items():
        print(f"{name:24} {pred.mean():>7.1f}mm {term(pred, False):>10.3f} {term(pred, True):>14.3f}")
    print(f"\nminimize-extrapolation effect (straight): {term(cols['straight (all frags)'], False):.3f} -> "
          f"{term(cols['straight + minextrap'], False):.3f}  (the off-screen overshoot fix)")
    print(f"bend effect WITH minextrap: {term(cols['straight + minextrap'], False):.3f} -> "
          f"{term(cols['wave/bend + minextrap'], False):.3f}  (negative move = bend helps on top)")


if __name__ == "__main__":
    main()
