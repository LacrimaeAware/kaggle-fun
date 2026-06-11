"""Test each FL geometry method (and combinations) side by side on the 35-expert benchmark, true scale.
Reports per method: n, FL mean (bias vs expert), and FL term (raw + recentered). This is the apples-to-
apples comparison to find what gave zero bias and what broke it.

    python experiments/bench_fl_methods.py
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

TOL = 12.0


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / f"results/seg_{t}.pt", map_location="cpu"))
    return m.eval()


def fragment_fls(am, fm):
    """Straight fragment extrapolation to apo lines. (all-frags median, minimize-extrap median)."""
    bands = PGV.apo_bands(am)
    if len(bands) < 2:
        return None, None
    bt, bb = bands[0], bands[-1]
    sup = M.fit_line(bt["bot"], bt["ux"]); deep = M.fit_line(bb["top"], bb["ux"])
    nf, labf, st, _ = cv2.connectedComponentsWithStats(fm, 8)
    allf, me = [], []
    for i in range(1, nf):
        if st[i, 4] < M.FASC_MIN_AREA:
            continue
        ys, xs = np.where(labf == i)
        if len(xs) < 8:
            continue
        fs, _ = M.pca_line(ys, xs); cx, cy = float(xs.mean()), float(ys.mean()); b = cy - fs * cx
        up = M.line_intersection((fs, b), sup); lo = M.line_intersection((fs, b), deep)
        if up is None or lo is None:
            continue
        fl = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
        if not (10 <= fl <= 2000):
            continue
        if M.fragment_visible_length(xs, ys, fs) / fl >= 0.25:
            me.append(fl)
        allf.append(fl)
    return (float(np.median(allf)) if allf else None), (float(np.median(me)) if me else None)


def wave_fls(am, fm, W):
    """Per-gap wave trace. (all median, minimize-extrap median)."""
    bands = PGV.apo_bands(am)
    if len(bands) < 2:
        return None, None
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
        return None, None
    field = PGV._fit_slope_field(frags)
    allf, me = [], []
    for fs, cx, cy, ar, vlen in frags:
        _, L = PGV._trace_wave(field, sup, deep, cx)
        if 10 <= L <= 2000:
            allf.append(L)
            if vlen / L >= 0.25:
                me.append(L)
    return (float(np.median(allf)) if allf else None), (float(np.median(me)) if me else None)


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    methods = {k: [] for k in ["frag-all", "frag-minextrap", "facing(prod)", "wave-all", "wave-minextrap"]}
    expert = {k: [] for k in methods}
    for r in truth.itertuples():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        ppm = float(r.scale_px_per_cm) / 10.0; t = float(r.fl_mm_true)
        fa, fme = fragment_fls(am, fm)
        wa, wme = wave_fls(am, fm, img.shape[1])
        g = M.measure(am, fm); fac = g.get("fl_px") if g else None
        for k, px in [("frag-all", fa), ("frag-minextrap", fme), ("facing(prod)", fac),
                      ("wave-all", wa), ("wave-minextrap", wme)]:
            if px:
                methods[k].append(px / ppm); expert[k].append(t)

    def term(pred, true, recenter):
        pred = np.array(pred); true = np.array(true)
        p = pred * (true.mean() / pred.mean()) if recenter else pred
        return float(np.abs(p - true).mean() / TOL)
    print(f"expert FL mean {np.mean([float(r.fl_mm_true) for r in truth.itertuples()]):.1f}mm   tolerance 12mm\n")
    print(f"{'method':18} {'n':>3} {'FL mean':>9} {'bias':>7} {'term RAW':>9} {'term RECENTER':>13}")
    for k in methods:
        pred, true = methods[k], expert[k]
        if not pred:
            print(f"{k:18} {'0':>3}  (no values)"); continue
        bias = np.mean(pred) - np.mean(true)
        print(f"{k:18} {len(pred):>3} {np.mean(pred):>7.1f}mm {bias:>+6.1f} {term(pred, true, False):>9.3f} {term(pred, true, True):>13.3f}")
    print("\n(bias ~0 + low term = the good method. facing was our zero-bias build; see which still is.)")


if __name__ == "__main__":
    main()
