"""Test: measure MT between the aponeurosis bands' INNER (muscle-facing) edges instead of their
centroid lines. Hypothesis (from scale_sensitivity): our MT is ~5-7% too high because we fit through
the middle of each thick band. Compare centroid-MT vs inner-edge-MT vs true on the 35 experts.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
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


def fitline(xs, ys):
    if xs.max() - xs.min() < 1e-6:
        return 0.0, float(ys.mean())
    s, b = np.polyfit(xs, ys, 1)
    return float(s), float(b)


def band_lines(am):
    """Return (superficial_centroid, deep_centroid, superficial_inner, deep_inner) as (slope,intercept),
    or None. Inner edge: bottom of the top band, top of the bottom band (the muscle-facing sides)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(am, connectivity=8)
    bands = sorted([(stats[i, 4], i) for i in range(1, n)], reverse=True)[:2]
    if len(bands) < 2:
        return None
    info = []
    for _, i in bands:
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            return None
        info.append((float(ys.mean()), i, xs, ys))
    info.sort()  # by mean y: info[0]=superficial (top), info[1]=deep (bottom)
    out = {}
    for role, (_, i, xs, ys) in zip(("sup", "deep"), info):
        out[role + "_cent"] = fitline(xs, ys)
        # per-column inner edge
        edge_x, edge_y = [], []
        for x in np.unique(xs):
            yy = ys[xs == x]
            edge_x.append(x)
            edge_y.append(yy.max() if role == "sup" else yy.min())  # sup inner = bottom, deep inner = top
        out[role + "_inner"] = fitline(np.array(edge_x, float), np.array(edge_y, float))
    return out


def gap_px(sup, deep, w):
    xc = w / 2.0
    return abs((deep[0] * xc + deep[1]) - (sup[0] * xc + sup[1])) / np.sqrt(1 + deep[0] ** 2)


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    rows = []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = M.predict_mask(apo, img)
        bl = band_lines(am)
        ppm = float(r.scale_px_per_cm) / 10.0
        if bl is None:
            rows.append((r.mt_mm_true, M.PRIOR["mt_mm"], M.PRIOR["mt_mm"]))
            continue
        mt_cent = gap_px(bl["sup_cent"], bl["deep_cent"], am.shape[1]) / ppm
        mt_inner = gap_px(bl["sup_inner"], bl["deep_inner"], am.shape[1]) / ppm
        rows.append((r.mt_mm_true, mt_cent, mt_inner))
    a = np.array(rows, float)
    for name, col in (("centroid (current)", 1), ("inner-edge (new)", 2)):
        mt = np.clip(a[:, col], M.MT_MIN, M.MT_MAX)
        term = np.abs(mt - a[:, 0]).mean() / 3.0
        bias = (mt - a[:, 0]).mean()
        print(f"{name:22s} MT-term {term:.3f}  mean-bias {bias:+.2f}mm  (pred mean {mt.mean():.1f} vs true {a[:,0].mean():.1f})")


if __name__ == "__main__":
    main()
